"""ArmoRM-Llama3-8B-v0.1 architecture without remote code (review finding F1).

The Hub checkpoint RLHFlow/ArmoRM-Llama3-8B-v0.1 ships its architecture as remote code (`modeling_custom.py`, written
for transformers 4.40) that imports `LLAMA_INPUTS_DOCSTRING` from `transformers.models.llama.modeling_llama`. That symbol
was removed before transformers 4.56 (present in 4.40 / 4.51, absent in 4.56 / 4.57 / 5.22), so
`AutoModelForSequenceClassification.from_pretrained(..., trust_remote_code=True)` fails with ImportError on every version
that requirements.txt allows (>= 4.56). This module re-implements the same architecture with the parameter names of the
checkpoint, so `ArmoRMForRewardWithGating.from_pretrained(model_id, ...)` loads the safetensors shards directly:

    model.*                      LlamaModel backbone (288 layer tensors + embed_tokens + norm)
    regression_layer.weight      bias-free linear head hidden -> num_objectives (19) on the last non-pad token
    gating.layers.{0..3}.*       gating MLP (3 x ReLU(Linear(hidden -> 1024)), Linear(1024 -> 19)) on the hidden state
    gating.logit_scale           of the last assistant-header token, softmax(x / temperature) * logit_scale
    reward_transform_matrix      fixed (19, 19) matrix; score = sum(gating * (rewards @ transform))

Verified against the checkpoint's model.safetensors.index.json (301 tensors; see tests/test_review_fixes.py). The forward
signature follows the original (the gating position is the last occurrence of the Llama-3 assistant header
"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n" in the input ids).
"""
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaModel, LlamaPreTrainedModel
from transformers.utils import ModelOutput

# tokenizer.encode("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False), Llama-3 tokenizer
GATING_TOKEN_PATTERN = [128009, 128006, 78191, 128007, 271]


class GatingNetwork(nn.Module):
    """n_hidden ReLU layers to hidden_dim, one linear layer to out_features, softmax(x / temperature) * logit_scale."""

    def __init__(self, in_features, out_features, bias=True, temperature=10.0, logit_scale=1.0, hidden_dim=1024, n_hidden=3):
        super().__init__()
        self.temperature = float(temperature)
        self.logit_scale = nn.Parameter(torch.ones(1) * logit_scale)
        layers = []
        for _ in range(int(n_hidden)):
            layers.append(nn.Linear(in_features, hidden_dim))
            in_features = hidden_dim
        layers.append(nn.Linear(in_features, out_features, bias=bias))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < len(self.layers) - 1 else layer(x)
        return F.softmax(x / self.temperature, dim=1) * self.logit_scale[0]


def find_gating_token(ids, pattern=GATING_TOKEN_PATTERN):
    """Index of the last occurrence of `pattern` (the assistant header) in a token id list."""
    ids, pattern = list(ids), list(pattern)
    n = len(pattern)
    for j in range(len(ids) - n, -1, -1):
        if ids[j:j + n] == pattern:
            return j
    raise ValueError("gating token pattern (Llama-3 assistant header) not found; ArmoRM inputs must come from the model's chat template")


@dataclass
class ArmoRMOutput(ModelOutput):
    rewards: torch.FloatTensor = None
    hidden_state: Optional[torch.FloatTensor] = None
    prompt_embedding: Optional[torch.FloatTensor] = None
    gating_output: Optional[torch.FloatTensor] = None
    score: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None


class ArmoRMForRewardWithGating(LlamaPreTrainedModel):
    """LlamaForRewardModelWithGating of RLHFlow/ArmoRM-Llama3-8B-v0.1, loadable on transformers >= 4.56 without remote code."""

    config_class = LlamaConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.num_objectives = int(getattr(config, "num_objectives", 19))
        self.regression_layer = nn.Linear(config.hidden_size, self.num_objectives, bias=False)
        eye = torch.zeros(self.num_objectives, self.num_objectives)
        eye[range(self.num_objectives), range(self.num_objectives)] = 1.0
        self.reward_transform_matrix = nn.Parameter(eye, requires_grad=False)
        self.gating = GatingNetwork(config.hidden_size, self.num_objectives, temperature=getattr(config, "gating_temperature", 10),
                                    hidden_dim=int(getattr(config, "gating_hidden_dim", 1024)), n_hidden=int(getattr(config, "gating_n_hidden", 3)))
        self.gating_token_pattern = list(getattr(config, "gating_token_pattern", None) or GATING_TOKEN_PATTERN)
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, inputs_embeds=None, **kwargs):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, inputs_embeds=inputs_embeds,
                         use_cache=False)
        hidden = out[0]
        batch = hidden.shape[0]
        rows = torch.arange(batch, device=hidden.device)
        pad = self.config.pad_token_id
        if input_ids is not None and pad is not None:
            # first pad token - 1, or the last column when the row holds no pad (modulo, as in the original)
            last = (torch.eq(input_ids, pad).int().argmax(-1) - 1) % input_ids.shape[-1]
        else:
            last = torch.full((batch,), hidden.shape[1] - 1, device=hidden.device)
        final = hidden[rows, last.to(hidden.device)]
        rewards = self.regression_layer(final)
        if input_ids is None:
            raise ValueError("ArmoRM needs input_ids (the gating position is found in the token ids)")
        gating_pos = torch.tensor([find_gating_token(ids.tolist(), self.gating_token_pattern) for ids in input_ids], device=hidden.device)
        prompt_embedding = hidden[rows, gating_pos]
        gating_output = self.gating(prompt_embedding)
        score = torch.sum(gating_output * (rewards @ self.reward_transform_matrix), dim=1)
        return ArmoRMOutput(rewards=rewards, hidden_state=final, prompt_embedding=prompt_embedding, gating_output=gating_output,
                            score=score, logits=score)
