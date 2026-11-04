"""Mass-mean directions and steering vectors against the reference repo on identical synthetic features; unit norm,
sigma with ddof=1, the sum |lambda| <= 1 constraint and the npz round trip."""
import os
import sys

import numpy as np
import pytest
import torch

from valueheads_pythia.directions import (load_directions, mass_mean_directions, save_directions, sigma_along,
                                          steering_vectors, unit_directions)
from valueheads_pythia.hooks import HeadSteer
from valueheads_pythia.probing import group_by_layer

from test_probe_paper_mode import D, H, L, REPO, make_features, write_shards


@pytest.fixture(scope="module")
def feats(tmp_path_factory):
    rng = np.random.RandomState(11)
    base = tmp_path_factory.mktemp("dirs")
    out = {}
    for split, n in (("train", 200), ("val", 80)):
        a, y, g = make_features(rng, n)
        d = str(base / f"honesty_{split}")
        write_shards(d, a, y, g)
        out[split] = (d, a, y)
    return out


def test_mass_mean_matches_repo(feats):
    if not os.path.isdir(os.path.join(REPO, "valueheads")):
        pytest.skip("reference repo not found")
    sys.path.insert(0, REPO)
    import valueheads.directions as rd
    (tr, a_tr, y_tr), (va, a_va, y_va) = feats["train"], feats["val"]
    ref = rd.mass_mean_directions([a_tr, a_va], [y_tr, y_va], H)
    ours = mass_mean_directions([tr, va])
    assert ours.shape == ref.shape == (L * H, D) and ours.dtype == np.float32
    assert np.array_equal(ours, ref)
    ours_train = mass_mean_directions(tr)
    assert not np.array_equal(ours_train, ours)
    # steering vectors: same std set, same top heads
    heads = {"honesty": [1 * H + 2, 2 * H + 0, 5]}
    ref_vec = rd.steering_vectors(heads, {"honesty": ref}, {"honesty": 1.0}, a_va, H, D)
    our_vec = steering_vectors(heads, {"honesty": ours}, {"honesty": 1.0}, va, H, D)
    assert set(ref_vec) == set(our_vec) == set(group_by_layer(heads["honesty"], H))
    for layer in ref_vec:
        assert torch.allclose(ref_vec[layer], our_vec[layer], atol=1e-6), layer


def test_unit_norm_sigma_and_lambda(feats):
    tr, va = feats["train"][0], feats["val"][0]
    dirs = mass_mean_directions(tr)
    unit = unit_directions(dirs)
    assert np.allclose(np.linalg.norm(unit, axis=1), 1, atol=1e-6)
    sig = sigma_along(dirs, va)
    assert sig.shape == (L * H,) and np.all(sig > 0)
    # ddof=1 and projection onto the unit direction, checked by hand for one head
    from valueheads_pythia.extract import load_features
    acts = load_features(va)[0].head(1, 2)
    assert np.isclose(sig[1 * H + 2], np.std(acts @ unit[1 * H + 2], ddof=1), atol=1e-6)
    assert np.isnan(sigma_along(dirs, va, heads=[3])[0]) and not np.isnan(sigma_along(dirs, va, heads=[3])[3])
    # multi-value composition: lambda constraint enforced, composite = normalized weighted sum, sigma along it
    heads = {"a": [6], "b": [6, 7]}
    dirs_b = -dirs
    vec = steering_vectors(heads, {"a": dirs, "b": dirs_b}, {"a": 0.75, "b": 0.25}, va, H, D)
    comp = 0.75 * dirs[6] + 0.25 * dirs_b[6]
    comp = comp / np.linalg.norm(comp)
    layer, head = divmod(6, H)
    expected = np.std(load_features(va)[0].head(layer, head) @ comp, ddof=1) * comp
    assert np.allclose(vec[layer][head * D:(head + 1) * D].numpy(), expected, atol=1e-5)
    assert vec[layer][:head * D].abs().sum() == 0
    with pytest.raises(AssertionError):
        steering_vectors(heads, {"a": dirs, "b": dirs_b}, {"a": 0.8, "b": 0.3}, va, H, D)
    with pytest.raises(ValueError):
        steering_vectors(heads, {"a": dirs}, {"a": 1.0}, va, H, D)
    # zero-weight values are ignored
    assert set(steering_vectors(heads, {"a": dirs, "b": dirs_b}, {"a": 1.0, "b": 0.0}, va, H, D)) == {layer}


def test_npz_roundtrip_and_hook_shapes(feats, tmp_path):
    tr, va = feats["train"][0], feats["val"][0]
    dirs = mass_mean_directions(tr)
    path = save_directions(str(tmp_path / "d.npz"), dirs, {"honesty": sigma_along(dirs, va)}, {"value": "honesty"})
    d2, sig, meta = load_directions(path)
    assert np.array_equal(d2, dirs) and set(sig) == {"honesty"} and meta["value"] == "honesty"
    # vectors plug into HeadSteer of a model with the same H * D width (tiny Llama with 4 heads x 8 dims)
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(vocab_size=64, hidden_size=H * D, intermediate_size=64, num_hidden_layers=L, num_attention_heads=H,
                      num_key_value_heads=H, max_position_embeddings=32)
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg).eval()
    vec = steering_vectors({"honesty": [1 * H + 2]}, {"honesty": dirs}, {"honesty": 1.0}, va, H, D)
    ids = torch.randint(0, 64, (1, 6))
    with torch.no_grad():
        base = model(input_ids=ids).logits
        with HeadSteer(model, vec, alpha=1.0, positions="all"):
            steered = model(input_ids=ids).logits
    assert (base - steered).abs().max() > 0
