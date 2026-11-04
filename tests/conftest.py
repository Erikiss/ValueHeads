"""Shared fixtures: pythia-160m step143000 in float32 / eager on the CPU (DESIGN 3.1 numerics policy).

Run with the HF cache configured, e.g.
    HF_HOME=<cache> python -m pytest tests -q
"""
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

torch.set_num_threads(min(4, os.cpu_count() or 1))

TEST_MODEL = os.environ.get("VH_TEST_MODEL", "pythia-160m")
TEST_REVISION = os.environ.get("VH_TEST_REVISION", "step143000")


@pytest.fixture(scope="session")
def model_tok():
    from valueheads_pythia.models import load_model
    return load_model(TEST_MODEL, revision=TEST_REVISION, device="cpu", dtype="float32", attn_impl="eager")
