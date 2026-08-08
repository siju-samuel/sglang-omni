# SPDX-License-Identifier: Apache-2.0
"""Encoder stages must land on the card their placement assigned."""

from __future__ import annotations

import torch

from sglang_omni.models.qwen3_omni.stages import _encoder_device
from sglang_omni.platforms import PlatformEnum, ResolvedPlatformSpec

_CUDA = ResolvedPlatformSpec(PlatformEnum.CUDA, "cuda", "nccl")
_XPU = ResolvedPlatformSpec(PlatformEnum.XPU, "xpu", "xccl")


def test_bare_literal_uses_the_assigned_gpu_id() -> None:
    """config.py declares device="cuda" with no index, so the card comes from the
    stage's gpu_id -- defaulting to 0 would put every encoder on card 0."""
    assert _encoder_device("cuda", _CUDA, 5) == torch.device("cuda", 5)
    assert _encoder_device("cuda", _XPU, 5) == torch.device("xpu", 5)


def test_index_survives_and_gpu_id_wins_over_it() -> None:
    assert _encoder_device("cuda:2", _XPU, None) == torch.device("xpu", 2)
    assert _encoder_device("cuda:2", _XPU, 6) == torch.device("xpu", 6)


def test_an_explicit_cpu_request_is_never_rerouted() -> None:
    assert _encoder_device("cpu", _XPU, 5) == torch.device("cpu")
