# SPDX-License-Identifier: Apache-2.0
"""The torch profiler must record the build's own accelerator activity.

Asking for ``ProfilerActivity.CUDA`` on a non-CUDA build neither raises nor
warns -- the trace simply comes back with zero device kernels, so a profile
looks complete while every device timing is missing.
"""

from __future__ import annotations

import pytest
from torch.profiler import ProfilerActivity

from sglang_omni.profiler import torch_profiler


@pytest.mark.parametrize(
    ("supported", "expected"),
    [
        ({ProfilerActivity.CPU, ProfilerActivity.CUDA}, ["CUDA"]),
        ({ProfilerActivity.CPU, ProfilerActivity.XPU}, ["XPU"]),
        ({ProfilerActivity.CPU}, []),
    ],
)
def test_accelerator_activities_follow_the_build(
    monkeypatch: pytest.MonkeyPatch, supported: set, expected: list[str]
) -> None:
    monkeypatch.setattr(torch_profiler, "supported_activities", lambda: supported)

    activities = torch_profiler._accelerator_activities()

    assert [activity.name for activity in activities] == expected
