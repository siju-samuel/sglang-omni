# SPDX-License-Identifier: Apache-2.0
"""Runtime accelerator probes for ``accelerator``-marked tests.

Probed in the test body, not at collection time, so the marker still assigns
the test to the accelerator CI job (see tests/README.md).
"""

from __future__ import annotations

import pytest
import torch


def require_cuda(min_devices: int = 1) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if torch.cuda.device_count() < min_devices:
        pytest.skip(f"requires {min_devices} visible CUDA devices")


def require_graph_device(min_devices: int = 1) -> torch.device:
    """The local accelerator that records device graphs, or skip.

    For a test that captures a graph through a platform backend rather than
    torch.cuda, so the same body covers CUDA, ROCm and XPU.
    """
    from sglang_omni.platforms import current_platform

    device = torch.device(current_platform.device_type, 0)
    module = torch.get_device_module(device)
    if not getattr(module, "is_available", lambda: False)():
        pytest.skip(f"{device.type} is unavailable")
    if current_platform.get_device_graph_backend(device) is None:
        pytest.skip(f"{device.type} names no device graph backend")
    if module.device_count() < min_devices:
        pytest.skip(f"requires {min_devices} visible {device.type} devices")
    return device
