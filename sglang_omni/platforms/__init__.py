# SPDX-License-Identifier: Apache-2.0
"""SGLang-Omni platform abstraction."""

from __future__ import annotations

from typing import Any

import torch

from sglang_omni.platforms.cuda_platform import CudaDeviceMixin, CudaOmniPlatform
from sglang_omni.platforms.interface import (
    CpuDeviceMixin,
    CpuOmniPlatform,
    DeviceMixin,
    OmniPlatform,
    PlatformEnum,
    ResolvedPlatformSpec,
    TransferPolicy,
)
from sglang_omni.platforms.xpu_platform import XpuDeviceMixin, XpuOmniPlatform


def resolve_current_platform(torch_module: Any = torch) -> OmniPlatform:
    return OmniPlatform.detect(torch_module)


__all__ = [
    "CpuDeviceMixin",
    "CpuOmniPlatform",
    "CudaDeviceMixin",
    "CudaOmniPlatform",
    "DeviceMixin",
    "OmniPlatform",
    "PlatformEnum",
    "ResolvedPlatformSpec",
    "TransferPolicy",
    "XpuDeviceMixin",
    "XpuOmniPlatform",
    "resolve_current_platform",
]
