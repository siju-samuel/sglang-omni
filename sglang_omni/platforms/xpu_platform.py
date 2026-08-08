# SPDX-License-Identifier: Apache-2.0
"""Intel XPU device operations for the Omni platform layer.

Unlike ROCm -- which reuses ``CudaDeviceMixin`` because PyTorch exposes HIP through
``torch.cuda.*`` -- XPU has its own ``torch.xpu`` surface, so ``XpuDeviceMixin``
implements the device ops directly. Capability predicates keep the conservative
``OmniPlatform`` defaults (no same-device weight sharing, no cross-node transport):
both are unverified on XPU, and the defaults already say so.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, MutableMapping
from contextlib import suppress
from typing import Any

import torch

from sglang_omni.platforms.interface import (
    DeviceMixin,
    OmniPlatform,
    PlatformEnum,
    TransferPolicy,
)

logger = logging.getLogger(__name__)

_ZE_AFFINITY_MASK = "ZE_AFFINITY_MASK"  # cleared, never used to isolate ranks


class XpuDeviceMixin(DeviceMixin):
    _enum = PlatformEnum.XPU
    device_name = "xpu"
    device_type = "xpu"

    def device_count(self) -> int:
        return int(torch.xpu.device_count())

    def set_device(self, device: torch.device) -> None:
        # torch.xpu needs an explicit index; keep the type so a stray "cuda:0"
        # raises here instead of silently binding an XPU card.
        if device.type == "xpu":
            device = torch.device("xpu", device.index or 0)
        torch.xpu.set_device(device)

    def get_device_properties(self, device_id: int = 0) -> Any:
        return torch.xpu.get_device_properties(device_id)

    def synchronize(self) -> None:
        torch.xpu.synchronize()

    def empty_cache(self) -> None:
        torch.xpu.empty_cache()


class XpuOmniPlatform(XpuDeviceMixin, OmniPlatform):
    # None on purpose: ZE_AFFINITY_MASK would isolate a rank but hides its peers
    # and hangs XCCL discovery -- the opposite of CUDA_VISIBLE_DEVICES with NCCL.
    device_control_env_var = None
    distributed_backend = "xccl"

    def transfer_policy(self) -> TransferPolicy:
        return TransferPolicy.HOST_STAGED  # cuda_ipc is CUDA-only

    def inherited_visibility_count(self, env: Mapping[str, str]) -> int | None:
        """How many devices an inherited mask exposes, or None if unmasked.

        Read from the string, not ``device_count()``: querying the runtime would
        initialize it under the mask and freeze the truncated count.
        """
        mask = env.get(_ZE_AFFINITY_MASK)
        if not mask:
            return None
        return len([item for item in mask.split(",") if item.strip()])

    def clear_inherited_visibility(
        self, env: MutableMapping[str, str], *, log: Any = None
    ) -> str | None:
        inherited = env.pop(_ZE_AFFINITY_MASK, None)
        if inherited is not None:
            (log or logger).warning(
                "Cleared inherited %s=%s: XPU ranks address devices by id and "
                "need every card visible for XCCL discovery",
                _ZE_AFFINITY_MASK,
                inherited,
            )
        return inherited

    def reclaim_process_memory(
        self, device: torch.device, *, suppress_errors: bool = False
    ) -> None:
        self.set_device(device)
        if suppress_errors:
            with suppress(Exception):
                self.synchronize()
        else:
            self.synchronize()
        self.empty_cache()

    def get_available_memory(self, device_id: int = 0) -> tuple[int, int]:
        """Free/total bytes, floored by the allocator's reserved pool.

        Arc Pro B60's driver reports full capacity regardless of live allocations,
        and ``memory_allocated`` misses cached-but-freed blocks, so without the
        floor the KV pool is sized against headroom that does not exist.
        """
        free, total = (int(value) for value in torch.xpu.mem_get_info(device_id))
        reserved_floor = total - int(torch.xpu.memory_reserved(device_id))
        return min(free, max(reserved_floor, 0)), total
