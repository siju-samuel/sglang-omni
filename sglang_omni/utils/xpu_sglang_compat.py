# SPDX-License-Identifier: Apache-2.0
"""XPU corrections to SGLang runtime helpers that assume CUDA semantics."""

from __future__ import annotations

import logging
import sys
import threading

import torch

logger = logging.getLogger(__name__)

_PATCH_LOCK = threading.Lock()
_MEM_PATCHED = False


def patch_available_gpu_memory_for_xpu() -> bool:
    """Cap XPU free memory at ``total - memory_reserved`` before the KV pool is sized.

    SGLang caps at ``total - memory_allocated``, which counts the allocator's
    cached-but-freed blocks as free, and the driver cannot catch it: Arc Pro B60's
    ``mem_get_info`` always reports full capacity (measured 17.91 GiB against 13.91
    GiB real), so the pool over-commits. No-op off XPU; idempotent; thread-safe.
    """
    global _MEM_PATCHED
    if _MEM_PATCHED:
        return True
    if not torch.xpu.is_available():
        return False
    with _PATCH_LOCK:
        if _MEM_PATCHED:  # re-check under lock
            return True
        try:
            from sglang.srt.utils import common as _common
        except Exception:  # noqa: BLE001 - layout drift must not crash startup
            return False

        _orig = _common.get_available_gpu_memory

        def get_available_gpu_memory(  # noqa: ANN001, ANN201
            device, gpu_id, distributed=False, empty_cache=True, cpu_group=None
        ):
            # Accept torch.device or "xpu:0" alike.
            dev_type = getattr(device, "type", None) or str(device).split(":", 1)[0]
            if dev_type != "xpu":
                return _orig(
                    device,
                    gpu_id,
                    distributed=distributed,
                    empty_cache=empty_cache,
                    cpu_group=cpu_group,
                )
            if empty_cache:
                torch.xpu.empty_cache()
            free, total = torch.xpu.mem_get_info(gpu_id)
            free = min(free, total - torch.xpu.memory_reserved(gpu_id))
            if distributed:  # match the CUDA path: min free across the group
                tensor = torch.tensor(float(free), dtype=torch.float32)
                torch.distributed.all_reduce(
                    tensor, op=torch.distributed.ReduceOp.MIN, group=cpu_group
                )
                free = tensor.item()
            return free / (1 << 30)  # SGLang's callers expect GB

        # Rebind the source module and every module that imported it by value.
        _common.get_available_gpu_memory = get_available_gpu_memory
        for mod in list(sys.modules.values()):
            try:
                if getattr(mod, "get_available_gpu_memory", None) is _orig:
                    mod.get_available_gpu_memory = get_available_gpu_memory
            except Exception:  # noqa: BLE001 - lazy-import shims raise on getattr
                continue

        _MEM_PATCHED = True
    logger.info("Intel XPU: capped get_available_gpu_memory at total - reserved")
    return True


__all__ = ["patch_available_gpu_memory_for_xpu"]
