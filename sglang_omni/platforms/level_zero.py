# SPDX-License-Identifier: Apache-2.0
"""Level-Zero IPC for Intel GPUs, provided by sglang.srt.utils.xpu_ipc_utils.

The sglang pinned here may predate that module. Then every name below is None
and the probe answers False, so the XPU platform never picks the level_zero_ipc
relay and every stage edge keeps the shm path.
"""
from __future__ import annotations

import logging

try:
    from sglang.srt.utils.xpu_ipc_utils import (
        LevelZeroIpcContext,
        XpuIpcExportedPool,
        XpuIpcPeerPools,
        XpuIpcPoolReference,
        ensure_xpu_peer_access,
        is_xpu_ipc_available,
        wait_for_xpu_event,
    )
except ImportError:
    LevelZeroIpcContext = None
    XpuIpcExportedPool = None
    XpuIpcPeerPools = None
    XpuIpcPoolReference = None
    ensure_xpu_peer_access = None
    wait_for_xpu_event = None

    def is_xpu_ipc_available() -> bool:
        logging.getLogger(__name__).info(
            "Level-Zero IPC needs sglang.srt.utils.xpu_ipc_utils, which the "
            "installed sglang does not have"
        )
        return False
