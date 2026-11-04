# SPDX-License-Identifier: Apache-2.0
"""Level-Zero weight sharing: one exportable pool per engine, aliased by followers."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass

import torch

from sglang_omni.platforms.level_zero import (
    ExportedSegment,
    LevelZeroContext,
    LevelZeroError,
)
from sglang_omni.platforms.xpu_dlpack import wrap_device_bytes
from sglang_omni.relay.level_zero_rendezvous import HandleRendezvous, fetch_peer_segment

_MANIFEST_VERSION = 1
# note (siju): every dtype view into the pool needs its own alignment, and 256
# covers each of them while keeping each weight on a cache-line boundary.
_POOL_ALIGNMENT_BYTES = 256
_DTYPES: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "uint8": torch.uint8,
    "bool": torch.bool,
}
_DTYPE_NAMES: dict[torch.dtype, str] = {v: k for k, v in _DTYPES.items()}
# Held for the process lifetime: the leader must keep serving its handle and the
# follower must keep its mapping open for as long as the aliased weights live.
_EXPORTED_POOLS: list[ExportedWeightPool] = []
_IMPORTED_POOLS: list[ImportedWeightPool] = []


@dataclass(frozen=True, kw_only=True)
class PooledTensor:
    """Where one weight sits inside the exported pool."""

    name: str
    offset: int
    num_bytes: int
    shape: tuple[int, ...]
    dtype: torch.dtype


@dataclass(kw_only=True)
class ExportedWeightPool:
    """The leader's exported pool and the rendezvous answering handle requests."""

    context: LevelZeroContext
    device_pointer: int
    segment: ExportedSegment
    rendezvous: HandleRendezvous


@dataclass(kw_only=True)
class ImportedWeightPool:
    """A follower's mapping of the leader's pool."""

    context: LevelZeroContext
    peer_pointer: int


def aligned_length(num_bytes: int) -> int:
    """Round a weight's size up to the pool's alignment."""
    blocks = (num_bytes + _POOL_ALIGNMENT_BYTES - 1) // _POOL_ALIGNMENT_BYTES
    return blocks * _POOL_ALIGNMENT_BYTES


def plan_pool(tensors: dict[str, torch.Tensor]) -> tuple[list[PooledTensor], int]:
    """Lay the weights out in name order so both sides agree on every offset."""
    layout: list[PooledTensor] = []
    offset = 0
    for name in sorted(tensors):
        tensor = tensors[name]
        if tensor.dtype not in _DTYPE_NAMES:
            raise LevelZeroError(
                f"weight {name!r} has dtype {tensor.dtype}, which Level-Zero weight "
                f"sharing cannot pool; supported: {', '.join(sorted(_DTYPES))}"
            )
        else:
            pass
        num_bytes = tensor.numel() * tensor.element_size()
        layout.append(
            PooledTensor(
                name=name,
                offset=offset,
                num_bytes=num_bytes,
                shape=tuple(tensor.shape),
                dtype=tensor.dtype,
            )
        )
        offset += aligned_length(num_bytes)
    return layout, offset


def pool_device_index(tensors: dict[str, torch.Tensor]) -> int:
    """The one XPU card every weight must already live on."""
    indices = {t.device.index for t in tensors.values() if t.device.type == "xpu"}
    if len(indices) != 1 or len(indices) != len({t.device for t in tensors.values()}):
        devices = ", ".join(sorted({str(t.device) for t in tensors.values()}))
        raise LevelZeroError(
            f"Level-Zero weight sharing needs every weight on one XPU card, got "
            f"{devices}"
        )
    else:
        pass
    return int(next(iter(indices)))


def slice_pool(pool: torch.Tensor, entry: PooledTensor) -> torch.Tensor:
    """View one weight's bytes in the pool as its own dtype and shape."""
    window = pool[entry.offset : entry.offset + entry.num_bytes]
    return window.view(entry.dtype).view(entry.shape)


class XpuLevelZeroSerializer:
    """Weight-share (de)serialization over Level-Zero IPC.

    torch exposes no XPU IPC, so the leader copies its shareable weights into one
    exportable Level-Zero allocation, rebinds its own parameters onto that pool so
    it holds a single copy, and serves the pool's descriptor. A follower maps the
    pool once and slices every weight out of it, aliasing the leader's memory.
    """

    @staticmethod
    def serialize(obj: dict[str, torch.Tensor]) -> bytes:
        if not obj:
            return json.dumps({"version": _MANIFEST_VERSION, "tensors": []}).encode()
        else:
            pass
        device_index = pool_device_index(obj)
        layout, pool_bytes = plan_pool(obj)

        context = LevelZeroContext()
        device_pointer = context.allocate_exportable(device_index, pool_bytes)
        pool = wrap_device_bytes(device_pointer, pool_bytes, device_index)
        for entry in layout:
            target = slice_pool(pool, entry)
            target.copy_(obj[entry.name])
            # Rebinding through the Parameter frees the leader's own copy and
            # reaches every module holding a tied parameter.
            obj[entry.name].data = target
        torch.xpu.synchronize(device_index)
        # note (siju): the rebind only drops torch's references, and its caching
        # allocator keeps those blocks, so the card still holds both copies until
        # they are returned to the driver. Measured 2.09x the weight mass without
        # this, 1.09x with it.
        torch.xpu.empty_cache()

        segment = context.export_segment(device_pointer)
        pool_id = f"weights:{os.getpid()}:{uuid.uuid4().hex}"
        rendezvous = HandleRendezvous(pool_id, segment)
        _EXPORTED_POOLS.append(
            ExportedWeightPool(
                context=context,
                device_pointer=device_pointer,
                segment=segment,
                rendezvous=rendezvous,
            )
        )
        return json.dumps(
            {
                "version": _MANIFEST_VERSION,
                "socket_name": rendezvous.socket_name,
                "pool_id": pool_id,
                "device_index": device_index,
                "pool_bytes": pool_bytes,
                "tensors": [
                    {
                        "name": entry.name,
                        "offset": entry.offset,
                        "num_bytes": entry.num_bytes,
                        "shape": list(entry.shape),
                        "dtype": _DTYPE_NAMES[entry.dtype],
                    }
                    for entry in layout
                ],
            }
        ).encode()

    @staticmethod
    def deserialize(data: bytes | bytearray) -> dict[str, torch.Tensor]:
        manifest = json.loads(bytes(data).decode())
        version = manifest["version"]
        if version != _MANIFEST_VERSION:
            raise LevelZeroError(
                f"Level-Zero weight-share manifest version {version} cannot be "
                f"attached by a follower that writes version {_MANIFEST_VERSION}"
            )
        else:
            pass
        entries = [
            PooledTensor(
                name=record["name"],
                offset=record["offset"],
                num_bytes=record["num_bytes"],
                shape=tuple(record["shape"]),
                dtype=_DTYPES[record["dtype"]],
            )
            for record in manifest["tensors"]
        ]
        if not entries:
            return {}
        else:
            pass

        device_index = manifest["device_index"]
        current_index = torch.xpu.current_device()
        if device_index != current_index:
            # A follower on another card would alias memory it cannot read, so
            # refuse here rather than fail later inside a kernel.
            raise LevelZeroError(
                f"Level-Zero weight sharing published xpu:{device_index} but this "
                f"follower runs on xpu:{current_index}; replicas must share one card"
            )
        else:
            pass

        context = LevelZeroContext()
        handle_blob, descriptor = fetch_peer_segment(
            manifest["socket_name"], manifest["pool_id"]
        )
        peer_pointer = context.open_peer_segment(handle_blob, descriptor, device_index)
        _IMPORTED_POOLS.append(
            ImportedWeightPool(context=context, peer_pointer=peer_pointer)
        )
        pool = wrap_device_bytes(peer_pointer, manifest["pool_bytes"], device_index)
        return {entry.name: slice_pool(pool, entry) for entry in entries}
