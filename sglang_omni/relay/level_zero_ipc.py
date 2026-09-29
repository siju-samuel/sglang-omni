# SPDX-License-Identifier: Apache-2.0
"""Level-Zero IPC relay backed by a bounded sender-side XPU slot pool.

This is the Intel GPU device-to-device stage transport: a payload stages into the
sender's pool and the receiver pulls it straight out of that pool over the peer
mapping, so a GPU-to-GPU edge never round-trips through host memory the way the
shm relay does.

Unlike CUDA, an XPU event carries no interprocess handle, so the sender cannot
hand the receiver something to wait on. It therefore waits for its own staging
copy before it publishes the metadata that lets the receiver read those bytes.
"""
from __future__ import annotations

import asyncio
import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import torch

from sglang_omni.comm.data_ref import required
from sglang_omni.profiler.comm_trace import elapsed_ms as comm_trace_elapsed_ms
from sglang_omni.profiler.comm_trace import emit as comm_trace_emit
from sglang_omni.profiler.comm_trace import enabled as comm_trace_enabled
from sglang_omni.profiler.comm_trace import now_ns as comm_trace_now_ns
from sglang_omni.relay.base import Relay, RelayOperation, register_relay
from sglang_omni.relay.level_zero import (
    ExportedSegment,
    HandleRendezvous,
    LevelZeroContext,
    fetch_peer_segment,
    wrap_device_bytes,
)
from sglang_omni.relay.slot_pool import (
    ContiguousSlotAllocator,
    SlotAllocation,
    slots_for_size,
)

logger = logging.getLogger(__name__)

DEFAULT_WAIT_THREADS = 8
WAIT_THREADS_ENV = "SGLANG_OMNI_LEVEL_ZERO_IPC_WAIT_THREADS"
PEER_VISIBILITY_WARNED: set[tuple[int, int, int]] = set()

# A staging copy into the sender's own pool only waits on work the sender already
# queued, so exceeding this means the device stopped making progress.
STAGE_COPY_TIMEOUT_S = 30.0

DEFAULT_SLOT_SIZE_KB = 64
# The pool is a staging window for stage payloads, not a cache, and it is device
# memory that the model no longer gets. Intel cards in service here carry 24 GB, so
# the default stays small enough to be invisible next to the weights and the KV pool.
DEFAULT_POOL_SIZE_MB = 64


def wait_thread_count() -> int:
    value = os.getenv(WAIT_THREADS_ENV)
    if value is None:
        return DEFAULT_WAIT_THREADS
    else:
        pass
    threads = int(value)
    if threads <= 0:
        raise ValueError(f"{WAIT_THREADS_ENV} must be positive")
    else:
        pass
    return threads


def parse_device_index(device: str) -> int:
    if device.startswith("xpu:"):
        return int(device.split(":", 1)[1])
    else:
        return 0


def synchronize_xpu_event(event: torch.xpu.Event, device_index: int) -> None:
    with torch.xpu.device(device_index):
        event.synchronize()


async def wait_for_xpu_event(
    event: torch.xpu.Event,
    *,
    device_index: int,
    wait_executor: ThreadPoolExecutor,
    timeout: float,
) -> None:
    """Wait for an XPU event without blocking the asyncio event loop."""
    if event.query():
        return
    else:
        pass
    wait_future = asyncio.get_running_loop().run_in_executor(
        wait_executor,
        synchronize_xpu_event,
        event,
        device_index,
    )
    await asyncio.wait_for(wait_future, timeout=timeout)


def ensure_peer_access(source_device_index: int, destination_device_index: int) -> None:
    """Reject a peer copy the importing device cannot perform."""
    if source_device_index == destination_device_index:
        return
    else:
        pass
    device_count = torch.xpu.device_count()
    if source_device_index >= device_count:
        warn_key = (destination_device_index, source_device_index, device_count)
        if warn_key not in PEER_VISIBILITY_WARNED:
            PEER_VISIBILITY_WARNED.add(warn_key)
            logger.warning(
                f"level_zero_ipc source device {source_device_index} is outside this "
                f"receiver's visible range [0, {device_count}); peer-access "
                f"validation skipped. This is expected only when sender and receiver "
                f"use different ZE_AFFINITY_MASK namespaces."
            )
        else:
            pass
        return
    elif not torch.xpu.can_device_access_peer(
        destination_device_index, source_device_index
    ):
        raise RuntimeError(
            f"level_zero_ipc cross-GPU transfer requires peer access, but XPU "
            f"{destination_device_index} cannot access XPU {source_device_index}; "
            f"use the shm relay for host-staged transfer"
        )
    else:
        pass


@dataclass(frozen=True, kw_only=True)
class TransferPlacement:
    """Where one payload sits inside the sender's pool."""

    size: int
    offset: int
    slot_index: int
    slot_size: int
    num_slots: int


@dataclass(frozen=True, kw_only=True)
class PeerPoolReference:
    """How a receiver reaches the pool that holds a payload."""

    pool_id: str
    socket_name: str
    segment_offset: int
    pool_bytes: int
    source_device_index: int


def parse_metadata(
    metadata: dict[str, object],
) -> tuple[TransferPlacement, PeerPoolReference]:
    transfer_info = required(metadata, "transfer_info", dict)
    pool_info = required(metadata, "level_zero_ipc", dict)
    placement = TransferPlacement(
        size=required(transfer_info, "size", int),
        offset=required(transfer_info, "offset", int),
        slot_index=required(transfer_info, "slot_index", int),
        slot_size=required(transfer_info, "slot_size", int),
        num_slots=required(transfer_info, "num_slots", int),
    )
    reference = PeerPoolReference(
        pool_id=required(pool_info, "pool_id", str),
        socket_name=required(pool_info, "socket_name", str),
        segment_offset=required(pool_info, "segment_offset", int),
        pool_bytes=required(pool_info, "pool_bytes", int),
        source_device_index=required(pool_info, "source_device_index", int),
    )

    if placement.offset < 0:
        raise ValueError("level_zero_ipc transfer offset must be non-negative")
    elif placement.slot_size <= 0 or placement.num_slots <= 0:
        raise ValueError("level_zero_ipc slot_size and num_slots must be positive")
    elif placement.offset % placement.slot_size != 0:
        raise ValueError("level_zero_ipc transfer offset must be slot aligned")
    elif placement.num_slots < slots_for_size(placement.size, placement.slot_size):
        raise ValueError("level_zero_ipc num_slots is too small for transfer size")
    elif (
        placement.offset + placement.num_slots * placement.slot_size
        > reference.pool_bytes
    ):
        raise ValueError("level_zero_ipc allocation range exceeds pool size")
    else:
        pass
    return placement, reference


class LevelZeroIpcPutOperation(RelayOperation):
    """Sender-side handle; completion means the staged slots can be reused."""

    def __init__(
        self,
        metadata: dict[str, object],
        *,
        relay: LevelZeroIpcRelay,
        allocator: ContiguousSlotAllocator,
        placement: TransferPlacement,
        request_id: str | None,
    ) -> None:
        self.staged_metadata = metadata
        self.relay = relay
        self.allocator = allocator
        self.placement = placement
        self.request_id = request_id
        self.receiver_done = asyncio.get_running_loop().create_future()
        self.completed = False

    @property
    def metadata(self) -> dict[str, object]:
        return self.staged_metadata

    async def wait_for_completion(self, timeout: float = 30.0) -> None:
        if self.completed:
            return
        else:
            pass
        wait_start = comm_trace_now_ns()
        try:
            await asyncio.wait_for(self.receiver_done, timeout=timeout)
        except Exception as error:
            # The receiver may still be reading, so the slots stay reserved and the
            # whole relay fails rather than handing this range to another payload.
            self.completed = True
            self.relay.mark_failed(error)
            raise
        self.completed = True
        self.allocator.release(self.placement.offset, self.placement.num_slots)
        comm_trace_emit(
            "level_zero_ipc_put_wait_ack",
            request_id=self.request_id,
            slot_index=self.placement.slot_index,
            num_slots=self.placement.num_slots,
            bytes=self.placement.size,
            elapsed_ms=round(comm_trace_elapsed_ms(wait_start), 6),
        )

    def mark_receiver_done(self) -> None:
        if not self.receiver_done.done():
            self.receiver_done.set_result(None)
        else:
            pass

    def mark_receiver_failed(self, error: BaseException) -> None:
        if not self.receiver_done.done():
            self.receiver_done.set_exception(error)
        else:
            pass


class LevelZeroIpcGetOperation(RelayOperation):
    """Receiver-side handle; completion means the peer copy finished."""

    def __init__(
        self,
        *,
        event: torch.xpu.Event,
        peer_pool: torch.Tensor,
        placement: TransferPlacement,
        device_index: int,
        wait_executor: ThreadPoolExecutor,
        request_id: str | None,
    ) -> None:
        self.event = event
        self.peer_pool: torch.Tensor | None = peer_pool
        self.placement = placement
        self.device_index = device_index
        self.wait_executor = wait_executor
        self.request_id = request_id
        self.completed = False

    @property
    def metadata(self) -> None:
        return None

    async def wait_for_completion(self, timeout: float = 30.0) -> None:
        if self.completed:
            return
        else:
            pass
        wait_start = comm_trace_now_ns()
        try:
            await wait_for_xpu_event(
                self.event,
                device_index=self.device_index,
                wait_executor=self.wait_executor,
                timeout=timeout,
            )
        finally:
            self.completed = True
            self.peer_pool = None
        comm_trace_emit(
            "level_zero_ipc_get_wait_copy",
            request_id=self.request_id,
            slot_index=self.placement.slot_index,
            num_slots=self.placement.num_slots,
            bytes=self.placement.size,
            elapsed_ms=round(comm_trace_elapsed_ms(wait_start), 6),
        )


@dataclass(kw_only=True)
class PeerPool:
    """A peer pool this process has mapped, kept for the relay's lifetime."""

    peer_pointer: int
    pool_tensor: torch.Tensor


@register_relay("level_zero_ipc")
class LevelZeroIpcRelay(Relay):
    def __init__(
        self,
        engine_id: str,
        device: str = "xpu",
        slot_size_kb: int = DEFAULT_SLOT_SIZE_KB,
        pool_size_mb: int = DEFAULT_POOL_SIZE_MB,
    ) -> None:
        self.engine_id = engine_id
        if device == "cpu":
            raise ValueError(
                "level_zero_ipc relay requires an XPU device; got 'cpu'. Use the shm "
                "relay for host-memory stages."
            )
        else:
            pass
        self.device = device
        self.device_index = parse_device_index(device)
        self.slot_size = int(slot_size_kb) * 1024
        if self.slot_size <= 0:
            raise ValueError("level_zero_ipc slot_size_kb must be positive")
        else:
            pass
        requested_pool_size = int(pool_size_mb) * 1024 * 1024
        if requested_pool_size <= 0:
            raise ValueError("level_zero_ipc pool_size_mb must be positive")
        else:
            pass
        self.slot_count = requested_pool_size // self.slot_size
        if self.slot_count <= 0:
            raise ValueError("level_zero_ipc pool size must fit at least one slot")
        else:
            pass
        self.pool_size = self.slot_count * self.slot_size

        self.level_zero = LevelZeroContext()
        self.pool_tensor: torch.Tensor | None = None
        self.pool_id: str | None = None
        self.rendezvous: HandleRendezvous | None = None
        self.exported_segment: ExportedSegment | None = None
        self.segment_offset = 0
        self.peer_pools: dict[str, PeerPool] = {}
        self.allocator: ContiguousSlotAllocator | None = None
        self.failed_error: BaseException | None = None
        self.failed_event = asyncio.Event()
        self.wait_executor = ThreadPoolExecutor(
            max_workers=wait_thread_count(),
            thread_name_prefix=f"level-zero-ipc-wait-{engine_id}",
        )

    def local_pool_state(
        self,
    ) -> tuple[torch.Tensor, str, ContiguousSlotAllocator, HandleRendezvous]:
        """Allocate, export, and publish the sender pool on first use."""
        if self.pool_tensor is None:
            start = comm_trace_now_ns()
            device = torch.device(self.device)
            with torch.xpu.device(device):
                pool_tensor = torch.empty(
                    self.pool_size, dtype=torch.uint8, device=device
                )
            segment = self.level_zero.export_segment(pool_tensor.data_ptr())
            self.segment_offset = pool_tensor.data_ptr() - segment.base_pointer
            self.pool_id = f"{self.engine_id}:{os.getpid()}:{uuid.uuid4().hex}"
            self.exported_segment = segment
            self.rendezvous = HandleRendezvous(self.pool_id, segment)
            self.allocator = ContiguousSlotAllocator(
                slot_count=self.slot_count,
                slot_size=self.slot_size,
            )
            self.pool_tensor = pool_tensor
            logger.info(
                f"[{self.engine_id}] Allocated Level-Zero IPC pool: "
                f"{self.pool_size / 1024**2:.2f} MB on {self.device} "
                f"({self.slot_count} x {self.slot_size}B slots)"
            )
            comm_trace_emit(
                "level_zero_ipc_pool_alloc",
                engine_id=self.engine_id,
                device=self.device,
                slot_size=self.slot_size,
                slot_count=self.slot_count,
                total_pool_bytes=self.pool_size,
                elapsed_ms=round(comm_trace_elapsed_ms(start), 6),
            )
        else:
            pass

        pool_tensor = self.pool_tensor
        pool_id = self.pool_id
        allocator = self.allocator
        rendezvous = self.rendezvous
        if (
            pool_tensor is None
            or pool_id is None
            or allocator is None
            or rendezvous is None
        ):
            raise RuntimeError("level_zero_ipc local pool was not initialized")
        else:
            pass
        return pool_tensor, pool_id, allocator, rendezvous

    def mark_failed(self, error: BaseException) -> None:
        if self.failed_error is None:
            self.failed_error = error
            self.failed_event.set()
        else:
            pass

    def raise_if_failed(self) -> None:
        if self.failed_error is not None:
            raise RuntimeError("level_zero_ipc relay failed") from self.failed_error
        else:
            pass

    async def acquire_slots(
        self, allocator: ContiguousSlotAllocator, num_slots: int
    ) -> SlotAllocation:
        """Wait for pool space, giving up as soon as the relay fails."""
        self.raise_if_failed()
        acquire_task = asyncio.create_task(
            allocator.acquire_async(num_slots, capture_layout=comm_trace_enabled())
        )
        failure_task = asyncio.create_task(self.failed_event.wait())
        try:
            done, _pending = await asyncio.wait(
                {acquire_task, failure_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if failure_task in done:
                if acquire_task in done:
                    allocator.release(acquire_task.result().offset, num_slots)
                else:
                    pass
                self.raise_if_failed()
                raise RuntimeError("level_zero_ipc relay failed")
            else:
                pass
            allocation = acquire_task.result()
            try:
                self.raise_if_failed()
            except RuntimeError:
                allocator.release(allocation.offset, num_slots)
                raise
            return allocation
        finally:
            for task in (acquire_task, failure_task):
                if not task.done():
                    task.cancel()
                else:
                    pass

    async def put_async(
        self,
        tensor: torch.Tensor,
        request_id: str | None = None,
        dst_rank: int | None = None,
        receiver_id: str | None = None,
    ) -> LevelZeroIpcPutOperation:
        self.raise_if_failed()
        if tensor.device.type != "xpu":
            raise ValueError(
                f"level_zero_ipc relay can only transfer XPU tensors; got tensor on "
                f"{tensor.device}"
            )
        else:
            pass
        pool_tensor, pool_id, allocator, rendezvous = self.local_pool_state()
        flat = tensor.contiguous().view(torch.uint8).reshape(-1)
        size = int(flat.numel())
        num_slots = slots_for_size(size, self.slot_size)
        if num_slots > allocator.slot_count:
            raise ValueError(
                f"Tensor size {size} requires {num_slots} level_zero_ipc slots, but "
                f"pool has {allocator.slot_count}"
            )
        else:
            pass

        acquire_start = comm_trace_now_ns()
        allocation = await self.acquire_slots(allocator, num_slots)
        acquire_ms = comm_trace_elapsed_ms(acquire_start)
        offset = allocation.offset
        placement = TransferPlacement(
            size=size,
            offset=int(offset),
            slot_index=int(offset // self.slot_size),
            slot_size=self.slot_size,
            num_slots=num_slots,
        )

        copy_start = comm_trace_now_ns()
        device = torch.device(self.device)
        stream = torch.xpu.current_stream(device)
        staged_event = torch.xpu.Event()
        try:
            with torch.xpu.device(device), torch.xpu.stream(stream):
                pool_tensor[offset : offset + size].copy_(flat, non_blocking=True)
                staged_event.record(stream)
        except BaseException:
            allocator.release(offset, num_slots)
            raise

        # Without an interprocess XPU event the receiver has nothing to wait on, so
        # the bytes must reach the pool before its metadata goes out.
        try:
            await wait_for_xpu_event(
                staged_event,
                device_index=self.device_index,
                wait_executor=self.wait_executor,
                timeout=STAGE_COPY_TIMEOUT_S,
            )
        except BaseException as error:
            # The copy owns these slots until it drains, and it cannot be cancelled,
            # so the pool is no longer safe to hand out.
            self.mark_failed(error)
            raise
        copy_ms = comm_trace_elapsed_ms(copy_start)

        comm_trace_emit(
            "level_zero_ipc_put_async",
            request_id=request_id,
            engine_id=self.engine_id,
            device=self.device,
            receiver_id=receiver_id,
            bytes=size,
            slot_index=placement.slot_index,
            num_slots=num_slots,
            acquire_wait_rounds=allocation.wait_rounds,
            free_slots_before=allocation.free_slots_before,
            acquire_ms=round(acquire_ms, 6),
            stage_copy_ms=round(copy_ms, 6),
        )
        metadata: dict[str, object] = {
            "engine_id": self.engine_id,
            "transfer_info": {
                "size": size,
                "offset": placement.offset,
                "slot_index": placement.slot_index,
                "slot_size": self.slot_size,
                "num_slots": num_slots,
                "allocation_size": num_slots * self.slot_size,
            },
            "level_zero_ipc": {
                "pool_id": pool_id,
                "socket_name": rendezvous.socket_name,
                "segment_offset": int(self.segment_offset),
                "pool_bytes": int(self.pool_size),
                "source_device_index": self.device_index,
            },
        }
        return LevelZeroIpcPutOperation(
            metadata,
            relay=self,
            allocator=allocator,
            placement=placement,
            request_id=request_id,
        )

    def peer_pool_for(
        self,
        reference: PeerPoolReference,
        destination_device_index: int,
    ) -> torch.Tensor:
        """Map a sender's pool once and reuse that mapping for later payloads."""
        mapped = self.peer_pools.get(reference.pool_id)
        if mapped is not None:
            return mapped.pool_tensor
        else:
            pass
        handle_blob, file_descriptor = fetch_peer_segment(
            reference.socket_name, reference.pool_id
        )
        try:
            peer_pointer = self.level_zero.open_peer_segment(
                handle_blob, file_descriptor, destination_device_index
            )
        finally:
            # The driver holds its own reference to the mapping once it is open.
            os.close(file_descriptor)
        pool_tensor = wrap_device_bytes(
            peer_pointer + reference.segment_offset,
            reference.pool_bytes,
            destination_device_index,
        )
        self.peer_pools[reference.pool_id] = PeerPool(
            peer_pointer=peer_pointer,
            pool_tensor=pool_tensor,
        )
        logger.info(
            f"[{self.engine_id}] Mapped Level-Zero IPC pool {reference.pool_id} "
            f"from XPU {reference.source_device_index} onto XPU "
            f"{destination_device_index}"
        )
        return pool_tensor

    async def get_async(
        self,
        metadata: dict[str, object],
        dest_tensor: torch.Tensor,
        request_id: str | None = None,
    ) -> LevelZeroIpcGetOperation:
        if dest_tensor.device.type != "xpu":
            raise ValueError(
                f"level_zero_ipc relay can only receive into XPU tensors; dest is on "
                f"{dest_tensor.device}"
            )
        else:
            pass
        start = comm_trace_now_ns()
        placement, reference = parse_metadata(metadata)
        destination_device_index = int(dest_tensor.device.index or 0)
        ensure_peer_access(reference.source_device_index, destination_device_index)

        destination = dest_tensor.view(torch.uint8).reshape(-1)
        if destination.numel() < placement.size:
            raise ValueError(
                f"level_zero_ipc destination buffer has {destination.numel()} bytes, "
                f"but transfer requires {placement.size} bytes"
            )
        else:
            pass

        peer_pool = self.peer_pool_for(reference, destination_device_index)
        source = peer_pool[placement.offset : placement.offset + placement.size]
        stream = torch.xpu.current_stream(dest_tensor.device)
        event = torch.xpu.Event()
        with torch.xpu.device(dest_tensor.device), torch.xpu.stream(stream):
            destination[: placement.size].copy_(source, non_blocking=True)
            event.record(stream)
        comm_trace_emit(
            "level_zero_ipc_get_async",
            request_id=request_id,
            engine_id=self.engine_id,
            source_device=reference.source_device_index,
            destination_device=destination_device_index,
            bytes=placement.size,
            slot_index=placement.slot_index,
            num_slots=placement.num_slots,
            elapsed_ms=round(comm_trace_elapsed_ms(start), 6),
        )
        return LevelZeroIpcGetOperation(
            event=event,
            peer_pool=peer_pool,
            placement=placement,
            device_index=destination_device_index,
            wait_executor=self.wait_executor,
            request_id=request_id,
        )

    def cleanup(self, request_id: str) -> None:
        pass

    def close(self) -> None:
        for pool_id, mapped in self.peer_pools.items():
            self.level_zero.close_peer_segment(mapped.peer_pointer)
            logger.debug(f"[{self.engine_id}] Released peer pool {pool_id}")
        self.peer_pools.clear()
        if self.rendezvous is not None and self.exported_segment is not None:
            self.rendezvous.close()
            self.level_zero.release_segment(self.exported_segment)
            self.rendezvous = None
            self.exported_segment = None
        else:
            pass
        self.pool_tensor = None
        self.level_zero.destroy()
        self.wait_executor.shutdown(wait=False)
