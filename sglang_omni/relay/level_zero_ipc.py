# SPDX-License-Identifier: Apache-2.0
"""Level-Zero IPC relay: card-to-card stage transfers out of a sender slot pool."""
from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor

import torch

from sglang_omni.platforms.level_zero import (
    LevelZeroIpcContext,
    XpuIpcExportedPool,
    XpuIpcPeerPools,
    XpuIpcPoolReference,
    ensure_xpu_peer_access,
    wait_for_xpu_event,
)
from sglang_omni.profiler.comm_trace import elapsed_ms as comm_trace_elapsed_ms
from sglang_omni.profiler.comm_trace import emit as comm_trace_emit
from sglang_omni.profiler.comm_trace import enabled as comm_trace_enabled
from sglang_omni.profiler.comm_trace import now_ns as comm_trace_now_ns
from sglang_omni.relay.base import Relay, register_relay
from sglang_omni.relay.cuda_ipc import (
    ContiguousSlotAllocator,
    SlotAllocation,
    slots_for_size,
)
from sglang_omni.relay.level_zero_ipc_metadata import (
    LevelZeroIpcMetadata,
    PeerPoolReference,
    TransferPlacement,
    parse_metadata,
)
from sglang_omni.relay.level_zero_ipc_operations import (
    LevelZeroIpcGetOperation,
    LevelZeroIpcPutOperation,
)

logger = logging.getLogger(__name__)

DEFAULT_WAIT_THREADS = 8
WAIT_THREADS_ENV = "SGLANG_OMNI_LEVEL_ZERO_IPC_WAIT_THREADS"

# The staging copy waits only on the sender's own work, so this means a stall.
STAGE_COPY_TIMEOUT_S = 30.0


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


@register_relay("level_zero_ipc")
class LevelZeroIpcRelay(Relay):
    def __init__(
        self,
        engine_id: str,
        device: str,
        slot_size_kb: int,
        pool_size_mb: int,
    ) -> None:
        self.engine_id = engine_id
        parsed_device = torch.device(device)
        if parsed_device.type != "xpu":
            raise ValueError(
                f"level_zero_ipc relay requires an XPU device; got {device!r}. Use the "
                f"shm relay for host-memory stages."
            )
        else:
            pass
        self.device = device
        self.device_index = 0 if parsed_device.index is None else parsed_device.index
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
        # Read first, since nothing destroys a context that a later check abandons.
        wait_threads = wait_thread_count()

        if LevelZeroIpcContext is None:
            raise RuntimeError(
                "level_zero_ipc needs sglang.srt.utils.xpu_ipc_utils, which the "
                "installed sglang does not have"
            )
        else:
            pass
        self.level_zero = LevelZeroIpcContext()
        self.exported_pool: XpuIpcExportedPool | None = None
        self.allocator: ContiguousSlotAllocator | None = None
        self.failed_error: BaseException | None = None
        self.failed_event = asyncio.Event()
        self.wait_executor = ThreadPoolExecutor(
            max_workers=wait_threads,
            thread_name_prefix=f"level-zero-ipc-wait-{engine_id}",
        )
        self.peer_pools = XpuIpcPeerPools(self.level_zero)
        # Each sender pool is mapped once, by one worker, whoever asks first.
        self.peer_mappings: dict[str, Future[torch.Tensor]] = {}

    def local_pool_state(
        self,
    ) -> tuple[torch.Tensor, XpuIpcPoolReference, ContiguousSlotAllocator]:
        """Allocate, export, and publish the sender pool on first use."""
        if self.exported_pool is None:
            start = comm_trace_now_ns()
            allocator = ContiguousSlotAllocator(
                slot_count=self.slot_count,
                slot_size=self.slot_size,
            )
            # A pool that fails to come up frees itself, and the next send retries.
            self.exported_pool = XpuIpcExportedPool(
                self.level_zero, self.device_index, self.pool_size
            )
            self.allocator = allocator
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
        pool = self.exported_pool
        if pool.tensor is None or self.allocator is None:
            raise RuntimeError("level_zero_ipc local pool is closed")
        else:
            pass
        return pool.tensor, pool.reference, self.allocator

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
        handed_over = False
        try:
            await asyncio.wait(
                {acquire_task, failure_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            self.raise_if_failed()
            allocation = acquire_task.result()
            handed_over = True
            return allocation
        finally:
            for task in (acquire_task, failure_task):
                if not task.done():
                    task.cancel()
                else:
                    pass
            # Slots granted to a caller that failed or was cancelled go back.
            if (
                not handed_over
                and acquire_task.done()
                and not acquire_task.cancelled()
                and acquire_task.exception() is None
            ):
                allocator.release(acquire_task.result().offset, num_slots)
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
        pool_tensor, pool_reference, allocator = self.local_pool_state()
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

        # Made before the slots are taken, since neither touches the pool.
        device = torch.device(self.device)
        stream = torch.xpu.current_stream(device)
        staged_event = torch.xpu.Event()
        acquire_start = comm_trace_now_ns()
        allocation = await self.acquire_slots(allocator, num_slots)
        acquire_ms = comm_trace_elapsed_ms(acquire_start)
        offset = allocation.offset
        placement = TransferPlacement(
            size_bytes=size,
            offset_bytes=int(offset),
            slot_index=int(offset // self.slot_size),
            slot_size_bytes=self.slot_size,
            num_slots=num_slots,
        )
        copy_start = comm_trace_now_ns()
        try:
            with torch.xpu.device(device), torch.xpu.stream(stream):
                pool_tensor[offset : offset + size].copy_(flat, non_blocking=True)
                staged_event.record(stream)
            # The receiver has no XPU event to wait on, so the bytes land first.
            await wait_for_xpu_event(
                staged_event,
                device_index=self.device_index,
                executor=self.wait_executor,
                timeout=STAGE_COPY_TIMEOUT_S,
            )
        except BaseException as error:
            # The copy may still write these slots, so they stay reserved.
            self.mark_failed(error)
            raise
        copy_ms = comm_trace_elapsed_ms(copy_start)
        try:
            operation = self.publish_staged_transfer(
                placement,
                pool_reference=pool_reference,
                allocator=allocator,
                request_id=request_id,
            )
        except BaseException:
            # The copy has drained and nothing was published, so the slots go back.
            allocator.release(offset, num_slots)
            raise
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
        return operation

    def publish_staged_transfer(
        self,
        placement: TransferPlacement,
        *,
        pool_reference: XpuIpcPoolReference,
        allocator: ContiguousSlotAllocator,
        request_id: str | None,
    ) -> LevelZeroIpcPutOperation:
        """Describe a payload that has reached the pool, for the receiver to read."""
        metadata: LevelZeroIpcMetadata = {
            "engine_id": self.engine_id,
            "transfer_info": {
                "size": placement.size_bytes,
                "offset": placement.offset_bytes,
                "slot_index": placement.slot_index,
                "slot_size": placement.slot_size_bytes,
                "num_slots": placement.num_slots,
                "allocation_size": placement.num_slots * placement.slot_size_bytes,
            },
            "level_zero_ipc": {
                "pool_id": pool_reference.segment_id,
                "socket_name": pool_reference.socket_name,
                "segment_offset": pool_reference.segment_offset,
                "pool_bytes": pool_reference.pool_bytes,
                "source_device_index": pool_reference.device_index,
            },
        }
        return LevelZeroIpcPutOperation(
            metadata,
            on_failure=self.mark_failed,
            allocator=allocator,
            placement=placement,
            request_id=request_id,
        )

    async def get_async(
        self,
        metadata: Mapping[str, object],
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
        ensure_xpu_peer_access(reference.source_device_index, destination_device_index)

        destination = dest_tensor.view(torch.uint8).reshape(-1)
        if destination.numel() < placement.size_bytes:
            raise ValueError(
                f"level_zero_ipc destination buffer has {destination.numel()} bytes, "
                f"but transfer requires {placement.size_bytes} bytes"
            )
        else:
            pass

        peer_pool = await self.peer_pool_tensor(reference, destination_device_index)
        # The cached mapping is sized by the pool's first transfer; nothing widens it.
        allocation_end = (
            placement.offset_bytes + placement.num_slots * placement.slot_size_bytes
        )
        if allocation_end > peer_pool.numel():
            raise ValueError(
                f"level_zero_ipc transfer ends at byte {allocation_end}, past the "
                f"{peer_pool.numel()} bytes mapped for pool {reference.socket_name}"
            )
        else:
            pass
        source = peer_pool[
            placement.offset_bytes : placement.offset_bytes + placement.size_bytes
        ]
        stream = torch.xpu.current_stream(dest_tensor.device)
        event = torch.xpu.Event()
        with torch.xpu.device(dest_tensor.device), torch.xpu.stream(stream):
            destination[: placement.size_bytes].copy_(source, non_blocking=True)
            event.record(stream)
        comm_trace_emit(
            "level_zero_ipc_get_async",
            request_id=request_id,
            engine_id=self.engine_id,
            source_device=reference.source_device_index,
            destination_device=destination_device_index,
            bytes=placement.size_bytes,
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

    async def peer_pool_tensor(
        self, reference: PeerPoolReference, destination_device_index: int
    ) -> torch.Tensor:
        """Map a sender's pool once, off the event loop, and reuse the mapping."""
        mapping = self.peer_mappings.get(reference.pool_id)
        if mapping is None:
            pool_reference = XpuIpcPoolReference(
                socket_name=reference.socket_name,
                segment_id=reference.pool_id,
                segment_offset=reference.segment_offset,
                pool_bytes=reference.pool_bytes,
                device_index=reference.source_device_index,
            )
            mapping = self.wait_executor.submit(
                self.map_peer_pool, pool_reference, destination_device_index
            )
            self.peer_mappings[reference.pool_id] = mapping
        else:
            pass
        try:
            # A cancelled read must not abandon a mapping the worker is still opening.
            return await asyncio.shield(asyncio.wrap_future(mapping))
        except Exception:
            if self.peer_mappings.get(reference.pool_id) is mapping:
                self.peer_mappings.pop(reference.pool_id)
            else:
                pass
            raise

    def map_peer_pool(
        self, reference: XpuIpcPoolReference, destination_device_index: int
    ) -> torch.Tensor:
        """Map a sender's pool; blocks on the sender's socket."""
        pool_tensor = self.peer_pools.open(reference, destination_device_index)
        logger.info(
            f"[{self.engine_id}] Mapped Level-Zero IPC pool {reference.socket_name} "
            f"from XPU {reference.device_index} onto XPU {destination_device_index}"
        )
        return pool_tensor

    def cleanup(self, request_id: str) -> None:
        pass

    def close(self) -> None:
        # Drain the device and the mapping workers first; a failed step skips none.
        first_error: Exception | None = None
        for step in (
            lambda: torch.xpu.synchronize(self.device_index),
            lambda: self.wait_executor.shutdown(wait=True),
            self.close_peer_pools,
            self.close_exported_pool,
            self.level_zero.destroy,
        ):
            try:
                step()
            except Exception as error:
                if first_error is None:
                    first_error = error
                else:
                    logger.warning(
                        f"[{self.engine_id}] Level-Zero IPC teardown step also "
                        f"failed: {error}"
                    )
        if first_error is not None:
            raise first_error
        else:
            pass

    def close_peer_pools(self) -> None:
        self.peer_mappings.clear()
        self.peer_pools.close()

    def close_exported_pool(self) -> None:
        pool, self.exported_pool = self.exported_pool, None
        if pool is not None:
            # A peer's mapping keeps the memory alive on its own.
            pool.close()
        else:
            pass
