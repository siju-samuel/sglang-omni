# SPDX-License-Identifier: Apache-2.0
"""What the Level-Zero IPC relay keeps or gives back when a transfer goes wrong."""
from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch

import sglang_omni.relay.level_zero_ipc as level_zero_ipc
from sglang_omni.relay.cuda_ipc import ContiguousSlotAllocator
from sglang_omni.relay.level_zero_ipc import LevelZeroIpcRelay
from sglang_omni.relay.level_zero_ipc_metadata import (
    PeerPoolReference,
    TransferPlacement,
)
from sglang_omni.relay.level_zero_ipc_operations import LevelZeroIpcPutOperation

SLOT_SIZE_KB = 64
POOL_SIZE_MB = 8
POOL_BYTES = 262144


def peer_reference() -> PeerPoolReference:
    return PeerPoolReference(
        pool_id="sender:1:abc",
        socket_name="sglang-xpu-ipc-1-abc",
        segment_offset=0,
        pool_bytes=POOL_BYTES,
        source_device_index=0,
    )


def receiver_relay(peer_pools: SimpleNamespace) -> LevelZeroIpcRelay:
    """A receiving relay over fake peer pools, with no XPU behind it."""
    relay = LevelZeroIpcRelay.__new__(LevelZeroIpcRelay)
    relay.engine_id = "receiver"
    relay.device_index = 0
    relay.exported_pool = None
    relay.wait_executor = ThreadPoolExecutor(max_workers=2)
    relay.peer_pools = peer_pools
    relay.peer_mappings = {}
    return relay


def test_a_bad_wait_thread_setting_fails_before_any_context_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contexts: list[str] = []
    monkeypatch.setenv("SGLANG_OMNI_LEVEL_ZERO_IPC_WAIT_THREADS", "0")
    monkeypatch.setattr(
        level_zero_ipc, "LevelZeroIpcContext", lambda: contexts.append("created")
    )

    with pytest.raises(ValueError, match="must be positive"):
        LevelZeroIpcRelay(
            engine_id="sender",
            device="xpu:0",
            slot_size_kb=SLOT_SIZE_KB,
            pool_size_mb=POOL_SIZE_MB,
        )
    assert contexts == []


def test_a_cancelled_ack_wait_fails_the_relay_and_keeps_the_slots() -> None:
    """The receiver may still be reading, and cancelling the wait also drops the
    ACK that would release the slots, so the relay must stop handing them out."""

    async def run() -> None:
        allocator = ContiguousSlotAllocator(slot_count=2, slot_size=8)
        allocation = await allocator.acquire_async(1)
        failures: list[BaseException] = []
        operation = LevelZeroIpcPutOperation(
            {},
            on_failure=failures.append,
            allocator=allocator,
            placement=TransferPlacement(
                size_bytes=8,
                offset_bytes=allocation.offset,
                slot_index=0,
                slot_size_bytes=8,
                num_slots=1,
            ),
            request_id="r",
        )
        waiter = asyncio.create_task(operation.wait_for_completion(timeout=30))
        await asyncio.sleep(0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert len(failures) == 1
        assert allocator.free_slots == 1

    asyncio.run(run())


def test_slots_granted_to_a_cancelled_sender_go_back_to_the_pool() -> None:
    async def run() -> None:
        allocator = ContiguousSlotAllocator(slot_count=1, slot_size=8)
        held = await allocator.acquire_async(1)
        relay = LevelZeroIpcRelay.__new__(LevelZeroIpcRelay)
        relay.failed_error = None
        relay.failed_event = asyncio.Event()
        sender = asyncio.create_task(relay.acquire_slots(allocator, 1))
        await asyncio.sleep(0)
        # The grant lands on the loop's next turn, after the sender is cancelled.
        allocator.release(held.offset, 1)
        sender.cancel()
        with pytest.raises(asyncio.CancelledError):
            await sender
        await asyncio.sleep(0)
        assert allocator.free_slots == 1

    asyncio.run(run())


def test_close_waits_for_a_mapping_still_being_opened(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cancelled read leaves its mapping worker running, and the worker uses the
    context that close destroys, so close has to drain it before it unmaps."""
    events: list[str] = []

    def map_slowly() -> torch.Tensor:
        time.sleep(0.2)
        events.append("mapped")
        return torch.empty(0)

    relay = receiver_relay(SimpleNamespace(close=lambda: events.append("unmapped")))
    relay.level_zero = SimpleNamespace(destroy=lambda: events.append("destroyed"))
    relay.peer_mappings["pool"] = relay.wait_executor.submit(map_slowly)
    monkeypatch.setattr(torch.xpu, "synchronize", lambda device=None: None)

    relay.close()

    assert events == ["mapped", "unmapped", "destroyed"]
    assert relay.peer_mappings == {}


def test_concurrent_reads_map_a_pool_once_and_a_failed_mapping_is_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Stands in for sglang's XpuIpcPoolReference, which an older sglang lacks.
    monkeypatch.setattr(level_zero_ipc, "XpuIpcPoolReference", SimpleNamespace)
    opened: list[str] = []
    first_attempt = threading.Event()

    def open_pool(reference: SimpleNamespace, device_index: int) -> torch.Tensor:
        opened.append(reference.segment_id)
        if not first_attempt.is_set():
            first_attempt.set()
            raise RuntimeError("sender not serving yet")
        else:
            time.sleep(0.1)
            return torch.zeros(POOL_BYTES, dtype=torch.uint8)

    relay = receiver_relay(SimpleNamespace(open=open_pool))

    async def read_twice() -> list[torch.Tensor]:
        with pytest.raises(RuntimeError, match="not serving"):
            await relay.peer_pool_tensor(peer_reference(), 0)
        return await asyncio.gather(
            relay.peer_pool_tensor(peer_reference(), 0),
            relay.peer_pool_tensor(peer_reference(), 0),
        )

    first, second = asyncio.run(read_twice())
    assert first is second
    assert opened == ["sender:1:abc", "sender:1:abc"]
    relay.wait_executor.shutdown()


def test_a_transfer_past_a_cached_peer_mapping_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The mapping was sized by the pool's first transfer, and a later message that
    claims a larger pool must not read past it: the slice would be cut short and
    copy_ could broadcast the remainder into the destination."""
    cached_mapping = torch.zeros(POOL_BYTES, dtype=torch.uint8)

    async def cached_pool(
        reference: PeerPoolReference, destination_device_index: int
    ) -> torch.Tensor:
        return cached_mapping

    relay = LevelZeroIpcRelay.__new__(LevelZeroIpcRelay)
    relay.engine_id = "receiver"
    relay.peer_pool_tensor = cached_pool
    monkeypatch.setattr(
        level_zero_ipc, "ensure_xpu_peer_access", lambda source, destination: None
    )
    destination = SimpleNamespace(
        device=SimpleNamespace(type="xpu", index=0),
        view=lambda dtype: torch.zeros(4096, dtype=torch.uint8),
    )
    slot_bytes = SLOT_SIZE_KB * 1024
    metadata = {
        "engine_id": "sender",
        "transfer_info": {
            "size": 4096,
            "offset": POOL_BYTES,
            "slot_index": POOL_BYTES // slot_bytes,
            "slot_size": slot_bytes,
            "num_slots": 1,
            "allocation_size": slot_bytes,
        },
        "level_zero_ipc": {
            "pool_id": "sender:1:abc",
            "socket_name": "sglang-xpu-ipc-1-abc",
            "segment_offset": 0,
            "pool_bytes": 4 * POOL_BYTES,
            "source_device_index": 0,
        },
    }

    with pytest.raises(ValueError, match="past the"):
        asyncio.run(relay.get_async(metadata, destination, request_id="r"))
