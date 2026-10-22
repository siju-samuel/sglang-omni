# SPDX-License-Identifier: Apache-2.0
"""How the Level-Zero IPC sender takes, keeps and gives back pool slots."""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import sglang_omni.relay.level_zero_ipc as level_zero_ipc
from sglang_omni.relay.cuda_ipc import ContiguousSlotAllocator
from sglang_omni.relay.level_zero_ipc import LevelZeroIpcRelay

STAGED_BYTES = 4096


def unallocated_sender(
    engine_id: str, level_zero: SimpleNamespace
) -> LevelZeroIpcRelay:
    """A sender relay that has not made its pool yet, over a fake Level-Zero context."""
    relay = LevelZeroIpcRelay.__new__(LevelZeroIpcRelay)
    relay.engine_id = engine_id
    relay.device = "xpu:0"
    relay.device_index = 0
    relay.pool_size = 1 << 20
    relay.slot_size = 64 * 1024
    relay.slot_count = 16
    relay.exported_pool = None
    relay.allocator = None
    relay.level_zero = level_zero
    return relay


def pool_reference() -> SimpleNamespace:
    """A pool reference, as sglang's XpuIpcPoolReference describes one."""
    return SimpleNamespace(
        socket_name="unused",
        segment_id="pool",
        segment_offset=0,
        pool_bytes=4 * STAGED_BYTES,
        device_index=0,
    )


def test_close_attempts_every_step_and_raises_the_first_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A step close skips is never retried, so its pool or context would outlive
    the relay."""
    events: list[str] = []

    def refuse_to_unmap() -> None:
        raise RuntimeError("unmap failed")

    relay = unallocated_sender(
        "sender", SimpleNamespace(destroy=lambda: events.append("destroyed"))
    )
    relay.wait_executor = ThreadPoolExecutor(max_workers=1)
    relay.peer_pools = SimpleNamespace(close=refuse_to_unmap)
    relay.peer_mappings = {}
    relay.exported_pool = SimpleNamespace(close=lambda: events.append("pool closed"))
    monkeypatch.setattr(torch.xpu, "synchronize", lambda device=None: None)

    with pytest.raises(RuntimeError, match="unmap failed"):
        relay.close()
    assert events == ["pool closed", "destroyed"]
    assert relay.exported_pool is None


def test_a_pool_that_fails_to_come_up_is_not_kept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """sglang frees a pool that fails to come up; the relay must not hold on to it
    either, so the next send allocates again."""
    pool = SimpleNamespace(tensor=torch.zeros(1), reference=pool_reference())
    make_pool = Mock(side_effect=[RuntimeError("export failed"), pool])
    monkeypatch.setattr(level_zero_ipc, "XpuIpcExportedPool", make_pool)
    relay = unallocated_sender("sender", SimpleNamespace())

    with pytest.raises(RuntimeError, match="export failed"):
        relay.local_pool_state()
    assert relay.exported_pool is None
    assert relay.allocator is None

    pool_tensor, reference, allocator = relay.local_pool_state()
    assert pool_tensor is pool.tensor
    assert reference is pool.reference
    assert allocator.slot_count == relay.slot_count
    assert make_pool.call_count == 2


async def no_wait(
    event: SimpleNamespace,
    *,
    device_index: int,
    executor: ThreadPoolExecutor,
    timeout: float,
) -> None:
    """The staging copy into a host tensor has already finished."""


def staging_sender(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[LevelZeroIpcRelay, ContiguousSlotAllocator]:
    """A sender whose pool is up, staging into a host tensor with the XPU calls
    stubbed out, so each one can be made to fail on its own."""
    allocator = ContiguousSlotAllocator(slot_count=4, slot_size=STAGED_BYTES)
    relay = unallocated_sender("sender", SimpleNamespace())
    relay.failed_error = None
    relay.failed_event = asyncio.Event()
    relay.wait_executor = ThreadPoolExecutor(max_workers=1)
    relay.slot_size = STAGED_BYTES
    pool = torch.zeros(4 * STAGED_BYTES, dtype=torch.uint8)
    reference = pool_reference()
    relay.local_pool_state = lambda: (pool, reference, allocator)
    monkeypatch.setattr(torch.xpu, "current_stream", lambda device=None: None)
    monkeypatch.setattr(
        torch.xpu, "Event", lambda: SimpleNamespace(record=lambda stream: None)
    )
    monkeypatch.setattr(torch.xpu, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.xpu, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(level_zero_ipc, "wait_for_xpu_event", no_wait)
    return relay, allocator


def xpu_payload() -> SimpleNamespace:
    """A payload the relay takes for an XPU tensor; its bytes live on the host."""
    host_bytes = torch.ones(STAGED_BYTES, dtype=torch.uint8)
    return SimpleNamespace(
        device=SimpleNamespace(type="xpu"), contiguous=lambda: host_bytes
    )


@pytest.mark.parametrize("failing_call", ["current_stream", "Event"])
def test_a_send_takes_no_slots_when_its_stream_or_event_cannot_be_made(
    monkeypatch: pytest.MonkeyPatch, failing_call: str
) -> None:
    relay, allocator = staging_sender(monkeypatch)
    monkeypatch.setattr(
        torch.xpu,
        failing_call,
        Mock(side_effect=RuntimeError(f"{failing_call} failed")),
    )

    with pytest.raises(RuntimeError, match=f"{failing_call} failed"):
        asyncio.run(relay.put_async(xpu_payload(), request_id="r"))
    assert allocator.free_slots == allocator.slot_count
    assert relay.failed_error is None
    relay.wait_executor.shutdown()


def test_slots_go_back_when_a_staged_transfer_cannot_be_published(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relay, allocator = staging_sender(monkeypatch)
    monkeypatch.setattr(
        level_zero_ipc,
        "LevelZeroIpcPutOperation",
        Mock(side_effect=RuntimeError("publish failed")),
    )

    with pytest.raises(RuntimeError, match="publish failed"):
        asyncio.run(relay.put_async(xpu_payload(), request_id="r"))
    assert allocator.free_slots == allocator.slot_count
    assert relay.failed_error is None
    relay.wait_executor.shutdown()


def test_a_staged_transfer_keeps_its_slots_until_the_receiver_is_done(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relay, allocator = staging_sender(monkeypatch)

    async def stage_and_finish() -> tuple[int, int]:
        operation = await relay.put_async(xpu_payload(), request_id="r")
        held = allocator.free_slots
        operation.mark_receiver_done()
        await operation.wait_for_completion()
        return held, allocator.free_slots

    held, released = asyncio.run(stage_and_finish())
    assert held == allocator.slot_count - 1
    assert released == allocator.slot_count
    relay.wait_executor.shutdown()


def test_slots_stay_reserved_when_the_queued_copy_cannot_be_tracked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Once the copy is queued it may still be writing the slots, and without its
    event nothing says when it stops, so handing them out again would let it
    overwrite a later payload."""
    relay, allocator = staging_sender(monkeypatch)

    def refuse_to_record(stream: None) -> None:
        raise RuntimeError("record failed")

    monkeypatch.setattr(
        torch.xpu, "Event", lambda: SimpleNamespace(record=refuse_to_record)
    )

    with pytest.raises(RuntimeError, match="record failed"):
        asyncio.run(relay.put_async(xpu_payload(), request_id="r"))
    assert allocator.free_slots == allocator.slot_count - 1
    assert isinstance(relay.failed_error, RuntimeError)
    relay.wait_executor.shutdown()
