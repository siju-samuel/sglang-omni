# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

import sglang_omni.relay.nccl as nccl_relay
from sglang_omni.platforms.xpu import XPUOmniPlatform
from sglang_omni.relay.base import CreditAllocator
from sglang_omni.relay.nccl import NcclRelay


class CompletedWork:
    def is_completed(self) -> bool:
        return True

    def wait(self) -> bool:
        return True


@pytest.mark.parametrize(
    ("send_to_ranks", "recv_from_ranks"), [([1], [1]), ([1, 2], [])]
)
def test_xccl_relay_refuses_a_second_peer_or_direction_before_joining_a_group(
    monkeypatch: pytest.MonkeyPatch,
    send_to_ranks: list[int],
    recv_from_ranks: list[int],
) -> None:
    def join_group(backend: str, **kwargs: object) -> None:
        raise AssertionError(f"joined a {backend} group before checking the topology")

    monkeypatch.setattr(nccl_relay, "current_platform", XPUOmniPlatform())
    monkeypatch.setattr(dist, "init_process_group", join_group)

    with pytest.raises(NotImplementedError, match="one peer in one direction"):
        NcclRelay(
            engine_id="engine",
            send_to_ranks=send_to_ranks,
            recv_from_ranks=recv_from_ranks,
            rank=0,
            world_size=3,
        )


def test_put_returns_its_metadata_while_a_blocking_send_waits_for_the_peer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An XCCL isend returns only once the receiver posts its irecv, and the receiver
    posts it on the metadata put_async returns."""
    receiver_posted = threading.Event()
    sent_to: list[int] = []

    def blocking_isend(tensor: torch.Tensor, dst: int, group: None) -> CompletedWork:
        assert receiver_posted.wait(timeout=10.0)
        sent_to.append(dst)
        return CompletedWork()

    monkeypatch.setattr(dist, "isend", blocking_isend)
    relay = object.__new__(NcclRelay)
    relay.engine_id = "sender"
    relay.device_id = 0
    relay.connection = SimpleNamespace(
        send_ranks=[1],
        group=None,
        get_agent_metadata=lambda: {"rank": 0, "engine_id": "sender"},
    )
    relay.allocator = CreditAllocator(credits=1)
    relay.issue_thread = ThreadPoolExecutor(1)

    async def run() -> None:
        op = await asyncio.wait_for(relay.put_async(torch.zeros(4)), timeout=1.0)
        assert op.metadata["transfer_info"]["size"] == 16
        assert sent_to == []
        receiver_posted.set()
        await op.wait_for_completion(timeout=5.0)
        assert sent_to == [1]

    try:
        asyncio.run(run())
    finally:
        relay.close()
