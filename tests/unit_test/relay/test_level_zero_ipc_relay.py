# SPDX-License-Identifier: Apache-2.0
"""Device-to-device round-trip tests for the Level-Zero IPC relay.

The relay exports its XPU pool as a dma_buf handle, hands the descriptor to the
receiver over a Unix socket, and the receiver copies straight out of that peer
mapping. These run two real processes because a Level-Zero handle is only
meaningful to a different process than the one that exported it.
"""
from __future__ import annotations

import asyncio
import multiprocessing as mp
import traceback

import pytest
import torch

# The XPU runtime has to come up before the imports below, because the relay package
# pulls in a second Level-Zero loader that leaves this process with no XPU device if
# it lands first. A spawned worker imports this module the same way, so placing the
# call here covers the workers too.
if torch.xpu.is_available():
    try:
        torch.xpu.init()
    except RuntimeError:
        # Collection still has to succeed; the tests below skip themselves.
        pass
else:
    pass

from sglang_omni.relay.level_zero import (  # noqa: E402 - must follow the XPU init
    is_level_zero_ipc_available,
)
from sglang_omni.relay.level_zero_ipc import (  # noqa: E402 - must follow the XPU init
    LevelZeroIpcRelay,
    PeerPoolReference,
    TransferPlacement,
    parse_metadata,
)

PAYLOAD_SIZES = (1024 * 1024, 64 * 1024, 3 * 1024 * 1024)
POOL_SIZE_MB = 8
MESSAGE_TIMEOUT_S = 60
RESULT_TIMEOUT_S = 120
JOIN_TIMEOUT_S = 30


def skip_without_level_zero_ipc() -> None:
    """Probe at call time: collection must not bring the XPU runtime up."""
    if not is_level_zero_ipc_available():
        pytest.skip("requires an Intel GPU with Level-Zero IPC")
    else:
        pass


def make_expected(num_bytes: int) -> torch.Tensor:
    return (torch.arange(num_bytes, dtype=torch.int64) % 251).to(torch.uint8)


def valid_metadata() -> dict[str, object]:
    return {
        "engine_id": "sender",
        "transfer_info": {
            "size": 4096,
            "offset": 65536,
            "slot_index": 1,
            "slot_size": 65536,
            "num_slots": 1,
            "allocation_size": 65536,
        },
        "level_zero_ipc": {
            "pool_id": "sender:1:abc",
            "socket_name": "sglang-omni-level-zero-1-abc",
            "segment_offset": 0,
            "pool_bytes": 262144,
            "source_device_index": 0,
        },
    }


def test_level_zero_ipc_relay_refuses_a_host_device() -> None:
    with pytest.raises(ValueError, match="requires an XPU device"):
        LevelZeroIpcRelay(engine_id="sender", device="cpu")


def test_parse_metadata_reads_a_valid_transfer() -> None:
    placement, reference = parse_metadata(valid_metadata())

    assert placement == TransferPlacement(
        size=4096, offset=65536, slot_index=1, slot_size=65536, num_slots=1
    )
    assert reference == PeerPoolReference(
        pool_id="sender:1:abc",
        socket_name="sglang-omni-level-zero-1-abc",
        segment_offset=0,
        pool_bytes=262144,
        source_device_index=0,
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("offset", -64, "must be non-negative"),
        ("offset", 1024, "must be slot aligned"),
        ("slot_size", 0, "must be positive"),
        ("num_slots", 0, "must be positive"),
        ("size", 1 << 20, "num_slots is too small"),
    ],
)
def test_parse_metadata_rejects_an_inconsistent_transfer(
    field: str, value: int, message: str
) -> None:
    metadata = valid_metadata()
    metadata["transfer_info"][field] = value

    with pytest.raises(ValueError, match=message):
        parse_metadata(metadata)


def test_parse_metadata_rejects_a_range_past_the_pool() -> None:
    metadata = valid_metadata()
    metadata["level_zero_ipc"]["pool_bytes"] = 65536

    with pytest.raises(ValueError, match="exceeds pool size"):
        parse_metadata(metadata)


def level_zero_sender(
    source_gpu: int,
    metadata_queue: mp.Queue,
    ack_queue: mp.Queue,
    result_queue: mp.Queue,
) -> None:
    relay = None
    try:
        torch.xpu.set_device(source_gpu)
        relay = LevelZeroIpcRelay(
            engine_id="sender",
            device=f"xpu:{source_gpu}",
            pool_size_mb=POOL_SIZE_MB,
        )

        async def run() -> None:
            for num_bytes in PAYLOAD_SIZES:
                payload = make_expected(num_bytes).to(f"xpu:{source_gpu}")
                operation = await relay.put_async(
                    payload,
                    request_id=f"r{num_bytes}",
                    receiver_id="receiver",
                )
                metadata_queue.put(operation.metadata)
                ack_status, ack_value = ack_queue.get(timeout=MESSAGE_TIMEOUT_S)
                if ack_status == "ok":
                    operation.mark_receiver_done()
                elif ack_status == "err":
                    operation.mark_receiver_failed(RuntimeError(ack_value))
                else:
                    raise RuntimeError(
                        f"receiver returned invalid ACK status {ack_status!r}"
                    )
                await operation.wait_for_completion()

        asyncio.run(run())
        result_queue.put(("sender", "ok", True))
    except Exception as error:
        result_queue.put(("sender", "err", traceback.format_exc()))
    finally:
        if relay is not None:
            relay.close()
        else:
            pass


def level_zero_receiver(
    destination_gpu: int,
    metadata_queue: mp.Queue,
    ack_queue: mp.Queue,
    result_queue: mp.Queue,
) -> None:
    relay = None
    try:
        torch.xpu.set_device(destination_gpu)
        relay = LevelZeroIpcRelay(engine_id="receiver", device=f"xpu:{destination_gpu}")
        mapped_pools: set[int] = set()

        async def receive(num_bytes: int) -> None:
            metadata = metadata_queue.get(timeout=MESSAGE_TIMEOUT_S)
            size = metadata["transfer_info"]["size"]
            if size != num_bytes:
                raise AssertionError(f"expected {num_bytes} bytes, got {size}")
            else:
                pass
            destination = torch.zeros(
                size, dtype=torch.uint8, device=f"xpu:{destination_gpu}"
            )
            operation = await relay.get_async(
                metadata, destination, request_id=f"r{num_bytes}"
            )
            await operation.wait_for_completion()
            if not torch.equal(destination.cpu(), make_expected(size)):
                raise AssertionError(
                    f"received {size} Level-Zero IPC bytes do not match the source"
                )
            else:
                pass
            mapped_pools.add(len(relay.peer_pools))

        async def run() -> None:
            for num_bytes in PAYLOAD_SIZES:
                try:
                    await receive(num_bytes)
                except Exception as error:
                    ack_queue.put(("err", repr(error)))
                    raise
                ack_queue.put(("ok", None))

        asyncio.run(run())
        # Every payload came from one sender pool, so the mapping is opened once.
        if mapped_pools != {1}:
            raise AssertionError(f"expected one cached peer pool, saw {mapped_pools}")
        else:
            pass
        result_queue.put(("receiver", "ok", True))
    except Exception as error:
        result_queue.put(("receiver", "err", traceback.format_exc()))
    finally:
        if relay is not None:
            relay.close()
        else:
            pass


def run_case(source_gpu: int, destination_gpu: int) -> None:
    context = mp.get_context("spawn")
    metadata_queue = context.Queue()
    ack_queue = context.Queue()
    result_queue = context.Queue()
    sender = context.Process(
        target=level_zero_sender,
        args=(source_gpu, metadata_queue, ack_queue, result_queue),
    )
    receiver = context.Process(
        target=level_zero_receiver,
        args=(destination_gpu, metadata_queue, ack_queue, result_queue),
    )
    sender.start()
    receiver.start()
    results: dict[str, tuple[str, object]] = {}
    try:
        for _ in range(2):
            process, status, value = result_queue.get(timeout=RESULT_TIMEOUT_S)
            results[process] = (status, value)
    finally:
        sender.join(timeout=JOIN_TIMEOUT_S)
        receiver.join(timeout=JOIN_TIMEOUT_S)
        for process in (sender, receiver):
            if process.is_alive():
                process.terminate()
                process.join(timeout=JOIN_TIMEOUT_S)
            else:
                pass

    assert results.keys() == {"sender", "receiver"}
    for name in ("sender", "receiver"):
        status, value = results[name]
        assert status == "ok", value
        assert value is True
    assert sender.exitcode == 0
    assert receiver.exitcode == 0


@pytest.mark.accelerator
def test_level_zero_ipc_same_gpu_round_trip() -> None:
    skip_without_level_zero_ipc()
    run_case(0, 0)


@pytest.mark.accelerator
def test_level_zero_ipc_cross_gpu_round_trip() -> None:
    skip_without_level_zero_ipc()
    if torch.xpu.device_count() < 2:
        pytest.skip("requires two Intel GPUs")
    else:
        pass
    run_case(0, 1)
