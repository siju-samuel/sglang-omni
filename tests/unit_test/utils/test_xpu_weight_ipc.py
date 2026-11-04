# SPDX-License-Identifier: Apache-2.0
"""Level-Zero weight sharing: pool layout, refusals, and the platform default."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import tempfile
import time

import pytest
import torch
from torch import nn

from sglang_omni.platforms import current_platform
from sglang_omni.platforms.level_zero import LevelZeroError
from sglang_omni.utils.ipc_weights import (
    SglangIpcSerializer,
    default_alias_predicate,
    default_serializer,
)
from sglang_omni.utils.xpu_weight_ipc import (
    XpuLevelZeroSerializer,
    plan_pool,
    pool_device_index,
)


def test_the_pool_layout_packs_every_weight_without_overlap() -> None:
    """Both sides slice the pool from this layout alone, so each weight needs its
    whole size, an alignment every dtype view accepts, and no shared bytes.
    """
    tensors = {
        "b.weight": torch.empty(3, 5, dtype=torch.bfloat16),
        "a.weight": torch.empty(7, dtype=torch.float32),
        "c.bias": torch.empty(1, dtype=torch.uint8),
    }

    layout, pool_bytes = plan_pool(tensors)

    assert [entry.name for entry in layout] == ["a.weight", "b.weight", "c.bias"]
    occupied: set[int] = set()
    for entry in layout:
        assert entry.offset % 256 == 0
        assert entry.num_bytes == tensors[entry.name].numel() * (
            tensors[entry.name].element_size()
        )
        window = set(range(entry.offset, entry.offset + entry.num_bytes))
        assert not (window & occupied)
        occupied |= window
        assert entry.offset + entry.num_bytes <= pool_bytes


def test_an_unpoolable_dtype_is_refused_rather_than_mislabelled() -> None:
    """The manifest carries a dtype by name, so a dtype with no name would be
    reconstructed as something else in the follower.
    """
    with pytest.raises(LevelZeroError, match="cannot pool"):
        plan_pool({"w": torch.empty(4, dtype=torch.complex64)})


def test_weights_spread_over_two_cards_are_refused() -> None:
    """One pool lives on one card; exporting weights from another would hand the
    follower memory that card cannot read.
    """
    first = torch.empty(2, device="meta")
    second = torch.empty(2, device="cpu")

    with pytest.raises(LevelZeroError, match="one XPU card"):
        pool_device_index({"a": first, "b": second})


def test_a_manifest_from_another_card_is_refused(monkeypatch) -> None:
    """Replicas share weights only when they share a card, and a cross-card alias
    would fail later inside a kernel instead of naming weight sharing.
    """
    manifest = {
        "version": 1,
        "socket_name": "unused",
        "pool_id": "unused",
        "device_index": 3,
        "pool_bytes": 256,
        "tensors": [
            {
                "name": "w",
                "offset": 0,
                "num_bytes": 8,
                "shape": [2],
                "dtype": "float32",
            }
        ],
    }
    monkeypatch.setattr(torch.xpu, "current_device", lambda: 0)

    with pytest.raises(LevelZeroError, match="must share one card"):
        XpuLevelZeroSerializer.deserialize(json.dumps(manifest).encode())


def test_a_manifest_version_the_follower_cannot_read_is_refused() -> None:
    """A leader and follower on different revisions must not guess the layout."""
    manifest = {"version": 999, "tensors": []}

    with pytest.raises(LevelZeroError, match="version 999"):
        XpuLevelZeroSerializer.deserialize(json.dumps(manifest).encode())


def test_an_empty_handle_round_trips_without_a_pool() -> None:
    """A model whose every tensor is replica-private publishes no pool, and the
    follower must read that as no shared tensors rather than fail.
    """
    blob = XpuLevelZeroSerializer.serialize({})

    assert XpuLevelZeroSerializer.deserialize(blob) == {}


def test_the_platform_chooses_a_handle_format_it_can_alias() -> None:
    """torch publishes no XPU IPC handle, so an XPU engine that fell back to the
    CUDA serializer embedded every weight by value and shared nothing.
    """
    serializer = default_serializer()
    predicate = default_alias_predicate()

    # A host tensor has no handle to publish on any platform.
    assert predicate(torch.empty(2)) is False
    if current_platform.is_xpu():
        assert serializer is XpuLevelZeroSerializer
    else:
        assert serializer is SglangIpcSerializer


class Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(256, 256, bias=False) for _ in range(4)])
        self.register_buffer("scale", torch.ones(256, dtype=torch.float32))


def attach_in_follower(dir_path: str, out: mp.Queue) -> None:
    import torch as follower_torch

    from sglang_omni.utils.ipc_weights import follower_attach

    follower_torch.xpu.set_device(0)
    model = Backbone().to("xpu:0")
    before = model.layers[0].weight.data_ptr()
    follower_attach(model, dir_path, validate_secure=False, timeout_s=240)
    follower_torch.xpu.synchronize()
    weight = model.layers[0].weight
    out.put(
        ("attached", float(weight.flatten()[0].item()), weight.data_ptr() != before)
    )
    time.sleep(6)
    out.put(("reread", float(model.layers[0].weight.flatten()[0].item()), True))


@pytest.mark.accelerator
def test_the_leader_keeps_one_copy_of_the_shared_weights() -> None:
    """Pooling the weights leaves torch holding the originals in its caching
    allocator, so without returning them to the driver the leader pays for both
    copies and cancels out what a follower saves. Measured 2.09x the weight mass
    before, 1.09x after. Only the driver's counter sees this: the pool is outside
    torch's allocator and the freed blocks stay inside it.
    """
    from sglang_omni.utils.ipc_weights import leader_export

    if not current_platform.is_xpu():
        pytest.skip("Level-Zero weight sharing needs an XPU host")
    else:
        pass

    torch.xpu.set_device(0)
    # Large enough that a second copy dwarfs the Level-Zero context the export
    # builds, which also lands in this measurement.
    model = nn.Sequential(
        *[nn.Linear(2048, 2048, bias=False) for _ in range(32)]
    ).to("xpu:0")
    torch.xpu.synchronize()
    mass = sum(t.numel() * t.element_size() for t in model.state_dict().values())
    free_loaded = torch.xpu.mem_get_info(0)[0]

    with tempfile.TemporaryDirectory() as dir_path:
        leader_export(model, dir_path, validate_secure=False)
        torch.xpu.synchronize()
        used_by_export = free_loaded - torch.xpu.mem_get_info(0)[0]

    assert used_by_export < mass * 0.5, (
        f"export added {used_by_export} bytes for {mass} bytes of weights, which "
        f"is a second copy"
    )


@pytest.mark.accelerator
def test_a_follower_aliases_the_leader_weights_instead_of_copying_them() -> None:
    """The handle file carried the whole checkpoint on XPU and shared nothing.
    A follower that aliases sees the leader mutate a weight in place.
    """
    from sglang_omni.utils.ipc_weights import leader_export

    if not current_platform.is_xpu():
        pytest.skip("Level-Zero weight sharing needs an XPU host")
    else:
        pass

    torch.xpu.set_device(0)
    model = Backbone().to("xpu:0")
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(1.5)
    torch.xpu.synchronize()
    mass = sum(t.numel() * t.element_size() for t in model.state_dict().values())

    with tempfile.TemporaryDirectory() as dir_path:
        leader_export(model, dir_path, validate_secure=False)
        torch.xpu.synchronize()
        handle = os.path.join(dir_path, "Backbone.weights-ipc")
        assert os.path.getsize(handle) < mass / 100

        context = mp.get_context("spawn")
        queue = context.Queue()
        follower = context.Process(target=attach_in_follower, args=(dir_path, queue))
        follower.start()
        try:
            _tag, value, rebound = queue.get(timeout=300)
            assert value == 1.5
            assert rebound is True

            with torch.no_grad():
                model.layers[0].weight.fill_(9.25)
            torch.xpu.synchronize()
            _tag, after, _ = queue.get(timeout=300)
            assert after == 9.25
        finally:
            follower.join(90)
