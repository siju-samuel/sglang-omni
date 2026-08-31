# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import sglang_omni.models.fun_asr.encoder_cuda_graph as encoder_cuda_graph
from sglang_omni.models.fun_asr.encoder_cuda_graph import (
    FunASREncoderCudaGraphRunner,
    _bucket_batch,
    _bucket_t,
    _new_graph,
)
from sglang_omni.models.fun_asr.sglang_model import FunAsrNanoForConditionalGeneration
from sglang_omni.models.fun_asr.tool_funcs.audio_lengths import (
    fun_asr_low_frame_rate_length,
)


def test_bucket_batch_rounds_up_within_max() -> None:
    assert _bucket_batch(1, 8) == 1
    assert _bucket_batch(2, 8) == 2
    assert _bucket_batch(3, 8) == 4
    assert _bucket_batch(5, 8) == 8
    assert _bucket_batch(8, 8) == 8
    # max_batch not a power of two: fall through to max itself
    assert _bucket_batch(5, 6) == 6
    # over the max -> no bucket
    assert _bucket_batch(9, 8) is None


def test_bucket_t_rounds_up_to_step() -> None:
    assert _bucket_t(1) == 64
    assert _bucket_t(64) == 64
    assert _bucket_t(65) == 128
    assert _bucket_t(500) == 512
    # beyond the 30s ceiling -> no bucket
    assert _bucket_t(513) is None


class _EagerTower(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.param = nn.Parameter(torch.zeros(1))
        self.calls: list[tuple] = []

    def forward(self, xs, mask):
        self.calls.append((xs.shape, None if mask is None else mask.shape))
        return xs


class _EagerProjector(nn.Module):
    def __init__(self, llm_dim: int = 4) -> None:
        super().__init__()
        self.llm_dim = llm_dim

    def forward(self, enc_out, mask):
        b, t, _ = enc_out.shape
        t_out = int(fun_asr_low_frame_rate_length(t))
        return torch.arange(b * t_out * self.llm_dim, dtype=torch.float32).reshape(
            b, t_out, self.llm_dim
        )


def _model_with(runner) -> FunAsrNanoForConditionalGeneration:
    model = object.__new__(FunAsrNanoForConditionalGeneration)
    nn.Module.__init__(model)
    model.audio_tower = _EagerTower()
    model.multi_modal_projector = _EagerProjector()
    if runner is not None:
        model.encoder_cuda_graph_runner = runner
    return model


def _item(num_frames: int) -> SimpleNamespace:
    return SimpleNamespace(
        feature=torch.randn(1, 560, num_frames),
        feature_attention_mask=torch.ones(1, num_frames, dtype=torch.long),
    )


def test_get_audio_feature_routes_through_graph_runner() -> None:
    observed = {}

    class _Runner:
        def run(self, xs, lengths):
            observed["xs_shape"] = tuple(xs.shape)
            observed["lengths"] = list(lengths)
            b = xs.shape[0]
            t_out = int(fun_asr_low_frame_rate_length(xs.shape[1]))
            return torch.ones(b, t_out, 4)

    model = _model_with(_Runner())
    out = model.get_audio_feature([_item(17), _item(9)])

    assert observed["xs_shape"] == (2, 17, 560)
    assert observed["lengths"] == [17, 9]
    expected_rows = int(fun_asr_low_frame_rate_length(17)) + int(
        fun_asr_low_frame_rate_length(9)
    )
    assert out.shape == (expected_rows, 4)
    # eager tower must not have run
    assert model.audio_tower.calls == []


def test_get_audio_feature_falls_back_to_eager_when_runner_declines() -> None:
    class _DecliningRunner:
        def run(self, xs, lengths):
            return None

    model = _model_with(_DecliningRunner())
    out = model.get_audio_feature([_item(17), _item(9)])

    # eager path ran, with a mask (batched input)
    assert len(model.audio_tower.calls) == 1
    xs_shape, mask_shape = model.audio_tower.calls[0]
    assert tuple(xs_shape) == (2, 17, 560)
    assert tuple(mask_shape) == (2, 1, 17)
    expected_rows = int(fun_asr_low_frame_rate_length(17)) + int(
        fun_asr_low_frame_rate_length(9)
    )
    assert out.shape == (expected_rows, 4)


def test_get_audio_feature_without_runner_matches_previous_behavior() -> None:
    model = _model_with(None)
    out = model.get_audio_feature([_item(12)])

    # single unpadded item keeps the maskless fast path
    assert model.audio_tower.calls == [((1, 12, 560), None)]
    assert out.shape == (int(fun_asr_low_frame_rate_length(12)), 4)


class _FakeGraph:
    def __init__(self) -> None:
        self.replays = 0

    def replay(self) -> None:
        self.replays += 1


class _FakeStream:
    def __init__(self, label: str) -> None:
        self.label = label
        self.joined: list[str] = []

    def wait_stream(self, other: "_FakeStream") -> None:
        self.joined.append(other.label)


class _FakeEvent:
    def __init__(self) -> None:
        self.recorded: list[str] = []
        self.awaited: list[str] = []

    def record(self, stream: _FakeStream) -> None:
        self.recorded.append(stream.label)

    def wait(self, stream: _FakeStream) -> None:
        self.awaited.append(stream.label)


class _FakeDeviceModule:
    """Stands in for torch.cuda / torch.xpu.

    The runner may only reach the device through this object, so anything it
    needs that is missing here shows up as an AttributeError in these tests
    rather than as a CUDA-only runtime on someone else's accelerator.
    """

    def __init__(
        self,
        log: list[str],
        *,
        free_bytes: int = 40 * 1024**3,
        total_bytes: int = 40 * 1024**3,
        reserved_bytes: int = 0,
    ) -> None:
        self.__name__ = "fake_device_module"
        self.log = log
        self.free_bytes = free_bytes
        self.total_bytes = total_bytes
        self.reserved_bytes = reserved_bytes
        self.capture_kwargs: list[dict] = []
        self.event = _FakeEvent()

    def Event(self) -> _FakeEvent:  # noqa: N802 - mirrors the torch spelling
        return self.event

    def Stream(self, device=None) -> _FakeStream:  # noqa: N802 - ditto
        return _FakeStream("side")

    def current_stream(self, device=None) -> _FakeStream:
        return _FakeStream("current")

    @contextlib.contextmanager
    def stream(self, stream: _FakeStream):
        self.log.append("warmup-stream:enter")
        yield
        self.log.append("warmup-stream:exit")

    def synchronize(self, device=None) -> None:
        self.log.append("synchronize")

    def graph_pool_handle(self) -> str:
        return "pool-token"

    @contextlib.contextmanager
    def device(self, device):
        self.log.append("device:enter")
        yield
        self.log.append("device:exit")

    def mem_get_info(self, device=None) -> tuple[int, int]:
        return self.free_bytes, self.total_bytes

    def memory_reserved(self, device=None) -> int:
        return self.reserved_bytes


class _XpuLikeModule(_FakeDeviceModule):
    XPUGraph = _FakeGraph

    @contextlib.contextmanager
    def graph(self, graph, pool=None, stream=None):
        self.capture_kwargs.append({"pool": pool, "stream": stream})
        self.log.append("capture:enter")
        yield
        self.log.append("capture:exit")


class _CudaLikeModule(_FakeDeviceModule):
    CUDAGraph = _FakeGraph

    @contextlib.contextmanager
    def graph(self, graph, pool=None, stream=None, capture_error_mode="global"):
        self.capture_kwargs.append(
            {"pool": pool, "stream": stream, "capture_error_mode": capture_error_mode}
        )
        self.log.append("capture:enter")
        yield
        self.log.append("capture:exit")


def _runner_on(module: _FakeDeviceModule) -> FunASREncoderCudaGraphRunner:
    runner = FunASREncoderCudaGraphRunner(
        _EagerTower(), _EagerProjector(), max_batch_size=4
    )
    runner._device_module = module
    runner._done_event = module.Event()
    return runner


def test_runner_resolves_its_device_module_from_the_model() -> None:
    runner = FunASREncoderCudaGraphRunner(_EagerTower(), _EagerProjector())

    # A CPU-resident tower must not reach for torch.cuda, which is how this
    # constructor used to fail on an accelerator without a CUDA runtime.
    assert runner._device_module is torch.get_device_module(torch.device("cpu"))


def test_runner_declines_every_bucket_on_a_device_without_graphs() -> None:
    runner = FunASREncoderCudaGraphRunner(_EagerTower(), _EagerProjector())

    # torch.cpu has no graph API; the caller then runs the encoder eager rather
    # than raising once per request from inside the capture path.
    assert runner.run(torch.zeros(1, 17, 560), [17]) is None
    assert runner._graphs == {}


def test_new_graph_prefers_the_device_s_own_graph_type() -> None:
    assert _new_graph(SimpleNamespace(CUDAGraph=lambda: "cuda-graph")) == "cuda-graph"
    assert _new_graph(SimpleNamespace(XPUGraph=lambda: "xpu-graph")) == "xpu-graph"

    with pytest.raises(RuntimeError, match="no graph type"):
        _new_graph(SimpleNamespace(__name__="torch.cpu"))


def test_capture_error_mode_is_passed_only_where_the_signature_takes_it() -> None:
    cuda_like = _CudaLikeModule([])
    runner = _runner_on(cuda_like)
    runner._pool = "pool-token"
    with runner._capture_context(_FakeGraph()):
        pass
    assert cuda_like.capture_kwargs == [
        {"pool": "pool-token", "stream": None, "capture_error_mode": "thread_local"}
    ]

    xpu_like = _XpuLikeModule([])
    runner = _runner_on(xpu_like)
    runner._pool = "pool-token"
    with runner._capture_context(_FakeGraph()):
        pass
    assert xpu_like.capture_kwargs == [{"pool": "pool-token", "stream": None}]


def test_free_vram_check_clamps_a_runtime_that_reports_the_whole_card_free() -> None:
    over_reporting = _XpuLikeModule(
        [],
        free_bytes=24 * 1024**3,
        total_bytes=24 * 1024**3,
        reserved_bytes=23 * 1024**3,
    )
    enough, free = _runner_on(over_reporting)._enough_free_vram()
    assert (enough, free) == (False, 1024**3)

    # Where mem_get_info is the tighter number it is the one that counts.
    accounting = _XpuLikeModule(
        [],
        free_bytes=5 * 1024**3,
        total_bytes=24 * 1024**3,
        reserved_bytes=1024**3,
    )
    enough, free = _runner_on(accounting)._enough_free_vram()
    assert (enough, free) == (True, 5 * 1024**3)


def test_capture_warms_up_and_records_under_the_platform_sdpa_context(
    monkeypatch,
) -> None:
    log: list[str] = []
    module = _XpuLikeModule(log)
    runner = _runner_on(module)

    @contextlib.contextmanager
    def _recording_sdpa_context():
        log.append("sdpa:enter")
        yield
        log.append("sdpa:exit")

    monkeypatch.setattr(
        encoder_cuda_graph.current_platform,
        "sdpa_capture_context",
        _recording_sdpa_context,
        raising=False,
    )

    xs = torch.zeros(1, 17, 560)
    out = runner.run(xs, [17])

    assert out is not None
    assert out.shape == (1, int(fun_asr_low_frame_rate_length(64)), 4)
    # The warmup is inside the pinned dispatch, so what was captured is what
    # was exercised, and the pin is released once the capture closes.
    assert log.index("sdpa:enter") < log.index("warmup-stream:enter")
    assert log.index("capture:exit") < log.index("sdpa:exit")
    assert module.capture_kwargs == [{"pool": "pool-token", "stream": None}]

    graph, _, static_ilens, _ = runner._graphs[(1, 64)]
    assert graph.replays == 1
    assert static_ilens.tolist() == [17]

    # A second call replays the cached bucket instead of capturing again.
    runner.run(xs, [17])
    assert graph.replays == 2
    assert len(module.capture_kwargs) == 1
