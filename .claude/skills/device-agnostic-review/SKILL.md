---
name: device-agnostic-review
description: "Review or rewrite a change so shared sglang-omni code runs on every accelerator the repo serves (CUDA, ROCm, MUSA, XPU, NPU, MPS, CPU) and only code that is platform-specific by design names one. Covers what counts as platform-specific, a scanner for the device-hardcoded spellings a diff adds, the device-neutral replacement for each, the branching traps (is_cuda() is False on ROCm; the no-accelerator fallback is not is_cpu()), when a difference belongs in an OmniPlatform hook, and how tests stay device-neutral. Use when reviewing a PR or branch for device portability, when asked to make code device agnostic, or before adding torch.cuda, \"cuda\" device strings, streams, events, graphs, memory queries or NCCL outside sglang_omni/platforms/."
---

# Device-agnostic review

## Scope

Shared code asks for its device; it never names one. This skill finds the places a change names one, decides whether that is allowed, and gives the device-neutral form when it is not. It does not judge whether a graph capture is valid on a given device, measure performance, or run hardware it does not have.

These are platform-specific by design, so naming a vendor there is not a finding:

- `sglang_omni/platforms/<vendor>.py` and the per-vendor graph backends in `sglang_omni/platforms/device_graph.py`.
- Vendor kernel sources: `.cu` files and the CuTe DSL kernels under `*/cutedsl/`.
- `sglang_omni/mps/`, which is NVIDIA's CUDA Multi-Process Service. It is not Apple's `mps` device.
- A transport that only one platform's `get_intra_node_transport()` selects, such as `sglang_omni/relay/cuda_ipc.py`. When a new module like that lands, add it to `PLATFORM_SPECIFIC_PATH` in the scanner. `sglang_omni/relay/nccl.py` does not qualify: users select it by name.
- `sglang_omni/vendor/`, `tests/unit_test/platforms/`, `tests/unit_test/test_platforms.py` and `tests/unit_test/xpu/`.
- Code under a positive predicate for the platform it serves (`if current_platform.is_cuda():`, `assert x.is_cuda`, a test behind `require_cuda()`), provided every other platform still has a path: a fallback, a hook that returns `None`, or a clear error.
- Features that exist on one vendor only: FlashInfer, CUTLASS MoE, NVML telemetry, SM-version checks. They stay, behind a predicate or a hook.
- External names: `disable_cuda_graph`, `cuda_graph_max_bs`, `cuda_graph_backend_decode`, `enable_encoder_cuda_graph`, and file names such as `encoder_cuda_graph.py`. These are SGLang's or the config's spelling, and coding-style.md forbids renaming an external contract.

Everything else is shared: models, the model runner, scheduling, pipeline, comm, relay, utils, profiler, CLI, benchmarks and tests.

## The rule

Get the device from the data (`tensor.device`), from the stage (`resolve_concrete_device(device, index)` in `sglang_omni/utils/device.py`), or from the host (`current_platform.device_type`). Then reach the operation through that device: `torch.get_device_module(device)`, `torch.accelerator`, or a `current_platform` method. When behavior really differs by platform, the difference goes into an `OmniPlatform` hook, not into an `if` in the model.

Two constraints apply to every rewrite:

- **The CUDA path stays identical.** It keeps the same kernels, streams, captures, sync points and allocations. A new hook's default on `OmniPlatform` returns what the code did before the change, and only the platform that needs something else overrides it. Do not add a synchronize, a host read, a copy or a per-step lookup to make the neutral spelling work. In a hot loop, resolve the device module once at construction.
- **Neutral is not enabled.** A rewrite that makes a graph capture, a stream overlap or a fused kernel reachable on XPU or NPU for the first time can turn a working eager fallback into a crash or a stale replay. Gate the new reach behind the existing hook (`get_device_graph_backend`, `enable_talker_graph`, `supports_graph_captured_host_read`, `supports_graph_captured_fft` and so on), and have the PR say which platforms were actually run.

## Workflow

1. Check out the change. For a branch, `git fetch origin main` and stay on it. For a PR, run `gh pr checkout <number>` or `git fetch origin pull/<number>/head:pr-<number>`.
2. Run the scanner from the repo root:

   ```bash
   # working tree against main, uncommitted and untracked files included
   python .claude/skills/device-agnostic-review/scan_device_hardcoding.py
   # a fetched PR ref, without checking it out
   python .claude/skills/device-agnostic-review/scan_device_hardcoding.py --head pr-1234
   # whole files or directories, when porting existing code
   python .claude/skills/device-agnostic-review/scan_device_hardcoding.py sglang_omni/models/foo/
   # also list the hits that sit under a platform guard
   python .claude/skills/device-agnostic-review/scan_device_hardcoding.py --include-guarded
   ```

   It reads only the lines the diff adds and skips the platform-specific paths above. A hit is marked `[guarded]` when an enclosing `if`, an earlier `assert` or early exit, a decorator, a module-level `pytestmark` or a vendor class name names a platform. Exit status 1 means unguarded hits remain.
3. Triage every hit with the tables below. An unguarded hit in shared code is a finding unless Scope covers it. For a `[guarded]` hit, check the guard. It must name the same vendor the line calls, and every other platform must still get a correct path. A guard that turns the other platforms into a silent no-op or a wrong value is still a finding. For example, a memory log that reports `"cpu"` on XPU.
4. Read the diff for what a regex cannot see: the `else` arms, capability gaps, and the silent degradations listed below.
5. Report, or fix and verify, as described in the last two sections.

## Replacements

### Device and placement

| Device-hardcoded | Device-neutral |
|---|---|
| `"cuda"`, `"cuda:0"`, `f"cuda:{index}"`, a `device="cuda"` default | `tensor.device`; `resolve_concrete_device(device, index)`; `torch.device(current_platform.device_type)` for the host's type |
| `.cuda()` | `.to(device)` |
| `torch.cuda.is_available()` meaning "on an accelerator" | `has_device_streams(device)` from `sglang_omni/utils/device.py`, or the predicate of the platform the path serves |
| `torch.cuda.set_device(index)` | `current_platform.set_device(device)` |
| `torch.cuda.current_device()`, `device_count()` | `torch.get_device_module(device).current_device()`, `torch.accelerator.device_count()` |
| `with torch.cuda.device(device):` | `with device_guard(device):`, which is a no-op off an accelerator |
| `torch.cuda.get_device_name()`, `get_device_properties()` | `current_platform.get_device_name()`, `get_device_total_memory()`. An SM-version check is a CUDA feature gate and belongs in a hook. |

### Streams, events and synchronization

| Device-hardcoded | Device-neutral |
|---|---|
| `torch.cuda.Stream()`, `stream(s)`, `current_stream()` | `torch.get_device_module(device).Stream(device=device)`, `.stream(s)`, `.current_stream(device)`. If the path also runs on CPU, gate it on `has_device_streams(device)`. |
| `torch.cuda.Event()` | `new_device_event(device)` from `sglang_omni/utils/cuda_staging.py` |
| `torch.cuda.synchronize()` | `torch.get_device_module(device).synchronize(device)`. With no device in hand, use `current_platform.synchronize()`. |
| `torch.cuda.Stream` in an annotation | `torch.Stream` |

`torch.cpu` has `Stream`, `Event`, `synchronize` and `current_stream`, but no `empty_cache` and no `device` context. That is why `device_guard` and `has_device_streams` exist: a bare `torch.get_device_module(device).empty_cache()` breaks the CPU path.

### Graphs

| Device-hardcoded | Device-neutral |
|---|---|
| `torch.cuda.CUDAGraph()` with `torch.cuda.graph(...)` | `current_platform.get_device_graph_backend(device)`, then `backend.capture(pool=..., stream=...)`. `None` means run eager. |
| `torch.cuda.graph_pool_handle()` | `backend.graph_pool_handle()` |
| `torch.backends.cuda.sdp_kernel(...)` around a capture | `current_platform.graph_capture_attention()` |
| `torch.cuda.is_current_stream_capturing()` | `torch.get_device_module(device).is_current_stream_capturing()` |

```python
graph_backend = current_platform.get_device_graph_backend(device)
if graph_backend is None:
    return self.forward_eager(inputs)
else:
    pass
with current_platform.graph_capture_attention():
    with graph_backend.capture(pool=self.graph_pool, stream=capture_stream) as graph:
        static_outputs = self.forward_eager(static_inputs)
```

### Memory

| Device-hardcoded | Device-neutral |
|---|---|
| `torch.cuda.empty_cache()` | `current_platform.empty_cache()` or `torch.accelerator.empty_cache()` |
| `torch.cuda.mem_get_info(index)` | `current_platform.get_available_memory(index)` or `torch.accelerator.get_memory_info(device)`. Use `avail_gpu_mem(gpu_id)` from `sglang_omni/utils/misc.py` when "unknown" is an acceptable answer. |
| `torch.cuda.memory_reserved`, `max_memory_reserved`, `memory_allocated`, `reset_peak_memory_stats` | the same names on `torch.accelerator`, given the device |
| `pin_memory=True` on a path that runs without a card | `is_pin_memory_available(device)` from `sglang.srt.utils.common` |
| `pynvml`, `nvidia-smi` | NVIDIA-only. Keep it behind `current_platform.is_cuda()`, and give other platforms `current_platform.get_available_memory` and `get_device_name`. |

`NPUOmniPlatform` does not compose `NPUDeviceMixin` yet. On NPU, `current_platform.empty_cache()` and `synchronize()` are silent no-ops and the memory queries raise `NotImplementedError`. Prefer the device's own module there, and do not count `current_platform.*` as NPU coverage.

### Numerics, seeds and profiling

| Device-hardcoded | Device-neutral |
|---|---|
| `torch.cuda.amp.autocast(...)` | `torch.amp.autocast(device.type, ...)` |
| `torch.cuda.manual_seed(_all)` | `set_random_seed(seed)` from `sglang_omni/utils/misc.py` |
| `torch.cuda.OutOfMemoryError` | `torch.OutOfMemoryError`; the CUDA name is an alias of it |
| `torch.backends.cuda.matmul.allow_tf32` | `torch.set_float32_matmul_precision(...)`. A `torch.backends.cudnn` flag is CUDA-only, so scope it where a reader can see that. |
| `torch.version.cuda` | a build query, `None` on XPU and NPU builds. Gate on the platform predicate. |
| `ProfilerActivity.CUDA` | `profiler_activities()` from `sglang_omni/profiler/torch_profiler.py`. On XPU a CUDA-only list records zero kernel events. Upstream still leaves `get_torch_profiler_activity()` unimplemented. |
| `torch.cuda.nvtx.range(name)` | `torch.profiler.record_function(name)` |
| `torch.float64` on a device path | check `current_platform.is_float64_supported()`; MPS has no float64 |

### Distributed and processes

| Device-hardcoded | Device-neutral |
|---|---|
| `backend="nccl"` | `current_platform.get_torch_distributed_backend_str()`: xccl on XPU, hccl on NPU, gloo on CPU |
| setting `CUDA_VISIBLE_DEVICES` for a child | `current_platform.get_stage_process_env(spec, env)`. XPU masks with `ZE_AFFINITY_MASK`, ROCm with `HIP_VISIBLE_DEVICES`, NPU with `ASCEND_RT_VISIBLE_DEVICES`. |
| choosing a stage-to-stage transport | `current_platform.get_intra_node_transport()` |

The neutral backend string is necessary but not sufficient. XCCL point-to-point blocks the host until the peer posts its side, so a two-way concurrent isend/irecv protocol that works on NCCL can deadlock on XPU. `ZE_AFFINITY_MASK` is applied in ascending order: mask `5,2` makes physical card 2 logical 0. `CUDA_VISIBLE_DEVICES` keeps the order given, so do not port index arithmetic between the two.

### Kernels

| Device-hardcoded | Device-neutral |
|---|---|
| `import flashinfer`, `flash_attn`, `cutlass`, or `from sgl_kernel import op` at the top of a shared module | a platform hook (`get_rmsnorm`, `get_fused_qk_norm_rope`, `get_fused_qk_norm_rope_with_cos_sin_cache`, `get_joint_rope_inplace_kernel`, or a new one) with a torch reference fallback. The vendor file does the import. |
| a Triton launch passing `enable_reflect_ftz=...` | NVIDIA-only; Intel Triton raises `KeyError`. Pass `**({"enable_reflect_ftz": False} if x.is_cuda else {})`, as `sglang_omni/utils/snake_beta.py` does. `enable_fp_fusion` is accepted on both. |

A module-scope vendor import makes the whole module fail to import wherever that package is missing. `sgl_kernel` on XPU is a different build with a different op set, so `from sgl_kernel import op` can fail there even though the package exists.

## Branching on the platform

When a branch is the right answer, these are the traps:

- `is_cuda()` is False on ROCm and MUSA, though both present through `torch.cuda` and usually share the CUDA path. `is_cuda_alike()` is True for CUDA, ROCm and MUSA. An `else` hung off `if current_platform.is_cuda():` silently covers ROCm, MUSA, XPU, NPU and MPS at once.
- `tensor.is_cuda` is True on ROCm and False on XPU, NPU and MPS.
- With no accelerator visible, `current_platform` is `OmniSRTPlatform`. Its `device_type` is `"cpu"`, but `is_cpu()` and every other `is_*()` return False. A "not CUDA, so X" branch and an "is_cpu(), so skip" branch both send it down the wrong arm.
- `torch.accelerator.current_accelerator()` reports what torch was built for, not a usable card. On a torch 2.14 XPU build with the card hidden, it still answers `xpu` while `is_available()` is False, and `torch.accelerator.synchronize()` and `get_memory_info()` then segfault the process. Hold a real device, or check `is_available()`, first.

So:

- Route on the positive predicate for the platform you serve, and keep the old path as the `else`: `if current_platform.is_xpu() or current_platform.is_npu(): <new path> else: <old path>`.
- Prefer a capability to a vendor. `get_device_graph_backend(device) is None`, `has_device_streams(device)` and `supports_graph_captured_fft()` read better, and stay correct when a platform is added.
- Every `if` under `sglang_omni/` carries an `else` (the `if-else-required` hook), so the other arm is always on the page. Read it.

## When the difference belongs in a hook

If two platforms need different behavior, add a method to `OmniPlatform` in `sglang_omni/platforms/interface.py` rather than branching in shared code:

- Name it by capability, not vendor: `supports_graph_captured_fft()`, `enable_talker_graph()`, `get_rmsnorm()`.
- The base default returns today's behavior, so CUDA and every platform you do not override stay unchanged. Override it in the platform file that needs something else.
- For an optional kernel, return `None` and let the model decide whether the capability is required.
- Test it in `tests/unit_test/test_platforms.py` or `tests/unit_test/platforms/`. Assert the override, and assert that at least one other platform keeps the default.
- A platform class composes the upstream `*DeviceMixin` for its device, as `class XPUOmniPlatform(XpuDeviceMixin, OmniPlatform)` does. Without it, `empty_cache()` and `synchronize()` are silent no-ops and the memory queries raise `NotImplementedError`. `test_cpu_cuda_rocm_xpu_and_musa_implement_the_shared_device_operations` guards the platforms it names.

## Silent degradations

These do not crash, so look for them on purpose:

- A graph path whose backend is `None` runs eager: slower, not wrong. That is fine when intended, and a finding when the PR claims a speedup on that platform.
- A fused-kernel module swapped in on a platform whose kernel then rejects the tensor falls back to eager on every call. A "fused N modules" log line is not proof the kernel runs.
- `empty_cache()` and `synchronize()` on a platform class without its mixin do nothing.
- A profiler activity list without the device activity records no kernels.
- A guarded CUDA-only telemetry or memory path that returns a placeholder elsewhere, so a log or a budget is quietly wrong on XPU or NPU.

## Tests

- A test of shared behavior that needs an accelerator carries `@pytest.mark.accelerator` unconditionally, and gets its device in the body, never from `device="cuda"`. For a test that needs streams or graphs, call `require_device_streams()` from `tests/unit_test/fixtures/accelerator.py`, which returns the host's CUDA, MUSA or XPU device and skips elsewhere. Otherwise use `torch.device(current_platform.device_type)`.
- `require_cuda()` is for behavior that is CUDA-only by design.
- Keep accelerator probes out of module scope and `skipif` conditions, as `tests/README.md` requires: marker filtering happens after import. The scanner flags a probe in a decorator or `pytestmark` even though it guards the code below it.
- The main unit lane in `.github/workflows/test.yaml` runs `pytest tests/ -m "not benchmark and not accelerator" -x` with `CUDA_VISIBLE_DEVICES=""`. The XPU lane in `.github/workflows/omni-xpu-ci.yaml` runs an explicit file list behind a path filter. A new device-neutral accelerator test runs on XPU CI only once its file is in that list and its directory is in the filter. An XPU-only test goes under `tests/unit_test/xpu/` and skips off XPU.
- To build the no-card fallback in a test, use `load_platform_class("sglang.srt.platforms.interface.SRTPlatform")()` rather than a fake.

## Reporting

Write one finding per construct. Give `path:line`, then what breaks on which platform (crash, wrong result, silent no-op, or slower), then the replacement from the tables. Lead with what breaks. Do not report external names, platform-specific files, or guarded hits whose guard is right. If the change makes a path newly reachable on a platform, ask which platforms were run.

## Fixing

1. Rewrite each unguarded hit with its replacement. Where behavior must differ, add or reuse a hook as described above.
2. Re-run the scanner until only intended guarded hits remain.
3. Run `python scripts/check_if_else.py <files>` and `python scripts/check_leading_underscore.py <files>` on the changed Python files. Pre-commit runs both.
4. Run the touched unit tests with the accelerator hidden. Use `CUDA_VISIBLE_DEVICES=""` on a CUDA host; `ZE_AFFINITY_MASK=99` gives the same no-card fallback on an Intel host. Then run the `accelerator` tests on every platform you have.
5. In the PR, list the platforms you ran, and separately the ones you changed only by spelling.
