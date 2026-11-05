#!/usr/bin/env bash
# Intel XPU unit suite, split into the tier that gates a pull request and the tier
# that does not.
#
# Usage: xpu_unit_tests.sh [pr|nightly] [extra pytest args...]
#
# The lists live here rather than inline in omni-xpu-ci.yaml so a model enablement
# adds its coverage in one place and a developer can run the same set locally.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}" || exit 2

# Device and platform layer: the abstraction XPU rides on, so a regression here
# breaks every model at once.
DEVICE_LAYER_TESTS=(
  tests/unit_test/xpu
  tests/unit_test/cpu/test_device_layer.py
  tests/unit_test/test_platforms.py
  tests/unit_test/test_stage_device_contract.py
  tests/unit_test/platforms/test_device_graph.py
  tests/unit_test/utils/test_device.py
  tests/unit_test/diagnostics/test_gpu.py
  tests/unit_test/scheduling/test_server_args_builder_device.py
  tests/unit_test/pipeline/test_stage_process_env.py
  tests/unit_test/pipeline/test_comm_router.py
  tests/unit_test/pipeline/test_gpu_memory.py
  tests/unit_test/pipeline/test_payload_device_restore.py
  tests/unit_test/model_runner/test_base_hooks.py
  tests/unit_test/model_runner/test_weight_share_load_path.py
)

# Per-model tests that assert something XPU-specific, or that gate a stage XPU
# serves. Weight-free: they exercise the CPU side of each pipeline.
MODEL_TESTS=(
  tests/unit_test/qwen3_asr/test_pipeline.py
  tests/unit_test/qwen3_asr/test_encoder_service.py
  tests/unit_test/qwen3_tts/test_pipeline.py
  tests/unit_test/qwen3_tts/test_predictor_cuda_graph.py
  tests/unit_test/qwen3_omni/test_code2wav_cuda_graph.py
  tests/unit_test/qwen3_omni/test_fp8_backend_config.py
  tests/unit_test/qwen3_omni/test_thinker_fused_rope.py
  tests/unit_test/dots_tts/test_flow_head.py
  tests/unit_test/dots_tts/test_tail.py
  tests/unit_test/dots_tts/test_vocoder_cuda_graph.py
  tests/unit_test/nemotron3_5_asr/test_hf_compat.py
  tests/unit_test/nemotron3_5_asr/test_model_runner.py
  tests/unit_test/nemotron_diarization/test_stages.py
  tests/unit_test/ming_tts/test_stages.py
  tests/unit_test/ming3_tts/test_streaming_vocoder.py
  tests/unit_test/ming_omni/test_audio_encoder.py
  tests/unit_test/fun_cosyvoice3/test_vocoder.py
  tests/unit_test/audar_tts/test_pipeline.py
  tests/unit_test/zonos2/test_pipeline.py
  tests/unit_test/whisper_asr/test_pipeline.py
  tests/unit_test/arkasr/test_encoder_cuda_graph.py
  tests/unit_test/moss_transcribe_diarize/test_encoder_cuda_graph.py
  tests/unit_test/minimax_music3/test_core.py
)

# Nightly-only, and in their own interpreter: MiniCPM-o Code2Wav runs linear on a priority stream, which segfaults on the container's first XPU card under torch 2.14, and a shared process would lose every result above it.
QUARANTINED_TESTS=(
  tests/unit_test/minicpm_o/test_code2wav.py
)

TIER="${1:-pr}"
shift 2>/dev/null || true

case "${TIER}" in
  pr|nightly) LANES=("${DEVICE_LAYER_TESTS[@]}" "${MODEL_TESTS[@]}") ;;
  *)          echo "usage: $0 [pr|nightly] [extra pytest args...]" >&2; exit 1 ;;
esac

# pytest-randomly would shuffle across files, and tests/unit_test/pipeline leaves
# torch.xpu unusable for whatever runs after it in the same process.
PYTEST_ARGS=(-v -p no:cacheprovider -p no:randomly "$@")

echo "== XPU unit suite (${TIER}) =="
python -c 'import torch; print(f"  torch {torch.__version__}, xpu={torch.xpu.is_available()}, devices={torch.xpu.device_count() if torch.xpu.is_available() else 0}")'

python -m pytest "${LANES[@]}" "${PYTEST_ARGS[@]}"
status=$?

# Exit 5 is "no tests collected", which would otherwise pass silently after a rename.
if [[ "${status}" -eq 5 ]]; then
  echo "::error::no tests collected; a path in scripts/ci/xpu/xpu_unit_tests.sh is gone"
  exit 1
fi

# Reported, not gating: the crash below is a known defect, and a lane that is red every night stops being read.
if [[ "${TIER}" == "nightly" ]]; then
  python -m pytest "${QUARANTINED_TESTS[@]}" "${PYTEST_ARGS[@]}"
  quarantined_status=$?
  echo "::warning::quarantined XPU tests exited ${quarantined_status}: ${QUARANTINED_TESTS[*]}"
fi

exit "${status}"
