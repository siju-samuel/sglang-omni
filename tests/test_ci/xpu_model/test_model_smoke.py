# SPDX-License-Identifier: Apache-2.0
"""Minimal functional smoke: every model Intel XPU serves answers one request.

Content is checked, not just status; the request shapes and their assertions live
in exercises.py. No WER, MOS, or throughput floors -- those need the eval extra
and belong in tests/test_model/.

    pytest tests/test_ci/xpu_model -v                     # every model with weights
    pytest tests/test_ci/xpu_model -v -k "asr or tts"     # a subset
    REQUIRE_FREE_GPU=0 pytest tests/test_ci/xpu_model -v  # ignore who else is on the card
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from tests.test_ci.xpu_model.conftest import resolve_checkpoint
from tests.test_ci.xpu_model.exercises import EXERCISES
from tests.test_ci.xpu_model.model_table import XPU_MODELS, XpuModel

pytestmark = pytest.mark.xpu_serving


def serve_plan(
    model: XpuModel, claim: Callable[[int], list[int]]
) -> tuple[list[str], dict[str, str]]:
    """The serve arguments and environment this row needs.

    The flags themselves come from the row, which carries its section of the XPU
    install guide; this only claims the cards and hands the row's own
    environment over. A stage's gpu id indexes into ZE_AFFINITY_MASK rather than
    the host, which is why the mask is set here and the rows count lanes from
    zero.
    """
    cards = claim(model.cards)
    env = {"ZE_AFFINITY_MASK": ",".join(str(card) for card in cards)}
    env["SGLANG_OMNI_STARTUP_TIMEOUT"] = str(model.startup_seconds)
    env.update(dict(model.env))
    return list(model.serve_args), env


@pytest.mark.parametrize("model", XPU_MODELS, ids=[m.key for m in XPU_MODELS])
def test_the_model_serves_and_answers_on_xpu(
    model: XpuModel, xpu_host: None, serve_xpu_model, claim_cards
) -> None:
    """Serve ``model`` on the XPU, send one request, and check what came back."""
    if model.serving:
        pytest.skip(model.serving)
    else:
        pass
    checkpoint = resolve_checkpoint(model)
    extra_args, env = serve_plan(model, claim_cards)

    with serve_xpu_model(model, checkpoint, extra_args, env) as base_url:
        verdict = EXERCISES[model.kind](base_url, checkpoint, model)

    print(f"\n{model.key}: {verdict}")
