# SPDX-License-Identifier: Apache-2.0
"""Compatibility shims for upstream qwen-tts."""

from __future__ import annotations

import inspect
import threading
from typing import Any, Callable

import torch

_APPLY_LOCK = threading.Lock()
_PATCHED_FLAG = "_sglang_omni_qwen_tts_compat_patched"
_MASK_PATCHED_FLAG = "_sglang_omni_qwen_tts_mask_patched"


def _patch_qwen_tts_mask_builders() -> None:
    """Drop mask kwargs transformers 5.x no longer accepts.

    qwen-tts passes ``input_embeds``, renamed to ``inputs_embeds`` upstream, so the
    unpatched call raises TypeError. Idempotent; no-op if qwen-tts is absent.
    """
    try:
        from qwen_tts.core.tokenizer_12hz import modeling_qwen3_tts_tokenizer_v2 as _mod
    except Exception:  # noqa: BLE001 - qwen-tts optional / layout drift
        return

    def _wrap(orig: Callable[..., Any]) -> Callable[..., Any]:
        accepted = frozenset(inspect.signature(orig).parameters)

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if "input_embeds" in kwargs and "input_embeds" not in accepted:
                kwargs.setdefault("inputs_embeds", kwargs.pop("input_embeds"))
            return orig(*args, **{k: v for k, v in kwargs.items() if k in accepted})

        setattr(wrapper, _MASK_PATCHED_FLAG, True)
        return wrapper

    for name in ("create_causal_mask", "create_sliding_window_causal_mask"):
        original = getattr(_mod, name, None)
        if original is not None and not getattr(original, _MASK_PATCHED_FLAG, False):
            setattr(_mod, name, _wrap(original))


def _compute_default_rope_parameters(
    config: Any,
    device: torch.device | None = None,
    seq_len: int | None = None,
    layer_type: str | None = None,
) -> tuple[torch.Tensor, float]:
    del seq_len, layer_type
    base = getattr(config, "rope_theta", getattr(config, "default_theta", 10000.0))
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        head_dim = config.hidden_size // config.num_attention_heads
    dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, dim, 2, dtype=torch.int64).to(
                device=device, dtype=torch.float
            )
            / dim
        )
    )
    return inv_freq, 1.0


def _patch_check_model_inputs(generic: Any) -> None:
    """Let transformers 5.x ``check_model_inputs`` be used as a bare decorator."""
    current = generic.check_model_inputs
    if getattr(current, _PATCHED_FLAG, False):
        return

    try:
        signature = inspect.signature(current)
    except (TypeError, ValueError):
        return

    params = list(signature.parameters.values())
    needs_func_arg = (
        len(params) == 1
        and params[0].default is inspect.Parameter.empty
        and params[0].kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    )
    if not needs_func_arg:
        return

    original = current

    def check_model_inputs_compat(
        func: Callable[..., Any] | None = None,
    ) -> Callable[..., Any]:
        if func is None:

            def decorator(inner: Callable[..., Any]) -> Callable[..., Any]:
                return original(inner)

            return decorator
        return original(func)

    check_model_inputs_compat.__name__ = getattr(
        original, "__name__", "check_model_inputs"
    )
    check_model_inputs_compat.__doc__ = getattr(original, "__doc__", None)
    setattr(check_model_inputs_compat, _PATCHED_FLAG, True)
    generic.check_model_inputs = check_model_inputs_compat


def apply_qwen_tts_transformers_compatibility_patches() -> None:
    """Patch Transformers APIs expected by qwen-tts."""
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    from transformers.utils import generic

    with _APPLY_LOCK:
        ROPE_INIT_FUNCTIONS.setdefault("default", _compute_default_rope_parameters)
        _patch_check_model_inputs(generic)
        # Last: importing qwen-tts needs the shim above already installed, or the
        # import raises and this silently patches nothing.
        _patch_qwen_tts_mask_builders()
