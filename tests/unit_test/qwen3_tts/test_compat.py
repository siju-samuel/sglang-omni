# SPDX-License-Identifier: Apache-2.0
"""Qwen3-TTS transformers compatibility shims."""

from __future__ import annotations

import builtins
import sys
from types import ModuleType
from typing import Any

import pytest

from sglang_omni.models.qwen3_tts import compat

_PARENT = "qwen_tts.core.tokenizer_12hz"
_LEAF = f"{_PARENT}.modeling_qwen3_tts_tokenizer_v2"


def _install_fake_qwen_tts(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Stand in for qwen-tts, importable only once the shim is installed.

    The real leaf module is decorated with a bare ``@check_model_inputs()`` at
    import time, so it raises TypeError until the shim exists. A finder that
    checks the shim on each import attempt reproduces that ordering dependency.
    """
    leaf = ModuleType(_LEAF)
    leaf.create_causal_mask = lambda inputs_embeds=None, cache_position=None: (
        inputs_embeds,
        cache_position,
    )
    leaf.create_sliding_window_causal_mask = lambda inputs_embeds=None: inputs_embeds

    names = ("qwen_tts", "qwen_tts.core", _PARENT)
    chain = {name: ModuleType(name) for name in names}
    for child in ("qwen_tts.core", _PARENT):
        parent, _, attr = child.rpartition(".")
        setattr(chain[parent], attr, chain[child])
    for name, module in chain.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delitem(sys.modules, _LEAF, raising=False)

    real_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> ModuleType:
        if name.startswith("qwen_tts"):
            from transformers.utils import generic

            if not getattr(generic.check_model_inputs, compat._PATCHED_FLAG, False):
                raise TypeError(
                    "check_model_inputs() missing 1 required positional argument"
                )
            # Shim present: publish the fake leaf and serve it without touching
            # the real (installed) qwen-tts package.
            setattr(chain[_PARENT], _LEAF.rpartition(".")[2], leaf)
            sys.modules[_LEAF] = leaf
            return chain.get(name, chain[_PARENT])
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    return leaf


@pytest.fixture
def unshimmed_transformers(monkeypatch: pytest.MonkeyPatch):
    """A check_model_inputs still needing the shim, so apply() installs one."""
    from transformers.utils import generic

    monkeypatch.setattr(generic, "check_model_inputs", lambda func: func)
    return generic


def test_mask_builders_are_patched_by_the_first_apply(
    monkeypatch: pytest.MonkeyPatch, unshimmed_transformers
) -> None:
    """One apply() must patch the mask builders, not two.

    Guards a regression where the mask patch ran before the check_model_inputs
    shim its own import depends on, so the import raised, the broad except
    swallowed it, and the builders were silently left unpatched.
    """
    leaf = _install_fake_qwen_tts(monkeypatch)

    compat.apply_qwen_tts_transformers_compatibility_patches()

    assert getattr(unshimmed_transformers.check_model_inputs, compat._PATCHED_FLAG)
    assert getattr(leaf.create_causal_mask, compat._MASK_PATCHED_FLAG, False)
    assert getattr(
        leaf.create_sliding_window_causal_mask, compat._MASK_PATCHED_FLAG, False
    )


def test_mask_builders_are_patched_even_when_the_shim_is_already_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The early return for an installed shim must not skip the mask patch."""
    from transformers.utils import generic

    def already_shimmed(func=None):
        return func

    setattr(already_shimmed, compat._PATCHED_FLAG, True)
    monkeypatch.setattr(generic, "check_model_inputs", already_shimmed)
    leaf = _install_fake_qwen_tts(monkeypatch)

    compat.apply_qwen_tts_transformers_compatibility_patches()

    assert getattr(leaf.create_causal_mask, compat._MASK_PATCHED_FLAG, False)


def test_patched_mask_builder_renames_input_embeds_and_drops_extra_kwargs(
    monkeypatch: pytest.MonkeyPatch, unshimmed_transformers
) -> None:
    leaf = _install_fake_qwen_tts(monkeypatch)

    compat.apply_qwen_tts_transformers_compatibility_patches()

    # qwen-tts passes the pre-5.x name plus kwargs transformers no longer accepts.
    assert leaf.create_causal_mask(
        input_embeds="embeds", cache_position=3, position_ids=[0]
    ) == ("embeds", 3)


def test_applying_twice_does_not_double_wrap(
    monkeypatch: pytest.MonkeyPatch, unshimmed_transformers
) -> None:
    leaf = _install_fake_qwen_tts(monkeypatch)

    compat.apply_qwen_tts_transformers_compatibility_patches()
    once = leaf.create_causal_mask
    compat.apply_qwen_tts_transformers_compatibility_patches()

    assert leaf.create_causal_mask is once
