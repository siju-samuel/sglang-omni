#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""List the device-hardcoded constructs a change adds to shared code.

Reads the lines a diff adds, or whole files when paths are given, skips the files
that are platform-specific by design, and marks a hit that sits under a platform
guard so the reviewer checks the guard instead of the line. Exits 1 while any
unguarded hit remains.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

VENDOR = r"(?:cuda|xpu|npu|musa|mps)"
TORCH_VENDOR = rf"torch\.{VENDOR}\."

PLATFORM_SPECIFIC_PATH = re.compile(
    r"^(?:sglang_omni/platforms/(?:apple|cpu|cuda|device_graph|musa|npu|rocm|xpu)\.py$"
    r"|sglang_omni/mps/"
    r"|sglang_omni/relay/cuda_ipc\.py$"
    r"|sglang_omni/.+/cutedsl/"
    r"|sglang_omni/vendor/"
    r"|tests/unit_test/platforms/"
    r"|tests/unit_test/test_platforms\.py$"
    r"|tests/unit_test/xpu/"
    r"|\.claude/)"
)

PLATFORM_PREDICATE = re.compile(
    r"\bis_(?:cuda|cuda_alike|rocm|musa|xpu|npu|mps)\(\)"
    r"|\.is_(?:cuda|xpu|npu|mps)\b"
    rf"|\.type\s*==\s*[\"']{VENDOR}[\"']"
    rf"|{TORCH_VENDOR}is_available\(\)"
    r"|\brequire_cuda\("
)
PLATFORM_CLASS = re.compile(
    r"^\s*class\s+\w*(?:Cuda|CUDA|Xpu|XPU|Npu|NPU|Rocm|ROCm|ROCM|Musa|MUSA|Mps|MPS|Apple)"
)
NEGATIVE_DEVICE_TYPE = re.compile(rf"\.type\s*!=\s*[\"']{VENDOR}[\"']")
EARLY_EXIT = re.compile(r"^\s*(?:raise|return|continue|pytest\.skip)\b")
NEGATED_PREDICATE = re.compile(r"\bnot\s+[\w.]+(?:\(\))?")
HUNK_HEADER = re.compile(r"^@@ -\d+(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

GRAPH_NAMES = (
    "CUDAGraph|XPUGraph|NPUGraph|graph|graphs|graph_pool_handle"
    "|make_graphed_callables|is_current_stream_capturing"
)
STREAM_NAMES = (
    "Stream|ExternalStream|StreamContext|stream|current_stream|default_stream"
    "|set_stream|Event"
)
MEMORY_NAMES = (
    "empty_cache|mem_get_info|memory_allocated|memory_reserved|max_memory_allocated"
    "|max_memory_reserved|reset_peak_memory_stats|memory_stats|memory_summary"
)
DEVICE_NAMES = (
    "is_available|device_count|current_device|set_device|device"
    "|get_device_properties|get_device_capability|get_device_name"
)
SEED_NAMES = "manual_seed|manual_seed_all|seed|seed_all|get_rng_state|set_rng_state"
NAMED_ELSEWHERE = "|".join(
    (
        GRAPH_NAMES,
        STREAM_NAMES,
        "synchronize",
        MEMORY_NAMES,
        DEVICE_NAMES,
        SEED_NAMES,
        "amp|nvtx|OutOfMemoryError",
    )
)
DEVICE_BRANCH = re.compile(
    rf"[!=]=\s*[\"']{VENDOR}[\"']|[\"']{VENDOR}[\"']\s*[!=]="
    rf"|\bin\s*[(\[{{][^)\]}}]*[\"']{VENDOR}[\"']"
)


@dataclass(frozen=True, kw_only=True)
class Rule:
    name: str
    pattern: re.Pattern[str]
    advice: str
    unless: re.Pattern[str] | None = None


@dataclass(frozen=True, kw_only=True)
class Hit:
    path: str
    line_number: int
    rule: Rule
    code: str
    is_guarded: bool


RULES = (
    Rule(
        name="graph",
        pattern=re.compile(rf"{TORCH_VENDOR}(?:{GRAPH_NAMES})\b"),
        advice="current_platform.get_device_graph_backend(device); None means run eager.",
    ),
    Rule(
        name="stream",
        pattern=re.compile(rf"{TORCH_VENDOR}(?:{STREAM_NAMES})\b"),
        advice="torch.get_device_module(device).Stream/stream/current_stream, new_device_event(device); gate CPU-reachable paths on has_device_streams(device).",
    ),
    Rule(
        name="synchronize",
        pattern=re.compile(rf"{TORCH_VENDOR}synchronize\b"),
        advice="torch.get_device_module(device).synchronize(device); with no device in hand, current_platform.synchronize().",
    ),
    Rule(
        name="memory",
        pattern=re.compile(rf"{TORCH_VENDOR}(?:{MEMORY_NAMES})\b"),
        advice="current_platform.empty_cache() / get_available_memory(index), or the same names on torch.accelerator with the device.",
    ),
    Rule(
        name="device-query",
        pattern=re.compile(rf"{TORCH_VENDOR}(?:{DEVICE_NAMES})\b"),
        advice="current_platform.device_type / set_device / get_device_name, torch.get_device_module(device).current_device(), device_guard(device).",
    ),
    Rule(
        name="seed",
        pattern=re.compile(rf"{TORCH_VENDOR}(?:{SEED_NAMES})\b"),
        advice="set_random_seed(seed) from sglang_omni.utils.misc.",
    ),
    Rule(
        name="autocast",
        pattern=re.compile(rf"{TORCH_VENDOR}amp\b"),
        advice="torch.amp.autocast(device.type, ...).",
    ),
    Rule(
        name="nvtx",
        pattern=re.compile(rf"{TORCH_VENDOR}nvtx\b"),
        advice="torch.profiler.record_function(name).",
    ),
    Rule(
        name="out-of-memory",
        pattern=re.compile(rf"{TORCH_VENDOR}OutOfMemoryError\b"),
        advice="torch.OutOfMemoryError.",
    ),
    Rule(
        name="vendor-module",
        pattern=re.compile(rf"{TORCH_VENDOR}(?!(?:{NAMED_ELSEWHERE})\b)\w+"),
        advice="Reach the operation through torch.get_device_module(device) or torch.accelerator.",
    ),
    Rule(
        name="dot-cuda",
        pattern=re.compile(r"\.cuda\("),
        advice=".to(device) with the tensor's or the stage's resolved device.",
    ),
    Rule(
        name="device-branch",
        pattern=DEVICE_BRANCH,
        advice="Ask for the capability (has_device_streams, get_device_graph_backend, a hook); if it is vendor-only, check the other arm.",
    ),
    Rule(
        name="device-literal",
        pattern=re.compile(
            rf"(?<![\w.])[\"']{VENDOR}(?::\d+)?[\"']|\bf[\"']{VENDOR}:\{{"
        ),
        advice="tensor.device, resolve_concrete_device(device, index), or torch.device(current_platform.device_type).",
        unless=DEVICE_BRANCH,
    ),
    Rule(
        name="tensor-predicate",
        pattern=re.compile(r"\.is_(?:cuda|xpu|npu|mps)\b(?!\s*\()"),
        advice="tensor.is_cuda is True on ROCm and False on XPU/NPU/MPS; for 'on an accelerator' use has_device_streams(tensor.device).",
    ),
    Rule(
        name="negated-platform",
        pattern=re.compile(
            r"\bnot\s+(?:[\w.]+\.)?is_(?:cuda|cuda_alike|rocm|musa|xpu|npu|mps)\(\)"
        ),
        advice="'not is_cuda()' also covers ROCm, MUSA, XPU, NPU, MPS and the no-card fallback; name the platform you mean.",
    ),
    Rule(
        name="cuda-branch",
        pattern=re.compile(r"^\s*(?:if|elif)\b.*\bis_cuda\(\)"),
        advice="Read the else arm: is_cuda() is False on ROCm and MUSA, which is_cuda_alike() covers.",
        unless=re.compile(r"\bnot\s+(?:[\w.]+\.)?is_cuda\(\)"),
    ),
    Rule(
        name="distributed-backend",
        pattern=re.compile(r"[\"']nccl[\"']"),
        advice="current_platform.get_torch_distributed_backend_str().",
    ),
    Rule(
        name="visible-devices",
        pattern=re.compile(
            r"\b(?:CUDA|HIP)_VISIBLE_DEVICES\b|\bZE_AFFINITY_MASK\b|\bASCEND_RT_VISIBLE_DEVICES\b"
        ),
        advice="current_platform.get_stage_process_env(spec, env); each vendor has its own mask.",
    ),
    Rule(
        name="nvml",
        pattern=re.compile(r"\bpynvml\b|\bnvml[A-Z]\w*|nvidia-smi"),
        advice="NVIDIA-only; keep behind current_platform.is_cuda() and give other platforms current_platform memory/name queries.",
    ),
    Rule(
        name="profiler-activity",
        pattern=re.compile(r"ProfilerActivity\.CUDA\b"),
        advice="profiler_activities() from sglang_omni.profiler.torch_profiler.",
    ),
    Rule(
        name="torch-backends",
        pattern=re.compile(r"torch\.backends\.(?:cuda|cudnn)\b"),
        advice="torch.nn.attention.sdpa_kernel / current_platform.graph_capture_attention(); torch.set_float32_matmul_precision.",
    ),
    Rule(
        name="build-version",
        pattern=re.compile(r"torch\.version\.(?:cuda|hip|xpu)\b"),
        advice="A build query, None on other builds; gate on the platform predicate.",
    ),
    Rule(
        name="triton-option",
        pattern=re.compile(r"\benable_reflect_ftz\s*="),
        advice='NVIDIA-only Triton option (KeyError on Intel): **({"enable_reflect_ftz": False} if x.is_cuda else {}).',
    ),
    Rule(
        name="vendor-import",
        pattern=re.compile(
            r"^(?:from|import)\s+(?:flashinfer|flash_attn|cutlass|cuda|cupy|apex|xformers"
            r"|sgl_kernel|torch_npu|intel_extension_for_pytorch)\b"
        ),
        advice="A module-scope vendor import breaks import where it is missing; reach the kernel through a platform hook.",
    ),
)


def run_git(repository_root: Path, *git_arguments: str) -> str:
    completed = subprocess.run(
        ["git", *git_arguments],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def strip_comment(code: str) -> str:
    """The line up to a # that is not inside a string literal."""
    quote: str | None = None
    for index, character in enumerate(code):
        if quote is not None:
            if character == quote and code[index - 1] != "\\":
                quote = None
        elif character in "\"'":
            quote = character
        elif character == "#":
            return code[:index]
    return code


def parse_added_lines(diff_text: str) -> dict[str, list[int]]:
    """Map each changed file to the line numbers its diff adds."""
    added_lines: dict[str, list[int]] = {}
    path: str | None = None
    new_line_number = 0
    remaining_old = 0
    remaining_new = 0
    for diff_line in diff_text.splitlines():
        if remaining_old > 0 or remaining_new > 0:
            if diff_line.startswith("+"):
                if path is not None:
                    added_lines.setdefault(path, []).append(new_line_number)
                new_line_number += 1
                remaining_new -= 1
            elif diff_line.startswith("-"):
                remaining_old -= 1
            elif diff_line.startswith(" "):
                new_line_number += 1
                remaining_old -= 1
                remaining_new -= 1
            continue
        if diff_line.startswith("+++ "):
            target = diff_line[4:]
            path = target[2:] if target.startswith("b/") else None
            continue
        hunk = HUNK_HEADER.match(diff_line)
        if hunk is not None:
            remaining_old = int(hunk.group(1) or "1")
            new_line_number = int(hunk.group(2))
            remaining_new = int(hunk.group(3) or "1")
    return added_lines


def indent_of(text: str) -> int:
    return len(text) - len(text.lstrip())


def is_early_exit_guard(lines: list[str], index: int) -> bool:
    """An assert or require_cuda() naming a platform, or an 'if' that exits off it."""
    text = strip_comment(lines[index])
    if re.match(r"\s*(?:assert\b|require_cuda\()", text):
        return bool(PLATFORM_PREDICATE.search(text))
    if not re.match(r"\s*if\b", text):
        return False
    exits_off_platform = NEGATIVE_DEVICE_TYPE.search(text) or (
        re.search(r"\bnot\b", text) and PLATFORM_PREDICATE.search(text)
    )
    if not exits_off_platform:
        return False
    if re.search(r":\s*(?:raise|return|continue|pytest\.skip)\b", text):
        return True
    following = next((line for line in lines[index + 1 :] if line.strip()), "")
    return bool(EARLY_EXIT.match(following))


def decorators_name_platform(lines: list[str], definition_index: int) -> bool:
    """Whether the decorators above a def, multi-line ones included, name a platform."""
    definition_indent = indent_of(lines[definition_index])
    decorator_lines: list[str] = []
    for text in reversed(lines[:definition_index]):
        stripped = text.strip()
        if not stripped:
            break
        if indent_of(text) == definition_indent and not stripped.startswith(("@", ")", "]")):
            break
        decorator_lines.append(text)
    return bool(PLATFORM_PREDICATE.search("\n".join(decorator_lines)))


def is_guarded(lines: list[str], line_index: int, file_is_guarded: bool) -> bool:
    """Whether an enclosing block, a decorator or an earlier early exit names a platform."""
    if file_is_guarded:
        return True
    current_indent = indent_of(lines[line_index])
    for index in range(line_index - 1, -1, -1):
        text = lines[index]
        stripped = text.strip()
        if not stripped or stripped.startswith(("#", ")", "]", "}")):
            continue
        indent = indent_of(text)
        if indent > current_indent:
            continue
        if indent == current_indent:
            if is_early_exit_guard(lines, index):
                return True
            continue
        current_indent = indent
        header = NEGATED_PREDICATE.sub("", strip_comment(text))
        if PLATFORM_PREDICATE.search(header) or PLATFORM_CLASS.search(header):
            return True
        if re.match(r"\s*(?:async\s+)?def\b", text) and decorators_name_platform(lines, index):
            return True
        if indent == 0:
            break
    return False


def platform_pytestmark_lines(lines: list[str]) -> set[int]:
    """0-based lines of a module-level pytestmark statement that names a platform."""
    for index, text in enumerate(lines):
        if not text.startswith("pytestmark"):
            continue
        statement = [index]
        for continuation_index in range(index + 1, len(lines)):
            if lines[continuation_index].startswith((" ", "\t", "]", ")")):
                statement.append(continuation_index)
            else:
                break
        if PLATFORM_PREDICATE.search("\n".join(lines[i] for i in statement)):
            return set(statement)
    return set()


def scan_file(path: str, source: str, line_numbers: list[int] | None) -> list[Hit]:
    """Hits on the given 1-based lines of one file, or on every line when None.

    A platform-naming pytestmark guards the rest of the file, but its own probe
    stays a hit: tests/README.md keeps accelerator probes out of module scope.
    """
    lines = source.splitlines()
    pytestmark_lines = platform_pytestmark_lines(lines)
    targets = range(1, len(lines) + 1) if line_numbers is None else line_numbers
    hits: list[Hit] = []
    for line_number in targets:
        if line_number > len(lines):
            continue
        code = strip_comment(lines[line_number - 1])
        if not code.strip():
            continue
        for rule in RULES:
            if not rule.pattern.search(code):
                continue
            if rule.unless is not None and rule.unless.search(code):
                continue
            hits.append(
                Hit(
                    path=path,
                    line_number=line_number,
                    rule=rule,
                    code=code.strip(),
                    is_guarded=(line_number - 1) not in pytestmark_lines
                    and is_guarded(lines, line_number - 1, bool(pytestmark_lines)),
                )
            )
    return hits


def collect_targets(
    repository_root: Path, base: str, head: str | None, paths: list[str]
) -> dict[str, list[int] | None]:
    """Files to scan, each with the added lines to read, or None for the whole file."""
    if paths:
        targets: dict[str, list[int] | None] = {}
        for raw_path in paths:
            resolved = Path(raw_path).resolve()
            candidates = sorted(resolved.rglob("*.py")) if resolved.is_dir() else [resolved]
            for candidate in candidates:
                targets[candidate.relative_to(repository_root).as_posix()] = None
        return targets
    diff_options = ("diff", "--unified=0", "--no-color", "--no-ext-diff", "--diff-filter=d")
    if head is not None:
        diff_text = run_git(repository_root, *diff_options, f"{base}...{head}", "--", "*.py")
        return dict(parse_added_lines(diff_text))
    merge_base = run_git(repository_root, "merge-base", base, "HEAD").strip()
    diff_text = run_git(repository_root, *diff_options, merge_base, "--", "*.py")
    targets = dict(parse_added_lines(diff_text))
    untracked = run_git(repository_root, "ls-files", "--others", "--exclude-standard", "--", "*.py")
    for path in untracked.splitlines():
        targets[path] = None
    return targets


def read_source(repository_root: Path, path: str, head: str | None) -> str:
    """The file at head, else on disk, else in the index (a sparse checkout's skip-worktree file)."""
    if head is not None:
        return run_git(repository_root, "show", f"{head}:{path}")
    on_disk = repository_root / path
    if on_disk.exists():
        return on_disk.read_text(encoding="utf-8", errors="replace")
    return run_git(repository_root, "show", f":{path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*", help="scan these files or directories whole instead of a diff")
    parser.add_argument("--base", default="origin/main", help="diff base (default: origin/main)")
    parser.add_argument("--head", default=None, help="diff head ref (default: the working tree, untracked files included)")
    parser.add_argument("--include-guarded", action="store_true", help="also list hits under a platform guard")
    options = parser.parse_args()

    repository_root = Path(run_git(Path.cwd(), "rev-parse", "--show-toplevel").strip())
    targets = collect_targets(repository_root, options.base, options.head, options.paths)
    skipped_paths = sorted(path for path in targets if PLATFORM_SPECIFIC_PATH.match(path))
    hits: list[Hit] = []
    for path, line_numbers in sorted(targets.items()):
        if path in skipped_paths:
            continue
        hits.extend(scan_file(path, read_source(repository_root, path, options.head), line_numbers))

    unguarded = [hit for hit in hits if not hit.is_guarded]
    guarded = [hit for hit in hits if hit.is_guarded]
    shown = hits if options.include_guarded else unguarded
    for hit in shown:
        marker = " [guarded]" if hit.is_guarded else ""
        print(f"{hit.path}:{hit.line_number}: {hit.rule.name}{marker}: {hit.code}")

    print(
        f"\n{len(unguarded)} unguarded and {len(guarded)} guarded hit(s) in "
        f"{len({hit.path for hit in hits})} file(s); "
        f"{len(skipped_paths)} platform-specific file(s) skipped."
    )
    if guarded and not options.include_guarded:
        print("Pass --include-guarded to list the guarded hits and check each guard.")
    rules_shown = {hit.rule.name: hit.rule for hit in shown}
    for name, rule in rules_shown.items():
        print(f"  {name}: {rule.advice}")
    return 1 if unguarded else 0


if __name__ == "__main__":
    sys.exit(main())
