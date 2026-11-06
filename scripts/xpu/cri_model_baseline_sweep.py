#!/usr/bin/env python3
"""Drive xpu_model_baseline.py --record across the models CRI can host.

The baseline script does not manage the Hugging Face cache, and this host has roughly
7 GB free on its root disk against a model table of several hundred GB. So each model is
downloaded, measured and deleted in turn, and the cache is pruned to just what the next
measurement needs. Pulls come through the Intel proxy at ~300 MB/s, which makes a
re-download cheaper than keeping anything resident.

Resumable twice over: the baseline file is the script's own state (a key already recorded
is skipped), and this driver additionally skips the download for those keys. Re-run the
same command after a Ctrl-C or a reboot.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/sgl-workspace/sglang-omni")
SCRIPT = REPO / "scripts" / "xpu" / "xpu_model_baseline.py"
HF_HUB = Path(os.environ.get("HF_HOME", "/work/hf")) / "hub"
WORK = Path("/work")
BASELINE = WORK / "xpu_model_baseline.json"
LEDGER = WORK / "sweep_ledger.json"

JUDGE_REPO = "Qwen/Qwen3-ASR-1.7B"

# Per-model serve overrides this host needs, passed through the script's --serve-extra so
# the baseline entry records them.
#
# Qwen3-TTS: disable_cuda_graph zeroes reference_encoder_graph_bucket_frames, which is what
# stops Qwen3TTSRefCodeBatcher capturing a graph on its private encode stream. Capturing it
# on this pre-release driver kills the device, and the failure surfaces later as
# UR_RESULT_ERROR_DEVICE_LOST from handoff.synchronize() in synchronize_outcomes, failing
# server warmup. Measured: with this flag the probe round-trips at WER 0.0000. Disabling the
# vocoder's async_decode instead does NOT help; the stream belongs to the request builder.
SERVE_EXTRA: dict[str, str] = {
    "qwen3-tts": "--tts_engine.engine.disable_cuda_graph true",
    "qwen3-tts-cv": "--tts_engine.engine.disable_cuda_graph true",
    "qwen3-tts-vd": "--tts_engine.engine.disable_cuda_graph true",
}

# (key, repo id, GiB of weights to pull, allow_patterns or None)
# Ordered cheapest-first so a partial sweep still covers the most models. Entries whose
# weights cannot coexist with the judge in 7 GB are absent; see SKIPPED_TOO_LARGE.
PLAN: tuple[tuple[str, str, float, tuple[str, ...] | None], ...] = (
    ("qwen3-asr", JUDGE_REPO, 4.4, None),
    ("fun-asr", "FunAudioLLM/Fun-ASR-Nano-2512-hf", 1.6, None),
    ("moss-transcribe-diarize", "OpenMOSS-Team/MOSS-Transcribe-Diarize", 1.7, None),
    ("whisper", "openai/whisper-large-v3", 2.9, ("*.json", "*.txt", "model.safetensors")),
    ("nemotron35-asr", "nvidia/nemotron-3.5-asr-streaming-0.6b", 3.1, None),
    ("qwen3-tts", "Qwen/Qwen3-TTS-12Hz-1.7B-Base", 4.3, None),
    ("qwen3-tts-cv", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", 4.3, None),
    ("qwen3-tts-vd", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", 4.3, None),
    ("dots-tts", "dots-studio/dots.tts-soar", 4.9, None),
    ("dots-tts-base", "dots-studio/dots.tts-base", 4.9, None),
    ("cosyvoice3", "FunAudioLLM/Fun-CosyVoice3-0.5B-2512", 6.1, None),
    ("auk", "tencent/AuK", 6.3, None),
    ("auk-flash", "tencent/AuK-Flash", 6.3, None),
    ("arkasr", "AutoArk-AI/ARK-ASR-3B", 7.6, None),
)

# Recorded in the report rather than attempted: weights that this host's disk cannot hold
# even alone, or a topology it does not have.
SKIPPED_TOO_LARGE = {
    "higgs": "8.7 GiB + 4.4 GiB judge exceeds the free disk",
    "zonos2": "14.3 GiB of .pt weights exceeds the free disk",
    "minicpm-o": "17.5 GiB exceeds the free disk",
    "ming-tts": "33.3 GiB exceeds the free disk",
    "minimax-music3": "53.4 GiB exceeds the free disk",
    "qwen3-omni-text": "65.7 GiB and needs 8 cards; this host has 4",
    "ming-flash-omni": "220 GiB exceeds the free disk",
}

# Models whose probe also serves the judge for round-trip scoring.
JUDGED = {
    "qwen3-tts",
    "qwen3-tts-cv",
    "qwen3-tts-vd",
    "dots-tts",
    "dots-tts-base",
    "cosyvoice3",
    "auk",
    "auk-flash",
}


def cache_dir(repo: str) -> Path:
    return HF_HUB / ("models--" + repo.replace("/", "--"))


def free_gib() -> float:
    usage = shutil.disk_usage("/")
    return usage.free / 2**30


def recorded_keys() -> set[str]:
    if not BASELINE.exists():
        return set()
    stored = json.loads(BASELINE.read_text())
    keys: set[str] = set()
    for entry in stored.get("environments", {}).values():
        keys.update(entry.get("models", {}))
    return keys


def read_ledger() -> dict:
    if LEDGER.exists():
        return json.loads(LEDGER.read_text())
    return {"runs": {}}


def write_ledger(ledger: dict) -> None:
    LEDGER.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")


def prune_cache(keep: set[str]) -> float:
    """Delete every cached repo except the ones named, returning GiB reclaimed.

    Through the hub's own API rather than by removing models-- directories: this cache
    keeps file contents in one shared hub/blobs store that the per-repo directories only
    symlink into, so deleting a repo directory frees almost nothing and the store grows
    until the disk is full. delete_revisions drops the blobs a revision owns.
    """
    before = free_gib()
    try:
        from huggingface_hub import scan_cache_dir

        info = scan_cache_dir(HF_HUB)
        doomed = [
            revision.commit_hash
            for repo in info.repos
            if repo.repo_id not in keep
            for revision in repo.revisions
        ]
        if doomed:
            info.delete_revisions(*doomed).execute()
    except Exception as exc:
        print(f"  cache scan failed ({type(exc).__name__}: {exc}); clearing all of it")
        shutil.rmtree(HF_HUB, ignore_errors=True)
    return free_gib() - before


def download(repo: str, patterns: tuple[str, ...] | None) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(repo, max_workers=8, allow_patterns=patterns)


def cards_in_use(count: int) -> list[float]:
    """MiB in use on the first `count` cards, read from inside this container.

    Through torch rather than xpu-smi: xpu-smi is not usable in this container, and XPU's
    mem_get_info reports device-wide usage, so it sees another container's allocation on a
    card this one also maps.
    """
    probe = (
        "import json, torch; torch.xpu.init();\n"
        "print(json.dumps([(t - f) / 2**20 for f, t in "
        "(torch.xpu.mem_get_info(i) for i in range(min(%d, torch.xpu.device_count())))]))"
        % count
    )
    try:
        out = subprocess.run(
            [sys.executable, "-I", "-c", probe],
            capture_output=True,
            text=True,
            timeout=180,
            env=dict(os.environ, PYTHONPATH=str(REPO)),
        )
        return json.loads(out.stdout.strip().splitlines()[-1])
    except Exception as exc:
        print(f"  could not read card usage ({type(exc).__name__}: {exc})", flush=True)
        return []


def wait_for_idle_cards(count: int, ceiling_mib: float, wait_s: int) -> tuple[bool, str]:
    """Hold until the cards this measurement needs are quiet.

    These cards are shared with other users' containers, and a neighbour's workload does not
    fail a run, it slows it: a sweep measured against a busy card recorded Qwen3-ASR at
    0.520 s where a quiet card gives 0.065 s. A baseline carrying that is worse than none,
    because the inflated floor hides every regression underneath it.
    """
    deadline = time.time() + wait_s
    while True:
        usage = cards_in_use(count)
        if not usage:
            return True, "card usage unreadable; measuring anyway"
        busy = [f"dev{i}={mib:.0f}MiB" for i, mib in enumerate(usage) if mib > ceiling_mib]
        if not busy:
            return True, f"cards quiet ({', '.join(f'{m:.0f}' for m in usage)} MiB)"
        if time.time() >= deadline:
            return False, f"cards still busy after {wait_s}s: {', '.join(busy)}"
        print(f"  waiting for {', '.join(busy)} to go quiet", flush=True)
        time.sleep(30)

def run_one(key: str, repeats: int, tolerance: float, fresh: bool) -> tuple[bool, str]:
    log = WORK / f"sweep_{key}.log"
    command = [
        sys.executable,
        str(SCRIPT),
        "--record",
        "--models",
        key,
        "--card-pool",
        "0,1,2,3",
        "--repeats",
        str(repeats),
        "--speed-tolerance",
        str(tolerance),
        "--baseline",
        str(BASELINE),
    ]
    if key in SERVE_EXTRA:
        command += ["--serve-extra", f"{key}={SERVE_EXTRA[key]}"]
    if fresh:
        command += ["--fresh"]
    env = dict(os.environ, PYTHONPATH=str(REPO))
    with log.open("w") as handle:
        completed = subprocess.run(
            command, cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT
        )
    tail = ""
    for line in log.read_text(errors="replace").splitlines():
        if line.startswith(("  FAILED", "  median", "  accuracy", "REGRESSION")):
            tail += line.strip() + " | "
    return completed.returncode == 0, tail or f"exit {completed.returncode}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--speed-tolerance", type=float, default=0.15)
    parser.add_argument("--only", default="", help="comma-separated subset of keys")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="re-measure keys already in the baseline, passing --fresh through",
    )
    parser.add_argument(
        "--idle-ceiling-mib",
        type=float,
        default=1024.0,
        help="a card holding more than this is treated as in use by someone else",
    )
    parser.add_argument(
        "--idle-wait-s",
        type=int,
        default=600,
        help="how long to wait for a busy card before skipping the model",
    )
    args = parser.parse_args()

    plan = PLAN
    if args.only:
        wanted = set(args.only.split(","))
        plan = tuple(item for item in PLAN if item[0] in wanted)

    ledger = read_ledger()
    done = recorded_keys()
    print(f"baseline {BASELINE} already holds: {sorted(done) or 'nothing'}", flush=True)
    print(f"free disk {free_gib():.1f} GiB, {len(plan)} model(s) planned", flush=True)

    for index, (key, repo, weight_gib, patterns) in enumerate(plan, start=1):
        print(f"\n===== [{index}/{len(plan)}] {key} ({repo}, ~{weight_gib} GiB) =====", flush=True)
        if key in done and not args.fresh:
            print("  already recorded; skipping download and run", flush=True)
            continue

        keep = {repo} | ({JUDGE_REPO} if key in JUDGED else set())
        reclaimed = prune_cache(keep)
        print(f"  pruned cache (+{reclaimed:.1f} GiB), free now {free_gib():.1f} GiB", flush=True)

        # Only what still has to be fetched: a repo already in the cache has already been
        # subtracted from the free space, and counting it twice declined models that fit.
        pending = [
            (target, size)
            for target, size in (
                [(repo, weight_gib)]
                + ([(JUDGE_REPO, 4.4)] if key in JUDGED and repo != JUDGE_REPO else [])
            )
            if not cache_dir(target).exists()
        ]
        needed = sum(size for _, size in pending)
        # This root disk is shared with other people's containers. Decline a model the
        # disk cannot take rather than filling it and taking their jobs down with mine.
        margin = 1.0
        if needed + margin > free_gib():
            message = (
                f"needs ~{needed:.1f} GiB + {margin:.0f} GiB margin, "
                f"only {free_gib():.1f} GiB free"
            )
            print(f"  DECLINED: {message}", flush=True)
            ledger["runs"][key] = {"ok": False, "detail": f"declined: {message}"}
            write_ledger(ledger)
            continue
        started = time.time()
        try:
            for target, pattern in [(repo, patterns)] + (
                [(JUDGE_REPO, None)] if key in JUDGED and repo != JUDGE_REPO else []
            ):
                if not cache_dir(target).exists():
                    print(f"  downloading {target}", flush=True)
                    download(target, pattern)
        except Exception as exc:
            message = f"download failed: {type(exc).__name__}: {exc}"
            print(f"  {message}", flush=True)
            ledger["runs"][key] = {"ok": False, "detail": message}
            write_ledger(ledger)
            continue
        print(
            f"  cache ready in {time.time() - started:.0f}s "
            f"(needed ~{needed:.1f} GiB, free {free_gib():.1f} GiB)",
            flush=True,
        )

        cards_needed = 2 if key in JUDGED else 1
        quiet, why = wait_for_idle_cards(cards_needed, args.idle_ceiling_mib, args.idle_wait_s)
        print(f"  {why}", flush=True)
        if not quiet:
            ledger["runs"][key] = {"ok": False, "detail": f"not measured: {why}"}
            write_ledger(ledger)
            continue
        started = time.time()
        ok, detail = run_one(key, args.repeats, args.speed_tolerance, args.fresh)
        elapsed = time.time() - started
        after = cards_in_use(cards_needed)
        if after and max(after) > args.idle_ceiling_mib:
            detail += f" | WARNING neighbour on card during run ({max(after):.0f} MiB left after)"
            print(f"  {detail}", flush=True)
        print(f"  {'OK' if ok else 'FAILED'} in {elapsed:.0f}s: {detail}", flush=True)
        ledger["runs"][key] = {
            "ok": ok,
            "detail": detail,
            "seconds": round(elapsed, 1),
            "weights_gib": weight_gib,
        }
        write_ledger(ledger)
        done = recorded_keys()
        print(
            f"  TALLY recorded={len(done)} "
            f"ok={sum(1 for r in ledger['runs'].values() if r['ok'])} "
            f"failed={sum(1 for r in ledger['runs'].values() if not r['ok'])}",
            flush=True,
        )

    print("\n===== sweep summary =====", flush=True)
    for key, info in sorted(ledger["runs"].items()):
        print(f"  {'ok    ' if info['ok'] else 'FAILED'} {key:26s} {info['detail'][:110]}")
    for key, why in sorted(SKIPPED_TOO_LARGE.items()):
        print(f"  skip   {key:26s} {why}")
    print(f"\nrecorded in baseline: {sorted(recorded_keys())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
