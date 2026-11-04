# Intel XPU weight sharing over Level-Zero IPC

Replicas of one engine on one card each load their own copy of the same read-only weights. On CUDA, `weight_share: "on"` publishes the leader's weights over CUDA IPC and every follower aliases them. This document records why that path produced nothing on Intel GPUs, what was measured on an Arc Pro B60 host, and the design that makes it work.

## The problem

`export_weights` defaults its alias predicate to `t.is_cuda` and embeds everything else by value, which its docstring describes as the rare case of a CPU-resident buffer. On any other accelerator it is not rare: no weight satisfies the predicate, so every weight takes the by-value path. The handle file grows to the size of the checkpoint, each follower copies out of it, and nothing is shared. Measured on `main` with a 192 MiB module, the handle file was 192.0 MiB.

torch offers no way out. On torch 2.14.0+xpu, `torch.xpu` exposes no IPC API at all, and `torch.multiprocessing.reductions.reduce_tensor` on an XPU tensor returns the generic by-value rebuild rather than a device-handle rebuild. Sharing XPU memory between processes has to go through Level Zero directly.

## What was verified first

Before any design, two properties were measured on 8x Arc Pro B60, torch 2.14.0+xpu, compute-runtime 26.22:

- **An unrelated process can alias a torch XPU tensor.** The owner exported the Level-Zero segment behind a torch pointer, passed the dma_buf descriptor over a Unix socket, and the importer mapped it and wrapped it through DLPack. The owner then mutated the tensor in place, 7.0 to 42.0, and the importer read 42.0 through its own mapping. That is a zero-copy alias, not a copy.
- **A model's weights span many allocations.** For a module shaped like a 1.7B ASR backbone, 169 tensors over 5.32 GiB were backed by 114 distinct Level-Zero segments, each exporting cleanly.

## Design: one exportable pool per engine

114 segments would mean 114 descriptors, 114 mappings, and a handle server per segment, because `HandleRendezvous` serves exactly one segment. Instead the leader allocates **one** exportable Level-Zero allocation sized to the shareable weight mass, copies each weight into its own window, and rebinds its own parameter onto that window. One segment is exported, so one descriptor crosses to each follower.

Rebinding is what keeps the leader at a single copy. `named_shared_tensors` returns the live `nn.Parameter` objects, and assigning `.data` through one of them reaches every module holding a tied parameter, so the original storage loses its last reference and is freed.

The handle file carries only the layout: socket name, pool id, device index, pool size, and per-weight offset, size, shape and dtype, as JSON. A follower maps the one segment and slices every weight out of it.

```text
leader                                    follower
  weights (torch allocator)
    |  copy + rebind .data
    v
  one exportable L0 segment  --fd over SCM_RIGHTS-->  mapped once
    |                                                   |
  handle file (layout only, KiB)  -------------------> sliced per weight
```

Two details are forced by the platform:

- **A descriptor cannot ride the handle file.** A dma_buf fd has to be passed over a Unix socket with `SCM_RIGHTS`, and the handle blob's first four bytes are rewritten with the receiver's local descriptor number. So the leader runs a rendezvous thread for its process lifetime, which matches the documented lifecycle where the leader must outlive its followers.
- **Weights are laid out in name order, aligned to 256 bytes.** Both sides derive every offset from the layout alone, and 256 satisfies the alignment of every dtype view the pool serves.

## Measured result

A 0.375 GiB module, leader and follower in separate processes on one B60:

| | before | after |
|---|---|---|
| handle file | 384 MiB (whole checkpoint) | **3.9 KiB** |
| leader memory growth at export | — | **+0.000 GiB** (originals freed by the rebind) |
| follower weights | its own second copy | **aliased**, attach allocated nothing new |
| leader mutates a weight in place | invisible to the follower | **follower reads the new value** |

All 25 tensors rebound in the follower, and values matched the leader.

## What this does not change

The architecture allowlist, the replica-private tensor policy, the manifest hash, the leader liveness monitor, and the whole-group lifecycle are device-independent and apply unchanged. CUDA and ROCm keep CUDA IPC: the serializer and alias predicate are chosen from the live platform, and every non-XPU platform resolves to exactly what it used before.

Two guards became device-neutral rather than CUDA-only in passing: `gpu_uuid()` now reports the XPU UUID, which binds a publication to the physical card it came from, and the arrived-on-the-wrong-card check in the attach path now covers any device with handles.

## Limits and open work

- **Requires the Level-Zero IPC primitives.** This builds on `sglang_omni/platforms/level_zero.py`, `platforms/xpu_dlpack.py` and `relay/level_zero_rendezvous.py` from the stage-transport work, which is not yet merged.
- **One card per sharing group.** A follower on another card is refused, which is the same constraint the feature already has: replicas share weights only when they share a GPU.
- **Driver floor.** The descriptor rewrite is ignored below compute-runtime 26.18, where the probe declines and the engine keeps its own copy.
- **No model is validated end to end yet.** The documented bar for calling a model supported is a `launch.sh` run under a private MPS daemon, which is CUDA-only. The in-pipeline `weight_share: "on"` route does work on XPU and plans leader and follower roles correctly, so that becomes the XPU validation recipe. Until a model passes it, no XPU row should be added to the supported-architecture table.
- **Transient copy at export.** The pool is filled before each original is released, so the leader briefly holds the shared mass twice.

## Why it is worth more on Intel than on CUDA

The saving is one copy of the shared mass per extra replica, and B60 cards are 24 GB where the validated CUDA hosts are 80 GB. Qwen3-ASR shares 3.83 GiB and Fun-ASR 1.57 GiB, so DP2 on Qwen3-ASR returns about 16% of a B60 and DP3 about 32%. The same saving on an 80 GB card is a rounding error.
