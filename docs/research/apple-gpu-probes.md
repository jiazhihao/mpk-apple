# Apple GPU execution model — measured facts

Apple documents almost none of what an in-GPU runtime needs to know: how threadgroups map to cores, how many
SIMD-groups run at once, what a dispatch boundary costs, whether the GPU is shared while a dispatch runs, how fast
memory really streams. This report records what we measured, per chip, and what each number means for the
[design](../design/design.md).

| Chip | Status | Results file |
|---|---|---|
| **M3 Pro**, 18-core GPU, 36 GB, Apple9, macOS 26.6.2 | measured 2026-09-19 (on battery, Low Power Mode off) | [`probes/results/Apple-M3-Pro_18c_…txt`](../../probes/results) |
| **M4 family** | **not measured — next** (§3) | — |
| **M5 family** | not measured; no bare-metal rental found as of 2026-09 | — |

Every number here is an observation on one chip and one OS build — *firmware behaviour, not an API contract*. The design
treats them as per-chip profile values, and correctness never depends on them.

**Running the suite** (13 probes, ~4 min, Xcode Command Line Tools only — shaders compile at runtime):

```bash
./probes/run_all.sh                          # on the machine itself; output is saved to probes/results/<chip>_<cores>c_…txt
./probes/remote_run.sh user@host             # on another bare-metal Mac over SSH; fetches the results file
./probes/run_all.sh p10_claim_protocol p11_interop_overlap     # a subset
```

Every probe derives its launch geometry from the GPU core count (`gpu-core-count` in the IORegistry; override with
`GPU_CORES=<n>`), so results are comparable across chips. Do not use virtualized macOS (hosted CI runners, Tart/Anka
VMs): a paravirtual GPU does not schedule like the real one. The GPU does not preempt a running dispatch (P6), so every
probe is bounded and the longest single dispatch is ~1.5 s — expect brief display stalls during `p3`, `p6`, `p6b`.
Never add an unbounded spin.

---

## 1. Cross-chip results

Ranges are the spread over repeated runs on the same day. Fill a column per chip; the probe/section that prints each
number is in the first column.

| Metric (probe) | M3 Pro 18c | M4 … | M5 … |
|---|---|---|---|
| GPU cores · nominal GB/s · **GB/s per core** | 18 · 153.6 · **8.5** | | |
| SIMD width · max threads/threadgroup · threadgroup memory (`p1`) | 32 · 1024 · 32 KB | | |
| GPU family · highest MSL that compiles (`p1`) | Apple9 · 4.0 | | |
| Same-SIMD-group handoff deadlocks? · cross-SIMD-group handoff (`p2`) | yes · 195–230 ns | | |
| SIMD-groups in flight at once (`p3a`) | 1,536 ok, 2,048 not | | |
| Full-speed SIMD-groups per core · threadgroups per core (`p4`) | ~12 (384 threads) · 1 — flat to G = 18, clean 2× at G = 20 | | |
| Saturated ALU throughput, in full-speed-SIMD-group equivalents (`p3b`) | 205–215 (≈ 11.5–12 per core) | | |
| Streaming, crew geometry, 64 KB block sweep (`p5b`) | **124–134 GB/s (81–87 % of nominal)** | | |
| … lanes on far-apart stripes · lanes interleaved per word (`p5b`) | 59–60 · 84–90 GB/s | | |
| … best conventional geometry, 192·C SIMD-groups (`p5`, `p5b`) | 106–126 GB/s | | |
| Naive NVFP4-style decode + dot (`p5`, fmt 1) | 49–58 GB/s | | |
| Small dispatch behind ONE long dispatch (`p6`) | waits 1.24–1.42 s, even with 1/3 of the cores idle | | |
| Small dispatch behind many short dispatches (`p6b`) | typical 0.3–0.5 ms; **worst case one whole command buffer** (130–140 ms behind ~155 ms buffers; once 1.13 s behind a 1.25 s buffer) | | |
| CPU encode · GPU cost per dispatch · ICB replay per 1,240 dispatches · sync round trip (`p7`) | 0.13 µs · 1.3–1.8 µs · 0.01–0.02 ms · 0.12–0.13 ms | | |
| Hot read: threadgroup memory vs device memory (`p8`) | 61–64 ns vs 62–64 ns (thread-private array: 116 ns) | | |
| Clock SIMD-group tick (`p9`) | 47–53 ns, ±1 % within a session, durations linear within 1 % | | |
| Barrier cost per op: **dispatch boundary** vs in-kernel static slices vs in-kernel stealing (`p10`, protocol only) | **1.6–2.7 µs** vs 2.1–2.6 µs (3.1–3.9 µs spec-compliant) vs 4.7–6.1 µs | | |
| … same with ~30 µs blocks (`p10`) | 92–96.5 µs/op for every variant | | |
| Exactly-once under missing / surplus threadgroups (`p10`) | yes, 0 timeouts | | |
| One core's uncontended streaming rate (`p11` §1) | 18.0–18.7 GB/s | | |
| Cores needed for ≥ 97 % of peak bandwidth (`p11` §1) | 12 of 18 (9 cores already give 114–126 GB/s) | | |
| ALU-bound op hidden inside a bus-bound op, no barrier, both at full geometry (`p11` §3b) | 86–114 % hidden — i.e. fully hidden within noise — in either encode order, up to 87 % of the partner's length | | |
| … same pair with cores split by hand (`p11` §3) | 29–32 ms vs 27 ms serial — *slower* | | |
| Two bus-bound ops overlapped (`p11` §4) | no consistent gain (−3 %…+10 %, within noise) | | |

---

## 2. What the M3 Pro numbers mean

| # | Question | Result | Design consequence |
|---|---|---|---|
| P1 | Device limits | SIMD-group = 32 threads; ≤ 1024 threads/threadgroup; threadgroup memory ≤ 32 KB; tier-2 argument buffers; function pointers and dynamic libraries; MSL 4.0 compiles, 4.1 needs macOS 27 | The SIMD-group is the 32-lane lockstep unit — the analogue of an NVIDIA *warp* |
| P2 | Can execution units of **one dispatch** synchronize through device atomics? | Same SIMD-group: **deadlock** (lockstep). Different SIMD-groups, same or different threadgroups: **works, ~200 ns per handoff**, ~2 spins per wait | In-kernel events are feasible; the unit of independent control flow is the SIMD-group |
| P3a | How many SIMD-groups of one dispatch are in flight at once? | 1,536 all see each other within 9 spins; at 2,048 most time out | Anything that waits in-kernel must stay well under ~1,500 SIMD-groups; the crew geometry uses 12 per core |
| P3b/P4 | Physical concurrency and threadgroup → core mapping | Exactly **18 slots = 18 cores**. With threadgroups ≥ 384 threads **one threadgroup runs per core**; a core runs **~12 SIMD-groups at full speed** (16 → 1.5× slower, 32 → 2.6–2.8×). 18 × 384 threads all run at full speed; the 19th+ threadgroup waits for a core | **GPU core ≈ MPK "SM"; threadgroup of 384 = 12 × 32 ≈ MPK "CTA"**. Launch geometry = `cores × 384` (design D4) |
| P5 | Streaming bandwidth | Conventional geometry peaks at 106–126 GB/s; a naive NVFP4 decode+dot kernel is **ALU/load-latency-bound at 49–58 GB/s** | The decode GEMV must be engineered (wide packed loads, activation reuse); it is not automatically bandwidth-bound |
| P5b | Does the lanes' access pattern matter? | Crew geometry: far-apart stripes 59–60 GB/s; interleaved per word 84–90; **64 KB blocks with each lane on a contiguous 2 KB sub-range: 124–134 GB/s** | Block-lane-major weight packs (D8). Advantage over the best conventional geometry: **+5 % to +17 %** depending on the run — to be confirmed with real kernels (plan M1) |
| P6 | Is a running dispatch preempted or shared? | **No.** A 30 µs dispatch on a second queue waited 1.24–1.42 s for one long dispatch, even when it used 12 of 18 cores | A never-returning kernel is not viable; long dispatches freeze every other GPU client |
| P6b | At what granularity *is* the GPU shared? | **Usually per dispatch** (0.3–0.5 ms behind 800 short dispatches in one command buffer, 5 of 6 trials), but **the worst case is a whole command buffer**: 130–140 ms outliers behind ~155 ms buffers, and once the full 1.13 s | Short dispatches **and** short command buffers (design D6): ≤ ~16–33 ms of work per command buffer while a display is attached |
| P6' | Watchdogs | An 11 s dispatch whose threads retired continuously completed without error; others report a ~5 s kill of a non-progressing kernel (MLX #4475) and an *interactivity* kill at ~0.5–1.2 s with the display on (MLX #3267) | Rely on none of it: dispatches are sub-millisecond, command buffers tens of milliseconds |
| P7 | Launch overhead for a 1,240-dispatch "token" | CPU encode 0.13 µs/dispatch; GPU 1.3–1.8 µs/dispatch; ICB replay 0.01–0.02 ms per token; one CPU↔GPU sync 0.12–0.13 ms | ~2 ms/token ≈ 1–2 % of a 27B token here: launch overhead alone does not justify a single-kernel design (design §2) |
| P8 | Is threadgroup memory a fast scratchpad (an SMEM analogue)? | **No**: 61–64 ns per hot read from either; a thread-private array is slower | No software-managed memory level ⇒ no page planner, no loader/storer roles, nothing to pre-stage into |
| P9 | In-kernel clock (MSL has none)? | A dedicated **clock SIMD-group** incrementing an atomic: 47–53 ns/tick, ±1 %, durations linear within 1 % | MPK-style tracing is recoverable at the cost of 1 SIMD-group, in profiling builds |
| P10 | Whole step inside **one** dispatch with in-kernel barriers — does it beat one dispatch per op? | **It works, but it does not win.** Own-slice + steal is exactly-once with zero timeouts even with half the crew missing or 2×/4× surplus threadgroups. Per barrier: dispatch boundary **1.6–2.7 µs** (including the launch of 6,912 threads) vs in-kernel 2.1–2.6 µs with static slices (**3.1–3.9 µs** in the spec-compliant form) and 4.7–6.1 µs with stealing; a global claim cursor is 27 % slower. With 30 µs blocks all variants are within noise | **One dispatch per fused op** (D5). In-kernel sync is never meaningfully cheaper, the spec-compliant form is always dearer, and a multi-op kernel is a long non-preemptible dispatch. Co-residency is a performance property, not a correctness requirement |
| P11 | Can ops overlap (MPK V2-style inter-op pipelining)? | The bus saturates with **half to two-thirds of the cores**. Un-barriered dispatches do overlap: an **ALU-bound op hides (86–114 %, i.e. fully within noise) inside a bus-bound one**; **two bus-bound ops gain nothing**; splitting cores by hand is slower than serial | No weight pre-staging across dependencies; overlap only ALU-bound with bus-bound siblings at full geometry, placement left to the firmware (D14, design §5.12) |

---

## 3. Continuing on M4 (and M5)

**Do this first:** `./probes/run_all.sh` on the M4, commit the results file, fill the M4 column of §1, then walk the
table below. M4 is the same GPU family as the M3 (Apple9, dynamic caching), so most mechanisms should carry over; what
changes is the *ratio of memory bandwidth to GPU cores*, which is what the design's performance arguments rest on:

| Chip | GPU cores | Nominal GB/s | GB/s per core |
|---|---|---|---|
| M3 Pro (measured) | 18 | 153.6 | 8.5 |
| M4 | 10 | 120 | 12.0 |
| M4 Pro | 16 / 20 | 273 | 17.1 / 13.7 |
| M4 Max | 32 / 40 | 410 / 546 | 12.8 / 13.7 |
| M5 · M5 Pro · M5 Max · M5 Ultra | 10 · 20 · 40 · 80 | 153.6 · 307 · 614 · 1229 | 15.4 |

One M3 Pro core streams ~18 GB/s uncontended. At 8.5 GB/s per core the bus saturates with half the cores and the rest
idle; at 13.7–17 GB/s per core a bus-bound op needs nearly every core.

| Hypothesis for M4 | Probe | If it fails |
|---|---|---|
| H1 Same SIMD width (32), limits, and lockstep deadlock | `p1`, `p2` | kernels assume 32 lanes (BLM stripe count) — make it a profile value |
| H2 One 384-thread threadgroup per core, ~12 SIMD-groups at full speed (`p4`: times flat up to G = cores, 2× just above) | `p4` | crew geometry becomes `cores × (SIMD-groups per core × 32)`; D4 is already profile-driven |
| H3 Crew-geometry block sweep ≥ 80 % of nominal and ≥ the conventional geometry | `p5b` | the BLM layout matters less; adopt MLX's GEMV geometry (plan M1 fallback) |
| H4 **A dispatch boundary is still no dearer than an in-kernel barrier** | `p10` (compare "ONE DISPATCH PER OP" with the static-slices and own+steal rows, both builds) | **revisit D5**: multi-op kernels with in-kernel barriers could pay on this chip |
| H5 Sharing: typically per dispatch, worst case per command buffer; never inside a dispatch | `p6`, `p6b` (run `p6b` 3–5 times) | tighten or relax the command-buffer length (D6) |
| H6 The bus needs **most** of the cores (little spare ALU) | `p11` §1: where does GB/s stop growing? | if there is as much slack as on the M3 Pro, the overlap rule (D14) is worth more than projected |
| H7 **An ALU-bound op is no longer fully hidden**, and may slow its bus-bound partner | `p11` §3 / §3b: compare "no barrier" with "A alone" | if still ~100 % hidden, enable sibling overlap on M4; if the partner slows, disable it there |
| H8 Threadgroup memory is not faster than device memory | `p8` | reconsider staging small hot data (activation stripes) |

Useful additions while on the M4 (not written yet): (a) `p11` with a real FP8/NVFP4 decode kernel as the bus-bound op —
with real ALU work per byte, is there any spare ALU at all? (this is the first half of plan M1); (b) a threadgroup-size
sweep for streaming (384 / 768 / 1024 threads) — higher occupancy hides load latency and may matter more at higher
bandwidth per core; (c) an on-screen frame-pacing check with command buffers of 8 / 16 / 33 / 66 ms.

Practicalities: the streaming probes allocate 3 GB buffers (have ~10 GB free). The probes run on any M4; the target
model needs ~21 GB of weights plus state, i.e. a ≥ 36 GB machine (M4 Pro 48 GB, M4 Max) — a 16–32 GB base M4 can
characterize the GPU but cannot host Qwen3.8-27B.

---

## 4. Details (M3 Pro)

### P2 — in-kernel synchronization (`p2_sync`)

Two designated threads of one dispatch alternately increment a shared `atomic_uint` 20,000 times each, with bounded
spins so the kernel always terminates.

```
same SIMD-group        (tid 0 vs 1)            : A done=1 B done=0   timeouts      <- lockstep: cannot hand off
diff SIMD-group, same threadgroup (0 vs 32)     : 40000 handoffs     218 ns/handoff (avg 2.0 spins/wait, max 4)
diff SIMD-group, same threadgroup (0 vs 512)    : 40000 handoffs     229 ns/handoff
diff threadgroups (2 x 64)                      : 40000 handoffs     195 ns/handoff
diff threadgroups far apart (TG0 vs TG15 of 16) : 40000 handoffs     206 ns/handoff
```

Through MSL 4.0 atomic operations accept only `memory_order_relaxed` (MSL 4.1 adds acquire/release orders) and are
32-bit except `ulong` min/max; that is sufficient for monotone counters and claim cursors.

### P3/P4 — the core model (`p3_residency`, `p4_core_model`)

Fixed ALU work per SIMD-group; GPU time of the dispatch with all SIMD-groups busy (one SIMD-group alone = 15.2 ms):

```
threads/TG   G=1    G=2    G=4    G=9    G=12   G=16   G=18   G=20   G=24   G=36   G=54   G=72
   32        15.2   15.2   15.2   15.2   15.2   15.2   15.2   15.2   15.2   15.2   15.2   15.2
  128        15.2   15.2   15.2   15.2   15.2   15.4   15.4   15.2   15.2   15.2   19.9   23.6
  256        15.2   15.2   15.2   15.2   15.2   15.2   15.2   23.0   23.2   23.6   31.4   43.2
  384        15.3   15.4   15.6   15.6   15.7   15.7   15.7   31.5   31.5   31.6   47.5   60.0    <- flat to 18, then 2x / 3x / 4x
  512        17.0   23.0   23.4   23.6   23.4   23.6   23.6   38.0   42.9   43.0   60.5   76.7
 1024        39.2   42.8   41.9   43.1   43.3   43.3   43.0   76.7   78.3   79.0  110.5  141.7
```

Inside one 1024-thread threadgroup, 1–14 busy SIMD-groups run at full speed; 16 → ×1.09; 20–28 → ×2.0; 32 → ×2.6.
The firmware hands whole threadgroups to cores: a 384-thread threadgroup fills one core's full-speed capacity and 18 of
them fill the GPU with no time-slicing. Small threadgroups are packed several per core, less efficiently (216
SIMD-groups as 54 × 128 threads reach throughput 160 vs 210 as 18 × 384).

In flight: with every SIMD-group spinning until all have checked in, 1,536 SIMD-groups (as 48 × 1024 or 1536 × 32
threads) all see each other within 9 spins; at 2,048 most time out (the rest are admitted only as residents exit).

### P5/P5b — bandwidth (`p5_bandwidth`, `p5b_access_pattern`)

3.2 GB streamed, raw `uint` sum:

```
crew geometry (18 x 384): lanes on far-apart stripes                  59 - 60 GB/s
crew geometry: lanes interleaved word by word                         84 - 90 GB/s
crew geometry: 64 KB blocks, lane = contiguous 2 KB sub-range        124 - 134 GB/s   <- best
crew geometry: 1 MB blocks, lane = 32 KB sub-range                   113 GB/s
conventional, 3456 SIMD-groups: blocked 64 KB / interleaved / striped 106-117 / 108-126 / 87-92 GB/s
```

A deliberately naive NVFP4-style kernel (byte loads, 16-entry LUT, per-16 block scale, `half` activations) peaks at
49–58 GB/s of weight bytes: ~1.5 memory loads per weight at ~40–60 ns per load per lane (P8 shows the same ~60 ns for a
hot 8 KB working set). The production GEMV must load packed `uint32/uint64` words, keep per-row accumulators in
registers, and keep a lane's activation stripe small and contiguous.

### P6 / P6b — sharing (`p6_preemption`, `p6b_interleave`)

```
small dispatch on an idle GPU                                          : mean 0.46 ms
one long dispatch on 18 / 16 / 12 of 18 cores; small one submitted 150 ms in : waited 1417 / 1402 / 1374 ms

queue A: 1 command buffer  x 800 dispatches of ~1.5 ms | queue B meanwhile: mean 0.32-0.42 ms, max 0.7-1.9 ms  (5 trials)
                                                        |                   mean 1134 ms (n=1)                 (1 trial)
queue A: 8 command buffers x 100 dispatches (~155 ms)   | mean 0.33-3.3 ms, max 0.8 / 1.5 / 10.9 / 130 / 140 ms
queue A: 80 command buffers x 10 dispatches (~15 ms)    | mean 0.33-1.0 ms, max 0.45-8.8 ms
```

Other work is normally scheduled between dispatches, but it can be held behind an entire in-flight command buffer; it
never gets in during a dispatch. Same process, second command queue — the window compositor was not measured directly.

### P7 — launch overhead (`p7_dispatch_overhead`)

```
(A0) 1 cmdbuf, 1 encoder, 1240 dispatches : CPU encode 0.16 ms (0.13 us each) | GPU 1.85-2.28 ms (1.5-1.8 us each)
(A1) concurrent encoder + barriers        : CPU encode 0.47 ms                | GPU 1.85 ms
(B)  ICB built once, replayed             : CPU encode 0.01-0.02 ms           | GPU 1.63-1.68 ms (1.3-1.4 us each)
(C)  the same 1240 ops inside ONE dispatch: CPU encode 0.003-0.008 ms         | GPU 0.002-0.003 ms   (no barriers between ops)
(D)  commit + waitUntilCompleted, 1 tiny dispatch : 0.12-0.13 ms wall         <- every CPU<->GPU sync point
```

Floor numbers for the Metal API itself (same pipeline and buffers rebound each time). Other reports put the GPU cost
per dispatch at 2.6–5 µs on other chips (see the survey).

### P9 — clock SIMD-group (`p9_clock_warp`)

One SIMD-group increments an `atomic_uint` in a tight loop; every other SIMD-group reads it before and after fixed
workloads of 1×–4×. Ticks are converted to time with the command buffer's `GPUStartTime/GPUEndTime`.

```
58.82 ms, 1,240,065 ticks -> 47.4 ns/tick | 1x 14.43  2x 28.99  3x 43.36  4x 58.15 ms
59.73 ms, 1,238,017 ticks -> 48.2 ns/tick | 1x 14.58  2x 29.01  3x 43.76  4x 58.75 ms
66.02 ms, 1,241,089 ticks -> 53.2 ns/tick | 1x 16.11  2x 32.29  3x 48.51  4x 65.28 ms   (reference run, on battery)
```

### P10 — in-kernel barriers vs dispatch boundaries (`p10_claim_protocol`)

One dispatch runs 320 ops in sequence, 544 blocks each (≈ one decode step: 17,408 rows / 32, five barriers per layer).
Lane 0 of each SIMD-group claims a block with a bounded `compare_exchange`, broadcasts it with `simd_broadcast_first`,
all 32 lanes run the block, lane 0 bumps `done[op]`; the op ends when `done == n_blocks` (bounded spin).
`hits[op][block]` must end at exactly 1 everywhere. "Fenced" is the spec-compliant build: MSL 3.2, `coherent(device)`
buffers, `atomic_thread_fence(mem_device, seq_cst, thread_scope_device)` before every publish and at every barrier
exit — required by the MSL memory model (§4.8, §6.16) for cross-threadgroup visibility of non-atomic data, although
the unfenced build also worked in every run.

```
                                              protocol only, us/op          with ~30 us blocks, us/op    exactly-once
ONE DISPATCH PER OP (static slices, no sync)   1.6 - 2.7                     92.0 - 94.2                   yes
one kernel, barriers, static slices            2.1 - 2.6   (fenced 3.1-3.9)  92.3 - 95.9                   yes
one kernel, barriers, own slice + steal        4.7 - 5.4   (fenced 5.4-6.1)  93.9 - 96.5                   yes
one kernel, barriers, global claim cursor      87 - 90                       125 - 126                     yes
own + steal, half the crew present             -                             187 - 195  (static re-partitioned: 179-221)   yes
own + steal, 2x / 4x surplus threadgroups      -                             98 - 114 / 121 - 126          yes
```

Victim order matters: with every thief scanning victims in the same order the half-crew case took 264 µs/op; a
per-SIMD-group stride coprime with the crew size brought it to 187–195. Timeouts: 0 in every configuration.

### P11 — inter-op overlap (`p11_interop_overlap`)

A = bus-bound 3.2 GB block sweep; B = ALU-only work, total work held constant across core counts (2.7 ms on 18 cores).

```
(1) A alone on G cores:  1: 18.3 GB/s | 2: 36.8 | 4: 67.8 | 6: 93.9 | 9: 120.4 | 12: 127.9 | 15: 131.6 | 18: 131.2   (other runs: 9 -> 114-126, 18 -> 127-133)
(3) serial A then B (all cores each)                      27.25 ms   (A 24.55 + B 2.71)
    no barrier, A || B, both on all cores                 24.72 ms   <- B fully hidden
    no barrier, A on 15 cores || B on 3 cores             31.17 ms   <- worse than serial
    no barrier, A on 12 cores || B on 6 cores             25.07 ms
    no barrier, B on 6 || A on 12 (B encoded first)       29.04 ms   <- order-sensitive when cores are split
(3b) both on all cores, B encoded first                   25.30 ms
    B = 2x / 4x / 8x work (22 % / 44 % / 87 % of A)       hidden 86-98 % / 88-114 % / 95-100 %
(4) two bus-bound ops: serial 118-131 GB/s | split cores 126-131 | both on all cores 127-130
```

---

## 5. Caveats

* One chip, one OS build, one day; the reference run was taken on battery. Threadgroup → core mapping, the in-flight
  limit, sharing granularity and the absence of a watchdog kill are observed firmware behaviour. The design stays
  *correct* if any of them changes (it relies only on Metal's documented dispatch ordering and ICB barriers) and only
  loses performance.
* Microsecond-level numbers move by tens of percent between runs (compare the ranges above); conclusions are drawn only
  where the ranges do not overlap, or from with-work measurements.
* Bandwidth numbers are raw-read microbenchmarks. The advantage of the crew geometry + blocked layout has to be
  reproduced with real NVFP4/FP8 GEMV kernels against MLX's `qmv` before it counts (plan M1).
* P6/P6b used two queues of one process; cross-process behaviour (the window compositor) was not measured directly.
