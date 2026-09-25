#!/usr/bin/env python3
"""Write (or refresh) this machine's chip profile from the kernel harnesses — the autotuner at install time (design
§5.7, #49). It measures what the compiler reads from ``profiles/<chip>-<cores>c.json``'s ``engine`` block:

* the pack's lane order (the T = 1 GEMV rate, interleaved16 vs contiguous) and the threadgroups per core (x1 vs x2),
* ``cost_T`` per format: the shader GEMV's pass cost at T = 1, 2, 4, 8 relative to T = 1, at the best geometry per T,
* ``accelerator_<fmt>``: the tensor-ops tile (gemm_tile) at 8 / 16 / 32 token rows in the same unit, and from the
  two the accelerator switch and ``accelerator_min_t`` per format (the smallest T the tile takes),
* the attention kernel (v1 vs v2 over context lengths and T).

Everything else in a profile is the probes' record (``./probes/run_all.sh``): ``sibling_order`` (p11) and
``max_cb_ms`` (p6/p6b) carry over from the existing file, or take the safe defaults on a new chip. Min-of-N over
>= 2 GB streamed per point (``--quick`` for a short run). The decisions are ``monolith.core.profile_writer``'s.

    python tools/profile_writer.py --dry-run                 # measure and print the engine block
    python tools/profile_writer.py                           # write profiles/<chip>-<cores>c.json (merged into an existing one)
    python tools/profile_writer.py --nominal-gbps 307        # a new chip: the spec bandwidth (else a measured stand-in)
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "bench"))

import gemm_bench  # noqa: E402
import gemv_bench  # noqa: E402
import gqa_bench  # noqa: E402

from monolith.core.profile import Profile, profiles_dir  # noqa: E402
from monolith.core.profile_writer import COST_TS, TILE_TMS, decide, merge_profile, profile_name  # noqa: E402
from monolith.formats import FORMATS  # noqa: E402
from monolith.runtime import _native as nt  # noqa: E402

GEOMETRIES: Tuple[Tuple[str, Dict[str, Any]], ...] = (("crew x1", {"tg_per_core": 1}), ("crew x2", {"tg_per_core": 2}),
                                                      ("1blk/SG tg64", {"one_block_per_sg": True, "tg": 64}))
DEFAULT_FORMATS = ("nvfp4", "fp8_e4m3", "int4_affine", "bf16")
DEFAULT_ATTENTION = (32, 4, 256)            # heads, kv heads, head_dim (the target's)


def device_facts(info: Any, *, target_gb: float = 21.0, nominal_gbps: Optional[float] = None) -> Dict[str, Any]:
    """The top-level facts of the profile from the Metal device (and the machine)."""
    ws, mb = info.recommended_working_set / 2 ** 30, info.max_buffer_length / 2 ** 30
    mem = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2 ** 30
    d = {"chip": info.name, "gpu_family": f"Apple{info.apple_family}", "gpu_cores": int(info.gpu_cores), "memory_gb": int(round(mem)),
         "os": f"macOS {platform.mac_ver()[0]}", "gpu_working_set_gb": round(ws, 2), "max_buffer_gb": round(mb, 2),
         "hosts_target_model": bool(ws >= target_gb),
         "hosts_target_model_note": f"{target_gb:g} GB of weights against recommendedMaxWorkingSetSize {ws:.2f} GB, maxBufferLength {mb:.2f} GB"}
    if nominal_gbps:
        d["nominal_gbps"] = float(nominal_gbps)
    return d


def measure(*, shape: Tuple[int, int] = (17408, 5120), formats: Sequence[str] = DEFAULT_FORMATS, ts: Sequence[int] = COST_TS,
            tms: Sequence[int] = TILE_TMS, copies: Optional[int] = None, reps: int = 3, accelerator: bool = True,
            attention: Optional[Tuple[int, int, int]] = DEFAULT_ATTENTION, ctxs: Sequence[int] = (1024, 4096), attn_ts: Sequence[int] = (1, 4),
            log: Callable[[str], None] = print) -> Dict[str, Any]:
    """Run the harnesses and return the measurement mapping ``monolith.core.profile_writer.decide`` reads."""
    b = gemv_bench.Bench()
    n, k = shape
    m: Dict[str, Any] = {"family": f"Apple{b.info.apple_family}", "chip": b.info.name, "shape": f"{n}x{k}", "copies": copies, "reps": reps}
    formats = [f for f in formats if f in FORMATS.names()]
    ref = "fp8_e4m3" if "fp8_e4m3" in formats else formats[0]
    # 1. the lane order: the reference format at T = 1, crew x1
    m["lane_order_gbps"] = {}
    for order in ("interleaved16", "contiguous"):
        r = b.run(ref, n, k, t=1, lane_order=order, copies=copies, reps=reps)
        m["lane_order_gbps"][order] = r["gbps"]
        log(f"lane order {order:13s} {ref} T=1: {r['ms']:.3f} ms {r['gbps']:.0f} GB/s{'' if r['ok'] in (None, True) else '  ORACLE FAIL'}")
    lane = max(m["lane_order_gbps"], key=lambda o: m["lane_order_gbps"][o])
    m["lane_order_measured_with"] = lane
    # 2. threadgroups per core at T = 1
    m["threadgroups_ms"] = {}
    for g in (1, 2):
        r = b.run(ref, n, k, t=1, lane_order=lane, tg_per_core=g, copies=copies, reps=reps)
        m["threadgroups_ms"][g] = r["ms"]
        log(f"threadgroups x{g} {ref} T=1: {r['ms']:.3f} ms {r['gbps']:.0f} GB/s")
    # 3. the shader GEMV's cost per format and T at the best geometry
    m["shader_ms"], m["shader_geometry"], m["shader_gbps"] = {}, {}, {}
    for fmt in formats:
        m["shader_ms"][fmt], m["shader_geometry"][fmt], m["shader_gbps"][fmt] = {}, {}, {}
        for t in ts:
            best = None
            for name, geo in GEOMETRIES:
                try:
                    r = b.run(fmt, n, k, rows=16, t=t, lane_order=lane, copies=copies, reps=reps, **geo)
                except Exception as e:  # noqa: BLE001 — a geometry the kernel refuses (register budget) is simply not a candidate
                    log(f"shader {fmt} T={t} {name}: skipped ({e})")
                    continue
                if r["ok"] is False:
                    log(f"shader {fmt} T={t} {name}: ORACLE FAIL (ulp {r['max_ulp_at_rms']}) — not a candidate")
                    continue
                if best is None or r["ms"] < best[0]:
                    best = (r["ms"], name, r["gbps"])
            if best is None:
                raise RuntimeError(f"profile_writer: no geometry ran for {fmt} at T = {t}")
            m["shader_ms"][fmt][t], m["shader_geometry"][fmt][t], m["shader_gbps"][fmt][t] = best
            log(f"shader {fmt:12s} T={t}: {best[0]:.3f} ms ({best[1]}, {best[2]:.0f} GB/s) = {best[0] / m['shader_ms'][fmt][ts[0]]:.2f}x T={ts[0]}")
    # 4. the tensor-ops tile (Apple10 with MSL 4.0; a compile failure means no accelerator on this chip)
    m["tile_ms"], m["tile_gbps"] = {}, {}
    if accelerator:
        try:
            gb = gemm_bench.GemmBench()
            for fmt in formats:
                m["tile_ms"][fmt], m["tile_gbps"][fmt] = {}, {}
                for tm in tms:
                    r = gb.run(fmt, n, k, tm=tm, lane_order=lane, copies=copies, reps=reps)
                    if r["ok"] is False:
                        log(f"tile {fmt} TM={tm}: ORACLE FAIL (ulp {r['max_ulp_at_rms']}) — not a candidate")
                        continue
                    m["tile_ms"][fmt][tm], m["tile_gbps"][fmt][tm] = r["ms"], r["gbps"]
                    log(f"tile   {fmt:12s} TM={tm}: {r['ms']:.3f} ms ({r['gbps']:.0f} GB/s) = {r['ms'] / m['shader_ms'][fmt][1]:.2f}x the T=1 shader pass")
        except Exception as e:  # noqa: BLE001
            m["tile_ms"], m["tile_error"] = {}, str(e)
            log(f"tile: not available on this device ({e})")
    # 5. the attention kernel: v1 vs v2 over the contexts and T
    if attention is not None:
        heads, kv, d = attention
        m["attention_ms"] = {"v1": {}, "v2": {}}
        for ctx in ctxs:
            for t in attn_ts:
                for v2 in (False, True):
                    try:
                        r = gqa_bench.run(b.dev, heads, kv, d, ctx, t, 64, 4, reps, v2=v2)
                    except Exception as e:  # noqa: BLE001
                        log(f"attention {'v2' if v2 else 'v1'} ctx={ctx} T={t}: skipped ({e})")
                        continue
                    m["attention_ms"]["v2" if v2 else "v1"][(ctx, t)] = r["ms"]
                    log(f"attention {'v2' if v2 else 'v1'} ctx={ctx} T={t}: {r['ms']:.3f} ms ({r['kv_gbps']:.0f} GB/s of KV)")
    return m


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shape", default="17408x5120", help="the GEMV shape measured (the target's largest projection)")
    ap.add_argument("--formats", default=",".join(DEFAULT_FORMATS))
    ap.add_argument("--ts", default=",".join(str(t) for t in COST_TS))
    ap.add_argument("--tms", default=",".join(str(t) for t in TILE_TMS))
    ap.add_argument("--copies", type=int, help="packs streamed per measurement (default: >= 2 GB)")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--attention", default=",".join(str(x) for x in DEFAULT_ATTENTION), help="heads,kv,head_dim or 'none'")
    ap.add_argument("--ctx", default="1024,4096")
    ap.add_argument("--attn-ts", default="1,4")
    ap.add_argument("--no-accelerator", action="store_true", help="skip the tile (the profile then keeps the shader path)")
    ap.add_argument("--quick", action="store_true", help="~1 minute: 4 packs per point, 2 reps, TM = 8 only, one context")
    ap.add_argument("--nominal-gbps", type=float, help="the chip's spec bandwidth (kept from an existing profile; else a measured stand-in)")
    ap.add_argument("--target-gb", type=float, default=21.0, help="the target model's resident weight bytes for hosts_target_model")
    ap.add_argument("--name", help="the profile name (default <chip>-<cores>c)")
    ap.add_argument("--out", type=Path, help="write here instead of profiles/<name>.json")
    ap.add_argument("--dry-run", action="store_true", help="measure and print; write nothing")
    a = ap.parse_args(argv)
    n, k = (int(x) for x in a.shape.lower().split("x"))
    formats = [f for f in a.formats.split(",") if f]
    ts = [int(x) for x in a.ts.split(",")]
    tms = [int(x) for x in a.tms.split(",")]
    ctxs = [int(x) for x in a.ctx.split(",")]
    attn_ts = [int(x) for x in a.attn_ts.split(",")]
    attention = None if a.attention.strip().lower() == "none" else tuple(int(x) for x in a.attention.split(","))
    copies, reps = a.copies, a.reps
    if a.quick:
        copies, reps, tms, ctxs = (copies or 4), min(reps, 2), tms[:1], ctxs[:1]
    info = nt.Device().info()
    name = a.name or profile_name(info.name, info.gpu_cores)
    src = profiles_dir() / f"{name}.json"                              # the existing record merges in even when --out goes elsewhere
    path = a.out or src
    existing = json.load(open(src)) if src.exists() else None
    print(f"{info.name}: {info.gpu_cores} cores, Apple{info.apple_family}; profile {name} ({'merging ' + str(src) if existing else 'new'}) -> {path}", flush=True)
    t0 = time.time()
    m = measure(shape=(n, k), formats=formats, ts=ts, tms=tms, copies=copies, reps=reps, accelerator=not a.no_accelerator,
                attention=attention, ctxs=ctxs, attn_ts=attn_ts, log=lambda s: print(s, flush=True))
    engine, notes = decide(m, existing.get("engine") if existing else None)
    device = device_facts(info, target_gb=a.target_gb, nominal_gbps=a.nominal_gbps)
    doc = merge_profile(existing, device=device, engine=engine, measurements=m, notes=notes, written=time.strftime("%Y-%m-%d"),
                        command="python " + " ".join(sys.argv))
    Profile.from_dict(name, doc)                                    # the loader's validation before anything is written
    print(f"\nmeasured in {time.time() - t0:.0f} s\nengine: {json.dumps(engine, indent=2)}\ndecisions:")
    for key, why in notes.items():
        print(f"  {key}: {why}")
    if a.dry_run:
        return 0
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
        f.write("\n")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
