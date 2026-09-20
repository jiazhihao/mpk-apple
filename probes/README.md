# GPU feasibility probes

Small, bounded Metal programs that measure the Apple-GPU behaviours the design depends on and Apple does not
document. Results and their design consequences: [`docs/research/apple-gpu-probes.md`](../docs/research/apple-gpu-probes.md).

```bash
./run_all.sh                 # build + run everything here (~4 min); output is saved under results/
./run_all.sh p10_claim_protocol p11_interop_overlap   # or a subset
./remote_run.sh user@host    # run on another bare-metal Apple-silicon Mac over SSH and fetch its results file
```

Every probe derives its geometry from the GPU core count (`gpu-core-count` in the IORegistry; override with
`GPU_CORES=<n>`), so results are comparable across a 10-core M4, an 18-core M3 Pro and a 40-core M5 Max. Do not use
virtualized macOS (hosted CI runners): its paravirtual GPU does not schedule like the real one.

| Probe | Question |
|---|---|
| `p1_limits` | SIMD width, threadgroup limits, GPU family, MSL versions |
| `p2_sync` | can units of one dispatch hand off through device atomics? (same SIMD-group / other SIMD-group / other threadgroup) |
| `p3_residency` | how many SIMD-groups are in flight at once; where does physical concurrency saturate |
| `p4_core_model` | threadgroup → core mapping; full-speed SIMD-groups per core |
| `p5_bandwidth` | streaming GB/s vs number of SIMD-groups; a naive NVFP4 decode+dot |
| `p5b_access_pattern` | striped vs interleaved vs block-cooperative lane access |
| `p6_preemption` | does a long dispatch block other GPU work? (runs three ~1.5 s dispatches) |
| `p6b_interleave` | at what granularity is the GPU shared with other clients: dispatch or command buffer? |
| `p7_dispatch_overhead` | CPU encode, GPU per-dispatch overhead, ICB replay, sync round trip |
| `p8_threadgroup_mem` | is threadgroup memory faster than device memory? |
| `p9_clock_warp` | a free-running SIMD-group as an in-kernel clock |
| `p11_interop_overlap` | cores needed to saturate the memory bus; do un-barriered dispatches overlap; does an ALU-bound op hide inside a bus-bound one; can two bus-bound ops overlap usefully (MPK V2-style inter-op pipelining) |
| `p10_claim_protocol` | in-kernel scheduler (own-slice + steal, fenced and unfenced) vs one dispatch per op: exactly-once, cost per barrier, robustness to missing/surplus threadgroups |

Safety: the GPU does not preempt a running dispatch (`p6`), so every loop here is bounded and the longest single
dispatch is ~1.5 s. Expect brief display stalls during `p3`/`p6`/`p6b`. Never add an unbounded spin to a probe.

To characterize a new chip: run everything, commit the results file, add its numbers to the report, and derive a
profile (`cores`, full-speed SIMD-groups per core, in-flight limit, best block size, GB/s, cores needed to saturate the
bus). `results/` holds the complete M3 Pro reference run.
