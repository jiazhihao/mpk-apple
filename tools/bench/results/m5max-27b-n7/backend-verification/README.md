# Eight-row backend forward evidence

Measured 2026-10-02 on Apple M5 Max (40 GPU cores, 48 GB).

See the [full report](../../../../../docs/research/m5max-27b-backend-verification.md) for scope, ranges, numerical checks, checkpoint adapters, and backend limitations. vLLM-Metal results are paged-prefill target-compute proxies: its hybrid speculative scheduler rejects this workload. These are not speculative-generation rates.

- `summary.csv` and `summary.json`: 20 cases, 20 samples each, 400 accepted timed samples total.
- `ollama.jsonl`, `vllm-bf16.jsonl`, `vllm-mxfp8.jsonl`: all accepted raw samples.
- `vllm-bf16-validation.jsonl`: independent numerical-validation run; excluded from performance summaries.
- `validation.json` and `serial-audit.jsonl`: serial-forward comparisons and snapshot/replay checks.
- `inputs.json`, `methodology.json`, `ollama-import-audit.json`: exact input IDs, configuration, and 193 NVFP4 matrix preservation checks.
- `source-manifest.json`, `output-manifest.json`: source/runtime and output tensor digests.
- `reproduce.md`, `vllm-requirements.txt`: reproduction instructions and pinned Python dependencies.
- `evidence.tar.gz`: benchmark source snapshots, preparation/audit/summary scripts, accepted and rejected logs, release metadata, and an internal SHA-256 manifest. No model weights, runtime binaries, or full output tensors.

The old Monolith/MLX-LM decoder-layer estimates omit embedding and the output head; the new backend forward measurements include both. Do not derive a full-model speedup from mixing those scopes.
