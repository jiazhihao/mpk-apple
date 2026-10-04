# Reproducing the eight-row target-forward measurements

Run on bare-metal Apple silicon with Metal 4 support. The recorded machine is an M5 Max with 40 GPU cores and 48 GB RAM, macOS 26.5.1. The report in the repository defines the timed scopes and unsupported hybrid-verification limitation.

## Pinned dependencies and checkpoint preparation

Use the official Ollama v0.35.1 `ollama-darwin.tgz` native payload and source tag, Go 1.27.1, and an isolated Python 3.12 environment with the official vLLM 0.30.0+cpu macOS arm64 and vLLM-Metal 0.30.0 wheels. Release asset URLs and metadata are in the three `*-release*.json` files. `vllm-requirements.txt` records the installed environment, including the pinned MLX-LM Git revision. Do not use the separate main-branch vLLM-Metal checkout for this measurement.

The original checkpoint is `nvidia/Qwen3.8-27B-NVFP4`, revision `482ca0f3832238542f8f5295dde86b5f22711d80`, downloaded to `/tmp/monolith-models/Qwen3.8-27B-NVFP4`. The working directory is `/tmp/monolith-m5max/baseline-verify`. Preparation/audit scripts have those machine-local paths as constants; update them if relocating.

1. Run `scripts/prepare_ollama_fp8.py` with the repository's Python environment. This creates `checkpoint-fp8-bf16` without modifying the original checkpoint.
2. Start the released Ollama server with `OLLAMA_MODELS` set to the working directory's `ollama-models`, `OLLAMA_HOST=127.0.0.1:18203`, and `OLLAMA_NO_CLOUD=1`. Import `Modelfile-fp8-bf16` as `monolith-benchmark-qwen38-nvfp4-bf16`. Set **both** `OLLAMA_MODELS` and `OLLAMA_HOST` on the CLI import command: import uses the CLI's local model directory.
3. Run `scripts/audit_import.py`; require all 193 NVFP4 matrices, scales, and global scales to match exactly.
4. Copy the repository's `tools/bench/ollama_target_verify_test.go` to the pinned Ollama checkout's `mlxrunner/target_verify_bench_test.go`. Compile its package test binary next to the released native payload:

```sh
repo=/Users/zhihaojia/lithos/mpk-apple
base=/tmp/monolith-m5max/baseline-verify
cp "$repo/tools/bench/ollama_target_verify_test.go" "$base/ollama/mlxrunner/target_verify_bench_test.go"
cd "$base/ollama"
CGO_ENABLED=1 GOCACHE="$base/go-cache" GOPATH="$base/go-path" \
  "$base/go/bin/go" test -c -o "$base/ollama-runtime/mlxrunner.test" ./mlxrunner
```

The server can then be stopped; the package benchmark directly loads the imported model. The only added Ollama source is the test overlay. The inference model and native kernel sources are unchanged.

## Measurement commands

The harnesses append JSONL, so always use a fresh output directory. Copy the exact saved input IDs; no tokenization change is required. Run engines sequentially without other GPU workloads.

```sh
repo=/Users/zhihaojia/lithos/mpk-apple
base=/tmp/monolith-m5max/baseline-verify
run_dir=$(mktemp -d "$base/rerun.XXXXXX")
cp "$base/inputs.json" "$run_dir/inputs.json"

OLLAMA_MODELS="$base/ollama-models" VERIFY_BENCH_DIR="$run_dir" VERIFY_REPS=20 \
  "$base/ollama-runtime/mlxrunner.test" \
  -test.run '^TestTargetVerifyBenchmark$' -test.v -test.timeout 30m \
  > "$run_dir/ollama.log" 2>&1

"$base/vllm-env/bin/python" "$repo/tools/bench/vllm_target_verify.py" \
  --model /tmp/monolith-models/Qwen3.8-27B-NVFP4 --work "$run_dir" \
  --fp8-mode bf16 --reps 20 > "$run_dir/vllm-bf16.log" 2>&1

"$base/vllm-env/bin/python" "$repo/tools/bench/vllm_target_verify.py" \
  --model /tmp/monolith-models/Qwen3.8-27B-NVFP4 --work "$run_dir" \
  --fp8-mode mxfp8 --reps 20 > "$run_dir/vllm-mxfp8.log" 2>&1
```

Each command performs five warmups per case. Ollama runs both final-state and per-token-snapshot scopes. The final vLLM harness also checks eight sequential forwards after each context. The recorded BF16 performance run used `scripts/vllm-first-pass.py`, before adding serial checks and per-replay memory samples. Its numerical check was run separately with the final harness and `--fp8-mode bf16 --reps 1 --validation-only`; all five output hashes match. That one-sample validation file is excluded from the latency tables.

Run `scripts/summarize.py` with the repository Python environment after updating its working-directory constant if needed. It summarizes only `ollama.jsonl`, `vllm-bf16.jsonl`, and `vllm-mxfp8.jsonl`. It checks output tensors when present. The compact archive contains their digests instead of full tensors; original full outputs remain in the machine-local working directory.

## Evidence conventions

`source-manifest.json` hashes the final benchmark sources, installed backend source files, and released native payload. The archived `scripts/vllm-first-pass.py` captures the earlier BF16 timing harness; the archived Ollama test overlay is the exact pre-gofmt source used for the recorded binary. Repository and compiled overlay differences are formatting only. `SHA256SUMS` covers each archived file; output tensor file hashes are in `output-manifest.json`. All original rejected-run logs are explicitly separate from accepted raw timing rows. No loading, prefill, sampling, acceptance, drafting, rewind, HTTP, or independent multi-prompt trial is included in the recorded latency.
