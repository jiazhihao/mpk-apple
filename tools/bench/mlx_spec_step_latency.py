"""Time full MLX-LM speculative rounds between native generator round-end yields.

Uses the unmodified speculative_generate_step, not an HTTP server. A non-draft
yield ends a round. Read its suspended frame to exclude shortened final rounds;
this instrumentation depends on the pinned MLX-LM generator's local names.
"""
import argparse
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import statistics
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--drafter', required=True)
    parser.add_argument('--prompts', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load

    generation = importlib.import_module('mlx_lm.generate')
    model, tokenizer = load(args.model)
    draft, draft_tokenizer = load(args.drafter)
    if tokenizer.get_vocab() != draft_tokenizer.get_vocab():
        raise ValueError('Target and drafter token-ID mappings differ')
    source_hash = hashlib.sha256(Path(generation.__file__).read_bytes()).hexdigest()
    versions = {p: importlib.metadata.version(p) for p in ('mlx', 'mlx-lm')}

    for prompt in json.loads(args.prompts.read_text()):
        ids = tokenizer.encode(prompt['prompt'], add_special_tokens=False)
        assert len(ids) == prompt['input_tokens']
        tokens, boundaries, intervals = [], [], []
        mx.synchronize()
        start = time.perf_counter()
        generator = generation.speculative_generate_step(
            mx.array(ids), model, draft, num_draft_tokens=7, max_tokens=128,
            prefill_step_size=512)
        with generation.wired_limit(model, [generation.generation_stream]):
            try:
                for token, _, from_draft in generator:
                    now = time.perf_counter()
                    token = int(token)
                    if token in tokenizer.eos_token_ids:
                        break
                    tokens.append(token)
                    if not from_draft:
                        state = generator.gi_frame.f_locals
                        boundary = dict(elapsed_ms=(now-start)*1000,
                                        proposed=int(state['num_draft']),
                                        accepted=int(state['n']),
                                        emitted_tokens=len(tokens))
                        if boundaries and boundaries[-1]['proposed'] == boundary['proposed'] == 7:
                            intervals.append(boundary['elapsed_ms'] - boundaries[-1]['elapsed_ms'])
                        boundaries.append(boundary)
            finally:
                generator.close()
        assert len(tokens) == 128, 'Early EOS: not comparable to the 128-token screen'
        assert intervals and min(intervals) > 0
        row = dict(engine='mlx-lm-n7', prompt_id=prompt['id'],
                   input_tokens=len(ids), output_tokens=len(tokens),
                   warmup=prompt['rep'] == 0,
                   timestamp=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                   model=args.model, drafter=args.drafter, versions=versions,
                   generator_source_sha256=source_hash,
                   prefill_step_size=512, num_draft_tokens=7,
                   ms_per_full_step=statistics.mean(intervals),
                   full_step_intervals_ms=intervals, round_end_boundaries=boundaries,
                   tokens=tokens, text=tokenizer.decode(tokens),
                   peak_memory_bytes=mx.get_peak_memory())
        with args.out.open('a') as f:
            f.write(json.dumps(row) + '\n')
        print(prompt['id'], round(row['ms_per_full_step'], 3),
              'ms/full step', len(intervals), 'timed rounds',
              'warmup' if row['warmup'] else '', flush=True)


if __name__ == '__main__':
    main()
