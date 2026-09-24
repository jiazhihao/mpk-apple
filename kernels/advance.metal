// advance: the per-step SERIAL op that closes a step (design §5.4): publishes the last position's sampled token to
// the ring as (sequence << 32) | token, makes it the next step's pending token, advances position and step, and
// sets `done` at EOS. The program's StepState struct is prepended by the compiler (StepStateLayout.to_msl()).
// One thread. T is `t_this_step` (the host sets it per prefill chunk; the advance leaves 1 for decode). A prompt
// longer than one step is fed in chunks: while `prefill_left > 0` the step only advances the position and emits
// nothing (the next chunk's tokens come from the prompt); the last chunk's last position is the first sampled token.
struct AdvanceParams { uint t_active; uint ring_cap; int eos; uint pad; };

kernel void advance(device const int* token [[buffer(0)]], device StepState* st [[buffer(1)]], device ulong* ring [[buffer(2)]],
                    constant AdvanceParams& p [[buffer(3)]], uint i [[thread_position_in_grid]]) {
  if (i != 0 || st->done) return;
  const uint t = st->t_this_step;
  if (st->prefill_left > 0u) {
    st->position = st->position + t;
    st->step = st->step + 1u;
    return;
  }
  const int tok = token[t - 1u];
  const uint head = st->ring_head;
  if (head - st->ring_tail >= p.ring_cap) { st->error = 1; st->done = 1; return; }     // ring overflow: the host fell behind
  ring[head % p.ring_cap] = (ulong(head + 1u) << 32) | ulong(uint(tok));
  st->ring_head = head + 1u;
  st->pending_tokens[0] = tok;
  st->position = st->position + t;
  st->step = st->step + 1u;
  st->t_this_step = 1u;
  if (p.eos >= 0 && tok == p.eos) st->done = 1;
}
