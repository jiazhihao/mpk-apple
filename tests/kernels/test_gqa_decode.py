"""gqa_decode + gqa_merge against (a) a numpy model of the kernel's own contract (chunked online softmax with the
reference's roundings) and (b) the GQAAttention layer oracle (torch, the HF-faithful semantics), on fresh and
filled caches, T = 1 and T > 1 (causal inside the step), several chunks and row groups, both head dims."""

import numpy as np
import pytest

from monolith import kernels
from monolith.formats.fp import bf16_to_f32, f32_to_bf16
from monolith.nn.rope import rope_tables_permuted
from monolith.packs.transforms import rope_head_perm
from monolith.runtime import _native as nt

EPS = 1e-6
THETA = 10000.0


def rbf(x):
    return bf16_to_f32(f32_to_bf16(np.asarray(x, dtype=np.float32)))


@pytest.fixture(scope="module")
def dev():
    return nt.Device()


class Cfg:
    def __init__(self, heads, kv, d, rot, ctx_max, t_max, chunk=64, rb=4, gate=True, v2=False, n_tg=None):
        self.heads, self.kv, self.d, self.rot, self.ctx_max, self.t_max = heads, kv, d, rot, ctx_max, t_max
        self.v2, self.n_tg = v2, n_tg                                 # v2: the partial granularity is 32 keys (the numpy model's chunk)
        if v2:
            chunk = 32
        self.chunk, self.rb, self.gate = chunk, rb, gate
        self.rep = heads // kv
        hd, kd = heads * d, kv * d
        self.q_off, self.k_off, self.v_off = 0, hd, hd + kd                    # the pack's order: q | k | v | gate
        self.gate_off = hd + 2 * kd
        self.n1 = hd + 2 * kd + (hd if gate else 0)
        self.rows_max = self.rep * t_max
        self.n_chunks_max = -(-ctx_max // chunk)
        self.scaling = d ** -0.5


class Harness:
    """The two dispatches with their caches, tables and workspace for one layer config."""

    def __init__(self, dev, cfg, qn, kn):
        self.dev, self.cfg = dev, cfg
        if cfg.v2:
            lib = nt.Library(dev, kernels.gqa_source(True), kernels.gqa_v2_macros(cfg.d, rmax=cfg.rows_max, rg=cfg.rb))
            self.p_dec, self.p_merge = nt.Pipeline(lib, "gqa_decode_v2"), nt.Pipeline(lib, "gqa_merge_v2")
        else:
            lib = nt.Library(dev, kernels.gqa_source(), kernels.gqa_macros(cfg.d, chunk=cfg.chunk, rb_max=cfg.rb))
            self.p_dec, self.p_merge = nt.Pipeline(lib, "gqa_decode"), nt.Pipeline(lib, "gqa_merge")
        cos, sin = rope_tables_permuted(THETA, cfg.d, cfg.rot, cfg.ctx_max)
        self.cos_b, self.sin_b = f32_to_bf16(cos), f32_to_bf16(sin)
        self.cos, self.sin = bf16_to_f32(self.cos_b), bf16_to_f32(self.sin_b)
        self.qn, self.kn = qn.astype(np.float32), kn.astype(np.float32)
        nbytes = self.cache_bytes = cfg.ctx_max * cfg.kv * cfg.d * 2
        self.k_cache, self.v_cache = nt.Buffer(dev, nbytes), nt.Buffer(dev, nbytes)
        self.k_cache.fill(0); self.v_cache.fill(0)
        po, pm = kernels.gqa_workspace(cfg.kv, cfg.n_chunks_max, cfg.rows_max, cfg.d)
        self.part_o, self.part_md = nt.Buffer(dev, po), nt.Buffer(dev, pm)
        self.bufs = {"cos": nt.Buffer(dev, self.cos_b.tobytes()), "sin": nt.Buffer(dev, self.sin_b.tobytes()),
                     "qn": nt.Buffer(dev, self.qn.tobytes()), "kn": nt.Buffer(dev, self.kn.tobytes())}
        self.n_sg = 12 * dev.info().gpu_cores
        self.n_tg = dev.info().gpu_cores if cfg.n_tg is None else cfg.n_tg      # v2: threadgroups (params.n_sg); the grid stays the crew

    def set_caches(self, k, v):
        self.k_cache.write(f32_to_bf16(k).tobytes(), 0)
        self.v_cache.write(f32_to_bf16(v).tobytes(), 0)

    def caches(self):
        c = self.cfg
        shape = (c.ctx_max, c.kv, c.d)
        return (bf16_to_f32(np.frombuffer(self.k_cache.read(0, self.cache_bytes), dtype=np.uint16).reshape(shape)),
                bf16_to_f32(np.frombuffer(self.v_cache.read(0, self.cache_bytes), dtype=np.uint16).reshape(shape)))

    def step(self, proj_bf16, position, t_active=None):
        c = self.cfg
        t = proj_bf16.shape[0]
        t_act = t if t_active is None else t_active
        params = kernels.gqa_params(heads=c.heads, kv_heads=c.kv, t_active=t_act, position=position, n_sg=self.n_tg if c.v2 else self.n_sg,
                                    q_off=c.q_off, gate_off=c.gate_off, k_off=c.k_off, v_off=c.v_off, in_stride=c.n1,
                                    out_stride=c.heads * c.d, ctx_max=c.ctx_max, eps=EPS, scaling=c.scaling, has_gate=c.gate,
                                    n_chunks_max=c.n_chunks_max, rows_max=c.rows_max)
        pb = nt.Buffer(self.dev, proj_bf16.tobytes())
        out = nt.Buffer(self.dev, t * c.heads * c.d * 2); out.fill(0)
        d1 = (nt.Dispatch().pipeline(self.p_dec).buffer(0, pb).buffer(1, self.k_cache).buffer(2, self.v_cache)
              .buffer(3, self.bufs["cos"]).buffer(4, self.bufs["sin"]).buffer(5, self.bufs["qn"]).buffer(6, self.bufs["kn"])
              .buffer(7, self.part_o).buffer(8, self.part_md).bytes(9, params).grid(-(-(self.n_sg * 32) // 384)).threadgroup(384).barrier())
        d2 = (nt.Dispatch().pipeline(self.p_merge).buffer(0, self.part_o).buffer(1, self.part_md).buffer(2, pb).buffer(3, out)
              .bytes(4, params).grid(t * c.heads).threadgroup(32))
        r = nt.Queue(self.dev).run([d1, d2])
        assert not r.error, r.error
        return bf16_to_f32(np.frombuffer(out.read(0, t * c.heads * c.d * 2), dtype=np.uint16).reshape(t, c.heads * c.d))


# ---- the kernel's contract in numpy -----------------------------------------------------------------------------

def norm_rope_np(x, nw, cos_row, sin_row, d):
    """``x [..., D]`` in the permuted layout: (1 + w) RMSNorm then full-width rotary pairs, the reference's roundings."""
    ss = (x.astype(np.float32) ** 2).sum(-1, keepdims=True)
    rstd = (1.0 / np.sqrt(ss / d + EPS)).astype(np.float32)
    x = rbf(x * rstd * nw)
    half = d // 2
    partner = np.concatenate([x[..., half:], x[..., :half]], axis=-1)
    a, b = rbf(x * cos_row), rbf(partner * sin_row)
    sign = np.concatenate([-np.ones(half), np.ones(half)]).astype(np.float32)
    return rbf(a + sign * b)


def ref_step(h, proj, position, k_cache, v_cache):
    """The chunked online-softmax semantics of gqa_decode + gqa_merge; ``k_cache``/``v_cache`` (numpy, BF16-valued)
    are advanced in place. Returns ``[T, heads·D]``."""
    c = h.cfg
    t = proj.shape[0]
    d = c.d
    q = proj[:, c.q_off: c.q_off + c.heads * d].reshape(t, c.heads, d)
    k = proj[:, c.k_off: c.k_off + c.kv * d].reshape(t, c.kv, d)
    v = proj[:, c.v_off: c.v_off + c.kv * d].reshape(t, c.kv, d)
    pos = np.arange(position, position + t)
    q = norm_rope_np(q, h.qn, h.cos[pos][:, None, :], h.sin[pos][:, None, :], d)
    k = norm_rope_np(k, h.kn, h.cos[pos][:, None, :], h.sin[pos][:, None, :], d)
    k_cache[position: position + t] = k
    v_cache[position: position + t] = v
    ctx = position + t
    out = np.zeros((t, c.heads, d), dtype=np.float32)
    for tt in range(t):
        for hh in range(c.heads):
            j = hh // c.rep
            keys = np.arange(0, ctx)
            s = rbf(rbf(k_cache[:ctx, j].astype(np.float64) @ q[tt, hh].astype(np.float64)) * c.scaling)
            s[keys > position + tt] = -np.inf
            parts = []
            for c0 in range(0, ctx, c.chunk):
                sc = s[c0: c0 + c.chunk]
                m = sc.max()
                if m == -np.inf:
                    continue
                p = np.exp((sc - m).astype(np.float32)).astype(np.float32)
                p[sc == -np.inf] = 0
                parts.append((m, float(p.sum(dtype=np.float32)), rbf(p).astype(np.float64) @ v_cache[c0: min(c0 + c.chunk, ctx), j].astype(np.float64)))
            m_g = max(m for m, _, _ in parts)
            d_g = sum(dc * np.exp(m - m_g) for m, dc, _ in parts)
            o = sum(oc * np.exp(m - m_g) for m, _, oc in parts) / d_g
            y = rbf(o)
            if c.gate:
                g = proj[tt, c.gate_off + hh * d: c.gate_off + (hh + 1) * d]
                y = rbf(y * rbf(1.0 / (1.0 + np.exp(-g))))
            out[tt, hh] = y
    return out.reshape(t, c.heads * d)


def _bars(got, ref):
    got, ref = got.astype(np.float64).ravel(), ref.astype(np.float64).ravel()
    cos = float(np.dot(got, ref) / (np.linalg.norm(got) * np.linalg.norm(ref) + 1e-30))
    return cos, float(np.abs(got - ref).max()), float(np.abs(ref).max())


def _random_proj(rng, cfg, t):
    return f32_to_bf16(rng.standard_normal((t, cfg.n1)).astype(np.float32))


def _norms(rng, d):
    return (1.0 + rng.standard_normal(d) * 0.1).astype(np.float32), (1.0 + rng.standard_normal(d) * 0.1).astype(np.float32)


@pytest.mark.parametrize("cfg", [Cfg(8, 2, 256, 64, 512, 8), Cfg(4, 1, 128, 32, 256, 8, chunk=32, rb=2), Cfg(6, 1, 256, 64, 200, 4, rb=8),
                                 Cfg(4, 2, 128, 128, 128, 4, gate=False)], ids=["d256_rep4", "d128_rep4_ch32_rb2", "rep6_rb8", "nogate_fullrope"])
def test_matches_kernel_contract(dev, cfg):
    rng = np.random.default_rng(cfg.heads * 31 + cfg.d)
    qn, kn = _norms(rng, cfg.d)
    h = Harness(dev, cfg, qn, kn)
    k_ref, v_ref = np.zeros((cfg.ctx_max, cfg.kv, cfg.d), np.float32), np.zeros((cfg.ctx_max, cfg.kv, cfg.d), np.float32)
    # a filled prefix of the caches (random BF16), then three steps: T = 5 prefill, T = 1, T = 4 (causal inside)
    pre = 70
    k_ref[:pre], v_ref[:pre] = rbf(rng.standard_normal((pre, cfg.kv, cfg.d)) * 0.5), rbf(rng.standard_normal((pre, cfg.kv, cfg.d)))
    h.set_caches(k_ref, v_ref)
    pos = pre
    for t in (5, 1, 4):
        if t > cfg.t_max:
            continue
        proj = _random_proj(rng, cfg, t)
        got = h.step(proj, pos)
        ref = ref_step(h, bf16_to_f32(proj), pos, k_ref, v_ref)
        cos, max_abs, scale = _bars(got, ref)
        assert cos > 0.99999 and max_abs <= 2e-3 * scale, (t, pos, cos, max_abs, scale)
        kc, vc = h.caches()
        assert np.abs(kc[: pos + t] - k_ref[: pos + t]).max() <= 1e-2 * np.abs(k_ref[: pos + t]).max()   # ≤ 1 ULP flips at rsqrt boundaries
        assert np.array_equal(vc[: pos + t], v_ref[: pos + t])
        pos += t


def test_repeat_runs_are_bit_identical_and_t_active(dev):
    cfg = Cfg(8, 2, 256, 64, 256, 4)
    rng = np.random.default_rng(5)
    h = Harness(dev, cfg, *_norms(rng, cfg.d))
    proj = _random_proj(rng, cfg, 4)
    a = h.step(proj, 0)
    h.set_caches(np.zeros((cfg.ctx_max, cfg.kv, cfg.d), np.float32), np.zeros((cfg.ctx_max, cfg.kv, cfg.d), np.float32))
    b = h.step(proj, 0)
    assert np.array_equal(a, b)
    h.set_caches(np.zeros((cfg.ctx_max, cfg.kv, cfg.d), np.float32), np.zeros((cfg.ctx_max, cfg.kv, cfg.d), np.float32))
    c = h.step(proj, 0, t_active=2)
    assert np.array_equal(c[:2], a[:2]) and np.all(c[2:] == 0)
    kc, _ = h.caches()
    assert np.all(kc[2:] == 0)                                             # only t_active positions appended


def test_matches_layer_oracle(dev):
    """The HF-faithful GQAAttention.mix (torch) vs the kernel on the same projection, caches and positions."""
    torch = pytest.importorskip("torch")
    from monolith.nn import GQAAttention

    heads, kv, d, rot, hidden, ctx_max = 8, 2, 256, 64, 64, 256
    cfg = Cfg(heads, kv, d, rot, ctx_max, 8)
    rng = np.random.default_rng(9)
    torch.manual_seed(9)
    mod = GQAAttention(hidden, heads, kv, d, rot, THETA, EPS, hf_prefix="x.", prefix="l.", max_context=ctx_max)
    qn_w, kn_w = (rng.standard_normal(d) * 0.1).astype(np.float32), (rng.standard_normal(d) * 0.1).astype(np.float32)
    mod.set_param("q_norm", torch.from_numpy(qn_w).to(torch.bfloat16))
    mod.set_param("k_norm", torch.from_numpy(kn_w).to(torch.bfloat16))
    perm = rope_head_perm(d, rot)
    qn_p, kn_p = (1.0 + bf16_to_f32(f32_to_bf16(qn_w)))[perm], (1.0 + bf16_to_f32(f32_to_bf16(kn_w)))[perm]
    h = Harness(dev, cfg, qn_p, kn_p)
    state = {"l.k_cache": torch.zeros(ctx_max, kv, d, dtype=torch.bfloat16), "l.v_cache": torch.zeros(ctx_max, kv, d, dtype=torch.bfloat16)}
    col_perm = mod.qkv.row_perm                                             # kernel column n = checkpoint column perm[n]
    inv = np.argsort(perm)
    pos = 0
    for t in (6, 1, 3):
        proj_hf = torch.from_numpy(rng.standard_normal((t, cfg.n1)).astype(np.float32)).to(torch.bfloat16)
        with torch.no_grad():
            ref = mod.mix(proj_hf, state, pos).float().numpy()
        proj_k = proj_hf.float().numpy()[:, col_perm]
        got = h.step(f32_to_bf16(proj_k), pos)
        cos, max_abs, scale = _bars(got, ref)
        # the kernel rounds P before normalization, per chunk (online softmax); the reference rounds the normalized
        # P — a 1–2 BF16 ULP difference at the output's magnitude, inside the composite bar (cos > 0.999, max-abs
        # bounded), not the leaf bar (that one is the kernel-contract test above)
        assert cos > 0.9999 and max_abs <= 1e-2 * scale, (t, pos, cos, max_abs, scale)
        kc, vc = h.caches()
        k_hf = state["l.k_cache"][: pos + t].float().numpy()
        assert np.abs(kc[: pos + t][..., inv] - k_hf).max() <= 1e-2 * np.abs(k_hf).max()
        assert np.array_equal(vc[: pos + t], state["l.v_cache"][: pos + t].float().numpy())
        pos += t


# ---- v2: the long-context structure (#34) --------------------------------------------------------------------------

@pytest.mark.parametrize("cfg", [Cfg(8, 2, 256, 64, 2048, 8, rb=4, v2=True), Cfg(8, 2, 256, 64, 2048, 8, rb=4, v2=True, n_tg=1),
                                 Cfg(4, 1, 128, 32, 4096, 8, rb=8, v2=True, n_tg=2), Cfg(4, 2, 128, 128, 1024, 4, gate=False, v2=True)],
                         ids=["v2_d256_ch32", "v2_d256_1tg_ch128", "v2_d128_2tg_ch64", "v2_nogate_fullrope"])
def test_v2_matches_kernel_contract(dev, cfg):
    """v2 against the numpy model of the kernel contract at chunk 32 (one partial per 32-key sub-chunk, folded
    exactly): contexts spanning several batches, T = 5 / 1 / 4 (causal inside the step), the new keys appended by
    their batch, and the chunk rule (n_tg = 1 → 128-key chunks, 2 → 64 at these contexts, the crew → 32)."""
    rng = np.random.default_rng(cfg.heads * 31 + cfg.d + cfg.ctx_max)
    qn, kn = _norms(rng, cfg.d)
    h = Harness(dev, cfg, qn, kn)
    k_ref, v_ref = np.zeros((cfg.ctx_max, cfg.kv, cfg.d), np.float32), np.zeros((cfg.ctx_max, cfg.kv, cfg.d), np.float32)
    pre = min(1500, cfg.ctx_max - 32)                                   # a filled prefix across several 12-chunk batches
    k_ref[:pre], v_ref[:pre] = rbf(rng.standard_normal((pre, cfg.kv, cfg.d)) * 0.5), rbf(rng.standard_normal((pre, cfg.kv, cfg.d)))
    h.set_caches(k_ref, v_ref)
    pos = pre
    for t in (5, 1, 4):
        if t > cfg.t_max:
            continue
        proj = _random_proj(rng, cfg, t)
        got = h.step(proj, pos)
        ref = ref_step(h, bf16_to_f32(proj), pos, k_ref, v_ref)
        cos, max_abs, scale = _bars(got, ref)
        assert cos > 0.99999 and max_abs <= 2e-3 * scale, (t, pos, cos, max_abs, scale)
        kc, vc = h.caches()
        assert np.abs(kc[: pos + t] - k_ref[: pos + t]).max() <= 1e-2 * np.abs(k_ref[: pos + t]).max()
        assert np.array_equal(vc[: pos + t], v_ref[: pos + t])
        pos += t


def test_v2_equals_v1_and_is_bit_stable(dev):
    """The two kernels agree within the contract bar on the same inputs at a long context (v1 at chunk 64 vs v2's
    hierarchical fold), v2 repeats bit-identically, and t_active limits the rows and the append."""
    heads, kv, d, ctx_max = 8, 2, 256, 4096
    rng = np.random.default_rng(77)
    qn, kn = _norms(rng, d)
    h1 = Harness(dev, Cfg(heads, kv, d, 64, ctx_max, 8), qn, kn)
    h2 = Harness(dev, Cfg(heads, kv, d, 64, ctx_max, 8, v2=True), qn, kn)
    pre = 3000
    k0, v0 = rbf(rng.standard_normal((pre, kv, d)) * 0.5), rbf(rng.standard_normal((pre, kv, d)))
    for h in (h1, h2):
        kk, vv = np.zeros((ctx_max, kv, d), np.float32), np.zeros((ctx_max, kv, d), np.float32)
        kk[:pre], vv[:pre] = k0, v0
        h.set_caches(kk, vv)
    proj = _random_proj(rng, Cfg(heads, kv, d, 64, ctx_max, 8), 4)
    a, b = h1.step(proj, pre), h2.step(proj, pre)
    cos, max_abs, scale = _bars(b, a)
    # v1 rounds p̃ per 64-key chunk, v2 per 32-key sub-chunk: the two contracts differ by BF16 ULPs of P — the
    # composite bar, as against the layer oracle (each kernel meets the leaf bar against its own numpy model)
    assert cos > 0.9999 and max_abs <= 1e-2 * scale, (cos, max_abs, scale)
    h2.set_caches(*(lambda kk, vv: (kk, vv))(*[np.concatenate([x, np.zeros((ctx_max - pre, kv, d), np.float32)]) for x in (k0, v0)]))
    b2 = h2.step(proj, pre)
    assert np.array_equal(b, b2)
    h2.set_caches(*[np.concatenate([x, np.zeros((ctx_max - pre, kv, d), np.float32)]) for x in (k0, v0)])
    c = h2.step(proj, pre, t_active=2)
    assert np.array_equal(c[:2], b[:2]) and np.all(c[2:] == 0)
    kc, _ = h2.caches()
    assert np.all(kc[pre + 2:] == 0)


def test_v2_matches_layer_oracle(dev):
    torch = pytest.importorskip("torch")
    from monolith.nn import GQAAttention

    heads, kv, d, rot, hidden, ctx_max = 8, 2, 256, 64, 64, 512
    cfg = Cfg(heads, kv, d, rot, ctx_max, 8, v2=True)
    rng = np.random.default_rng(9)
    torch.manual_seed(9)
    mod = GQAAttention(hidden, heads, kv, d, rot, THETA, EPS, hf_prefix="x.", prefix="l.", max_context=ctx_max)
    qn_w, kn_w = (rng.standard_normal(d) * 0.1).astype(np.float32), (rng.standard_normal(d) * 0.1).astype(np.float32)
    mod.set_param("q_norm", torch.from_numpy(qn_w).to(torch.bfloat16))
    mod.set_param("k_norm", torch.from_numpy(kn_w).to(torch.bfloat16))
    perm = rope_head_perm(d, rot)
    h = Harness(dev, cfg, (1.0 + bf16_to_f32(f32_to_bf16(qn_w)))[perm], (1.0 + bf16_to_f32(f32_to_bf16(kn_w)))[perm])
    state = {"l.k_cache": torch.zeros(ctx_max, kv, d, dtype=torch.bfloat16), "l.v_cache": torch.zeros(ctx_max, kv, d, dtype=torch.bfloat16)}
    col_perm = mod.qkv.row_perm
    pos = 0
    for t in (6, 1, 3, 8):
        proj_hf = torch.from_numpy(rng.standard_normal((t, cfg.n1)).astype(np.float32)).to(torch.bfloat16)
        with torch.no_grad():
            ref = mod.mix(proj_hf, state, pos).float().numpy()
        got = h.step(f32_to_bf16(proj_hf.float().numpy()[:, col_perm]), pos)
        cos, max_abs, scale = _bars(got, ref)
        assert cos > 0.9999 and max_abs <= 1e-2 * scale, (t, pos, cos, max_abs, scale)
        pos += t

