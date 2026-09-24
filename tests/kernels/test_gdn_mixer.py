"""gdn_mixer against the GatedDeltaNet layer oracle (torch): the kernel follows the reference's rounding order, so
the bars are the leaf gates — output ≤ 2 BF16 ULP at its RMS, recurrent state ≤ 8 FP32 ULP of its largest value,
conv state exact — on fresh and filled states, T = 1 / 4 / 8 (several token passes), v_heads = k_heads and 3×,
a|b from the same or a second projection buffer, continuation across steps, bit-identical repeat runs."""

import numpy as np
import pytest

from monolith import kernels
from monolith.bench import check_against_oracle
from monolith.formats.fp import bf16_to_f32, f32_to_bf16
from monolith.runtime import _native as nt

EPS = 1e-6
CW = 4


@pytest.fixture(scope="module")
def dev():
    return nt.Device()


def _module(hidden, hk, hv, dk, dv, seed):
    torch = pytest.importorskip("torch")
    from monolith.nn import GatedDeltaNet

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = GatedDeltaNet(hidden, hk, hv, dk, dv, CW, EPS, hf_prefix="x.", prefix="l.")
    conv_dim = 2 * hk * dk + hv * dv
    m.set_param("conv_w", torch.from_numpy((rng.standard_normal((conv_dim, 1, CW)) * 0.3).astype(np.float32)).to(torch.bfloat16))
    m.set_param("a_log", torch.from_numpy(rng.uniform(-2, 1, hv).astype(np.float32)).to(torch.bfloat16))
    m.set_param("dt_bias", torch.from_numpy(rng.standard_normal(hv).astype(np.float32)).to(torch.bfloat16))
    m.set_param("norm_w", torch.from_numpy((1 + rng.standard_normal(dv) * 0.2).astype(np.float32)).to(torch.bfloat16))
    return m, rng


class Harness:
    def __init__(self, dev, m, t_max, *, ab_separate=False, slice_cols=8, slices_per_block=4, tokens_per_pass=None):
        self.dev, self.m, self.ab_separate = dev, m, ab_separate
        macros = kernels.gdn_macros(m.dk, m.dv, conv_width=CW, t=t_max, slice_cols=slice_cols, slices_per_block=slices_per_block,
                                    tokens_per_pass=tokens_per_pass)
        lib = nt.Library(dev, kernels.gdn_source(), macros)
        self.pso, self.pso_norm = nt.Pipeline(lib, "gdn_mixer"), nt.Pipeline(lib, "gdn_norm")
        self.o_part = nt.Buffer(dev, kernels.gdn_workspace(t_max, m.v_heads, m.dv))
        self.conv_state = nt.Buffer(dev, m.conv_dim * (CW - 1) * 2); self.conv_state.fill(0)
        self.rec_state = nt.Buffer(dev, m.v_heads * m.dk * m.dv * 4); self.rec_state.fill(0)
        conv_w = m.param("conv_w").reshape(m.conv_dim, CW).float().numpy()
        self.aux = [nt.Buffer(dev, f32_to_bf16(conv_w).tobytes()),
                    nt.Buffer(dev, (-np.exp(m.param("a_log").float().numpy())).astype(np.float32).tobytes()),
                    nt.Buffer(dev, m.param("dt_bias").float().numpy().astype(np.float32).tobytes()),
                    nt.Buffer(dev, m.param("norm_w").float().numpy().astype(np.float32).tobytes())]
        self.n_sg = 12 * dev.info().gpu_cores

    def set_state(self, state):
        self.conv_state.write(f32_to_bf16(state["l.conv_state"].float().numpy()).tobytes(), 0)
        self.rec_state.write(state["l.rec_state"].float().numpy().astype(np.float32).tobytes(), 0)

    def get_state(self):
        m = self.m
        conv = bf16_to_f32(np.frombuffer(self.conv_state.read(0, m.conv_dim * (CW - 1) * 2), dtype=np.uint16)).reshape(m.conv_dim, CW - 1)
        rec = np.frombuffer(self.rec_state.read(0, m.v_heads * m.dk * m.dv * 4), dtype=np.float32).reshape(m.v_heads, m.dk, m.dv)
        return conv, rec

    def step(self, proj, t_active=None):
        """``proj`` torch BF16 [T, N1] in checkpoint column order."""
        m = self.m
        t = proj.shape[0]
        pf = f32_to_bf16(proj.float().numpy())
        kd, vd, hv = m.key_dim, m.value_dim, m.v_heads
        if self.ab_separate:                                   # the 27B layout: qkv|z in one slab, a|b in another
            main, ab = pf[:, : m.conv_dim + vd], np.ascontiguousarray(pf[:, m.conv_dim + vd:])
            a_off, b_off, ab_stride = 0, hv, 2 * hv
        else:
            main, ab = pf, pf
            a_off, b_off, ab_stride = m.conv_dim + vd, m.conv_dim + vd + hv, pf.shape[1]
        params = kernels.gdn_params(hv=hv, hk=m.k_heads, t_active=t if t_active is None else t_active, q_off=0, k_off=kd, v_off=2 * kd,
                                    z_off=m.conv_dim, a_off=a_off, b_off=b_off, in_stride=main.shape[1], ab_stride=ab_stride,
                                    ab_separate=self.ab_separate, out_stride=vd, n_sg=self.n_sg, key_dim=kd, eps=EPS)
        out = nt.Buffer(self.dev, t * vd * 2); out.fill(0)
        mb = nt.Buffer(self.dev, main.tobytes())
        abb = nt.Buffer(self.dev, ab.tobytes()) if self.ab_separate else mb
        d = (nt.Dispatch().pipeline(self.pso).buffer(0, mb).buffer(1, abb).buffer(2, self.conv_state).buffer(3, self.rec_state)
             .buffer(4, self.aux[0]).buffer(5, self.aux[1]).buffer(6, self.aux[2]).buffer(7, self.o_part)
             .bytes(9, params).grid(-(-(self.n_sg * 32) // 384)).threadgroup(384).barrier())
        d2 = (nt.Dispatch().pipeline(self.pso_norm).buffer(0, self.o_part).buffer(1, mb).buffer(2, self.aux[3]).buffer(3, out)
              .bytes(4, params).grid(t * hv).threadgroup(32))
        r = nt.Queue(self.dev).run([d, d2])
        assert not r.error, r.error
        return bf16_to_f32(np.frombuffer(out.read(0, t * vd * 2), dtype=np.uint16).reshape(t, vd))


def _proj(rng, torch, t, n1):
    return torch.from_numpy(rng.standard_normal((t, n1)).astype(np.float32)).to(torch.bfloat16)


def _check(got, ref_out, got_rec, ref_rec, got_conv, ref_conv):
    # the output is a BF16 tensor whose every element went through the same rounding chain as the oracle's (norm,
    # weight, gate), so the element-wise ULP is the meaningful gate here: FP32 summation-order noise flips a rounding
    # boundary in ~1 element per thousand by one ULP; the at-RMS metric of the GEMV gate over-weights such an
    # element when it is far above the RMS
    assert np.array_equal(got_conv, ref_conv)
    scale = float(np.abs(ref_rec).max())
    assert np.abs(got_rec - ref_rec).max() <= 8 * 2.0 ** -23 * max(scale, 1e-30), (np.abs(got_rec - ref_rec).max(), scale)
    chk = check_against_oracle(got, ref_out)
    assert chk.ok_rounded(), chk


@pytest.mark.parametrize("hk,hv,ab_separate", [(16, 16, False), (16, 48, True), (4, 4, False)], ids=["16x16", "16x48_ab_separate", "4x4"])
@pytest.mark.parametrize("t", [1, 4, 8])
def test_matches_layer_oracle(dev, hk, hv, ab_separate, t):
    torch = pytest.importorskip("torch")
    m, rng = _module(64, hk, hv, 128, 128, seed=hk * 7 + hv + t)
    h = Harness(dev, m, t, ab_separate=ab_separate)
    state = {"l.conv_state": torch.from_numpy((rng.standard_normal((m.conv_dim, CW - 1)) * 0.5).astype(np.float32)).to(torch.bfloat16),
             "l.rec_state": torch.from_numpy((rng.standard_normal((hv, 128, 128)) * 0.1).astype(np.float32))}
    h.set_state(state)
    for step in range(2):                                      # a step on the filled state, then a continuation
        proj = _proj(rng, torch, t, m.in_proj.n)
        with torch.no_grad():
            ref = m.mix(proj, state).float().numpy()
        got = h.step(proj)
        conv, rec = h.get_state()
        _check(got, ref, rec, state["l.rec_state"].numpy(), conv, state["l.conv_state"].float().numpy())


def test_fresh_state_passes_and_repeat_runs(dev):
    torch = pytest.importorskip("torch")
    m, rng = _module(64, 16, 16, 128, 128, seed=3)
    state = {"l.conv_state": torch.zeros(m.conv_dim, CW - 1, dtype=torch.bfloat16), "l.rec_state": torch.zeros(16, 128, 128)}
    proj = _proj(rng, torch, 6, m.in_proj.n)
    with torch.no_grad():
        ref = m.mix(proj, state).float().numpy()
    for tp, spb in ((2, 1), (3, 4), (6, 16)):                  # 3, 2 and 1 token passes; 1, 4 and 16 slices per block
        h = Harness(dev, m, 6, tokens_per_pass=tp, slices_per_block=spb)
        got = h.step(proj)
        conv, rec = h.get_state()
        _check(got, ref, rec, state["l.rec_state"].numpy(), conv, state["l.conv_state"].float().numpy())
    h = Harness(dev, m, 6)
    a = h.step(proj)
    h2 = Harness(dev, m, 6)
    assert np.array_equal(a, h2.step(proj)) and np.array_equal(h.get_state()[1], h2.get_state()[1])


def test_t_active_and_slice_width(dev):
    torch = pytest.importorskip("torch")
    m, rng = _module(64, 16, 16, 128, 128, seed=5)
    state = {"l.conv_state": torch.zeros(m.conv_dim, CW - 1, dtype=torch.bfloat16), "l.rec_state": torch.zeros(16, 128, 128)}
    proj = _proj(rng, torch, 4, m.in_proj.n)
    with torch.no_grad():
        ref = m.mix(proj[:2], state).float().numpy()
    h = Harness(dev, m, 4, slice_cols=16, slices_per_block=2)
    got = h.step(proj, t_active=2)
    assert np.all(got[2:] == 0)
    conv, rec = h.get_state()
    _check(got[:2], ref, rec, state["l.rec_state"].numpy(), conv, state["l.conv_state"].float().numpy())
