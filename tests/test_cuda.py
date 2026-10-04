"""The fused CUDA kernels (pianonn.cuda_ext) against the PyTorch reference: the same audio and gradients."""
import math

import pytest
import torch

from pianonn import NeuralPhysicalPiano, cuda_ext
from pianonn.dsp import linear_recurrence

from .conftest import small_cfg

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or cuda_ext.get() is None,
                                reason="needs CUDA and the compiled kernels")
DEV = "cuda"


def perf_with_history(model, n_samples, notes, hist_s=0.5, sustain=None):
    """A batch of one with ``hist_s`` s of control history before the window; ``notes`` (pitch, onset, offset,
    velocity) in s re the window."""
    cfg = model.cfg
    H = int(round(hist_s * cfg.sample_rate / cfg.hop))
    F = H + model.n_frames(n_samples)
    t = torch.tensor(notes, dtype=torch.float32)[None]
    tf = (torch.arange(F) - H) * cfg.hop / cfg.sample_rate
    sus = sustain(tf)[None] if sustain else torch.zeros(1, F)
    p = {"pitch": t[..., 0].long(), "onset": t[..., 1], "offset": t[..., 2], "velocity": t[..., 3],
         "mask": torch.ones(1, len(notes), dtype=torch.bool), "condition": torch.tensor([3]), "sustain": sus,
         "soft": torch.zeros(1, F), "sostenuto": torch.zeros(1, F), "hist_frames": torch.tensor(H)}
    return {k: v.to(DEV) for k, v in p.items()}


NOTES = [(60, 0.05, 0.40, 80), (43, -0.40, 0.70, 100), (100, 0.20, 0.30, 40), (60, 0.30, 0.90, 90),
         (36, 0.10, 0.25, 70), (72, -0.10, 0.60, 110)]


def render_both(model, perf, n, monkeypatch, residual=True):
    """Audio and parameter gradients ``(fused, reference)`` of one render with the same random draws."""
    res = []
    for fused in (True, False):
        with monkeypatch.context() as mp:
            if not fused:
                mp.setattr(cuda_ext, "get", lambda: None)
            model.zero_grad(set_to_none=True)
            out = model(perf, n, residual=residual, generator=torch.Generator(device=DEV).manual_seed(3))
            w = torch.randn(out["audio"].shape, device=DEV, generator=torch.Generator(device=DEV).manual_seed(4))
            ((out["audio"] * w).sum() + 100 * out["strings"].pow(2).sum() + 1e4 * out["symp"].pow(2).sum()).backward()
            res.append((out["audio"].detach(), {k: p.grad.clone() for k, p in model.named_parameters()
                                                if p.grad is not None}))
    return res


def assert_close(fus, ref, audio_db=-90.0, grad_rel=2e-3, loose=()):
    """``loose``: parameters held to 5 x ``grad_rel`` (the free longitudinal modes' frequency: a resonator's frequency
    gradient sums large terms of both signs, and the fused path runs that recurrence over the whole window, the
    reference per activity chunk)."""
    (a_f, g_f), (a_r, g_r) = fus, ref
    err = (a_f - a_r).abs().max() / a_r.abs().max()
    assert 20 * math.log10(float(err) + 1e-30) < audio_db
    assert g_f.keys() == g_r.keys()
    for k in g_r:
        rel = float((g_f[k] - g_r[k]).norm() / (g_r[k].norm() + 1e-30))
        assert rel < grad_rel * (5 if k in loose else 1) or float(g_r[k].norm()) < 1e-6, (k, rel)


@pytest.mark.parametrize("curve_partials", [0, 6])
def test_fused_matches_reference_with_the_aware_residual(monkeypatch, curve_partials):
    """Every oscillator rendered on both sides (activity_db 400: the reference pads slices with oscillators below the
    activity threshold, the fused bank does not); the residual's curves randomised, so the group gains are not 1."""
    cfg = small_cfg(residual_kind="aware", res_curve_partials=curve_partials, res_dim=32, res_heads=2, activity_db=400.0,
                    strike_level_db=1.0, strike_log_decay=0.2, strike_onset_ms=3.0)
    m = NeuralPhysicalPiano(cfg).to(DEV)
    with torch.no_grad():
        m.context.curve_head[-1].weight.normal_(0, 0.3)
        m.context.curve_head[-1].bias.normal_(0, 0.3)
    n = 6000
    perf = perf_with_history(m, n, NOTES, sustain=lambda t: ((t > 0.2) & (t < 0.5)).float())
    fus, ref = render_both(m, perf, n, monkeypatch)
    assert_close(fus, ref)


def test_fused_matches_reference_without_the_residual(monkeypatch):
    m = NeuralPhysicalPiano(small_cfg(activity_db=400.0)).to(DEV)
    n = 6000
    perf = perf_with_history(m, n, NOTES, sustain=lambda t: (t > 0.35).float())
    fus, ref = render_both(m, perf, n, monkeypatch, residual=False)
    assert_close(fus, ref)


def test_fused_activity_rule_is_the_references(monkeypatch):
    """At the default activity threshold the two differ only by the reference's padding: oscillators more than 70 dB
    under their note's peak."""
    m = NeuralPhysicalPiano(small_cfg()).to(DEV)
    n = 6000
    perf = perf_with_history(m, n, NOTES)
    fus, ref = render_both(m, perf, n, monkeypatch, residual=False)
    assert_close(fus, ref, audio_db=-70.0, grad_rel=5e-2)


@pytest.mark.parametrize("curve_partials", [0, 6])
def test_fused_coupled_bank_matches_reference(monkeypatch, curve_partials):
    """The coupled strings' bus bank (cuda_ext.BusBank) against oscbank.bus_bank: every new parameter moved off its
    start (unison spread, polarisations, admittance, longitudinal path, knock resonances, glide, image per key), the
    residual's curves randomised; every oscillator rendered on both sides (activity_db 400)."""
    cfg = small_cfg(string_model="coupled", residual_kind="aware", res_curve_partials=curve_partials, res_dim=32,
                    res_heads=2, activity_db=400.0, strike_evenness=0.1)
    m = NeuralPhysicalPiano(cfg).to(DEV)
    g = torch.Generator().manual_seed(5)
    with torch.no_grad():
        m.context.curve_head[-1].weight.normal_(0, 0.3)
        m.context.curve_head[-1].bias.normal_(0, 0.3)
        for k, p in list(m.physics.coupled.named_parameters()) + [("delay", m.room.raw_delay),
                                                                   ("pan", m.room.raw_pan_bus)]:
            p.add_(0.3 * torch.randn(p.shape, generator=g).to(DEV))
        m.physics.coupled.raw_glide.fill_(2.0)
        m.physics.coupled.kr_db.fill_(-20.0)
        m.physics.coupled.lm_gain_db.fill_(-10.0)
    n = 6000
    perf = perf_with_history(m, n, NOTES, sustain=lambda t: ((t > 0.2) & (t < 0.5)).float())
    fus, ref = render_both(m, perf, n, monkeypatch)
    assert_close(fus, ref, loose=("physics.raw_long_ratio", "physics.coupled.raw_lm1"))


def test_longitudinal_kernel_matches_reference(monkeypatch):
    """The longitudinal force (cuda_ext.Longitudinal) against NeuralPhysicalPiano._longitudinal's PyTorch recurrences:
    output and the gradients of the sums, the modes' frequencies, decay and gains, from a non-zero state."""
    m = NeuralPhysicalPiano(small_cfg(string_model="coupled"))
    sr = m.cfg.sample_rate
    g = torch.Generator().manual_seed(1)
    P, J, L = 37, 4, 3000
    Y = torch.randn(P, 2, L, generator=g) * torch.linspace(1, 0.1, L)
    lf = 150 + 900 * torch.rand(P, J, generator=g)
    la = 3 + 10 * torch.rand(P, generator=g)
    lg = torch.rand(P, J, generator=g)
    scale = torch.rand(P, generator=g) + 0.5
    hp = torch.randn(P, 2, generator=g)
    lm = torch.complex(torch.randn(P, J, generator=g), torch.randn(P, J, generator=g))
    w = torch.randn(P, L, generator=g)
    res = []
    for fused in (True, False):
        ins = [t.clone().to(DEV).requires_grad_(True) for t in (Y, lf, la, lg)]
        state = {"hp": hp.to(DEV), "lm": lm.to(DEV)}
        long = {"lm_f": ins[1], "lm_alpha": ins[2], "lm_gain": ins[3], "scale": scale.to(DEV)}
        with monkeypatch.context() as mp:
            if not fused:
                mp.setattr(cuda_ext, "get", lambda: None)
            F = m._longitudinal(ins[0], long, torch.arange(P, device=DEV), state)
        (F * w.to(DEV)).sum().backward()
        res.append((F.detach(), [t.grad for t in ins], state))
    (Ff, gf, sf), (Fr, gr, sr_) = res
    assert float((Ff - Fr).abs().max() / Fr.abs().max()) < 1e-4
    for a, b in zip(gf, gr):
        assert float((a - b).norm() / b.norm()) < 2e-3
    assert torch.allclose(sf["hp"], sr_["hp"], rtol=1e-3, atol=1e-4)
    assert torch.allclose(sf["lm"], sr_["lm"], rtol=1e-3, atol=1e-3)


def test_fused_block_rendering_matches_single_pass():
    m = NeuralPhysicalPiano(small_cfg(activity_db=400.0)).to(DEV).eval()
    n = 8000
    perf = perf_with_history(m, n, NOTES, sustain=lambda t: (t > 0.1).float())
    with torch.no_grad():
        one = m(perf, n, residual=False, generator=torch.Generator(device=DEV).manual_seed(1))
        blocks = m(perf, n, residual=False, block_seconds=0.3, generator=torch.Generator(device=DEV).manual_seed(1))
    for k in ("strings", "symp"):
        err = (one[k] - blocks[k]).abs().max() / one[k].abs().max()
        assert err < 1e-4, k


def _resonators64(drive, es, alpha, adamp, freq, gin, sr):
    """The resonators' definition, sequentially in float64."""
    dec = (alpha.double()[..., None] + es.double()[:, :, None, :] * adamp.double()[..., None]).clamp(max=1000.0) / sr
    a = torch.exp(torch.complex(-dec, (2 * math.pi / sr * freq.double())[..., None].expand_as(dec)))
    x = drive.double()[:, :, None, :] * gin.double()[..., None]
    z = torch.zeros(alpha.shape, dtype=torch.complex128, device=alpha.device)
    out = []
    for t in range(x.shape[-1]):
        z = a[..., t] * z + x[..., t]
        out.append(z.real.sum((1, 2)))
    return torch.stack(out, -1)


def test_resonators_match_the_float64_recurrence_and_its_gradients():
    g = torch.Generator(device=DEV).manual_seed(0)
    B, K, S, L, sr = 2, 3, 4, 1500, 6000.0
    drive = torch.randn(B, K, L, device=DEV, generator=g)
    es = (torch.rand(B, K, L, device=DEV, generator=g) > 0.995).float().cumsum(-1).clamp(max=1)
    alpha = torch.rand(B, K, S, device=DEV, generator=g) * 3 + 0.3
    adamp = torch.rand(B, K, S, device=DEV, generator=g) * 300  # some steps exceed the 1000/s clamp
    freq = torch.rand(B, K, S, device=DEV, generator=g) * 2000 + 30
    gin = torch.rand(B, K, S, device=DEV, generator=g) * 0.01
    ins = [t.clone().requires_grad_(True) for t in (drive, es, alpha, adamp, freq, gin)]
    ref = [t.clone().requires_grad_(True) for t in (drive, es, alpha, adamp, freq, gin)]
    w = torch.randn(B, L, device=DEV, generator=g)
    y, _, _ = cuda_ext.Resonators.apply(*ins, torch.zeros_like(alpha), torch.zeros_like(alpha), sr, 1000.0)
    y64 = _resonators64(*ref, sr)
    assert float((y - y64).abs().max() / y64.abs().max()) < 1e-4
    (y * w).sum().backward()
    (y64 * w.double()).sum().backward()
    for a, b in zip(ins, ref):
        assert float((a.grad.double() - b.grad).norm() / b.grad.norm()) < 5e-3
    # and the chunked PyTorch reference agrees too
    dec = (alpha[..., None] + es[:, :, None, :] * adamp[..., None]).clamp(max=1000.0) / sr
    la = torch.complex(-dec, (2 * math.pi / sr * freq)[..., None].expand_as(dec))
    x = drive[:, :, None, :] * gin[..., None]
    yr = linear_recurrence(torch.complex(x, torch.zeros_like(x)), la, None, 256)[0].real.sum((1, 2))
    assert float((y.detach() - yr).abs().max() / yr.abs().max()) < 1e-4
