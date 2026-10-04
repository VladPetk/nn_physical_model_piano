"""The coupled strings in the model (docs/physics_revamp.md): rendering, the image per key, the longitudinal path, the
glide, and starting from a mode model's weights."""

import math

import torch

from pianonn.config import PianoConfig
from pianonn.fit_init import distill_coupled
from pianonn.oscbank import bus_bank
from pianonn.render import load_weights
from pianonn.synth import NeuralPhysicalPiano
from tests.conftest import make_perf, small_cfg


def coupled_cfg(**kw):
    return PianoConfig(**{**small_cfg().to_dict(), "string_model": "coupled", **kw})


def test_renders_and_every_new_parameter_learns():
    m = NeuralPhysicalPiano(coupled_cfg(strike_evenness=0.1))
    n = m.cfg.sample_rate
    perf = make_perf(m, n, [(40, 0.05, 0.6, 90), (64, 0.1, 0.9, 70), (64, 0.5, 0.9, 50), (90, 0.2, 0.5, 100)], soft=0.5)
    out = m(perf, n)
    assert out["audio"].shape == (1, 2, n) and torch.isfinite(out["audio"]).all()
    out["audio"].pow(2).sum().backward()
    for k, p in m.named_parameters():
        if k.startswith(("physics.coupled.", "room.body_h", "room.raw_pan_bus", "room.raw_delay")):
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, k


def test_mode_model_unchanged_by_default():
    """The default config builds no coupled parameters (every earlier checkpoint loads as it was)."""
    m = NeuralPhysicalPiano(PianoConfig(**small_cfg().to_dict()))
    assert m.physics.coupled is None and not hasattr(m.room, "body_h")


def test_starts_from_a_mode_model():
    """Loading a mode model's weights into the coupled model copies its pan and body into both buses and its unison
    detuning into the strings' pitch offsets."""
    old = NeuralPhysicalPiano(PianoConfig(**small_cfg().to_dict()))
    with torch.no_grad():
        old.room.raw_pan.normal_()
        old.room.body.mul_(1.7)
    new = load_weights(NeuralPhysicalPiano(coupled_cfg()), old.state_dict(), log=lambda *_: None)
    assert torch.allclose(new.room.body_h, old.room.body) and torch.allclose(new.room.raw_pan_bus[:, 1], old.room.raw_pan)
    c = new.physics.coupled.string_cents()
    assert c.abs().max() > 0.05 and torch.allclose(c.sum(-1), torch.zeros(88), atol=1e-5)


def test_delay_between_the_microphones_is_a_phase():
    """A 1 ms delay of the vertical bus between the microphones shifts each mode's phase by 2 pi f x 1 ms: the cross
    correlation of the two channels peaks at 1 ms."""
    m = NeuralPhysicalPiano(coupled_cfg(use_noise=False, use_impulse=False, use_sympathetic=False, use_room=False))
    with torch.no_grad():
        m.room.raw_delay[3, 0, 60 - 21] = math.atanh(1.0 / 1.5)  # 1 ms
        m.room.raw_pan_bus.zero_()
        m.physics.coupled.raw_sh.fill_(-50.0)  # no horizontal motion
    sr = m.cfg.sample_rate
    n = sr // 2
    v = m(make_perf(m, n, [(60, 0.0, 0.5, 80)]), n)["strings_bus"][0, 0]  # [ch, T]
    a, b = v[0, 400:], v[1, 400:]
    lags = torch.arange(-20, 21)
    xc = torch.stack([(a[20:-20] * b[20 + k: len(b) - 20 + k]).sum() for k in lags.tolist()])
    assert lags[xc.argmax()].item() == -round(1e-3 * sr)  # channel 0 +0.5 ms, channel 1 -0.5 ms: 0 lags 1 by 1 ms


def test_longitudinal_force_is_square_law():
    """The longitudinal sums follow the strings' physical amplitude (a per-note level offset, as the strike's): +6 dB on
    the note doubles them, so the force (their square) gains 12 dB, while its scale (re the key's mf level, a
    calibration) stays; a change of the key's calibrated level moves the force as it moves everything else."""
    m = NeuralPhysicalPiano(coupled_cfg(n_partials=12))
    ki, u, z, cond = torch.tensor([[15]]), torch.tensor([[0.7]]), torch.zeros(1, 1), torch.tensor([3])
    with torch.no_grad():
        a = m.physics.modes(ki, u, z, cond)
        b = m.physics.modes(ki, u, z, cond, ctx={"gain_db": torch.full((1, 1), 6.0206)})
    ratio = b["bus_amp"][..., 2:, :].abs().sum() / a["bus_amp"][..., 2:, :].abs().sum()
    assert abs(ratio.item() - 2.0) < 1e-3
    assert torch.allclose(a["long"]["scale"], b["long"]["scale"])
    Y = torch.randn(3, 2, 500)
    long = {"lm_f": torch.full((3, 4), 300.0), "lm_alpha": torch.full((3,), 6.0), "lm_gain": torch.ones(3, 4),
            "scale": torch.ones(3)}
    st = lambda: {"hp": torch.zeros(3, 2), "lm": torch.zeros(3, 4, dtype=torch.complex64)}  # noqa: E731
    idx = torch.arange(3)
    F1, F2 = m._longitudinal(Y, long, idx, st()), m._longitudinal(2 * Y, long, idx, st())
    assert torch.allclose(F2, 4 * F1, atol=1e-4)


def test_glide_phase_matches_its_integral():
    """One oscillator gliding from f (1 + e) down to f: its phase leads the plain one by 2 pi f e (1 - exp(-beta t))
    / beta."""
    sr, L = 8000, 4000
    f, e, beta = torch.tensor([[200.0]]), 0.01, 3.0
    args = (torch.tensor([[0.0]]), torch.tensor([[[[1.0, 0.0]]]]), torch.zeros(1, 1), torch.tensor([1e-4]),
            torch.zeros(1, L), torch.zeros(1), torch.zeros(1), torch.zeros(1, dtype=torch.float64),
            torch.full((1, 1), math.inf), 0.0, sr)
    y0 = bus_bank(f, *args)[0, 0]
    y1 = bus_bank(f, *args, glide=(torch.tensor([e]), torch.tensor([beta])))[0, 0]
    t = torch.arange(L) / sr
    lead = 2 * math.pi * 200.0 * e * (1 - torch.exp(-beta * t)) / beta
    ref = torch.sin(2 * math.pi * 200.0 * (t - 0.5e-4) + lead)
    assert (y1[10:] - ref[10:]).abs().max() < 1e-3 and (y0 - y1).abs().max() > 0.5


def test_distillation_fits_a_mode_models_envelopes_and_phantom_level():
    """fit_init.distill_coupled: the coupled strings started from a mode model come closer to its partials' energy
    envelopes, and the longitudinal level is set to the phantoms' energy (less below_db) wherever both sound."""
    from pianonn.fit_init import calibrate_longitudinal, longitudinal_energy, phantom_energy

    teacher = NeuralPhysicalPiano(PianoConfig(**small_cfg(n_partials=8).to_dict()))
    with torch.no_grad():  # a mode model with its own aftersound levels and unison spread
        teacher.physics.raw_after.normal_(0, 0.5)
        teacher.physics.raw_unison.normal_(0, 1.0)
    model = load_weights(NeuralPhysicalPiano(coupled_cfg(n_partials=8)), teacher.state_dict(), log=lambda *_: None)
    first, last = distill_coupled(model, teacher, 3, steps=40, lr=0.05, log=None)
    assert last < 0.7 * first
    cal = calibrate_longitudinal(model, teacher, 3, below_db=0.0, log=None)
    e_l, e_p = longitudinal_energy(model, 0.6, 3), phantom_energy(teacher, 0.6, 3)
    ok = (e_l > 0) & (e_p > 0) & (cal.abs() < 59)
    assert ok.sum() > 20
    assert (10 * torch.log10(e_l[ok] / e_p[ok])).abs().max() < 0.1
