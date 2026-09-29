import math

import torch

from pianonn import NeuralPhysicalPiano
from pianonn.physics import PianoPhysics

from .conftest import make_perf, small_cfg


def energy(x, sr, t0, t1):
    return x[..., int(t0 * sr):int(t1 * sr)].pow(2).mean().item()


def test_forward_backward(model):
    n = 8000
    perf = make_perf(model, n, [(60, 0.1, 0.5, 80), (43, -0.3, 0.8, 100), (100, 0.2, 0.3, 40)],
                     sustain=lambda t: (t > 0.6).float())
    out = model(perf, n)
    assert out["audio"].shape == (1, n)
    for v in out.values():
        assert torch.isfinite(v).all()
    out["audio"].pow(2).mean().backward()
    for name in ["physics.raw_log_B", "physics.raw_cents", "physics.gain_db", "physics.raw_log_tc", "room.body",
                 "room.raw_log_t60", "symp.log_gain", "noise.knock", "noise.pedal", "context.head.2.weight",
                 "physics.pedal_theta", "physics.pedal_log_power", "physics.raw_order", "physics.raw_order_vel"]:
        g = dict(model.named_parameters())[name].grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, name


def test_fundamental_frequency_matches_physics(model):
    """A4's fundamental must land near 440 Hz, the 2nd partial slightly sharp of 880 (inharmonicity)."""
    cfg = small_cfg(use_noise=False, use_sympathetic=False, use_room=False)
    m = NeuralPhysicalPiano(cfg)
    n = 8000
    audio = m(make_perf(m, n, [(69, 0.0, 1.0, 80)]), n)["audio"][0]
    spec = torch.fft.rfft(audio * torch.hann_window(n)).abs()
    hz = torch.arange(len(spec)) * cfg.sample_rate / n

    def peak(lo, hi):
        band = (hz > lo) & (hz < hi)
        return hz[band][spec[band].argmax()].item()

    f1, f2 = peak(300, 600), peak(700, 1100)
    assert abs(f1 - 440) < 3
    assert f2 >= 2 * f1


def test_damper_and_sustain_pedal(model):
    cfg = small_cfg(use_noise=False)
    m = NeuralPhysicalPiano(cfg)
    n, sr = 12000, cfg.sample_rate
    notes = [(48, 0.0, 0.3, 90)]
    dry = m(make_perf(m, n, notes, sustain=0.0), n)["audio"]
    ped = m(make_perf(m, n, notes, sustain=1.0), n)["audio"]
    held = m(make_perf(m, n, [(48, 0.0, 2.0, 90)]), n)["audio"]
    # released without pedal: damped. with pedal: rings like a held key.
    assert energy(dry, sr, 1.0, 1.4) < 0.01 * energy(ped, sr, 1.0, 1.4)
    assert math.isclose(energy(ped, sr, 1.0, 1.4), energy(held, sr, 1.0, 1.4), rel_tol=0.2)


def test_top_keys_have_no_dampers():
    phys = PianoPhysics(small_cfg())
    assert phys.damper_strength[0] == 1 and phys.damper_strength[-1] == 0


def test_sympathetic_resonance_needs_pedal():
    cfg = small_cfg(use_noise=False)
    m = NeuralPhysicalPiano(cfg)
    n = 8000
    notes = [(48, 0.0, 0.8, 110)]
    off = m(make_perf(m, n, notes, sustain=0.0), n)["symp"]
    on = m(make_perf(m, n, notes, sustain=1.0), n)["symp"]
    assert on.pow(2).mean() > 3 * off.pow(2).mean()


def test_block_rendering_matches_single_pass():
    cfg = small_cfg()
    m = NeuralPhysicalPiano(cfg)
    n = 16000
    perf = make_perf(m, n, [(60, 0.1, 0.5, 80), (36, 0.3, 1.5, 100), (72, 1.2, 1.4, 60)],
                     sustain=lambda t: (t > 0.8).float())
    perf["sostenuto"] = perf["sustain"].flip(-1)
    with torch.no_grad():
        a = m(perf, n, generator=torch.Generator().manual_seed(0))["audio"]
        b = m(perf, n, block_seconds=0.37, generator=torch.Generator().manual_seed(0))["audio"]
    assert torch.allclose(a, b, atol=1e-5)


def test_soft_pedal_darkens(model):
    cfg = small_cfg(use_noise=False, use_sympathetic=False, use_room=False)  # the physics, not the hall
    m = NeuralPhysicalPiano(cfg)
    n = 4000
    notes = [(60, 0.0, 0.5, 100)]

    def centroid(x):
        s = torch.fft.rfft(x[0]).abs()
        f = torch.arange(len(s), dtype=torch.float32)
        return (s * f).sum() / s.sum()

    assert centroid(m(make_perf(m, n, notes, soft=1.0), n)["audio"]) < centroid(m(make_perf(m, n, notes), n)["audio"])


def test_sostenuto_latch_holds_only_keys_down_at_press():
    kd = torch.zeros(1, 88, 12)
    kd[0, 5, 2:4] = 1  # held when the pedal goes down at frame 3
    kd[0, 6, 5:7] = 1  # pressed after the pedal
    sost = torch.zeros(1, 12)
    sost[0, 3:9] = 1
    latch = NeuralPhysicalPiano.sostenuto_latch(kd, sost)
    assert latch[0, 5, 3:9].eq(1).all() and latch[0, 5, :3].eq(0).all() and latch[0, 5, 9:].eq(0).all()
    assert latch[0, 6].eq(0).all()


def test_sostenuto_sustains_released_note():
    cfg = small_cfg(use_noise=False, use_sympathetic=False, use_room=False)
    m = NeuralPhysicalPiano(cfg)
    n, sr = 12000, cfg.sample_rate
    notes = [(48, 0.0, 0.3, 90)]
    caught = make_perf(m, n, notes)
    caught["sostenuto"] = (torch.arange(caught["sostenuto"].shape[-1]) * cfg.hop / sr > 0.2).float()[None]
    missed = make_perf(m, n, notes)
    missed["sostenuto"] = (torch.arange(missed["sostenuto"].shape[-1]) * cfg.hop / sr > 0.5).float()[None]
    e_caught = energy(m(caught, n)["audio"], sr, 1.0, 1.4)
    assert energy(m(missed, n)["audio"], sr, 1.0, 1.4) < 0.01 * e_caught


def test_per_key_strings_sum_to_total():
    cfg = small_cfg(use_noise=False)
    m = NeuralPhysicalPiano(cfg)
    n = 4000
    perf = make_perf(m, n, [(60, 0.1, 0.3, 80), (60, 0.2, 0.4, 60), (40, 0.0, 0.4, 90)])
    with torch.no_grad():
        ki = perf["pitch"] - 21
        modes = m.physics.modes(ki, perf["velocity"] / 127, torch.zeros_like(perf["onset"]), perf["condition"])
        C = torch.zeros(1, 88, m.n_frames(n))
        total, per_key = m.render_strings(modes, ki, perf["onset"], m._ring_end(modes, perf["onset"]), C, 0, n,
                                          per_key=True)
    assert torch.allclose(per_key.sum(1), total, atol=1e-6)
    assert per_key[0, 39].abs().sum() > 0 and per_key[0, 19].abs().sum() > 0 and per_key[0, 50].abs().sum() == 0


def test_decay_profile_matches_measured_piano():
    """Analytic decay profile (power-summed decay partials, beats averaged) within 6 dB of the profile measured on
    the Iowa Steinway B (docs/calibration_iowa.md) in the bass and mid-range, and a real two-stage decay. Rendered
    notes are checked the same way the recordings were measured by pianonn.diagnostics."""
    import math

    from pianonn.calibration import PROFILE_TIMES, decay_partials

    phys = PianoPhysics(small_cfg(sample_rate=24000, n_partials=12))
    targets = {3: (-3, -6, -10, -18, -27, -33), 15: (-5, -10, -12, -22, -24, -35), 27: (-5, -10, -22, -24, -32, -45),
               39: (-12, -21, -25, -31, -45, -55)}
    for k, target in targets.items():
        m = phys.modes(torch.tensor([[k]]), torch.tensor([[0.5]]), torch.zeros(1, 1), torch.tensor([0]))
        idx = [n - 1 for n in decay_partials(k + 21)]
        a2, al = m["amp"][0, 0, idx].double() ** 2, m["alpha"][0, 0, idx].double()
        t = torch.tensor((0.05,) + PROFILE_TIMES, dtype=torch.float64)
        P = (a2[..., None] * torch.exp(-2 * al[..., None] * t)).sum((0, 1))
        prof = 10 * torch.log10(P[1:] / P[0])
        assert (prof - torch.tensor(target, dtype=torch.float64)).abs().max() <= 6, (k, prof)
        assert (m["alpha"][0, 0, 0, 0] / m["alpha"][0, 0, 0, 1]).item() > 3  # prompt much faster than aftersound


def test_physics_gradients_finite_all_keys_and_velocities():
    """Every physics parameter gets a finite gradient for every key at every velocity (full-rate config,
    96 partials: partials far above Nyquist must not overflow before they are masked)."""
    from pianonn import PianoConfig

    phys = PianoPhysics(PianoConfig())
    keys = torch.arange(88).repeat_interleave(22)
    vels = torch.linspace(1, 127, 22).repeat(88)
    ki, u = keys[None], (vels / 127)[None]
    for soft in (0.0, 1.0):
        phys.zero_grad()
        m = phys.modes(ki, u, torch.full(ki.shape, soft), torch.tensor([0]))
        loss = sum(v.float().pow(2).mean() for v in m.values() if v.is_floating_point())
        loss.backward()
        grads = {name: p.grad for name, p in phys.named_parameters() if p.grad is not None}  # pedals act in synth
        assert {"raw_order", "raw_order_vel", "raw_log_tc", "raw_log_B", "gain_db"} <= set(grads)
        for name, g in grads.items():
            assert torch.isfinite(g).all(), (name, soft)
