import math

import torch

from pianonn import NeuralPhysicalPiano, PianoConfig
from pianonn.physics import PianoPhysics

from .conftest import make_perf, small_cfg


def energy(x, sr, t0, t1):
    return x[..., int(t0 * sr):int(t1 * sr)].pow(2).mean().item()


def test_forward_backward():
    m = NeuralPhysicalPiano(small_cfg(use_floor=True))
    n = 8000
    # 60 is struck again at 0.4 s while still sounding (re-strike), 43 starts before the window
    perf = make_perf(m, n, [(60, 0.1, 0.5, 80), (43, -0.3, 0.8, 100), (100, 0.2, 0.3, 40), (60, 0.4, 0.9, 90)],
                     sustain=lambda t: (t > 0.6).float())
    out = m(perf, n)
    assert out["audio"].shape == (1, 2, n)
    for v in out.values():
        if torch.is_tensor(v):
            assert torch.isfinite(v).all()
    out["audio"].pow(2).mean().backward()
    for name in ["physics.raw_log_B", "physics.raw_cents", "physics.gain_db", "physics.raw_log_tc", "room.body",
                 "room.raw_log_t60", "symp.log_gain", "noise.knock", "noise.pedal", "context.head.2.weight",
                 "context.frame_head.2.weight", "noise.att", "physics.raw_pedal_theta", "physics.raw_pedal_power",
                 "physics.raw_order", "physics.raw_order_vel", "physics.raw_restrike", "physics.raw_phantom_db",
                 "physics.raw_impulse_db", "physics.raw_bridge_g", "physics.cond_damper_delay", "physics.color",
                 "room.raw_floor", "room.raw_pan", "room.mic_gain_db", "noise.raw_knock_tau", "physics.cond_vel_curve",
                 "physics.cond_vel_map"]:
        g = dict(m.named_parameters())[name].grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, name


def test_fundamental_frequency_matches_physics(model):
    """A4's fundamental must land near 440 Hz, the 2nd partial slightly sharp of 880 (inharmonicity)."""
    cfg = small_cfg(use_noise=False, use_sympathetic=False, use_room=False)
    m = NeuralPhysicalPiano(cfg)
    n = 8000
    audio = m(make_perf(m, n, [(69, 0.0, 1.0, 80)]), n)["audio"][0, 0]
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
    perf = make_perf(m, n, [(60, 0.1, 0.5, 80), (36, 0.3, 1.5, 100), (72, 1.2, 1.4, 60), (36, 1.0, 1.8, 70)],
                     sustain=lambda t: (t > 0.8).float())
    perf["sostenuto"] = perf["sustain"].flip(-1)
    with torch.no_grad():
        # a trained-looking residual: non-zero band gains, noise and per-note corrections
        for head in (m.context.head, m.context.frame_head):
            head[-1].bias.normal_(0, 0.5)
        a = m(perf, n, generator=torch.Generator().manual_seed(0))["audio"]
        b = m(perf, n, block_seconds=0.37, generator=torch.Generator().manual_seed(0))["audio"]
    assert torch.allclose(a, b, atol=1e-5), (a - b).abs().max()


def test_soft_pedal_darkens(model):
    cfg = small_cfg(use_noise=False, use_sympathetic=False, use_room=False, use_impulse=False)  # the physics, not the hall
    m = NeuralPhysicalPiano(cfg)
    n = 4000
    notes = [(60, 0.0, 0.5, 100)]

    def centroid(x):
        s = torch.fft.rfft(x[0, 0]).abs()
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
        rs = m.next_strikes(ki, perf["onset"], perf["mask"])
        pan = m.room.pan_gains(ki, perf["condition"])
        total, per_key = m.render_strings(modes, ki, perf["onset"], perf["mask"], C, 0, rs, pan, 0, n, per_key=True)
    assert torch.allclose(per_key.sum(1), total[:, 0], atol=1e-6)  # pan is 0 dB before training
    assert per_key[0, 39].abs().sum() > 0 and per_key[0, 19].abs().sum() > 0 and per_key[0, 50].abs().sum() == 0


def test_decay_profile_matches_measured_piano():
    """Analytic decay profile (power-summed decay partials, beats averaged) within 6 dB of the profile measured on
    the Iowa Steinway B (docs/calibration_iowa.md) in the bass and mid-range, and a real two-stage decay. Rendered
    notes are checked the same way the recordings were measured by pianonn.diagnostics."""
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
        assert {"raw_order", "raw_order_vel", "raw_log_tc", "raw_log_B", "gain_db", "raw_phantom_db", "raw_restrike",
                "raw_impulse_db", "raw_bridge_g", "raw_long_ratio"} <= set(grads)
        for name, g in grads.items():
            assert torch.isfinite(g).all(), (name, soft)


def test_restrike_does_not_accumulate():
    """A fast tremolo on one key under the pedal: without re-strike damping every blow adds to a string that still
    rings and the level climbs (+4 to +5 dB after eight strikes in the full-rate model); a real string loses part of
    its vibration to the hammer, so the level stays near that of a single strike."""
    cfg = small_cfg(use_noise=False, use_sympathetic=False, use_room=False, use_impulse=False)
    m = NeuralPhysicalPiano(cfg)
    sr, gap = cfg.sample_rate, 0.125
    n = int((8 * gap + 0.3) * sr)
    notes = [(60, gap * i, gap * i + 0.8 * gap, 90) for i in range(8)]
    with torch.no_grad():
        many = m(make_perf(m, n, notes, sustain=1.0), n)["audio"]
        one = m(make_perf(m, n, notes[:1], sustain=1.0), n)["audio"]
        m.physics.prior_restrike.fill_(0.0)  # re-strike damping off
        none = m(make_perf(m, n, notes, sustain=1.0), n)["audio"]
    first = energy(one, sr, 0.005, gap)
    db = lambda x: 10 * math.log10(energy(x, sr, 7 * gap + 0.005, 8 * gap) / first)
    assert db(many) < 2.0
    assert db(none) > db(many) + 1.5


def test_history_damps_note_released_before_window():
    """A note struck and released 3 s before the window is silent at the window start when the control
    curves cover that history (review 3, section 4.5), while a note still held rings."""
    cfg = small_cfg(use_noise=False, use_sympathetic=False, use_room=False, use_impulse=False)
    m = NeuralPhysicalPiano(cfg)
    n, sr = 4000, cfg.sample_rate
    H = int(4.0 * sr / cfg.hop)

    def render(notes):
        F = H + m.n_frames(n)
        t = torch.tensor(notes, dtype=torch.float32)[None]
        perf = {"pitch": t[..., 0].long(), "onset": t[..., 1], "offset": t[..., 2], "velocity": t[..., 3],
                "mask": torch.ones(1, len(notes), dtype=torch.bool), "condition": torch.tensor([3]),
                "sustain": torch.zeros(1, F), "soft": torch.zeros(1, F), "sostenuto": torch.zeros(1, F),
                "hist_frames": torch.tensor(H)}
        with torch.no_grad():
            return m(perf, n)["audio"]

    stale = energy(render([(48, -3.0, -2.5, 90)]), sr, 0.0, 0.2)
    held = energy(render([(48, -3.0, 2.0, 90)]), sr, 0.0, 0.2)
    assert stale < 1e-6 * held


def test_phase_is_exact_minutes_into_a_piece():
    """The oscillator bank forms the cycle count in float64: a note ten minutes in renders like one at t = 0
    (review 3, F11: float32 absolute time raised the inter-partial floor to -90 dB)."""
    from pianonn.oscbank import osc_bank

    m = PianoPhysics(small_cfg(sample_rate=24000, n_partials=32))
    modes = m.modes(torch.tensor([[75]]), torch.tensor([[0.7]]), torch.zeros(1, 1), torch.tensor([0]), phantoms=False)
    args = (modes["freq"].flatten(2)[0], modes["alpha"].flatten(2)[0], modes["amp"].flatten(2)[0],
            modes["alpha_damp"][..., None].expand(*modes["alpha_damp"].shape, 3).flatten(2)[0], modes["tc"][0])
    L = 4000
    ys = []
    for t0 in (0.0, 600.0):
        ys.append(osc_bank(*args, torch.zeros(1, L), torch.zeros(1), torch.zeros(1), torch.tensor([t0]),
                           torch.full((1, 2), math.inf), t0, 24000)[0])
    err = (ys[0] - ys[1]).pow(2).mean() / ys[0].pow(2).mean()
    assert 10 * math.log10(err.item() + 1e-30) < -100


def test_osc_bank_gradients_match_autograd():
    """The oscillator bank's hand-written backward agrees with finite differences."""
    from pianonn.oscbank import osc_bank

    torch.manual_seed(0)
    P, Q, L, sr = 3, 5, 64, 2000.0
    d = dict(dtype=torch.float64)
    inputs = ((100 + 300 * torch.rand(P, Q, **d)), (0.5 + 3 * torch.rand(P, Q, **d)), torch.randn(P, Q, **d),
              (1 + 5 * torch.rand(P, Q, **d)), (0.004 + 0.01 * torch.rand(P, **d)),
              torch.cumsum(torch.rand(P, L, **d), -1) / sr, 0.001 * torch.rand(P, **d), 0.5 + torch.rand(P, **d))
    inputs = tuple(x.requires_grad_() for x in inputs)
    onset = torch.tensor([0.0, 0.005, -0.01], **d)
    rs_delay = torch.tensor([[0.012, math.inf], [math.inf, math.inf], [0.02, 0.025]], **d)
    assert torch.autograd.gradcheck(lambda *a: osc_bank(*a, onset, rs_delay, 0.0, sr), inputs, eps=1e-7, atol=1e-5,
                                    rtol=1e-4)


def test_osc_bank_group_gains():
    """With group gain curves (the physics-aware residual): gains of 1 give the plain bank's output, and the
    hand-written backward, the curves' gradient included, agrees with finite differences."""
    from pianonn.oscbank import group_weights, osc_bank

    torch.manual_seed(0)
    P, Q, L, sr, Gn = 3, 5, 64, 2000.0, 3
    d = dict(dtype=torch.float64)
    inputs = ((100 + 300 * torch.rand(P, Q, **d)), (0.5 + 3 * torch.rand(P, Q, **d)), torch.randn(P, Q, **d),
              (1 + 5 * torch.rand(P, Q, **d)), (0.004 + 0.01 * torch.rand(P, **d)),
              torch.cumsum(torch.rand(P, L, **d), -1) / sr, 0.001 * torch.rand(P, **d), 0.5 + torch.rand(P, **d))
    onset = torch.tensor([0.0, 0.005, -0.01], **d)
    rs_delay = torch.tensor([[0.012, math.inf], [math.inf, math.inf], [0.02, 0.025]], **d)
    W = group_weights(inputs[0], [125.0, 250.0, 500.0])
    assert torch.allclose(W.sum(1), torch.ones(P, Q, **d))
    plain = osc_bank(*inputs, onset, rs_delay, 0.0, sr)
    ones = osc_bank(*inputs, onset, rs_delay, 0.0, sr, W, torch.ones(P, Gn, L, **d))
    assert torch.allclose(plain, ones, atol=1e-12)
    m = (0.5 + torch.rand(P, Gn, L, **d))
    inputs = tuple(x.requires_grad_() for x in inputs + (m,))
    assert torch.autograd.gradcheck(lambda *a: osc_bank(*a[:-1], onset, rs_delay, 0.0, sr, W, a[-1]), inputs,
                                    eps=1e-7, atol=1e-5, rtol=1e-4)


def test_osc_bank_control_rate_gains():
    """Log gains at a control rate, interpolated inside the bank: the same output as the sample-rate gains they
    interpolate to, and the backward through the interpolation agrees with finite differences."""
    from pianonn.dsp import frames_to_samples
    from pianonn.oscbank import group_weights, osc_bank

    torch.manual_seed(0)
    P, Q, L, sr, Gn, hop, start = 3, 5, 64, 2000.0, 3, 16, 5
    d = dict(dtype=torch.float64)
    inputs = ((100 + 300 * torch.rand(P, Q, **d)), (0.5 + 3 * torch.rand(P, Q, **d)), torch.randn(P, Q, **d),
              (1 + 5 * torch.rand(P, Q, **d)), (0.004 + 0.01 * torch.rand(P, **d)),
              torch.cumsum(torch.rand(P, L, **d), -1) / sr, 0.001 * torch.rand(P, **d), 0.5 + torch.rand(P, **d))
    onset = torch.tensor([0.0, 0.005, -0.01], **d)
    rs_delay = torch.tensor([[0.012, math.inf], [math.inf, math.inf], [0.02, 0.025]], **d)
    W = group_weights(inputs[0], [125.0, 250.0, 500.0])
    c = 0.3 * torch.randn(P, Gn, (start + L) // hop + 2, **d)
    a = osc_bank(*inputs, onset, rs_delay, 0.0, sr, W, torch.exp(frames_to_samples(c, start, L, hop)))
    b = osc_bank(*inputs, onset, rs_delay, 0.0, sr, W, c, start, hop)
    assert torch.allclose(a, b, atol=1e-12)
    inputs = tuple(x.requires_grad_() for x in inputs + (c,))
    assert torch.autograd.gradcheck(lambda *a: osc_bank(*a[:-1], onset, rs_delay, 0.0, sr, W, a[-1], start, hop),
                                    inputs, eps=1e-7, atol=1e-5, rtol=1e-4)


def test_damper_delay_is_learnable():
    """The release edge is fractional in frames, so the loss has a gradient w.r.t. the per-condition damper delay."""
    cfg = small_cfg(use_noise=False, use_sympathetic=False)
    m = NeuralPhysicalPiano(cfg)
    n = 6000
    out = m(make_perf(m, n, [(48, 0.0, 0.3, 90)]), n, residual=False)
    out["audio"][..., 2400:].pow(2).sum().backward()
    assert m.physics.cond_damper_delay.grad[3] > 0  # a later damper leaves more sound after the release


def test_bridge_end_comb_keeps_levels_and_delays_the_first_string_pulse():
    """(-1)^(n+1) on the strike comb: the bridge end's force. Same |amplitude| per partial (phantoms unchanged), but
    on A1 (T = 18.2 ms, x0 ~ 0.12) the first transverse pulse arrives after (1 - x0) T/2 instead of x0 T/2."""
    cfg = small_cfg(use_noise=False, use_sympathetic=False, use_room=False, use_impulse=False, use_context=False)
    m = NeuralPhysicalPiano(cfg)
    ki, u = torch.tensor([[12, 40, 60]]), torch.tensor([[0.8, 0.5, 0.3]])
    mode, audio = {}, {}
    n, sr = 4000, cfg.sample_rate
    perf = make_perf(m, n, [(33, 0.1, 0.4, 100)])
    for flag in (False, True):
        m.cfg.bridge_end_comb = m.physics.cfg.bridge_end_comb = flag
        mode[flag] = m.physics.modes(ki, u, torch.zeros_like(u), torch.tensor([3]))
        audio[flag] = m(perf, n)["audio"][0, 0]
    assert torch.allclose(mode[False]["amp"].abs(), mode[True]["amp"].abs())
    assert not torch.allclose(mode[False]["amp"], mode[True]["amp"])
    assert torch.allclose(mode[False]["ph_amp"], mode[True]["ph_amp"])
    T = 1 / 55.0
    start = int((audio[False].abs() > 1e-6 * audio[False].abs().max()).float().argmax())
    win = slice(start, start + int(0.6 * T * sr))
    t_peak = {f: audio[f][win].abs().argmax().item() / sr for f in audio}
    assert t_peak[False] < 0.2 * T and t_peak[True] > 0.35 * T


def test_body_q_cap_damps_a_narrow_mode_and_keeps_band_energy():
    """An undamped 1158 Hz mode in the body (as the round-2 bodies grew): with the Q capped at 50 its ringing at
    0.15-0.3 s falls by >= 30 dB, and the energy per octave moves by under 25 % of the total (each band keeps its
    energy; the crossovers overlap, so some shows in the neighbours); off (the default), nothing changes."""
    from pianonn.room import octave_masks

    m = NeuralPhysicalPiano(small_cfg(body_q_max=50.0, body_seconds=0.3))
    sr, L = m.cfg.sample_rate, m.room.body.shape[-1]
    t = torch.arange(L) / sr
    h = torch.zeros(L)
    h[int(0.003 * sr)] = 1.0
    h = h + 0.05 * torch.sin(2 * math.pi * 1158 * t) * (t > 0.003)
    out = m.room.limit_q(h[None, None])[0, 0]

    def tail_db(x):
        seg = x[int(0.15 * sr): int(0.3 * sr)]
        P = torch.fft.rfft(seg * torch.hann_window(len(seg))).abs() ** 2
        f = torch.fft.rfftfreq(len(seg), 1 / sr)
        return 10 * torch.log10(P[(f > 1100) & (f < 1220)].sum()).item()

    assert tail_db(out) < tail_db(h) - 30
    masks = octave_masks(2 * L, sr, [125, 250, 500, 1000, 2000, 4000]).float()
    e = lambda x: ((torch.fft.rfft(x, 2 * L).abs() ** 2) * masks).sum(-1)
    assert (e(out) - e(h)).abs().sum() < 0.25 * e(h).sum()
    off = NeuralPhysicalPiano(small_cfg())
    cond = torch.tensor([3])
    ir = off.room(cond)
    with torch.no_grad():
        off.room.body.data[cond] = off.room.body.data[cond] * 1.0
    assert torch.equal(ir, off.room(cond))


def test_velocity_map_is_the_prior_at_zero_monotone_and_anchored_at_mf():
    from pianonn.physics import hammer_velocity

    ph = PianoPhysics(small_cfg())
    u = torch.arange(1, 128).float()[None] / 127
    c = torch.tensor([3])
    assert torch.allclose(ph.hammer_speed(u, c), hammer_velocity(u), rtol=1e-5)
    with torch.no_grad():
        ph.cond_vel_map[3] = 5.0 * torch.randn(ph.cond_vel_map.shape[1])
    v = ph.hammer_speed(u, c)[0]
    assert (v[1:] > v[:-1]).all() and abs(float(v[63]) - 2.8) < 1e-4
    # faster hammers above mf shorten the contact of loud notes only
    with torch.no_grad():
        ph.cond_vel_map.zero_()
        ki, uu, soft = torch.tensor([[39, 39]]), torch.tensor([[40 / 127, 110 / 127]]), torch.zeros(1, 2)
        tc0 = ph.contact_time(ki, uu, soft, c)
        ph.cond_vel_map[3, 4:] = 0.5
        tc1 = ph.contact_time(ki, uu, soft, c)
    assert abs(float(tc1[0, 0] / tc0[0, 0]) - 1) < 1e-5 and float(tc1[0, 1] / tc0[0, 1]) < 0.9


def test_strike_variation_is_off_by_default_and_draws_the_configured_spread():
    """Per-strike variation (config ``strike_*``): off unless set; per-register knots are held flat beyond the ends;
    a 3 dB level sd and a 5 ms onset sd come out of the renderer as such, around the unvaried note."""
    from pianonn.synth import STRIKE_DIMS, strike_sd_table

    assert not NeuralPhysicalPiano(small_cfg()).strike_on
    tab = strike_sd_table(PianoConfig(strike_log_fc="0.1/0.2/0.3/0.4/0.5", strike_onset_ms=[1, 1, 1, 1, 5]))
    fc, on = tab[STRIKE_DIMS.index("log_fc")], tab[STRIKE_DIMS.index("onset_ms")]
    assert abs(fc[21 - 21] - 0.1) < 1e-6 and abs(fc[108 - 21] - 0.5) < 1e-6 and abs(fc[59 - 21] - 0.2 - 0.1 * 6 / 12.5) < 1e-5
    assert on[60 - 21] == 1 and on[100 - 21] == 5 and tab[STRIKE_DIMS.index("level_db")].abs().sum() == 0

    base = dict(use_noise=False, use_sympathetic=False, use_room=False, use_impulse=False, use_context=False)
    n, sr = 6000, 8000
    notes = [(60, 0.1, 0.6, 80)]
    ref = NeuralPhysicalPiano(small_cfg(**base))
    m = NeuralPhysicalPiano(small_cfg(**base, strike_level_db=3.0, strike_onset_ms=5.0))
    m.load_state_dict(ref.state_dict())
    perf = make_perf(m, n, notes)

    def first(x):
        return int((x.abs() > 1e-4 * x.abs().max()).float().argmax()) / sr

    y0 = ref(perf, n)["audio"][0, 0]
    e0, t0 = 10 * math.log10(energy(y0, sr, 0.0, 0.75)), first(y0)
    levels, shifts = [], []
    for seed in range(60):
        y = m(perf, n, generator=torch.Generator().manual_seed(seed))["audio"][0, 0]
        levels.append(10 * math.log10(energy(y, sr, 0.0, 0.75)) - e0)
        shifts.append(1000 * (first(y) - t0))
    lv, sh = torch.tensor(levels), torch.tensor(shifts)
    assert 2.3 < lv.std() < 3.7 and lv.median().abs() < 1.0
    assert 3.8 < sh.std() < 6.2 and sh.median().abs() < 1.5


def test_strike_brightness_and_decay_keep_the_note_level():
    """The per-strike brightness, decay and decay tilt change the spectrum and its evolution but keep the note's energy
    over its first 0.3 s (the level is a dimension of its own)."""
    base = dict(use_noise=False, use_sympathetic=False, use_room=False, use_impulse=False, use_context=False)
    n, sr = 6000, 8000
    ref = NeuralPhysicalPiano(small_cfg(**base))
    perf = make_perf(ref, n, [(60, 0.1, 0.6, 80)])

    def level_and_centroid(y):
        seg = y[int(0.1 * sr): int(0.4 * sr)]
        S = torch.fft.rfft(seg).abs() ** 2
        f = torch.fft.rfftfreq(len(seg), 1 / sr)
        return 10 * math.log10(seg.pow(2).mean().item()), float((S * f).sum() / S.sum())

    e0, c0 = level_and_centroid(ref(perf, n)["audio"][0, 0])
    for kw in (dict(strike_log_fc=0.2), dict(strike_log_decay=0.3), dict(strike_decay_tilt=0.3)):
        m = NeuralPhysicalPiano(small_cfg(**base, **kw))
        m.load_state_dict(ref.state_dict())
        lv, ce = [], []
        for seed in range(12):
            e, c = level_and_centroid(m(perf, n, generator=torch.Generator().manual_seed(seed))["audio"][0, 0])
            lv.append(e - e0), ce.append(1200 * math.log2(c / c0))
        assert torch.tensor(lv).abs().max() < 0.3, (kw, lv)
        if "strike_log_decay" not in kw:
            assert torch.tensor(ce).std() > 50, (kw, ce)


def _aware_setup():
    m = NeuralPhysicalPiano(small_cfg(residual_kind="aware", use_sympathetic=False, res_dim=32, res_heads=4))
    n = 8000
    perf = make_perf(m, n, [(60, 0.1, 0.5, 80), (43, -0.3, 0.8, 100), (100, 0.2, 0.3, 40), (60, 0.4, 0.9, 90),
                            (30, -2.0, -1.0, 60)], sustain=lambda t: (t > 0.6).float())
    return m, perf, n


def test_aware_residual_starts_neutral_and_scales_exactly():
    """The physics-aware residual starts at zero output (the strings as without it); a gain of 2 on every octave group
    (the groups sum to one per partial) doubles the strings exactly."""
    m, perf, n = _aware_setup()
    with torch.no_grad():
        base = m(perf, n, residual=False)["strings"]
        out = m(perf, n, residual=True)
        assert out["curves"].shape[:3] == (1, 5, m.cfg.res_groups)
        assert torch.allclose(out["strings"], base, atol=1e-6)
        bound = m.cfg.res_curve_db * math.log(10) / 20
        m.context.curve_head[-1].bias.fill_(bound * math.atanh(math.log(2) / bound))
        doubled = m(perf, n, residual=True)["strings"]
    assert torch.allclose(doubled, 2 * base, atol=1e-5)


def test_aware_residual_gradients():
    """Gradients reach every output head of the aware residual (through the oscillator bank for the curves)."""
    m, perf, n = _aware_setup()
    out = m(perf, n, residual=True)
    assert torch.isfinite(out["audio"]).all()
    out["audio"].pow(2).mean().backward()
    for name in ["context.curve_head.1.weight", "context.onset_head.2.weight", "context.mix_head.2.weight"]:
        g = dict(m.named_parameters())[name].grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, name


def _wide_setup():
    m = NeuralPhysicalPiano(small_cfg(residual_kind="aware", use_sympathetic=False, res_dim=32, res_heads=4,
                                      res_curve_partials=8, res_note_noise=4, res_latent=3))
    n = 8000
    perf = make_perf(m, n, [(60, 0.3, 0.8, 80), (43, 0.5, 0.9, 100)])
    return m, perf, n


def test_wide_residual_starts_neutral():
    """The wider outputs start at the physics: the strings exactly, the per-note noise 60 dB under the notes."""
    m, perf, n = _wide_setup()
    with torch.no_grad():
        base = m(perf, n, residual=False)["strings"]
        out = m(perf, n, residual=True, generator=torch.Generator().manual_seed(0))
    assert out["curves"].shape[:3] == (1, 2, 8)
    assert torch.allclose(out["strings"], base, atol=1e-6)
    ratio = out["noise_res"].pow(2).mean() / base.pow(2).mean()
    assert 1e-8 < ratio < 1e-5, ratio


def test_note_noise_follows_its_note():
    """The per-note noise is silent before its note's onset (and the control step after it) and comes in after."""
    m, perf, n = _wide_setup()
    with torch.no_grad():
        m.context.noise_head[-1].bias.fill_(4.0)
        pw = m(perf, n, residual=True)["frame_ctx"]["note_noise"][0].sum(0)  # [frames]
    t = torch.arange(pw.shape[-1]) * m.cfg.hop / m.cfg.sample_rate
    assert pw[t < 0.3].abs().max() == 0
    assert (pw[(t > 0.36) & (t < 0.8)] > 0).all()


def test_latent_varies_the_residual_by_seed():
    """The random inputs: the same seed gives the same residual, another seed another; gradients reach the new heads."""
    m, perf, n = _wide_setup()
    torch.nn.init.normal_(m.context.curve_head[-1].weight, std=0.5)
    curves = lambda seed: m(perf, n, residual=True, generator=torch.Generator().manual_seed(seed))["curves"]
    with torch.no_grad():
        a, b, c = curves(0), curves(0), curves(1)
    assert torch.equal(a, b) and not torch.allclose(a, c)
    out = m(perf, n, residual=True)
    out["audio"].pow(2).mean().backward()
    for name in ["context.curve_head.1.weight", "context.noise_head.1.weight"]:
        g = dict(m.named_parameters())[name].grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, name


def _parts_model(**kw):
    cfg = small_cfg(**{"sample_rate": 24000, "hop": 120, "noise_bands": 32, "use_sympathetic": False,
                       "attack_model": "parts", **kw})
    m = NeuralPhysicalPiano(cfg)
    m.noise.init_parts()
    return m


def test_attack_parts_start_with_the_knock_noise_energy():
    """With the thump and the precursor silenced, the parts' knock carries the old knock noise's energy in every band
    below its cap: init_parts keeps the decays, and each kernel keeps the step-onset exponential's energy."""
    m = _parts_model()
    nb = m.noise
    with torch.no_grad():
        nb.thump_log_gain.fill_(-30.0)
        nb.thump_spec.fill_(-30.0)
        nb.prec_spec.fill_(-30.0)
    sr, n = 24000, 24000
    ki = torch.tensor([[15, 39, 63, 80]])
    u = torch.tensor([[0.3, 0.6, 0.8, 0.5]])
    onset = torch.tensor([[0.05, 0.2, 0.31, 0.4]])
    w = torch.ones_like(u)
    with torch.no_grad():
        m.cfg.attack_model = "noise"  # the old knock noise, from the same parameters
        old, _ = nb.event_power(ki, u, onset, onset + 9.0, w, w * 0, {}, 0, n, False)
        m.cfg.attack_model = "parts"
        new = nb.attack_power(ki, u, onset, onset, w, {}, 0, n)
    below = nb.centers <= m.cfg.knock_max_hz
    ratio = new.sum((0, 2))[below] / old.sum((0, 2))[below]
    assert ((ratio - 1).abs() < 0.02).all(), ratio
    assert new[..., : int(0.05 * sr)].abs().max() < 1e-6 * new.max()  # nothing before the first strike (float32 FFT round-off, ~-73 dB)


def test_attack_parts_leave_the_top_octave_empty():
    """Smooth onsets and capped spectra: the parts put nothing near the top (the old step onset spread every band's
    onset over all frequencies)."""
    m = _parts_model(use_room=False, use_impulse=False)
    nb = m.noise
    sr, n = 24000, 12000
    ki, u, onset = torch.tensor([[39, 63]]), torch.tensor([[0.9, 0.9]]), torch.tensor([[0.1, 0.25]])
    w = torch.ones_like(u)
    white = torch.randn(1, n)
    with torch.no_grad():
        p = nb.attack_power(ki, u, onset, onset + 0.003, w, {}, 0, n)
        y = (nb.band_split(white, 0, n) * p.clamp(min=0).sqrt()).sum(1)[0]
    S = torch.fft.rfft(y).abs() ** 2
    f = torch.fft.rfftfreq(n, 1 / sr)
    top = S[f > 2 * m.cfg.precursor_max_hz].sum() / S[(f > 500) & (f < 2000)].sum()
    assert 10 * math.log10(float(top)) < -40


def test_attack_parts_block_rendering_and_gradients():
    m = _parts_model(sample_rate=8000, hop=40, noise_bands=16)
    with torch.no_grad():
        m.noise.thump_raw_tau.fill_(3.0)  # ~0.4 s: the thump's kernel spans the block boundaries
    n = 16000
    perf = make_perf(m, n, [(60, 0.1, 0.5, 80), (36, 0.3, 1.5, 100), (90, 1.2, 1.4, 60), (36, -0.2, 1.8, 70)])
    with torch.no_grad():
        a = m(perf, n, residual=False, generator=torch.Generator().manual_seed(0))["audio"]
        b = m(perf, n, residual=False, block_seconds=0.37, generator=torch.Generator().manual_seed(0))["audio"]
    assert torch.allclose(a, b, atol=1e-5), (a - b).abs().max()
    m(perf, n, residual=False)["audio"].pow(2).sum().backward()
    for name in ("knock", "knock_raw_tau", "knock_raw_rise", "thump_spec", "thump_reg", "thump_vel", "thump_raw_tau",
                 "prec_spec", "prec_reg", "prec_vel", "prec_raw_tau"):
        g = getattr(m.noise, name).grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, name


def test_cheap_sympathetic_bank_matches_the_full_rate_one_and_renders_in_blocks():
    """The decimated bank (resonators at sr / 4) gives the full-rate bank's response for the same keys and partials
    (energy within 1 dB), and blocks of a multiple of 4 samples render as one pass."""
    kw = dict(use_noise=False, use_impulse=False, use_room=False, symp_lo_midi=21, symp_hi_midi=59, symp_max_hz=800.0,
              symp_partials=6)
    full = NeuralPhysicalPiano(small_cfg(**kw))
    cheap = NeuralPhysicalPiano(small_cfg(**kw, symp_decimate=4))
    cheap.load_state_dict(full.state_dict())
    n = 16000
    perf = make_perf(full, n, [(48, 0.1, 0.5, 90), (55, 0.3, 1.5, 100), (60, 1.0, 1.4, 70)], sustain=1.0)
    with torch.no_grad():
        a = full(perf, n, residual=False)["symp"]
        b = cheap(perf, n, residual=False)["symp"]
        c = cheap(perf, n, residual=False, block_seconds=0.37)["symp"]
    assert torch.allclose(b, c, atol=1e-4 * float(b.abs().max())), (b - c).abs().max()  # float32 round-off: ~1e-5
    ea, eb = float(a[..., 4000:].pow(2).sum()), float(b[..., 4000:].pow(2).sum())
    assert abs(10 * math.log10(eb / ea)) < 1.0, (ea, eb)


def test_strike_per_partial_keeps_each_partials_energy_and_the_mean_aftersound():
    """Per strike and partial (config ``strike_partial_decay``, ``strike_after``): off by default; the per-note draws
    stay what they were; each partial's prompt energy over 0.3 s is kept under its decay draw; the aftersound's random
    part keeps its mean power and flips its sign in some partials."""
    from pianonn.physics import STRIKE_LEVEL_SECONDS

    assert not NeuralPhysicalPiano(small_cfg()).strike_partial_on
    base = dict(use_noise=False, use_sympathetic=False, use_room=False, use_impulse=False, use_context=False)
    ref = NeuralPhysicalPiano(small_cfg(**base, strike_level_db=2.0))
    m = NeuralPhysicalPiano(small_cfg(**base, strike_level_db=2.0, strike_partial_decay=0.5, strike_after=2.0))
    m.load_state_dict(ref.state_dict())
    ki = torch.tensor([[60 - 21, 40 - 21, 75 - 21]])
    v0 = ref.strike_offsets(ki, torch.Generator().manual_seed(3))
    v1 = m.strike_offsets(ki, torch.Generator().manual_seed(3))
    assert torch.equal(v0["level_db"], v1["level_db"]) and "after" not in v0
    assert v1["partial_decay"].shape == (1, 3, m.cfg.n_partials) and v1["after"].shape == (1, 3, m.cfg.n_partials, m.cfg.n_modes - 1)

    u = torch.full(ki.shape, 0.6)
    zero, cond = torch.zeros_like(u), torch.zeros(1, dtype=torch.long)
    plain = {f"strike_{d}": zero for d in ("log_fc", "log_decay", "decay_tilt")}
    with torch.no_grad():
        a = m.physics.modes(ki, u, zero, cond, dict(plain), phantoms=False)
        b = m.physics.modes(ki, u, zero, cond, {**plain, "strike_partial_decay": v1["partial_decay"]}, phantoms=False)
    e = lambda md: md["amp"][..., 0] ** 2 * -torch.expm1(-2 * md["alpha"][..., 0] * STRIKE_LEVEL_SECONDS) / (2 * md["alpha"][..., 0])
    ok = a["amp"][..., 0].abs() > 0
    assert torch.allclose(e(b)[ok], e(a)[ok], rtol=1e-4)
    assert (b["alpha"][..., 0][ok] / a["alpha"][..., 0][ok]).log().std() > 0.3

    draws = [m.strike_offsets(ki, torch.Generator().manual_seed(s))["after"] for s in range(40)]
    g = torch.stack(draws)[..., :40, :]  # partials 1-40
    assert abs(float((g ** 2).mean()) - 1) < 0.1 and 0.2 < float((g < 0).float().mean()) < 0.45


def test_body_ring_up_spreads_a_partials_onset_and_keeps_its_level():
    """Config ``body_ring_ms``: a decaying noise kernel of amplitude time constant tau in one band. A sinusoid switched on
    through it builds up as a random walk: over many frequencies its power grows as 1 - exp(-2 t / tau); a band without a kernel passes it at once; ``ring_gain`` is each frequency's steady power gain, the mean
    over the channels, so dividing it out keeps the partials' levels."""
    from pianonn.dsp import fft_convolve
    from pianonn.room import Q_BANDS, Room

    sr = 24000
    taus = ["0"] * len(Q_BANDS)
    taus[Q_BANDS.index(2000)] = "6"
    r0 = Room(PianoConfig(sample_rate=sr, body_seconds=0.3, hall_seconds=0.5))
    r1 = Room(PianoConfig(sample_rate=sr, body_seconds=0.3, hall_seconds=0.5, body_ring_ms="/".join(taus)))
    assert not r0.ring_on and r1.ring_on
    k = r1.ring_kernel[0].double()
    t = torch.arange(len(k), dtype=torch.float64) / sr

    def build_up(freqs):
        """Mean over ``freqs`` of |the kernel's transform up to t|^2, re its value at the end; and the steady gains."""
        z = torch.cumsum(k[None] * torch.exp(-2j * math.pi * freqs[:, None] * t[None]), -1).abs() ** 2
        return z.mean(0) / z[:, -1].mean(), z[:, -1]

    def settled(p):  # ms until the build-up stays above 90 % (the delta's share at 2 kHz cancels within ~1 ms)
        return 1000 * (int(torch.nonzero(p < 0.9).max()) + 1) / sr if bool((p < 0.9).any()) else 0.0

    p, steady = build_up(torch.linspace(1600, 2500, 300, dtype=torch.float64))
    assert 0.6 * 1.15 * 6 < settled(p) < 1.8 * 1.15 * 6, settled(p)  # 1 - exp(-2 t / tau) reaches 90 % at 1.15 tau
    p_low, _ = build_up(torch.linspace(300, 450, 50, dtype=torch.float64))
    assert settled(p_low) < 2.0  # 400 Hz: no kernel there, a delta (the 2 kHz band split leaks for ~1.5 ms)
    f = torch.tensor([1937.0, 2210.0, 2486.0])
    expect = torch.tensor([[_kernel_power(r1.ring_kernel[c], float(x), sr) for x in f] for c in range(2)]).mean(0)
    assert torch.allclose(r1.ring_gain(f), expect.float(), rtol=0.05)
    assert steady.std() / steady.mean() > 0.5  # resonant fine structure: what ring_gain divides out


def _kernel_power(k, f, sr):
    n = torch.arange(len(k), dtype=torch.float64) / sr
    z = (k.double() * torch.exp(-2j * math.pi * f * n)).sum()
    return float(z.abs() ** 2)



def test_partial_groups_follow_the_partial_number():
    """scripts/fit_passages.py's extended outputs: one gain curve per partial. Transverse oscillators are stored
    partial-major (q // modes), phantoms join the partial nearest in frequency, the last group takes the rest."""
    from pianonn.oscbank import partial_group_weights

    M, P = 3, 6
    f_part = 100.0 * torch.arange(1, P + 1, dtype=torch.float32)[None]  # partials at 100 ... 600 Hz
    freq = f_part.repeat_interleave(M, -1)  # [1, P*M]
    W = partial_group_weights(0, freq, f_part, M, 4)
    assert W.shape == (1, 4, P * M) and torch.all(W.sum(1) == 1)
    assert W[0, :, 0:3].argmax(0).tolist() == [0, 0, 0] and W[0, :, 6:9].argmax(0).tolist() == [2, 2, 2]
    assert W[0, 3, 9:].sum() == 9  # partials 4-6 share the last group
    ph = torch.tensor([[205.0, 395.0]])  # phantoms near partials 2 and 4
    Wp = partial_group_weights(1, ph, f_part, M, 8)
    assert Wp[0].argmax(0).tolist() == [1, 3]


def test_shared_board_aligns_the_direct_sound_and_the_hall_has_the_diffuse_coherence():
    """``shared_board``: both microphones' direct paths are the left body through real band gains, so their cross
    spectrum has (nearly) no phase (and +6 dB in one band shows as +6 dB there); ``hall_mic_d``: the two hall carriers' coherence
    follows sinc(2 f d / c) (near 1 at 100 Hz for d = 0.3 m, under 0.1 at 2-4 kHz), and d = 0 keeps the earlier
    independent tails exactly."""
    from pianonn.room import SPEED_OF_SOUND, band_carriers

    m = NeuralPhysicalPiano(small_cfg(shared_board=True, hall_mic_d=0.3))
    R, sr = m.room, m.cfg.sample_rate
    cond = torch.tensor([3])
    with torch.no_grad():
        R.mic_eq_db[3, 0, 1, 5] = 6.0  # the right microphone's 1 kHz band
        d = R.direct(R.body[cond], cond)[0]
    L = d.shape[-1]
    X = torch.fft.rfft(d.double(), 2 * L)
    f = torch.fft.rfftfreq(2 * L, 1 / sr)
    band = (f > 300) & (f < 4000) & (X[0].abs() > 1e-3 * X[0].abs().max())
    gain_db = 20 * torch.log10(X[1].abs() / X[0].abs()) - (R.mic_gain_db[3, 1] - R.mic_gain_db[3, 0])
    ph = torch.angle(X[1][band] * X[0][band].conj()).abs()  # the band gains are cut to the body's length: ~0.01 rad
    assert ph.median() < 0.01 and torch.quantile(ph, 0.95) < 0.05
    at = lambda hz: float(gain_db[int(torch.argmin((f - hz).abs()))])  # noqa: E731
    assert abs(at(1000) - 6.0) < 0.2 and abs(at(250)) < 0.2
    c = R.carriers.double().sum(1)  # [ch, L]: the band carriers sum to the noise
    Y = torch.fft.rfft(c, dim=-1)
    fy = torch.fft.rfftfreq(c.shape[-1], 1 / sr)

    def coh(lo, hi):
        b = (fy >= lo) & (fy < hi)
        cr = (Y[0, b] * Y[1, b].conj()).sum()
        return float(cr.abs() ** 2 / ((Y[0, b].abs() ** 2).sum() * (Y[1, b].abs() ** 2).sum()))

    want = float(torch.special.sinc(torch.tensor(2 * 100 * 0.3 / SPEED_OF_SOUND))) ** 2
    assert abs(coh(90, 110) - want) < 0.05 and coh(2000, 4000) < 0.1
    off = NeuralPhysicalPiano(small_cfg())
    sr_o, hs = off.cfg.sample_rate, off.cfg.hall_seconds
    assert torch.equal(off.room.carriers[1], band_carriers(sr_o, hs, seed=2))
