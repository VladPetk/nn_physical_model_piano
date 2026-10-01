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
