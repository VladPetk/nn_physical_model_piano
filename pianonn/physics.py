"""Per-key physical parameters of the piano.

Every quantity is ``prior(key) + bounded(learned offset)``. The priors and their
sources are specified in ``docs/physical_parameters.md``; they make the untrained
model sound like a concert grand and, more importantly, start the
frequency-related parameters inside the basin where spectral-loss gradients are
informative. Learned offsets are bounded so that gradient descent tunes the
instrument rather than inventing a different one.

String model (per key k, partial n, coupled mode m), evaluated in closed form:

    f_n      = n f0 sqrt(1 + B n^2)                        stiff-string inharmonicity
    f_nm     = f_n 2^(unison_cents_m / 1200)               unison mistuning -> beats
    alpha_n  = b1 + b3 f_r^2 (f_n / f_r)^p                  aftersound: internal + air losses (f_r = 1 kHz, p ~ 1.1)
    mode 0 (in-phase, "prompt sound") adds the bridge loss of all strings of the
    unison, alpha_n + (R - 1) b1 g_c(f_n), with g_c the bridge conductance over frequency
    of recording condition c (1/6-octave knots); modes 1.. are the weakly radiating
    aftersound (Weinreich's coupled strings in mode space).
    a_n      = gain(v) F(f_n; T_c(v)) sin(pi n x0/L) C_c(k, f_n)
                                                           bridge force: hammer pulse envelope x strike comb
                                                           x the bridge colouration where key k meets the board
    damper:  extra decay alpha_d,n whenever the key is up and neither pedal holds the damper off.
    phantom partials: tension modulation couples pairs of transverse partials into components at
    f_j + f_k (here j = k and k = j + 1), amplitude ~ a_j a_k (so they grow with velocity squared),
    emphasised around the longitudinal resonance at ~15 f_1.
    re-strike: a hammer striking a string that still rings takes out part of its vibration, so every
    earlier sounding instance of the key loses ``restrike`` nats when the key is struck again.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from .config import PianoConfig
from .dsp import bounded

N_KEYS = 88
LOWEST_MIDI = 21
LN10 = math.log(10)


def key_curve(points, n=N_KEYS):
    """Piecewise-linear curve over key index from ``[(key, value), ...]``."""
    xs = torch.tensor([p[0] for p in points], dtype=torch.float32)
    ys = torch.tensor([p[1] for p in points], dtype=torch.float32)
    k = torch.arange(n, dtype=torch.float32)
    idx = torch.searchsorted(xs, k, right=True).clamp(1, len(xs) - 1)
    x0, x1, y0, y1 = xs[idx - 1], xs[idx], ys[idx - 1], ys[idx]
    w = ((k - x0) / (x1 - x0)).clamp(0, 1)
    return y0 + w * (y1 - y0)


def _p(*shape, value=0.0):
    return nn.Parameter(torch.full(shape, float(value)))


def _atanh_init(value, r):
    """Raw value such that ``bounded(raw, r) == value``."""
    return r * math.atanh(value / r)


HAMMER_ORDER_VEL = 0.225  # Hall, Five Lectures, Fig. 15 (C4 slopes pp/mf/ff at vel 30/64/110)
# Two corners: roll-off order q at mf per key above 0.59/T_c, and a second, steeper corner at f T_c = HAMMER_X2
# (order HAMMER_Q2 more). Both fitted to the attack spectra of every partial below 10 kHz of the Iowa Steinway B
# (scripts/fit_spectra.py, mf + ff, through the body): a single power law cannot be flat enough in the middle
# and steep enough on top, and left the bass/tenor 15-25 dB too loud above 3.5 kHz (a harpsichord-like zing).
# The order rises from bass to treble, where the contact outlasts half the string period (Askenfelt & Jansson,
# Five Lectures, Fig. 8). The top two knots have 1-4 partials below 10 kHz and keep the earlier values.
HAMMER_ORDER_MF = [(0, 0.99), (15, 1.2), (27, 1.15), (39, 1.42), (51, 2.24), (63, 4.26), (75, 5.3), (87, 5.3)]
HAMMER_X2 = 3.98
HAMMER_Q2 = 2.58
# aftersound loss exponent over frequency (b3 f^2 has p = 2): fitted to the per-partial decays of the Iowa
# Steinway B (scripts/fit_decays.py), close to linear. With p = 2 the high partials of bass notes died 15-50 dB too early.
DECAY_EXPONENT = 1.092
# prompt-stage (bridge) loss vs frequency, normalised to a geometric mean of 1 over the knots: the unison's
# in-phase motion drains through the bridge at a rate ~ f0 Z0 G(f_n) (Weinreich), so one conductance curve
# times a per-key factor. Fitted with the decay priors; the 14 kHz knot (no data) repeats the 7 kHz one.
BRIDGE_G = [(55, 0.717), (110, 0.806), (220, 0.884), (440, 0.915), (880, 0.789), (1760, 0.755), (3520, 1.009),
            (7040, 1.53), (14080, 1.53)]
# The learned conductance has knots every 1/6 octave from A0, one curve per recording condition: a real
# soundboard's admittance varies on a resonance scale (review 3, item 3), and every MAESTRO year is a
# different instrument. The Iowa curve above is the prior.
BRIDGE_KNOTS_PER_OCT = 6
BRIDGE_N_KNOTS = 9 * BRIDGE_KNOTS_PER_OCT + 1  # 27.5 Hz .. 14.08 kHz
# Per-key bridge colouration: the soundboard's mobility where each key's strings meet the bridge (review 3,
# item 6). A smooth table over (key, log f), one per condition: 23 key knots (every 4 keys) x 36 frequency
# knots (every 1/4 octave from A0). Bounded to +-8 dB and kept zero-mean over keys, so it holds only what the
# condition's body FIR (the same for every key) cannot.
COLOR_KEY_STEP = 4
COLOR_N_KEYS = N_KEYS // COLOR_KEY_STEP + 1
COLOR_KNOTS_PER_OCT = 4
COLOR_N_F = 36
COLOR_BOUND = 0.92  # nats, 8 dB
PARTIAL_GAIN_BOUND = 0.92  # nats, 8 dB (was 3 nats: enough to replace the hammer model outright)
F_REF = 27.5  # A0: origin of the log-frequency knot grids
# per-condition correction of the level-vs-velocity law, dB at u = 0, 0.2, ..., 1 (linear in between, +-12 dB):
# a Disklavier's velocity map is not a straight line in dB, and one slope per year left soft playing ~2 dB quiet
VEL_KNOTS = 6
# per-condition velocity map: MIDI velocity -> hammer speed, piecewise log-linear over segments of 16 velocity steps
# (knots at 0, 16, ..., 128), each segment's slope the prior's (0.023 / step) x exp(+-1), anchored at 64 -> 2.8 m/s.
# Monotone by construction. Zero: ``hammer_velocity`` (a guess at the Disklavier's map, docs/tone_measures.md 10.3)
VEL_MAP_STEP = 16
VEL_MAP_SEGMENTS = 8
# per-strike brightness and decay offsets keep the note's energy over this long (N1's window): at equal key and velocity
# the piano's level barely follows its brightness or early decay (rank correlations -0.16 and +0.24, docs 12.7)
STRIKE_LEVEL_SECONDS = 0.3


def hammer_velocity(u):
    """MIDI velocity / 127 -> hammer speed in m/s: the prior of ``PianoPhysics.hammer_speed``.

    Exponential, anchored so that the dynamic labels of Askenfelt & Jansson (Five Lectures,
    Fig. 6) fall on the usual MIDI velocities: mf = 64 -> 2.8 m/s, f ~ 90 -> 5.1, ff ~ 115 -> 9.0,
    p ~ 40 -> 1.6. So "mf" means the same thing everywhere (contact time, roll-off order,
    key-bottom timing). The level-vs-velocity law is learned separately.
    """
    return 2.8 * torch.exp(0.023 * (127.0 * u - 64.0))


def hammer_spectrum(f, tc, order=1.0, x2=HAMMER_X2, order2=HAMMER_Q2):
    """Smooth magnitude envelope of the hammer force pulse of duration ``tc``.

    -3 dB at 0.59/tc (the corner of a half-sine pulse), then -6*order dB/oct, steepening by
    another -6*order2 dB/oct above f tc = x2 (the pulse is smooth, so its spectrum falls ever
    faster; fitted to recorded attack spectra, see HAMMER_X2). The felt is
    nonlinear (F ~ x^p, p = 2-3.5), so a harder blow drives it into its stiff range and the
    pulse gets sharper: the roll-off order falls with hammer speed (Hall, Five Lectures,
    Fig. 15: -18 / -15 / -11 dB/oct at pp / mf / ff for C4). The ideal half-sine's nulls are
    replaced by the smooth envelope: measured pulses are skewed and their nulls are filled.
    """
    # log domain: (f tc / 0.59)^(2 order) overflows float32 for partials far above the corner
    # (inf -> NaN gradient w.r.t. the order), even though those partials are masked out later
    x = torch.log(f * tc)
    return torch.exp(-0.5 * (F.softplus(2 * order * (x - math.log(0.59))) + F.softplus(2 * order2 * (x - torch.log(torch.as_tensor(x2, dtype=x.dtype))))))


def interp_knots(table, pos):
    """Linear interpolation of ``table[B, n]`` at fractional knot positions ``pos[B, ...]`` (clamped to the ends)."""
    n = table.shape[-1]
    pos = pos.clamp(0, n - 1)
    i0 = pos.detach().floor().long().clamp(max=n - 2)
    w = pos - i0
    flat = pos.reshape(pos.shape[0], -1)
    i0f, wf = i0.reshape(flat.shape), w.reshape(flat.shape)
    y = table.gather(1, i0f) * (1 - wf) + table.gather(1, i0f + 1) * wf
    return y.reshape(pos.shape)


def log_f_bumps(coef, f, lo=40.0, hi=10000.0):
    """Smooth log-frequency correction: ``coef[..., J]`` raised-cosine bumps spread evenly in log f over
    [lo, hi], evaluated at ``f[..., P]`` (broadcast over leading dims). Bumps sum to 1 inside the range."""
    J = coef.shape[-1]
    pos = (torch.log2(f.clamp(min=1.0) / lo) / math.log2(hi / lo) * (J - 1)).clamp(0, J - 1)
    d = pos[..., None] - torch.arange(J, device=f.device, dtype=f.dtype)
    w = torch.where(d.abs() < 1, 0.5 + 0.5 * torch.cos(math.pi * d), torch.zeros_like(d))
    return (w * coef[..., None, :]).sum(-1)


class PianoPhysics(nn.Module):
    def __init__(self, cfg: PianoConfig):
        super().__init__()
        self.cfg = cfg
        P, M, C = cfg.n_partials, cfg.n_modes, cfg.n_conditions
        k = torch.arange(N_KEYS)

        # --- priors (buffers, not trained); see docs/physical_parameters.md ---
        # Rigaud, David & Daudet (DAFx 2011) two-asymptote fit, m = MIDI pitch (as used by DDSP-Piano)
        m = (k + LOWEST_MIDI).float()
        # times the ratio measured on the Iowa Steinway B (median over +-2 keys): its bass strings are ~2x less
        # inharmonic than the curve. A concert grand's longer bass strings are lower still, so this is the
        # right direction for MAESTRO's pianos too (and the learned offset spans x/4.5).
        rigaud = torch.exp(0.0926 * m - 13.64) + torch.exp(-0.0847 * m - 5.82)
        self.register_buffer("prior_log_B", torch.log(rigaud * key_curve(
            [(0, 0.48), (9, 0.46), (15, 0.7), (21, 0.88), (39, 0.9), (45, 1.0), (87, 1.0)])))
        # stretch re A4, measured on the Iowa Steinway B (C8 extrapolated)
        self.register_buffer("prior_cents", key_curve(
            [(0, -16), (3, -15), (15, -4), (27, 0), (39, -1), (48, 0), (51, 0), (63, 6), (75, 14), (87, 25)]))
        # decay priors: b1, b3 (at 1 kHz), R and the aftersound amplitude, fitted together with DECAY_EXPONENT and
        # BRIDGE_G to the level of every partial below 10 kHz at 0.5-16 s in the Iowa Steinway B recordings
        # (scripts/fit_decays.py, docs/calibration_iowa.md)
        self.register_buffer("prior_log_b1", torch.log(key_curve(
            [(0, 0.129), (12, 0.143), (20, 0.174), (27, 0.193), (34, 0.204), (39, 0.211), (45, 0.228), (51, 0.261),
             (57, 0.318), (63, 0.405), (75, 0.562), (87, 0.687)])))
        self.register_buffer("prior_log_b3", torch.log(key_curve(
            [(0, 2.33e-7), (12, 1.51e-7), (20, 1.29e-7), (27, 1.34e-7), (34, 1.43e-7), (39, 1.4e-7), (45, 1.35e-7),
             (51, 1.23e-7), (57, 9.43e-8), (63, 5.79e-8), (75, 3.18e-8), (87, 1.65e-8)])))
        self.register_buffer("n_strings", torch.where(k < 8, 1, torch.where(k < 26, 2, 3)))
        # R ~ 10-15 (at g = 1): the in-phase mode of the unison loads the bridge much harder than one string
        # (Weinreich), so the prompt stage is fast and the aftersound much quieter
        self.register_buffer("prior_prompt_ratio", key_curve(
            [(0, 9.66), (12, 11.0), (20, 12.1), (27, 13.0), (34, 14.0), (39, 15.3), (45, 15.6), (51, 13.5), (57, 10.9),
             (63, 9.43), (75, 10.1), (87, 12.3)]))
        self.register_buffer("prior_log_after", torch.log(key_curve(
            [(0, 0.317), (12, 0.226), (20, 0.144), (27, 0.0954), (34, 0.0669), (39, 0.0487), (45, 0.0368), (51, 0.0279),
             (57, 0.0194), (63, 0.0106), (75, 0.00473), (87, 0.00225)])))
        # the Iowa conductance curve resampled on the 1/6-octave grid (flat beyond its end knots)
        knot_f = F_REF * 2 ** (torch.arange(BRIDGE_N_KNOTS, dtype=torch.float32) / BRIDGE_KNOTS_PER_OCT)
        gx = torch.log(torch.tensor([p[0] for p in BRIDGE_G], dtype=torch.float32))
        gy = torch.log(torch.tensor([p[1] for p in BRIDGE_G], dtype=torch.float32))
        lx = torch.log(knot_f)
        idx = torch.searchsorted(gx, lx).clamp(1, len(gx) - 1)
        w = ((lx - gx[idx - 1]) / (gx[idx] - gx[idx - 1])).clamp(0, 1)
        self.register_buffer("prior_log_bridge_g", gy[idx - 1] + w * (gy[idx] - gy[idx - 1]))
        # contact time at mf (2.8 m/s): Askenfelt & Jansson, Five Lectures, Fig. 7
        self.register_buffer("prior_log_tc", torch.log(1e-3 * key_curve(
            [(0, 3.7), (15, 3.0), (27, 2.8), (39, 2.1), (51, 1.45), (63, 1.1), (75, 0.6), (87, 0.5)])))
        # strike position d/L: Conklin, Five Lectures, Fig. 11 (contemporary grand)
        self.register_buffer("prior_strike", key_curve(
            [(0, 0.122), (27, 0.122), (39, 0.121), (49, 0.115), (54, 0.108), (59, 0.100), (69, 0.090), (79, 0.075), (87, 0.065)]))
        self.register_buffer("damper_strength", key_curve([(0, 1.0), (62, 1.0), (67, 0.3), (68, 0.0), (87, 0.0)]))
        self.register_buffer("prior_log_damp", torch.log(key_curve(
            [(0, 8.0), (12, 10.0), (24, 14.0), (36, 20.0), (48, 28.0), (60, 36.0), (67, 40.0), (87, 40.0)])))
        # una corda: lost string(s) of the unison -> quieter prompt, strong aftersound
        self.register_buffer("soft_gain_prior", key_curve([(0, 0.0), (7, 0.0), (8, -2.0), (25, -2.0), (26, -3.5), (87, -3.5)]))
        self.register_buffer("soft_log_after_prior", torch.log(key_curve([(0, 1.0), (7, 1.0), (8, 5.0), (25, 5.0), (26, 3.5), (87, 3.5)])))
        self.register_buffer("prior_log_order", torch.log(key_curve(HAMMER_ORDER_MF)))
        self.register_buffer("harmonic", torch.arange(1, P + 1, dtype=torch.float32))
        # re-strike: nats of the ringing vibration a new blow takes out (M). Weak in the bass, where the contact
        # is short against the period of the low partials; strong in the treble, where it outlasts the period.
        self.register_buffer("prior_restrike", key_curve([(0, 0.35), (39, 0.7), (87, 1.0)]))
        # phantom partials at mf, dB re the partial product (M): strongest in the wound bass strings, negligible
        # above C6. Longitudinal resonance near 15 f1 (L: Conklin, E1 longitudinal ~600 Hz = 14.6 f1; plain steel
        # c_L / 2L gives ~15.7 f1 at C4).
        self.register_buffer("prior_phantom_db", key_curve([(0, -26.0), (27, -30.0), (51, -38.0), (63, -50.0), (87, -60.0)]))
        # knock impulse: the hammer's force pulse reaching the board directly (before the partials build up),
        # dB re the note's gain (M, about -25 dB re the tone over the first 60 ms through the body)
        self.register_buffer("prior_impulse_db", key_curve([(0, -18.0), (39, -15.0), (87, -12.0)]))

        # --- per-key string / hammer / damper parameters (learned offsets) ---
        self.raw_log_B = _p(N_KEYS)
        self.raw_cents = _p(N_KEYS)
        self.raw_log_b1 = _p(N_KEYS)
        self.raw_log_b3 = _p(N_KEYS)
        self.raw_prompt = _p(N_KEYS)
        self.raw_bridge_g = _p(C, BRIDGE_N_KNOTS)  # bridge conductance per condition (one soundboard each)
        self.raw_decay_p = _p()
        self.raw_hammer_x2 = _p()  # second hammer corner (global)
        # unison mistuning (cents) of the aftersound modes: 0.2-2 cents with random sign, drawn per key -- tuners'
        # mistuning "varied randomly from note to note" (Kirk 1959, via Weinreich, Five Lectures). A shared pattern
        # would line up the beat nulls of neighbouring keys.
        g = torch.Generator().manual_seed(1234)
        mag = 0.2 + 1.8 * torch.rand(N_KEYS, M - 1, generator=g) ** 2  # skewed towards small mistuning
        sign = torch.where(torch.rand(N_KEYS, M - 1, generator=g) < 0.5, -1.0, 1.0)
        self.raw_unison = nn.Parameter(5.0 * torch.atanh(mag * sign / 5.0))  # so that bounded(raw, 5) = mag * sign
        self.raw_after = _p(N_KEYS, M - 1)
        self.raw_strike = _p(N_KEYS)
        self.raw_log_tc = _p(N_KEYS)
        self.raw_tc_vel = _p(N_KEYS)
        self.raw_order = _p(N_KEYS)  # hammer roll-off order at mf (offset on the HAMMER_ORDER_MF prior)
        self.raw_order_vel = _p(N_KEYS)  # how fast the roll-off flattens with hammer speed
        self.gain_db = _p(N_KEYS, value=cfg.init_gain_db)
        self.raw_vel_slope = _p(N_KEYS)
        self.partial_gain = _p(N_KEYS, P)  # what the hammer/comb model misses (bounded 8 dB, regularised, stage 2)
        self.color = _p(C, COLOR_N_KEYS, COLOR_N_F)  # per-key bridge colouration per condition (stage 2)
        self.raw_log_damp = _p(N_KEYS)
        self.raw_damp_tilt = _p(N_KEYS)
        self.raw_restrike = _p(N_KEYS)
        self.raw_phantom_db = _p(N_KEYS)
        self.raw_long_ratio = _p(N_KEYS)
        self.raw_long_peak = _p()
        self.raw_long_width = _p()
        self.raw_impulse_db = _p(N_KEYS)
        self.raw_impulse_vel = _p()

        # --- per recording condition (piano + hall + mics of one MAESTRO year) ---
        self.cond_cents = _p(C)
        self.cond_gain_db = _p(C)
        self.cond_vel_slope = _p(C)
        self.cond_log_tc = _p(C)
        self.cond_damper_delay = _p(C)  # damper contact after MIDI note-off: 15 ms + [-50, +50] ms (spec: per year)
        self.cond_vel_curve = _p(C, VEL_KNOTS)  # dB correction of the velocity law (zero: the slope alone)
        self.cond_vel_map = _p(C, VEL_MAP_SEGMENTS)  # log slope multipliers of the velocity map (zero: the prior map)

        # --- pedal mechanics (bounded: review 1 nit, the power was free to run off) ---
        self.raw_pedal_theta = _p()  # sustain value at which dampers lift half-way: 0.42 +- 0.3
        self.raw_pedal_width = _p()  # logistic width 0.06 x/e
        self.raw_pedal_power = _p()  # damping ~ (1 - lift)^power, power 2.5 in [1.6, 4]
        self.soft_log_tc = _p(value=-math.log(0.7))  # una corda: softer felt -> longer contact, darker
        self.soft_gain_db = _p()  # offsets on the per-register una corda priors
        self.soft_log_after = _p()

        # loudness prior: equal early string energy per key at mf (u = 0.6) -- the treble hammer's contact
        # outlasts the string period and the strike point moves, which alone would make C8 ~45 dB quieter.
        # Energy over the first 0.3 s (what is heard as loudness), not the total, which the slow bass
        # aftersound inflates (review 1, m1). Keys with no partial below Nyquist (tiny test sample rates) are left alone.
        with torch.no_grad():
            e = self._mf_energy_db()
            self.gain_db -= (e - e[39]).nan_to_num(0.0, 0.0, 0.0).clamp(-60, 60)
        self.register_buffer("prior_gain_db", self.gain_db.detach().clone())  # the level gauge (regularizer)

    def _mf_energy_db(self, seconds=0.3):
        ki = torch.arange(N_KEYS)[None]
        m = self.modes(ki, torch.full(ki.shape, 0.6), torch.zeros(ki.shape), torch.zeros(1, dtype=torch.long),
                       phantoms=False)
        a2, al = m["amp"] ** 2, m["alpha"]
        energy = (a2 * (1 - torch.exp(-2 * al * seconds)) / (2 * al)).sum((2, 3))[0]
        return 10 * torch.log10(energy)  # -inf where no partial is below Nyquist (tiny test sample rates)

    def hammer_speed(self, u, cond):
        """Hammer speed (m/s) of normalised MIDI velocities ``u[B, K]`` under condition ``cond[B]``'s velocity map."""
        slope = 0.023 * VEL_MAP_STEP * torch.exp(bounded(self.cond_vel_map[cond], 1.0))  # [B, S] nats per segment
        mid = VEL_MAP_SEGMENTS // 2  # the knot at 64
        up = torch.cumsum(slope[:, mid:], 1)
        down = -torch.cumsum(slope[:, :mid].flip(1), 1).flip(1)
        knots = math.log(2.8) + torch.cat([down, torch.zeros_like(slope[:, :1]), up], 1)  # [B, S + 1] at 0, 16, ..., 128
        return torch.exp(interp_knots(knots, 127.0 * u / VEL_MAP_STEP))

    # ------------------------------------------------------------------ pedals, dampers, bridge
    def pedal_lift(self, sustain: torch.Tensor) -> torch.Tensor:
        """Fraction the sustain pedal lifts the dampers (half-pedalling is continuous)."""
        theta = 0.42 + bounded(self.raw_pedal_theta, 0.3)
        width = 0.06 * torch.exp(bounded(self.raw_pedal_width, 1.0))
        return torch.sigmoid((sustain - theta) / width)

    def pedal_damping(self, lift: torch.Tensor) -> torch.Tensor:
        """How much damping remains at a given lift: felt pressure falls faster than the gap opens."""
        power = 2.5 * torch.exp(bounded(self.raw_pedal_power, 0.47))
        return (1 - lift).clamp(min=1e-6) ** power  # min > 0: d(x^p)/dp = x^p ln x

    def damper_delay(self, cond):
        """Damper contact after the MIDI note-off (s), per condition: 15 ms prior (Askenfelt Fig. 3), +-50 ms."""
        return 0.015 + bounded(self.cond_damper_delay[cond], 0.05)

    def bridge_conductance(self, f, cond):
        """Relative bridge conductance g_c(f): piecewise linear in log f on the 1/6-octave grid, per condition."""
        ys = self.prior_log_bridge_g + bounded(self.raw_bridge_g[cond], 1.0)  # [B, knots]
        pos = torch.log2(f.clamp(min=1.0) / F_REF) * BRIDGE_KNOTS_PER_OCT
        return torch.exp(interp_knots(ys, pos))

    def coloration(self, ki, f, cond):
        """Per-key bridge colouration (log amplitude) of condition ``cond[B]`` for keys ``ki[B,K]`` at ``f[B,K,P]``."""
        tab = bounded(self.color[cond], COLOR_BOUND)  # [B, Kk, Fk]
        B, Kk, Fk = tab.shape
        kp = (ki.float() / COLOR_KEY_STEP)[..., None].expand_as(f).clamp(0, Kk - 1)
        fp = (torch.log2(f.clamp(min=1.0) / F_REF) * COLOR_KNOTS_PER_OCT).clamp(0, Fk - 1)
        k0 = kp.floor().long().clamp(max=Kk - 2)
        f0 = fp.detach().floor().long().clamp(max=Fk - 2)
        wk, wf = kp - k0, fp - f0
        flat = tab.reshape(B, -1)
        k0f, f0f = k0.reshape(B, -1), f0.reshape(B, -1)
        wkf, wff = wk.reshape(B, -1), wf.reshape(B, -1)

        def at(dk, df):
            return flat.gather(1, (k0f + dk) * Fk + f0f + df)

        y = ((at(0, 0) * (1 - wff) + at(0, 1) * wff) * (1 - wkf) + (at(1, 0) * (1 - wff) + at(1, 1) * wff) * wkf)
        return y.reshape(f.shape)

    def contact_time(self, ki, u, soft, cond, ctx_log_fc=None):
        """Hammer-string contact time in seconds; shorter (brighter) for harder strikes."""
        q = 0.2 * torch.exp(bounded(self.raw_tc_vel[ki], 0.7))  # Askenfelt Fig. 6: slope ~ -0.19 at C4
        log_tc = (self.prior_log_tc[ki] + bounded(self.raw_log_tc[ki], 1.0) + bounded(self.cond_log_tc[cond[:, None]], 0.5)
                  - q * torch.log(self.hammer_speed(u, cond) / 2.8) + soft * self.soft_log_tc)
        if ctx_log_fc is not None:
            log_tc = log_tc - ctx_log_fc
        return log_tc.exp()

    def level_db(self, ki, u, soft, cond, ctx_gain_db=None):
        """Note level (dB) of keys ``ki`` at velocity ``u``: per-key gain + condition + velocity law + una corda."""
        cond_k = cond[:, None]
        u0 = u - 0.6
        db = (self.gain_db[ki] + bounded(self.cond_gain_db[cond_k], 12.0)
              + (40.0 * torch.exp(bounded(self.raw_vel_slope[ki], 0.7)) + bounded(self.cond_vel_slope[cond_k], 10.0)) * u0
              + interp_knots(bounded(self.cond_vel_curve[cond], 12.0), u * (VEL_KNOTS - 1))
              + soft * (self.soft_gain_prior[ki] + bounded(self.soft_gain_db, 3.0)))
        return db if ctx_gain_db is None else db + ctx_gain_db

    # ------------------------------------------------------------------ modes
    def modes(self, ki, u, soft, cond, ctx=None, phantoms=True):
        """Modal parameters for strikes of keys ``ki[B,K]`` at normalised velocity ``u[B,K]``.

        Returns ``freq``, ``alpha`` (1/s) and signed ``amp`` of shape ``[B, K, P, M]``,
        ``alpha_damp[B, K, P]``, the hammer contact time ``tc[B, K]``, the re-strike loss
        ``restrike[B, K]`` (nats), the knock impulse amplitude ``impulse[B, K]`` and, with
        ``phantoms``, the phantom partials ``ph_freq / ph_alpha / ph_amp / ph_alpha_damp`` ``[B, K, Q]``.
        ``ctx`` holds per-note corrections (all optional): the context network's, and the per-strike variation's
        (``impulse_db``, and ``strike_log_fc``, ``strike_log_decay``, ``strike_decay_tilt``: brightness and decay
        offsets under which the prompt partials' energy over ``STRIKE_LEVEL_SECONDS`` is kept).
        """
        cfg = self.cfg
        n = self.harmonic
        cond_k = cond[:, None]
        ctx = ctx or {}
        zero = torch.zeros_like(u)

        cents = self.prior_cents[ki] + bounded(self.raw_cents[ki], 30.0) + bounded(self.cond_cents[cond_k], 30.0)
        B = torch.exp(self.prior_log_B[ki] + bounded(self.raw_log_B[ki], 1.5))
        # the tuner sets the *sounding* fundamental f1 = f0 sqrt(1 + B), so the stretch applies to f1
        f1 = 440.0 * torch.pow(2.0, (ki + LOWEST_MIDI - 69).float() / 12 + cents / 1200)
        f0 = f1 / torch.sqrt(1 + B)
        fn = f0[..., None] * n * torch.sqrt(1 + B[..., None] * n**2)  # [B,K,P]

        detune = torch.cat([torch.zeros_like(fn[..., :1]), bounded(self.raw_unison[ki], 5.0)], -1)
        freq = fn[..., None] * torch.pow(2.0, detune[..., None, :] / 1200)  # [B,K,P,M]

        # decay: aftersound = internal/air losses; prompt adds the unison's bridge loss (additive, same for all n)
        b1 = torch.exp(self.prior_log_b1[ki] + bounded(self.raw_log_b1[ki], 1.5))
        b3 = torch.exp(self.prior_log_b3[ki] + bounded(self.raw_log_b3[ki], 1.5))
        log_decay = ctx.get("log_decay", zero)[..., None] + ctx.get("decay_tilt", zero)[..., None] * torch.log2(fn / 1000.0)
        strike = "strike_log_fc" in ctx
        if strike:
            s_decay = ctx["strike_log_decay"][..., None] + ctx["strike_decay_tilt"][..., None] * torch.log2(fn / 1000.0)
            log_decay = log_decay + s_decay
        decay_scale = torch.exp(log_decay)
        p = DECAY_EXPONENT * torch.exp(bounded(self.raw_decay_p, 0.25))
        alpha_after = (b1[..., None] + b3[..., None] * 1e6 * (fn / 1000.0) ** p) * decay_scale
        bridge = (self.prior_prompt_ratio[ki] - 1) * b1 * torch.exp(bounded(self.raw_prompt[ki], 1.5))
        alpha_prompt = alpha_after + (bridge[..., None] * self.bridge_conductance(fn, cond)) * decay_scale
        keep_partial = None
        if "strike_partial_decay" in ctx:  # each partial's prompt decay per strike, its energy over the first 0.3 s kept
            def e_of(al):
                return -torch.expm1(-2 * al * STRIKE_LEVEL_SECONDS) / (2 * al)
            a0 = alpha_prompt
            alpha_prompt = alpha_prompt * torch.exp(ctx["strike_partial_decay"])
            keep_partial = torch.sqrt(e_of(a0) / e_of(alpha_prompt))
        alpha = torch.cat([alpha_prompt[..., None], alpha_after[..., None].expand(*alpha_after.shape, cfg.n_modes - 1)], -1)

        # excitation: bridge force = gain(v) * hammer pulse spectrum * signed strike-position comb * colouration
        log_fc = ctx.get("log_fc")
        if strike:
            log_fc = ctx["strike_log_fc"] if log_fc is None else log_fc + ctx["strike_log_fc"]
        tc = self.contact_time(ki, u, soft, cond, log_fc)
        order = (torch.exp(self.prior_log_order[ki] + bounded(self.raw_order[ki], 0.9))
                 * (self.hammer_speed(u, cond) / 2.8) ** (-HAMMER_ORDER_VEL * torch.exp(bounded(self.raw_order_vel[ki], 0.7))))
        x2 = HAMMER_X2 * torch.exp(bounded(self.raw_hammer_x2, 0.7))
        hammer = hammer_spectrum(fn, tc[..., None], order[..., None], x2)
        x0 = self.prior_strike[ki] * torch.exp(bounded(self.raw_strike[ki], 0.5))
        comb = torch.sin(math.pi * n * x0[..., None])
        # the bridge end sees (-1)^(n+1) of the agraffe end's comb; the phantoms keep the agraffe-end signs
        parity = 1.0 - 2.0 * (n.remainder(2) == 0).to(comb.dtype) if cfg.bridge_end_comb else None
        if parity is not None:
            comb = comb * parity
        gain_db = self.level_db(ki, u, soft, cond, ctx.get("gain_db"))
        log_shape = bounded(self.partial_gain[ki], PARTIAL_GAIN_BOUND) + self.coloration(ki, fn, cond)
        if "spec" in ctx:  # context net: per-note spectral correction, smooth in log f
            log_shape = log_shape + log_f_bumps(ctx["spec"], fn)
        gain = torch.pow(10.0, gain_db / 20)
        if strike:  # the strike's brightness and decay keep the note's early energy (the level is its own offset)
            def energy(h, al):
                return ((h * comb * torch.exp(log_shape)) ** 2 * -torch.expm1(-2 * al * STRIKE_LEVEL_SECONDS) / (2 * al)
                        * (fn < 0.48 * cfg.sample_rate)).sum(-1)
            h0 = hammer_spectrum(fn, (tc * torch.exp(ctx["strike_log_fc"]))[..., None], order[..., None], x2)
            e_var, e_0 = energy(hammer, alpha_prompt), energy(h0, alpha_prompt * torch.exp(-s_decay))
            gain = gain * torch.where(e_var > 0, torch.sqrt(e_0 / e_var.clamp(min=1e-30)), torch.ones_like(e_var))
        base = gain[..., None] * hammer * comb * torch.exp(log_shape)
        if keep_partial is not None:
            base = base * keep_partial
        soft_after = torch.exp(soft * (self.soft_log_after_prior[ki] + bounded(self.soft_log_after, 1.0)))
        after = torch.exp(self.prior_log_after[ki][..., None] + bounded(self.raw_after[ki], 1.5)) * soft_after[..., None]
        after = after[..., None, :]  # [B, K, 1, M - 1]
        if "strike_after" in ctx:
            # the strings never meet the hammer alike: each partial's aftersound gets a random part on top of its key's
            # (either sign, so a beat may start at its minimum): a factor (1 + s z) / sqrt(1 + s^2) [B, K, P, M - 1]
            after = after * ctx["strike_after"]
        amp = base[..., None] * torch.cat([torch.ones_like(base[..., None]), after.expand(*base.shape, after.shape[-1])], -1)

        n_active = self.n_strings[ki].clamp(min=2)  # monochords still have two polarisations
        mode_ok = torch.arange(cfg.n_modes, device=ki.device) < n_active[..., None]
        ok = (freq < 0.48 * cfg.sample_rate) & mode_ok[..., None, :]
        amp = amp * ok

        # dampers: felt near the string end damps higher partials harder, saturating around n = 6
        damp = self.damper_strength[ki] * torch.exp(self.prior_log_damp[ki] + bounded(self.raw_log_damp[ki], 1.0))
        tilt = 0.6 + bounded(self.raw_damp_tilt[ki], 0.4)
        alpha_damp = damp[..., None] * n.clamp(max=6.0) ** tilt[..., None]

        restrike = self.prior_restrike[ki] * torch.exp(bounded(self.raw_restrike[ki], 1.0))
        impulse_db = (gain_db + self.prior_impulse_db[ki] + bounded(self.raw_impulse_db[ki], 20.0)
                      + bounded(self.raw_impulse_vel, 20.0) * (u - 0.6) + ctx.get("impulse_db", zero))
        out = {"freq": freq, "alpha": alpha, "amp": amp, "alpha_damp": alpha_damp, "tc": tc,
               "restrike": restrike, "impulse": torch.pow(10.0, impulse_db / 20)}
        if phantoms and cfg.n_phantoms > 0:
            ref_db = self.level_db(ki, torch.full_like(u, 64 / 127), zero, cond)
            ph_amp = amp if parity is None else amp * parity[..., None]
            out.update(self._phantoms(ki, freq, alpha, ph_amp, alpha_damp, f1, ref_db, cond))
        return out

    def _phantoms(self, ki, freq, alpha, amp, alpha_damp, f1, ref_db, cond):
        """Phantom partials from pairs (j, j) and (j, j+1) of the prompt and first aftersound modes.

        Tension modulation (the string's length change ~ y^2) drives the bridge at f_j + f_k with a
        force ~ a_j a_k, so their level re the transverse partials grows with the note level (~v^2).
        The longitudinal resonance of the string (near 15 f1) emphasises the ones close to it.
        Ordered by frequency (2f_1, f_1+f_2, 2f_2, ...) so that trailing slots above Nyquist are zero.
        """
        cfg = self.cfg
        Q = min(cfg.n_phantoms, cfg.n_partials - 1)
        fa, aa, al =freq[..., : Q + 1, :2], amp[..., : Q + 1, :2], alpha[..., : Q + 1, :2]
        ad = alpha_damp[..., : Q + 1, None]
        f_ph = torch.stack([2 * fa[..., :Q, :], fa[..., :Q, :] + fa[..., 1:, :]], -2)  # [B,K,Q,2 series,2 modes]
        a_ph = torch.stack([aa[..., :Q, :] ** 2, aa[..., :Q, :] * aa[..., 1:, :]], -2)
        al_ph = torch.stack([2 * al[..., :Q, :], al[..., :Q, :] + al[..., 1:, :]], -2)
        ad_ph = torch.stack([2 * ad[..., :Q, :], ad[..., :Q, :] + ad[..., 1:, :]], -2).expand_as(al_ph)
        level_db = self.prior_phantom_db[ki] + bounded(self.raw_phantom_db[ki], 20.0) - ref_db  # re the mf gain
        f_long = f1 * 15.0 * torch.exp(bounded(self.raw_long_ratio[ki], 0.3))
        width = 0.5 * torch.exp(bounded(self.raw_long_width, 0.7))  # octaves
        peak = 10 ** ((6.0 + bounded(self.raw_long_peak, 6.0)) / 20)
        d = torch.log2(f_ph / f_long[..., None, None, None]) / width
        emphasis = 1 + (peak - 1) / (1 + d**2)
        a_ph = a_ph * torch.pow(10.0, level_db / 20)[..., None, None, None] * emphasis
        ok = (f_ph < 0.48 * cfg.sample_rate) & (ki < 64)[..., None, None, None]
        flat = lambda x: x.reshape(*x.shape[:2], -1)
        return {"ph_freq": flat(f_ph), "ph_alpha": flat(al_ph), "ph_amp": flat(a_ph * ok), "ph_alpha_damp": flat(ad_ph)}

    # ------------------------------------------------------------------ regularisation
    def regularizer(self, cond=None) -> torch.Tensor:
        """Keep per-key tables smooth across the keyboard and the residual tables small.

        ``cond`` restricts the per-condition tables to the conditions being trained (the others get no data)."""
        def smooth(x, dim=0):
            x = x.movedim(dim, 0)
            return ((x[2:] - 2 * x[1:-1] + x[:-2]) ** 2).mean()
        tables = [self.raw_log_B, self.raw_cents, self.raw_log_b1, self.raw_log_b3, self.raw_prompt,
                  self.raw_strike, self.raw_log_tc, self.raw_tc_vel, self.raw_order, self.raw_order_vel, self.raw_vel_slope,
                  self.raw_log_damp, self.raw_damp_tilt, self.gain_db / 20, self.raw_restrike, self.raw_phantom_db / 20,
                  self.raw_long_ratio, self.raw_impulse_db / 20]
        reg = sum(smooth(t) for t in tables)
        conds = torch.arange(self.cfg.n_conditions) if cond is None else torch.unique(cond).cpu()
        g = self.raw_bridge_g[conds]
        reg = reg + 0.1 * smooth(g, 1)  # 1/6-octave detail is allowed; wiggles at the knot scale are not
        pg = bounded(self.partial_gain, PARTIAL_GAIN_BOUND)
        reg = reg + 1e-2 * (pg**2).mean() + smooth(pg, 0)  # smooth across keys at equal partial number
        reg = reg + smooth(bounded(self.cond_vel_curve[conds], 12.0) / 20, 1)
        reg = reg + smooth(bounded(self.cond_vel_map[conds], 1.0), 1)
        col = bounded(self.color[conds], COLOR_BOUND)
        reg = reg + 1e-2 * (col**2).mean() + smooth(col, 1) + smooth(col, 2) + (col.mean(1) ** 2).mean()
        # level gauge: the per-key gains, the condition gain and the velocity curve could all trade a constant dB
        # with the microphone gain, so their means are pinned and the mic gain carries the level (review 4, 6)
        reg = reg + ((self.gain_db - self.prior_gain_db).mean() / 10) ** 2
        reg = reg + (bounded(self.cond_gain_db[conds], 12.0) / 10).pow(2).mean()
        reg = reg + (bounded(self.cond_vel_curve[conds], 12.0).mean(1) / 10).pow(2).mean()
        return reg
