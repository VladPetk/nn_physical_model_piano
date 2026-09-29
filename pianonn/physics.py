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
    alpha_n  = b1 + b3 f_r^2 (f_n / f_r)^p                  aftersound: internal + air losses (f_r = 1 kHz, p ~ 1.6)
    mode 0 (in-phase, "prompt sound") adds the bridge loss of all strings of the
    unison, alpha_n + (R - 1) b1 g(f_n), with g the bridge conductance over frequency
    (one curve, normalised); modes 1.. are the weakly radiating aftersound
    (Weinreich's coupled strings in mode space).
    a_n      = gain(v) F(f_n; T_c(v)) sin(pi n x0/L)       bridge force: hammer pulse envelope x strike comb
    damper:  extra decay alpha_d,n whenever the key is up and neither pedal holds the damper off.
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


def hammer_velocity(u):
    """MIDI velocity / 127 -> hammer speed in m/s.

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
        self.register_buffer("bridge_g_log_f", torch.log(torch.tensor([p[0] for p in BRIDGE_G], dtype=torch.float32)))
        self.register_buffer("prior_log_bridge_g", torch.log(torch.tensor([p[1] for p in BRIDGE_G], dtype=torch.float32)))
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

        # --- per-key string / hammer / damper parameters (learned offsets) ---
        self.raw_log_B = _p(N_KEYS)
        self.raw_cents = _p(N_KEYS)
        self.raw_log_b1 = _p(N_KEYS)
        self.raw_log_b3 = _p(N_KEYS)
        self.raw_prompt = _p(N_KEYS)
        self.raw_bridge_g = _p(len(BRIDGE_G))  # bridge conductance curve (global: one soundboard)
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
        self.partial_gain = _p(N_KEYS, P)  # what the hammer/comb model misses (regularised)
        self.raw_log_damp = _p(N_KEYS)
        self.raw_damp_tilt = _p(N_KEYS)

        # --- per recording condition (piano + hall + mics of one MAESTRO year) ---
        self.cond_cents = _p(C)
        self.cond_gain_db = _p(C)
        self.cond_vel_slope = _p(C)
        self.cond_log_tc = _p(C)

        # --- pedal mechanics ---
        self.pedal_theta = _p(value=0.42)  # sustain value at which dampers lift half-way
        self.pedal_log_width = _p(value=math.log(0.06))
        self.pedal_log_power = _p(value=math.log(2.5))  # damping ~ (1 - lift)^power (felt pressure)
        self.soft_log_tc = _p(value=-math.log(0.7))  # una corda: softer felt -> longer contact, darker
        self.soft_gain_db = _p()  # offsets on the per-register una corda priors
        self.soft_log_after = _p()

        # loudness prior: equal string energy per key at mf (u = 0.6) -- the treble hammer's contact
        # outlasts the string period and the strike point moves, which alone would make C8 ~45 dB quieter.
        # Keys with no partial below Nyquist (tiny test sample rates) are left alone.
        with torch.no_grad():
            e = self._mf_energy_db()
            self.gain_db -= (e - e[39]).nan_to_num(0.0, 0.0, 0.0).clamp(-60, 60)

    def _mf_energy_db(self):
        ki = torch.arange(N_KEYS)[None]
        m = self.modes(ki, torch.full(ki.shape, 0.6), torch.zeros(ki.shape), torch.zeros(1, dtype=torch.long))
        energy = (m["amp"] ** 2 / (2 * m["alpha"])).sum((2, 3))[0]
        return 10 * torch.log10(energy)  # -inf where no partial is below Nyquist (tiny test sample rates)

    def pedal_lift(self, sustain: torch.Tensor) -> torch.Tensor:
        """Fraction the sustain pedal lifts the dampers (half-pedalling is continuous)."""
        return torch.sigmoid((sustain - self.pedal_theta) / self.pedal_log_width.exp())

    def pedal_damping(self, lift: torch.Tensor) -> torch.Tensor:
        """How much damping remains at a given lift: felt pressure falls faster than the gap opens."""
        return (1 - lift).clamp(min=1e-6) ** self.pedal_log_power.exp()  # min > 0: d(x^p)/dp = x^p ln x

    def bridge_conductance(self, f):
        """Relative bridge conductance g(f): piecewise linear in log f between the BRIDGE_G knots."""
        xs = self.bridge_g_log_f
        ys = self.prior_log_bridge_g + bounded(self.raw_bridge_g, 1.0)
        lf = torch.log(f.clamp(min=1.0))
        idx = torch.searchsorted(xs, lf.detach().contiguous()).clamp(1, len(xs) - 1)
        w = ((lf - xs[idx - 1]) / (xs[idx] - xs[idx - 1])).clamp(0, 1)
        return torch.exp(ys[idx - 1] + w * (ys[idx] - ys[idx - 1]))

    def contact_time(self, ki, u, soft, cond, ctx_log_fc=None):
        """Hammer-string contact time in seconds; shorter (brighter) for harder strikes."""
        q = 0.2 * torch.exp(bounded(self.raw_tc_vel[ki], 0.7))  # Askenfelt Fig. 6: slope ~ -0.19 at C4
        log_tc = (self.prior_log_tc[ki] + bounded(self.raw_log_tc[ki], 1.0) + bounded(self.cond_log_tc[cond[:, None]], 0.5)
                  - q * torch.log(hammer_velocity(u) / 2.8) + soft * self.soft_log_tc)
        if ctx_log_fc is not None:
            log_tc = log_tc - ctx_log_fc
        return log_tc.exp()

    def modes(self, ki, u, soft, cond, ctx=None):
        """Modal parameters for strikes of keys ``ki[B,K]`` at normalised velocity ``u[B,K]``.

        Returns ``freq``, ``alpha`` (1/s) and signed ``amp`` of shape ``[B, K, P, M]``,
        ``alpha_damp[B, K, P]`` and the hammer contact time ``tc[B, K]``.
        """
        cfg = self.cfg
        n = self.harmonic
        cond_k = cond[:, None]
        ctx = ctx or {}
        zero = torch.zeros_like(u)

        cents = self.prior_cents[ki] + bounded(self.raw_cents[ki], 30.0) + bounded(self.cond_cents[cond_k], 30.0)
        B = torch.exp(self.prior_log_B[ki] + bounded(self.raw_log_B[ki], 1.5))
        # the tuner sets the *sounding* fundamental f1 = f0 sqrt(1 + B), so the stretch applies to f1
        f0 = 440.0 * torch.pow(2.0, (ki + LOWEST_MIDI - 69).float() / 12 + cents / 1200) / torch.sqrt(1 + B)
        fn = f0[..., None] * n * torch.sqrt(1 + B[..., None] * n**2)  # [B,K,P]

        detune = torch.cat([torch.zeros_like(fn[..., :1]), bounded(self.raw_unison[ki], 5.0)], -1)
        freq = fn[..., None] * torch.pow(2.0, detune[..., None, :] / 1200)  # [B,K,P,M]

        # decay: aftersound = internal/air losses; prompt adds the unison's bridge loss (additive, same for all n)
        b1 = torch.exp(self.prior_log_b1[ki] + bounded(self.raw_log_b1[ki], 1.5))
        b3 = torch.exp(self.prior_log_b3[ki] + bounded(self.raw_log_b3[ki], 1.5))
        decay_scale = torch.exp(ctx.get("log_decay", zero))[..., None]
        p = DECAY_EXPONENT * torch.exp(bounded(self.raw_decay_p, 0.25))
        alpha_after = (b1[..., None] + b3[..., None] * 1e6 * (fn / 1000.0) ** p) * decay_scale
        bridge = (self.prior_prompt_ratio[ki] - 1) * b1 * torch.exp(bounded(self.raw_prompt[ki], 1.5))
        alpha_prompt = alpha_after + (bridge[..., None] * self.bridge_conductance(fn)) * decay_scale
        alpha = torch.cat([alpha_prompt[..., None], alpha_after[..., None].expand(*alpha_after.shape, cfg.n_modes - 1)], -1)

        # excitation: bridge force = gain(v) * half-sine pulse spectrum * signed strike-position comb
        tc = self.contact_time(ki, u, soft, cond, ctx.get("log_fc"))
        order = (torch.exp(self.prior_log_order[ki] + bounded(self.raw_order[ki], 0.9))
                 * (hammer_velocity(u) / 2.8) ** (-HAMMER_ORDER_VEL * torch.exp(bounded(self.raw_order_vel[ki], 0.7))))
        hammer = hammer_spectrum(fn, tc[..., None], order[..., None], HAMMER_X2 * torch.exp(bounded(self.raw_hammer_x2, 0.7)))
        x0 = self.prior_strike[ki] * torch.exp(bounded(self.raw_strike[ki], 0.5))
        comb = torch.sin(math.pi * n * x0[..., None])
        u0 = u - 0.6
        gain_db = (self.gain_db[ki] + bounded(self.cond_gain_db[cond_k], 12.0)
                   + (40.0 * torch.exp(bounded(self.raw_vel_slope[ki], 0.7)) + bounded(self.cond_vel_slope[cond_k], 10.0)) * u0
                   + soft * (self.soft_gain_prior[ki] + bounded(self.soft_gain_db, 3.0)) + ctx.get("gain_db", zero))
        base = torch.pow(10.0, gain_db / 20)[..., None] * hammer * comb * torch.exp(bounded(self.partial_gain[ki], 3.0))
        soft_after = torch.exp(soft * (self.soft_log_after_prior[ki] + bounded(self.soft_log_after, 1.0)))
        after = torch.exp(self.prior_log_after[ki][..., None] + bounded(self.raw_after[ki], 1.5)) * soft_after[..., None]
        amp = base[..., None] * torch.cat([torch.ones_like(after[..., :1]), after], -1)[..., None, :]

        n_active = self.n_strings[ki].clamp(min=2)  # monochords still have two polarisations
        mode_ok = torch.arange(cfg.n_modes, device=ki.device) < n_active[..., None]
        ok = (freq < 0.48 * cfg.sample_rate) & mode_ok[..., None, :]

        # dampers: felt near the string end damps higher partials harder, saturating around n = 6
        damp = self.damper_strength[ki] * torch.exp(self.prior_log_damp[ki] + bounded(self.raw_log_damp[ki], 1.0))
        tilt = 0.6 + bounded(self.raw_damp_tilt[ki], 0.4)
        alpha_damp = damp[..., None] * n.clamp(max=6.0) ** tilt[..., None]

        return {"freq": freq, "alpha": alpha, "amp": amp * ok, "alpha_damp": alpha_damp, "tc": tc}

    def regularizer(self) -> torch.Tensor:
        """Keep per-key tables smooth across the keyboard and the residual table small."""
        def smooth(x):
            return ((x[2:] - 2 * x[1:-1] + x[:-2]) ** 2).mean()
        tables = [self.raw_log_B, self.raw_cents, self.raw_log_b1, self.raw_log_b3, self.raw_prompt,
                  self.raw_strike, self.raw_log_tc, self.raw_tc_vel, self.raw_order, self.raw_order_vel, self.raw_vel_slope,
                  self.raw_log_damp, self.raw_damp_tilt, self.gain_db / 20]
        return sum(smooth(t) for t in tables) + smooth(self.raw_bridge_g) + 1e-2 * (self.partial_gain**2).mean()
