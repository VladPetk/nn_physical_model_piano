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
    alpha_n  = b1 + b3 f_n^2                               aftersound: internal + air losses
    mode 0 (in-phase, "prompt sound") adds the bridge loss of all strings of the
    unison, alpha_n + (R - 1) b1, the same for every partial; modes 1.. are the
    weakly radiating aftersound (Weinreich's coupled strings in mode space).
    a_n      = gain(v) F(f_n; T_c(v)) sin(pi n x0/L)       bridge force: hammer pulse envelope x strike comb
    damper:  extra decay alpha_d,n whenever the key is up and neither pedal holds the damper off.
"""

import math

import torch
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


HAMMER_ORDER_VEL = 0.2  # fitted to Hall, Five Lectures, Fig. 15 (C4 slopes pp/mf/ff)
# roll-off order at mf per key: 2.1 at C4 from Hall; the rest fitted to the early spectral slope of the
# Iowa Steinway B (radiated, near field) -- gentle in the bass, steep in the treble where the contact
# outlasts half the string period (Askenfelt & Jansson, Five Lectures, Fig. 8)
HAMMER_ORDER_MF = [(0, 1.5), (27, 1.5), (39, 2.1), (48, 1.9), (51, 2.6), (63, 3.2), (75, 4.5), (87, 4.5)]


def hammer_velocity(u):
    """MIDI velocity / 127 -> hammer speed in m/s (about 0.4 at pp, 5.5 at fff)."""
    return 5.5 * u.clamp(min=1e-3) ** 1.4


def hammer_spectrum(f, tc, order=1.0):
    """Smooth magnitude envelope of the hammer force pulse of duration ``tc``.

    -3 dB at 0.59/tc (the corner of a half-sine pulse), then -6*order dB/oct. The felt is
    nonlinear (F ~ x^p, p = 2-3.5), so a harder blow drives it into its stiff range and the
    pulse gets sharper: the roll-off order falls with hammer speed (Hall, Five Lectures,
    Fig. 15: -18 / -15 / -11 dB/oct at pp / mf / ff for C4). The ideal half-sine's nulls are
    replaced by the smooth envelope: measured pulses are skewed and their nulls are filled.
    """
    return torch.rsqrt(1 + (f * tc / 0.59) ** (2 * order))


class PianoPhysics(nn.Module):
    def __init__(self, cfg: PianoConfig):
        super().__init__()
        self.cfg = cfg
        P, M, C = cfg.n_partials, cfg.n_modes, cfg.n_conditions
        k = torch.arange(N_KEYS)

        # --- priors (buffers, not trained); see docs/physical_parameters.md ---
        # Rigaud, David & Daudet (DAFx 2011) two-asymptote fit, m = MIDI pitch (as used by DDSP-Piano)
        m = (k + LOWEST_MIDI).float()
        self.register_buffer("prior_log_B", torch.log(torch.exp(0.0926 * m - 13.64) + torch.exp(-0.0847 * m - 5.82)))
        # stretch re A4, measured on the Iowa Steinway B (C8 extrapolated)
        self.register_buffer("prior_cents", key_curve(
            [(0, -16), (3, -15), (15, -4), (27, 0), (39, -1), (48, 0), (51, 0), (63, 6), (75, 14), (87, 25)]))
        self.register_buffer("prior_log_b1", torch.log(key_curve(
            [(0, 0.062), (15, 0.067), (27, 0.118), (39, 0.216), (48, 0.25), (51, 0.30), (63, 0.40), (75, 0.6), (87, 1.0)])))
        self.register_buffer("prior_log_b3", torch.log(key_curve([(0, 2.5e-7), (20, 2.5e-7), (30, 1.2e-7), (55, 1.0e-7), (63, 5e-8), (87, 2.5e-8)])))
        self.register_buffer("n_strings", torch.where(k < 8, 1, torch.where(k < 26, 2, 3)))
        # prompt/aftersound decay ratio at the fundamental (R) and aftersound amplitude per mode: solved from the
        # decays and knee levels measured on the Iowa Steinway B (docs/calibration_iowa.md)
        self.register_buffer("prior_prompt_ratio", key_curve([(0, 4.3), (15, 3.7), (27, 3.5), (39, 3.0), (48, 3.5), (63, 2.8), (75, 3.0), (87, 3.0)]))
        self.register_buffer("prior_log_after", torch.log(key_curve([(0, 0.06), (20, 0.06), (27, 0.06), (36, 0.09), (48, 0.09), (55, 0.10), (63, 0.12), (75, 0.06), (87, 0.05)])))
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
        unison_init = torch.tensor([0.6, -0.4, 0.2, -0.1])[: M - 1]
        self.raw_unison = nn.Parameter(unison_init.repeat(N_KEYS, 1).clone())  # cents
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
        alpha_after = (b1[..., None] + b3[..., None] * fn**2) * decay_scale
        bridge = (self.prior_prompt_ratio[ki] - 1) * b1 * torch.exp(bounded(self.raw_prompt[ki], 1.5))
        alpha_prompt = alpha_after + bridge[..., None] * decay_scale
        alpha = torch.cat([alpha_prompt[..., None], alpha_after[..., None].expand(*alpha_after.shape, cfg.n_modes - 1)], -1)

        # excitation: bridge force = gain(v) * half-sine pulse spectrum * signed strike-position comb
        tc = self.contact_time(ki, u, soft, cond, ctx.get("log_fc"))
        order = (torch.exp(self.prior_log_order[ki] + bounded(self.raw_order[ki], 0.5))
                 * (hammer_velocity(u) / 2.8) ** (-HAMMER_ORDER_VEL * torch.exp(bounded(self.raw_order_vel[ki], 0.7))))
        hammer = hammer_spectrum(fn, tc[..., None], order[..., None])
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
        return sum(smooth(t) for t in tables) + 1e-2 * (self.partial_gain**2).mean()
