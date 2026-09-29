"""Per-key physical parameters of the piano.

Every quantity is ``prior(key) + bounded(learned offset)``. The priors are rough
concert-grand values from the acoustics literature; they make the untrained
model already sound like a (plain) piano and, more importantly, start the
frequency-related parameters inside the basin where spectral-loss gradients are
informative. Learned offsets are bounded so that gradient descent tunes the
instrument rather than inventing a different one.

String model (per key k, partial n, coupled mode m), evaluated in closed form:

    f_n     = n * f0 * sqrt(1 + B n^2)                     stiff-string inharmonicity
    f_nm    = f_n * 2^(unison_cents_m / 1200)              unison mistuning -> beats
    alpha_n = b1 + b3 f_n^2                                frequency-dependent loss
    mode 0 decays prompt_ratio times faster ("prompt sound", in-phase string
    motion that drives the bridge strongly); modes 1.. are the weakly radiating
    "aftersound" (Weinreich's coupled-string picture, expressed in mode space).
    a_n     = gain(v) * H(f_n; v) * |sin(pi n x0)|         hammer spectrum x strike-position comb
    damper: extra decay alpha_d(f) whenever the key is up and the pedal is not holding the damper off.
"""

import math

import torch
from torch import nn

from .config import PianoConfig
from .dsp import bounded

N_KEYS = 88
LOWEST_MIDI = 21
HIGHEST_DAMPED_MIDI = 88  # keys above roughly E6 have no dampers on a grand


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


class PianoPhysics(nn.Module):
    def __init__(self, cfg: PianoConfig):
        super().__init__()
        self.cfg = cfg
        P, M, C = cfg.n_partials, cfg.n_modes, cfg.n_conditions
        k = torch.arange(N_KEYS)
        ln10 = math.log(10)

        # --- priors (buffers, not trained) ---
        self.register_buffer("prior_log_B", ln10 * key_curve([(0, -3.6), (25, -4.0), (45, -3.7), (65, -3.1), (87, -2.0)]))
        self.register_buffer("prior_cents", key_curve([(0, -20), (24, -5), (48, 0), (66, 4), (87, 25)]))  # stretch tuning
        self.register_buffer("prior_log_b1", torch.log(key_curve([(0, 0.25), (40, 0.5), (60, 1.0), (87, 5.0)])))
        self.register_buffer("prior_log_fc", torch.log(key_curve([(0, 350.0), (40, 800.0), (87, 3500.0)])))
        self.register_buffer("n_strings", torch.where(k < 10, 1, torch.where(k < 28, 2, 3)))
        self.register_buffer("has_damper", (k + LOWEST_MIDI <= HIGHEST_DAMPED_MIDI).float())
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
        self.raw_log_fc = _p(N_KEYS)
        self.raw_fc_vel = _p(N_KEYS)
        self.raw_rolloff = _p(N_KEYS)
        self.gain_db = _p(N_KEYS, value=cfg.init_gain_db)
        self.raw_vel_slope = _p(N_KEYS)
        self.partial_gain = _p(N_KEYS, P)  # what the hammer/comb model misses (regularised)
        self.raw_log_damp = _p(N_KEYS)
        self.raw_damp_tilt = _p(N_KEYS)

        # --- per recording condition (piano + hall + mics of one MAESTRO year) ---
        self.cond_cents = _p(C)
        self.cond_gain_db = _p(C)
        self.cond_vel_slope = _p(C)
        self.cond_log_fc = _p(C)

        # --- pedal mechanics ---
        self.pedal_theta = _p(value=0.45)  # sustain value at which dampers lift half-way
        self.pedal_log_width = _p(value=math.log(0.06))
        self.soft_log_fc = _p(value=math.log(0.75))  # una corda: softer felt -> darker
        self.soft_gain_db = _p(value=-3.0)
        self.soft_log_after = _p(value=math.log(1.5))  # ...and unbalanced unison -> more aftersound

    def pedal_lift(self, sustain: torch.Tensor) -> torch.Tensor:
        """Fraction the sustain pedal lifts the dampers (half-pedalling is continuous)."""
        return torch.sigmoid((sustain - self.pedal_theta) / self.pedal_log_width.exp())

    def modes(self, ki, u, soft, cond, ctx=None):
        """Modal parameters for strikes of keys ``ki[B,K]`` at normalised velocity ``u[B,K]``.

        Returns tensors of shape ``[B, K, P, M]`` (freq in Hz, alpha in 1/s, amp)
        plus ``alpha_damp[B, K, P]``.
        """
        cfg = self.cfg
        n = self.harmonic
        cond = cond[:, None]
        ctx = ctx or {}
        zero = torch.zeros_like(u)

        cents = self.prior_cents[ki] + bounded(self.raw_cents[ki], 30.0) + bounded(self.cond_cents[cond], 30.0)
        f0 = 440.0 * torch.pow(2.0, (ki + LOWEST_MIDI - 69).float() / 12 + cents / 1200)
        B = torch.exp(self.prior_log_B[ki] + bounded(self.raw_log_B[ki], 1.5))
        fn = f0[..., None] * n * torch.sqrt(1 + B[..., None] * n**2)  # [B,K,P]

        detune = torch.cat([torch.zeros_like(ki, dtype=fn.dtype)[..., None], bounded(self.raw_unison[ki], 5.0)], -1)
        freq = fn[..., None] * torch.pow(2.0, detune[..., None, :] / 1200)  # [B,K,P,M]

        b1 = torch.exp(self.prior_log_b1[ki] + bounded(self.raw_log_b1[ki], 1.5))
        b3 = torch.exp(math.log(3e-7) + bounded(self.raw_log_b3[ki], 2.0))
        alpha_after = (b1[..., None] + b3[..., None] * fn**2) * torch.exp(ctx.get("log_decay", zero))[..., None]
        prompt = 1 + torch.exp(math.log(2.0) + bounded(self.raw_prompt[ki], 2.0))
        alpha = torch.cat([(alpha_after * prompt[..., None])[..., None],
                           alpha_after[..., None].expand(*alpha_after.shape, cfg.n_modes - 1)], -1)

        u0 = u - 0.6
        log_fc = (self.prior_log_fc[ki] + bounded(self.raw_log_fc[ki], 1.5) + bounded(self.cond_log_fc[cond], 1.0)
                  + 2.0 * torch.exp(bounded(self.raw_fc_vel[ki], 1.0)) * u0
                  + soft * self.soft_log_fc + ctx.get("log_fc", zero))
        rolloff = 1.5 * torch.exp(bounded(self.raw_rolloff[ki], 1.0))
        hammer = torch.pow(1 + (fn / log_fc.exp()[..., None]) ** 2, -rolloff[..., None] / 2)
        x0 = 0.12 * torch.exp(bounded(self.raw_strike[ki], 0.7))
        comb = torch.sin(math.pi * n * x0[..., None]).abs() + 1e-4
        gain_db = (self.gain_db[ki] + bounded(self.cond_gain_db[cond], 12.0)
                   + (40.0 * torch.exp(bounded(self.raw_vel_slope[ki], 0.7)) + bounded(self.cond_vel_slope[cond], 10.0)) * u0
                   + soft * self.soft_gain_db + ctx.get("gain_db", zero))
        base = torch.pow(10.0, gain_db / 20)[..., None] * hammer * comb * torch.exp(bounded(self.partial_gain[ki], 3.0))
        after = torch.exp(math.log(0.2) + bounded(self.raw_after[ki], 2.0)) * torch.exp(soft * self.soft_log_after)[..., None]
        amp = base[..., None] * torch.cat([torch.ones_like(after[..., :1]), after], -1)[..., None, :]

        n_active = self.n_strings[ki].clamp(min=2)  # monochords still have two polarisations
        mode_ok = torch.arange(cfg.n_modes, device=ki.device) < n_active[..., None]
        ok = (freq < 0.48 * cfg.sample_rate) & mode_ok[..., None, :]

        damp = self.has_damper[ki] * torch.exp(math.log(25.0) + bounded(self.raw_log_damp[ki], 1.5))
        tilt = 0.3 + bounded(self.raw_damp_tilt[ki], 0.3)
        alpha_damp = damp[..., None] * (fn / fn[..., :1]) ** tilt[..., None]

        return {"freq": freq, "alpha": alpha, "amp": amp * ok, "alpha_damp": alpha_damp}

    def regularizer(self) -> torch.Tensor:
        """Keep per-key tables smooth across the keyboard and the residual table small."""
        def smooth(x):
            return ((x[2:] - 2 * x[1:-1] + x[:-2]) ** 2).mean()
        tables = [self.raw_log_B, self.raw_cents, self.raw_log_b1, self.raw_log_b3, self.raw_prompt,
                  self.raw_strike, self.raw_log_fc, self.raw_fc_vel, self.raw_rolloff, self.raw_vel_slope,
                  self.raw_log_damp, self.raw_damp_tilt, self.gain_db / 20]
        return sum(smooth(t) for t in tables) + 1e-2 * (self.partial_gain**2).mean()
