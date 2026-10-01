"""Soundboard body and hall: bridge force -> sound pressure at the microphones.

One set per recording condition (MAESTRO year):

* ``body``: a learnable FIR (default 0.3 s) initialised as a soundboard. It has
  sparse modes with loss factor 0.02 below 1.1 kHz, a diffuse short response
  above, the radiation high-pass around 60-70 Hz and a 3 ms pre-delay.
* ``hall``: a parametric tail. Fixed octave-band noise carriers are shaped by a
  learnable T60 and gain per band, which is far more identifiable than tens of
  thousands of free FIR taps.
* ``floor``: the stationary noise at the microphones (hall, audience, preamps) plus mains-hum lines,
  measured on the silence before each piece's first note and refined within +-3 dB.

The hall hears what the soundboard radiates, so the two are in series:
``ir = body * (delta + hall)`` (convolution). The radiation high-pass of the
body therefore also shapes the reverberant field.

Everything is per microphone channel: MAESTRO is a spaced stereo pair, and its two
channels correlate at only 0.2-0.4. There are two body FIRs and two decorrelated hall
tails per condition, a gain per channel, a per-key channel balance, and a learned
stationary noise floor at the microphones.
"""

import math

import torch
from torch import nn

from .dsp import bounded, fft_convolve

BODY_TARGET_DB = [(20, -40), (30, -30), (40, -20), (55, -10), (70, -4), (100, 0), (1000, 0), (2000, -3),
                  (4000, -6), (8000, -11), (11000, -15)]  # HF droop: Wogram, Five Lectures, Fig. 5 (upright; low confidence)
GRAND_LOW_MODES = [62.0, 90.0, 105.0, 127.0, 187.0, 222.0, 245.0, 325.0]  # 2.90 m concert grand (Wogram, modal.html)
HALL_BANDS = [125, 250, 500, 1000, 2000, 4000, 8000]
HALL_T60 = [2.0, 1.8, 1.7, 1.6, 1.45, 1.2, 0.8]
Q_BANDS = [31.25, 62.5, 125, 250, 500, 1000, 2000, 4000, 8000]  # octave bands of the body's Q cap
FLOOR_BOUND_DB = 3.0  # the floor is a measurement (leading silence); training may refine it by this much
MAINS_HZ = 60.0  # MAESTRO: the Piano-e-Competition, Minneapolis (US mains); hum at 60, 120, 180 Hz
HUM_LINES = 3


def _interp_log_f(points, f):
    xs = torch.log(torch.tensor([p[0] for p in points], dtype=torch.float64))
    ys = torch.tensor([p[1] for p in points], dtype=torch.float64)
    lf = torch.log(f.clamp(min=1.0))
    idx = torch.searchsorted(xs, lf).clamp(1, len(xs) - 1)
    w = ((lf - xs[idx - 1]) / (xs[idx] - xs[idx - 1])).clamp(0, 1)
    return ys[idx - 1] + w * (ys[idx] - ys[idx - 1])


def _octave_smooth(power, freqs, width=1 / 3):
    """Average a power spectrum over a moving band ``width`` octaves wide."""
    lo = torch.searchsorted(freqs, freqs * 2 ** (-width / 2))
    hi = torch.searchsorted(freqs, freqs * 2 ** (width / 2), right=True).clamp(min=1)
    hi = torch.maximum(hi, lo + 1)
    c = torch.cat([power.new_zeros(1), torch.cumsum(power, 0)])
    return (c[hi] - c[lo]) / (hi - lo)


def _minimum_phase(log_mag):
    """Minimum-phase spectrum with the given log magnitude (real-cepstrum folding)."""
    n = 2 * (log_mag.shape[-1] - 1)
    cep = torch.fft.irfft(log_mag.to(torch.float64), n)
    fold = torch.zeros_like(cep)
    fold[0] = cep[0]
    fold[1: n // 2] = 2 * cep[1: n // 2]
    fold[n // 2] = cep[n // 2]
    return torch.exp(torch.fft.rfft(fold))


def soundboard_body(sr, seconds, seed=0, eta=0.02, predelay=0.003, crossover=1350.0):
    """Initial soundboard + case impulse response (see docs/physical_parameters.md, section 4)."""
    g = torch.Generator().manual_seed(seed)
    L = int(seconds * sr)
    t = torch.arange(L, dtype=torch.float64) / sr

    # measured low modes of a concert grand, then ~0.07 modes/Hz (Steinway D model, Boutillon et al.)
    # up to the plate/rib-strip transition (~1.35 kHz for a D); T60 = 2.2 / (eta f)
    n_random = int(0.07 * (crossover - 330))
    f = torch.cat([torch.tensor(GRAND_LOW_MODES, dtype=torch.float64),
                   330 + (crossover - 330) * torch.rand(n_random, generator=g, dtype=torch.float64)])
    f = torch.sort(f).values
    n_modes = len(f)
    tau = (1 / (math.pi * eta * f)).clamp(max=0.1)  # finished grand soundboard T60 ~0.6 s (Bader & Plath 2020)
    phase = 2 * math.pi * torch.rand(n_modes, generator=g, dtype=torch.float64)
    amp = torch.randn(n_modes, generator=g, dtype=torch.float64)
    modal = (amp[:, None] * torch.exp(-t / tau[:, None]) * torch.sin(2 * math.pi * f[:, None] * t + phase[:, None])).sum(0)

    # diffuse response above: noise with the same frequency-dependent decay
    n_fft, hop = 512, 64
    noise = torch.randn(L, generator=g, dtype=torch.float64)
    win = torch.hann_window(n_fft, dtype=torch.float64)
    S = torch.stft(noise, n_fft, hop, window=win, return_complex=True)
    fb = torch.fft.rfftfreq(n_fft, 1 / sr).to(torch.float64)
    tf = torch.arange(S.shape[-1], dtype=torch.float64) * hop / sr
    tau_b = 1 / (math.pi * eta * fb.clamp(min=crossover * 0.8))
    xfade = torch.clamp((torch.log2(fb.clamp(min=1)) - math.log2(crossover * 0.8)) / math.log2(1.6), 0, 1)
    diffuse = torch.istft(S * (xfade[:, None] * torch.exp(-tf[None] / tau_b[:, None])), n_fft, hop, window=win, length=L)

    h = modal / modal.abs().max() + diffuse / diffuse.abs().max()

    # impose the target envelope (1/3-octave smoothed) with a causal minimum-phase correction
    n = 2 * L
    H = torch.fft.rfft(h, n)
    freqs = torch.fft.rfftfreq(n, 1 / sr).to(torch.float64)
    smooth = _octave_smooth(H.abs() ** 2, freqs).clamp(min=1e-20).sqrt()
    target = 10 ** (_interp_log_f(BODY_TARGET_DB, freqs) / 20)
    correction = _minimum_phase(torch.log(target / smooth))
    h = torch.fft.irfft(H * correction, n)[:L]

    h = torch.cat([torch.zeros(int(predelay * sr), dtype=torch.float64), h])[:L]
    Hn = torch.fft.rfft(h, n).abs()
    plateau = Hn[(freqs >= 200) & (freqs <= 1000)].mean()
    return (h / plateau).float()


def octave_masks(n, sr, bands):
    """Raised-cosine crossover masks ``[bands, n//2+1]`` in log f between the band centres (the first open below,
    the last open above); they sum to one."""
    lf = torch.log2(torch.fft.rfftfreq(n, 1 / sr).clamp(min=1.0).to(torch.float64))
    centers = torch.log2(torch.tensor(bands, dtype=torch.float64))
    masks = []
    for i, c in enumerate(centers):
        m = torch.ones_like(lf)
        if i > 0:  # rising edge from the previous centre
            m = torch.where(lf < c, torch.sin(0.5 * math.pi * ((lf - centers[i - 1]) / (c - centers[i - 1])).clamp(0, 1)) ** 2, m)
        if i < len(centers) - 1:
            m = torch.where(lf > c, torch.cos(0.5 * math.pi * ((lf - c) / (centers[i + 1] - c)).clamp(0, 1)) ** 2, m)
        masks.append(m)
    return torch.stack(masks)


def band_values(v, name):
    """A config value per band of ``Q_BANDS``: one number for all, or one per band (a list or "/"-separated string)."""
    if isinstance(v, str):
        v = [float(x) for x in v.split("/") if x.strip()]
    v = [float(v)] * len(Q_BANDS) if isinstance(v, (int, float)) else [float(x) for x in v]
    assert len(v) == len(Q_BANDS), f"{name}: one value or one per band of {Q_BANDS}"
    return torch.tensor(v, dtype=torch.float64)


def ring_taus(cfg):
    """The ring-up time constants (s) per band of ``Q_BANDS`` from config ``body_ring_ms``, or None when off."""
    t = band_values(cfg.body_ring_ms, "body_ring_ms") / 1000
    return t if bool((t > 0).any()) else None


def band_carriers(sr, seconds, seed=1):
    """White noise split into octave bands (raised-cosine crossovers, bands sum to the original)."""
    g = torch.Generator().manual_seed(seed)
    L = int(seconds * sr)
    X = torch.fft.rfft(torch.randn(L, generator=g, dtype=torch.float64))
    return torch.stack([torch.fft.irfft(X * m, L) for m in octave_masks(L, sr, HALL_BANDS)]).float()


class Room(nn.Module):
    """Body, hall, microphone chain and noise floor, one set per recording condition and channel."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        sr, C, ch = cfg.sample_rate, cfg.n_conditions, cfg.channels
        # one body FIR per microphone: each hears a different mixture of the soundboard's modes
        bodies = torch.stack([soundboard_body(sr, cfg.body_seconds, seed=c) for c in range(ch)])
        self.body = nn.Parameter(bodies.repeat(C, 1, 1))  # [C, ch, L]
        L = int(cfg.hall_seconds * sr)
        # decorrelated tails per channel (a diffuse field), same T60s and band levels
        self.register_buffer("carriers", torch.stack([band_carriers(sr, cfg.hall_seconds, seed=1 + c) for c in range(ch)]))
        t = torch.arange(L) / sr
        self.register_buffer("t", t)
        # tail builds up between 10 and 40 ms after the direct sound
        self.register_buffer("ramp", torch.sin(0.5 * math.pi * ((t - 0.010) / 0.030).clamp(0, 1)) ** 2)
        self.register_buffer("prior_log_t60", torch.log(torch.tensor(HALL_T60)))
        self.raw_log_t60 = nn.Parameter(torch.zeros(C, len(HALL_BANDS)))
        self.band_log_gain = nn.Parameter(torch.zeros(C, len(HALL_BANDS)))
        # initial direct-to-reverberant ratio ~0 dB: energy of body * hall = body energy
        with torch.no_grad():
            hall = self._hall(torch.zeros(1, dtype=torch.long), gain=torch.zeros(1, ch))[:, 0]
            ratio = bodies[0].pow(2).sum() / fft_convolve(torch.cat([bodies[0], bodies.new_zeros(L)])[None], hall).pow(2).sum()
        self.log_gain = nn.Parameter(torch.full((C, ch), 0.5 * math.log(ratio.item())))
        # microphone chain gain (dB, unbounded): initialised from the recordings' level before training
        self.mic_gain_db = nn.Parameter(torch.zeros(C, ch))
        # per-key level difference between the channels (dB, +-6): where along the bridge each key radiates
        self.raw_pan = nn.Parameter(torch.zeros(C, 88))
        # stationary noise floor at the microphones (hall, audience, preamps), white-equivalent dBFS per band:
        # a model that renders digital silence is otherwise scored against the recordings' floor in every
        # quiet bin (review 3, F1). The reference is measured on the recordings' leading silence
        # (fit_init.floor_from_silence); training refines it within +-FLOOR_BOUND_DB. Unbounded, the trial's
        # floor was the fastest-moving parameter of the recording chain (review 4, section 4).
        self.register_buffer("floor_log2_centers", torch.linspace(math.log2(40.0), math.log2(sr / 2), cfg.noise_bands))
        self.register_buffer("floor_ref_db", torch.full((C, ch, cfg.noise_bands), -90.0))
        self.raw_floor = nn.Parameter(torch.zeros(C, ch, cfg.noise_bands))
        # mains hum: sinusoids at MAINS_HZ x (1, 2, 3), RMS dBFS per channel (-200 = none until measured)
        self.register_buffer("hum_ref_db", torch.full((C, ch, HUM_LINES), -200.0))
        self.raw_hum = nn.Parameter(torch.zeros(C, ch, HUM_LINES))
        self._ring_setup(cfg)
        q = band_values(cfg.body_q_max, "body_q_max")
        self.q_on = bool((q > 0).any())
        if self.q_on:
            Lb = self.body.shape[-1]
            self.register_buffer("q_masks", octave_masks(2 * Lb, sr, Q_BANDS).float(), persistent=False)
            # a band left at 0 is not capped
            sigma = torch.where(q > 0, math.pi * torch.tensor(Q_BANDS, dtype=torch.float64) / q.clamp(min=1e-9), torch.zeros_like(q))
            self.register_buffer("q_sigma", sigma.float(), persistent=False)

    def _ring_setup(self, cfg):
        """Buffers of the board's ring-up (config ``body_ring_ms``): ``ring_kernel [ch, Lk]`` and ``ring_power [G]``,
        the kernels' power over frequency (mean over the channels, on a fine grid of ``ring_df`` Hz), or None."""
        tau = ring_taus(cfg)
        self.ring_on = tau is not None
        if not self.ring_on:
            return
        sr, ch = cfg.sample_rate, cfg.channels
        Lk = int(round(min(0.25, 6 * float(tau.max())) * sr))
        n_k = 4 * Lk  # the band split is zero-phase: compute long, keep t >= 0 (its pre-ringing is ~1 / bandwidth)
        t = torch.arange(Lk, dtype=torch.float64) / sr
        masks = octave_masks(n_k, sr, Q_BANDS).double()  # [bands, n_k // 2 + 1]
        kernels = []
        for c in range(ch):
            g = torch.Generator().manual_seed(4321 + c)
            noise = torch.randn(Lk, generator=g, dtype=torch.float64)
            K = torch.ones(n_k // 2 + 1, dtype=torch.complex128)  # a delta, exact where no band rings
            for b, tb in enumerate(tau.tolist()):
                if tb <= 0:
                    continue
                Nb = torch.fft.rfft(noise * torch.exp(-t / tb), n_k)
                band = masks[b] > 0.5
                K = K + masks[b] * (Nb / Nb[band].abs().pow(2).mean().sqrt() - 1)  # unit mean power in its band
            kernels.append(torch.fft.irfft(K, n_k)[:Lk])
        k = torch.stack(kernels)
        n = 1 << 19
        self.ring_df = sr / n
        power = (torch.fft.rfft(k, n).abs() ** 2).mean(0)
        self.register_buffer("ring_kernel", k.float(), persistent=False)
        self.register_buffer("ring_power", power.float(), persistent=False)

    def ring_gain(self, freq):
        """The kernels' power at ``freq`` (Hz, any shape; linear interpolation on the fine grid): what the ring-up adds
        to a partial at that frequency, to be divided out."""
        x = (freq / self.ring_df).clamp(0, self.ring_power.shape[0] - 2)
        i = x.floor().long()
        w = x - i
        return self.ring_power[i] * (1 - w) + self.ring_power[i + 1] * w

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        old = prefix + "floor_db"  # checkpoints before round 2: an unbounded learned floor
        if old in state_dict and prefix + "floor_ref_db" not in state_dict:
            v = state_dict.pop(old)
            state_dict[prefix + "floor_ref_db"] = v
            state_dict[prefix + "raw_floor"] = torch.zeros_like(v)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def floor_db(self, cond):
        """Floor band levels ``[B, ch, bands]`` (white-equivalent dBFS)."""
        return self.floor_ref_db[cond] + bounded(self.raw_floor[cond], FLOOR_BOUND_DB)

    def hum_db(self, cond):
        """Hum line levels ``[B, ch, lines]`` (RMS dBFS)."""
        return self.hum_ref_db[cond] + bounded(self.raw_hum[cond], FLOOR_BOUND_DB)

    def _hall(self, cond, gain=None):
        """Hall impulse responses ``[B, ch, L]``."""
        t60 = torch.exp(self.prior_log_t60 + bounded(self.raw_log_t60[cond], 0.7))  # [B,7]
        env = torch.exp(-6.91 * self.t / t60[..., None]) * torch.exp(self.band_log_gain[cond])[..., None]  # [B,7,L]
        gain = self.log_gain[cond] if gain is None else gain  # [B, ch]
        return (self.carriers[None] * env[:, None]).sum(2) * self.ramp * torch.exp(gain)[..., None]

    def limit_q(self, body):
        """The body FIRs ``[B, ch, L]`` with their ringing capped at Q = ``cfg.body_q_max`` (see the config): each
        octave band (zero-phase crossovers) is multiplied by exp(-pi f_band (t - t0) / Q) after the direct arrival
        t0 (the first sample at 10 % of the peak), then rescaled to its energy before."""
        L = body.shape[-1]
        sr = self.cfg.sample_rate
        parts = torch.fft.irfft(torch.fft.rfft(body, 2 * L)[..., None, :] * self.q_masks, 2 * L)  # [B, ch, bands, 2L]
        with torch.no_grad():
            a = body.abs()
            t0 = (a >= 0.1 * a.amax(-1, keepdim=True)).float().argmax(-1) / sr  # [B, ch]
        t = torch.arange(2 * L, device=body.device) / sr
        w = torch.exp(-self.q_sigma[:, None] * (t - t0[..., None, None]).clamp(min=0))  # [B, ch, bands, 2L]
        cut = parts * w
        g = (parts.pow(2).sum(-1) / cut.pow(2).sum(-1).clamp(min=1e-30)).sqrt()
        return (cut * g[..., None]).sum(-2)[..., :L]

    def forward(self, cond):
        """Impulse responses ``[B, ch, L]`` for conditions ``cond[B]``: mic gain x body * (delta + hall)."""
        hall = self._hall(cond)
        hall = torch.cat([hall[..., :1] + 1.0, hall[..., 1:]], -1)  # + delta: the direct sound
        body = self.body[cond]
        if self.q_on:
            body = self.limit_q(body)
        if self.ring_on:  # the board's ring-up: each channel's body through its own kernel
            body = fft_convolve(body, self.ring_kernel[None].expand(body.shape[0], -1, -1))[..., : body.shape[-1]]
        body = body * torch.pow(10.0, self.mic_gain_db[cond] / 20)[..., None]
        return fft_convolve(torch.cat([body, body.new_zeros(*body.shape[:2], hall.shape[-1])], -1), hall)

    def pan_gains(self, ki, cond):
        """Per-note channel gains ``[B, N, ch]`` (equal-power around 0 dB)."""
        if self.cfg.channels == 1:
            return torch.ones(*ki.shape, 1, device=ki.device)
        p = bounded(self.raw_pan[cond[:, None], ki], 6.0)
        return torch.stack([torch.pow(10.0, p / 40), torch.pow(10.0, -p / 40)], -1)

    def floor_noise(self, cond, n, generator=None):
        """Stationary noise ``[B, ch, n]`` with the condition's floor spectrum (raised-cosine interpolation of
        the band levels in log f, the same partition of unity as the noise bank's bands), plus the hum lines
        (random phase)."""
        B, ch = cond.shape[0], self.cfg.channels
        dev = self.floor_ref_db.device
        white = torch.randn(B, ch, n, generator=generator, device=dev)
        phase = 2 * math.pi * torch.rand(B, ch, HUM_LINES, 1, generator=generator, device=dev)
        X = torch.fft.rfft(white)
        f = torch.fft.rfftfreq(n, 1 / self.cfg.sample_rate).to(dev)
        c = self.floor_log2_centers
        pos = ((torch.log2(f.clamp(min=1.0)) - c[0]) / (c[1] - c[0])).clamp(0, len(c) - 1)
        i0 = pos.floor().long().clamp(max=len(c) - 2)
        w = torch.sin(0.5 * math.pi * (pos - i0)) ** 2
        power = torch.pow(10.0, self.floor_db(cond) / 10)  # [B, ch, bands]
        psd = power[..., i0] * (1 - w) + power[..., i0 + 1] * w
        t = torch.arange(n, device=dev, dtype=torch.float64) / self.cfg.sample_rate
        k = torch.arange(1, HUM_LINES + 1, device=dev, dtype=torch.float64)
        cyc = torch.frac(MAINS_HZ * k[:, None] * t).float()  # float64 cycle count: exact at any length
        amp = (2 * torch.pow(10.0, self.hum_db(cond) / 10)).sqrt()[..., None]  # [B, ch, lines, 1]
        hum = (amp * torch.sin(2 * math.pi * cyc + phase)).sum(2)
        return torch.fft.irfft(X * psd.sqrt(), n) + hum
