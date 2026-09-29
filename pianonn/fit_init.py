"""Initialise a condition's recording chain from its recordings, before gradient descent.

Spectral-loss gradients are informative only close to the answer: a sinusoid's frequency only
within about an FFT bin, an onset's timing only within a window hop. So what gradient descent is
bad at finding from far away is estimated directly, by comparing renders of the untrained prior
with the recordings of the same MIDI:

1. latency: cross-correlation of onset-strength curves (mic distance + MIDI/audio offset);
2. tuning: cross-correlation of log-frequency spectra (reference pitch of that year's piano);
3. level and long-term spectrum per channel: the microphone gain, and a smooth EQ folded into
   the body FIR (third-octave, +-20 dB);
4. the stationary noise floor per channel and band (low percentile of the recordings' band energy).

Each is logged, so the per-year recording chain is also a measurement.
"""

import math

import numpy as np
import torch

from .dsp import bounded
from .losses import _mag
from .room import _minimum_phase, _octave_smooth


def _onset_strength(x, n_fft=1024, hop=48):
    S = torch.log(_mag(x, n_fft, hop) + 1e-4)
    return (S[..., 1:] - S[..., :-1]).clamp(min=0).sum(-2)  # [N, frames]


def _xcorr_lag(a, b, max_lag):
    """Lag (in frames) by which ``b`` must be delayed to line up with ``a``, summed over rows, with parabolic refinement."""
    a = a - a.mean(-1, keepdim=True)
    b = b - b.mean(-1, keepdim=True)
    lags = torch.arange(-max_lag, max_lag + 1)
    T = a.shape[-1]
    score = torch.stack([(a[..., max_lag: T - max_lag] * b[..., max_lag - l: T - max_lag - l]).sum() for l in lags.tolist()])
    i = int(score.argmax())
    frac = 0.0
    if 0 < i < len(lags) - 1:
        y0, y1, y2 = score[i - 1], score[i], score[i + 1]
        den = y0 - 2 * y1 + y2
        frac = float(0.5 * (y0 - y2) / den) if den != 0 else 0.0
    return float(lags[i]) + frac, score


def _log_f_spectrum(x, sr, f_lo=150.0, f_hi=4000.0, n_fft=16384):
    """Welch power spectrum on a 1-cent log-frequency grid, detrended over +-50 cents: ``[N, cents]``."""
    P = (_mag(x, n_fft, n_fft // 4) ** 2).mean(-1)  # [N, bins]
    f = torch.arange(P.shape[-1], device=x.device) * sr / n_fft
    grid = f_lo * 2 ** (torch.arange(int(1200 * math.log2(f_hi / f_lo)), device=x.device) / 1200)
    pos = grid / (sr / n_fft)
    i0 = pos.floor().long()
    w = pos - i0
    L = torch.log(P + 1e-12)
    y = L[..., i0] * (1 - w) + L[..., i0 + 1] * w
    k = torch.ones(1, 1, 101, device=x.device) / 101
    trend = torch.nn.functional.conv1d(y[:, None], k, padding=50)[:, 0]
    return y - trend


def _band_matrix(model, n_fft):
    """The noise bank's band masks on an ``n_fft``-point grid, normalised to white-equivalent power."""
    masks = model.noise.band_masks(n_fft)  # [bands, bins]
    return masks / (masks.sum(-1, keepdim=True) * 0.375 * n_fft)


@torch.no_grad()
def initialise_from_data(model, batches, log=print, max_latency=0.06, max_cents=60):
    """``batches``: collated examples of one condition, on the model's device. Returns a dict of estimates."""
    cfg = model.cfg
    sr = cfg.sample_rate
    cond = int(batches[0]["condition"][0])
    s = int(batches[0]["loss_start"][0])

    def render():
        return [model(b, b["audio"].shape[-1], residual=False, floor=False)["audio"][..., s:] for b in batches]

    targets = [b["audio"][..., s:] for b in batches]
    mono = lambda xs: torch.cat([x.mean(1) for x in xs])
    est = {}

    # 1. latency: shift the body FIRs (all channels) so the onsets line up
    hop = 48
    lag, _ = _xcorr_lag(_onset_strength(mono(targets)), _onset_strength(mono(render())), int(max_latency * sr / hop))
    shift = int(round(lag * hop))
    body = model.room.body.data[cond]
    if shift > 0:
        body.copy_(torch.cat([body.new_zeros(*body.shape[:-1], shift), body[..., :-shift]], -1))
    elif shift < 0:
        body.copy_(torch.cat([body[..., -shift:], body.new_zeros(*body.shape[:-1], -shift)], -1))
    est["latency_ms"] = 1000 * shift / sr
    log(f"init: latency {est['latency_ms']:+.1f} ms (model {'delayed' if shift > 0 else 'advanced'} to match)")

    # 2. tuning: one offset in cents for the condition
    lt, lm = _log_f_spectrum(mono(targets), sr), _log_f_spectrum(mono(render()), sr)
    cents, _ = _xcorr_lag(lt, lm, max_cents)
    cur = float(bounded(model.physics.cond_cents.data[cond], 30.0))
    new = max(-29.0, min(29.0, cur + cents))
    model.physics.cond_cents.data[cond] = 30.0 * math.atanh(new / 30.0)
    est["tuning_cents"] = new
    log(f"init: tuning {new:+.1f} cents re the prior's A440 stretch")

    # 3. level and long-term spectrum per channel -> mic gain + body EQ
    n_fft = 2048
    tgt, pred = torch.cat(targets), torch.cat(render())
    Pt = torch.stack([(_mag(tgt[:, c], n_fft) ** 2).mean((0, -1)) for c in range(cfg.channels)])
    Pm = torch.stack([(_mag(pred[:, c], n_fft) ** 2).mean((0, -1)) for c in range(cfg.channels)])
    freqs = torch.fft.rfftfreq(n_fft, 1 / sr).to(Pt.device).double()
    band = (freqs >= 100) & (freqs <= 5000)
    est["level_db"], est["eq_db"] = [], []
    for c in range(cfg.channels):
        ratio = _octave_smooth(Pt[c].double(), freqs) / _octave_smooth(Pm[c].double(), freqs).clamp(min=1e-30)
        eq_db = 10 * torch.log10(ratio.clamp(min=1e-30))
        level = float(eq_db[band].mean())
        eq_db = (eq_db - level).clamp(-20, 20)
        eq_db = torch.where(freqs < 30, eq_db[(freqs >= 30).nonzero()[0, 0]], eq_db)
        eq_db = torch.where(freqs > 11000, eq_db[(freqs <= 11000).nonzero()[-1, 0]], eq_db)
        # min-phase EQ on the body's FFT grid, applied by convolution (truncated to the body length)
        Lb = body.shape[-1]
        nb = 2 * Lb
        fb = torch.fft.rfftfreq(nb, 1 / sr).to(Pt.device).double()
        g = torch.from_numpy(np.interp(fb.cpu().numpy(), freqs.cpu().numpy(), eq_db.cpu().numpy())).to(Pt.device)
        H = _minimum_phase(g * math.log(10) / 20)
        body[c] = torch.fft.irfft(torch.fft.rfft(body[c].double(), nb) * H, nb)[:Lb].to(body.dtype)
        model.room.mic_gain_db.data[cond, c] += level
        est["level_db"].append(level)
        est["eq_db"].append({int(f): round(float(eq_db[(freqs - f).abs().argmin()]), 1) for f in (63, 125, 250, 500, 1000, 2000, 4000, 8000)})
    log(f"init: mic gain {['%+.1f dB' % v for v in est['level_db']]}; body EQ (dB) {est['eq_db']}")

    # 4. noise floor: 1st percentile of the recordings' band energy, 6 dB under (music is rarely absent from the
    # mid bands, so a higher percentile overestimates the floor; the level is learned from here)
    M = _band_matrix(model, n_fft).to(Pt.device)
    for c in range(cfg.channels):
        E = torch.einsum("kf,nft->nkt", M, _mag(tgt[:, c], n_fft, n_fft // 4) ** 2)  # [N, bands, frames]
        q = torch.quantile(E.permute(1, 0, 2).reshape(E.shape[1], -1), 0.01, dim=-1)
        model.room.floor_db.data[cond, c] = 10 * torch.log10(q.clamp(min=1e-14)) - 6.0
    fl = model.room.floor_db.data[cond].mean(0)
    est["floor_db"] = [round(float(v), 1) for v in fl[:: max(1, len(fl) // 8)]]
    log(f"init: noise floor (white-equivalent dBFS, every 4th band) {est['floor_db']}")
    return est
