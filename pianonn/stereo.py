"""How the two microphone channels relate (docs/physics_revamp.md 14): the measures of the stereo image.

Every function takes stereo audio ``x[T, 2]`` (left, right). Conventions: the cross spectrum is ``X_L conj(X_R)``, so
a left channel that lags the right by ``tau`` reads a phase of ``-360 f tau`` degrees and a GCC lag of ``+tau``.

- ``band_stereo``: per segment and band, the magnitude-squared coherence (power-weighted over the band's bins), the
  coherent part's phase (the circular mean of the cross spectrum's angle, weighted by its magnitude) and the level
  difference. A coherence estimated from K frames reads about 1 / K_eff for independent channels (``msc_bias``).
- ``gcc_phat``: the lag of the coherent part (ms) and the height of its peak.
- ``partial_coherence``: per partial of a note, the coherence over a few frames at the partial's bin.
- ``lr_motion``: per partial, how much its left/right relation moves within the note (sd of the level difference, dB;
  circular sd of the phase difference, degrees) and the partial's power over its surroundings (dB): noise in the bin
  moves the relation too, so compare at matched surroundings.

A stationary sound through any two fixed filters stays coherent (1) with a fixed level and phase difference; what
lowers the coherence and moves the relation is several components within a bin reaching the two channels in
different proportions (beating unison modes, a reverberant field whose response changes within ~1 Hz) or a diffuse
field.
"""

import math

import numpy as np

BANDS = ((250, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000))


def _frames(x, n, hop):
    idx = np.arange(0, len(x) - n + 1, hop)
    return np.stack([x[i: i + n] for i in idx]) if len(idx) else np.zeros((0, n, x.shape[1]))


def band_stereo(x, sr, bands=BANDS, seg=0.5, hop=0.25, nf=2048):
    """Per segment of ``seg`` s (hop ``hop``) and band: ``msc``, ``phase`` (deg), ``ild`` (dB, left re right) and
    ``power`` (mean power over the channels, for weighting) as ``[S, bands]`` arrays; frames of ``nf`` points, hop
    ``nf // 2`` within each segment."""
    n_seg, h_seg = int(seg * sr), int(hop * sr)
    win = np.hanning(nf)[:, None]
    f = np.fft.rfftfreq(nf, 1 / sr)
    sel = [(f >= lo) & (f < hi) for lo, hi in bands]
    out = {k: [] for k in ("msc", "phase", "ild", "power")}
    for s0 in range(0, len(x) - n_seg + 1, h_seg):
        S = np.fft.rfft(_frames(x[s0: s0 + n_seg], nf, nf // 2) * win[None], axis=1)  # [K, F, 2]
        P = (np.abs(S) ** 2).mean(0)
        cross = (S[..., 0] * np.conj(S[..., 1])).mean(0)
        msc = np.abs(cross) ** 2 / (P[:, 0] * P[:, 1] + 1e-30)
        row = {k: [] for k in out}
        for b in sel:
            pm = P[b].sum(1)
            row["msc"].append(float((msc[b] * pm).sum() / (pm.sum() + 1e-30)))
            row["phase"].append(float(np.degrees(np.angle(np.sum(cross[b])))))
            row["ild"].append(float(10 * np.log10((P[b, 0].sum() + 1e-30) / (P[b, 1].sum() + 1e-30))))
            row["power"].append(float(pm.mean() / 2))
        for k in out:
            out[k].append(row[k])
    return {k: np.array(v) for k, v in out.items()}


def msc_bias(sr, seg=0.5, nf=2048):
    """The coherence ``band_stereo`` reads for independent channels: 1 / K_eff for its K half-overlapping Hann frames
    (Welch: overlap correlation of a Hann window at 50 % is 1/6 in power, ~0.167)."""
    K = (int(seg * sr) - nf) // (nf // 2) + 1
    rho = 0.1667
    return 1.0 / (K / (1 + 2 * rho ** 2 * (K - 1) / K))


def gcc_phat(x, sr, max_ms=3.0, lo=200.0, hi=8000.0, up=16):
    """The lag (ms; positive: the left channel lags) and peak height of the phase transform of the cross spectrum
    over ``lo``-``hi`` Hz, interpolated ``up`` times."""
    n = 1 << int(math.ceil(math.log2(len(x))))
    X = np.fft.rfft(x, n, axis=0)
    f = np.fft.rfftfreq(n, 1 / sr)
    G = X[:, 0] * np.conj(X[:, 1])
    G = np.where((f >= lo) & (f <= hi), G / (np.abs(G) + 1e-20), 0)
    r = np.fft.irfft(G, n * up)
    m = int(max_ms * 1e-3 * sr * up)
    lags = np.concatenate([r[: m + 1], r[-m:]])
    lag_i = np.concatenate([np.arange(m + 1), np.arange(-m, 0)])
    j = int(np.argmax(lags))
    full = 2 * np.sum((f >= lo) & (f <= hi)) / (n * up)  # the peak of identical channels
    return 1000 * lag_i[j] / (sr * up), float(lags[j] / full)


def partial_coherence(xs, sr, t0, t1, freqs, n_frames=7):
    """Per partial ``freqs[K]`` (Hz; NaN skipped): the coherence over ``n_frames`` half-overlapping Hann frames
    spanning ``[t0, t1]`` s, at the bin nearest the partial (frames zero-padded x4)."""
    i0, i1 = int(round(t0 * sr)), int(round(t1 * sr))
    nfr = (i1 - i0) * 2 // (n_frames + 1)
    ff = np.fft.rfftfreq(nfr * 4, 1 / sr)
    S = np.array([np.fft.rfft(xs[i0 + q * nfr // 2: i0 + q * nfr // 2 + nfr] * np.hanning(nfr)[:, None], nfr * 4, axis=0)
                  for q in range(n_frames)])  # [K, F, 2]
    out = np.full(len(freqs), np.nan)
    for k, fk in enumerate(freqs):
        if not np.isfinite(fk) or fk > 0.45 * sr:
            continue
        b = int(np.argmin(np.abs(ff - fk)))
        L, R = S[:, b, 0], S[:, b, 1]
        out[k] = np.abs(np.sum(L * np.conj(R))) ** 2 / (np.sum(np.abs(L) ** 2) * np.sum(np.abs(R) ** 2) + 1e-30)
    return out


def lr_motion(xs, sr, t0, t1, freqs, f1=None, win=0.04, hop=0.01):
    """Per partial ``freqs[K]``: over ``win`` s Hann frames every ``hop`` s in ``[t0, t1]``, the sd of the left/right
    level difference (dB), the circular sd of the phase difference (deg) and the partial's power over the power half
    way to its neighbours (dB; ``f1`` the spacing, default ``freqs[0]``). Returns three ``[K]`` arrays."""
    n = int(round(win * sr))
    w = np.hanning(n)
    f1 = freqs[0] if f1 is None else f1
    starts = [int(round(c * sr)) - n // 2 for c in np.arange(t0, t1, hop)]
    starts = [s for s in starts if s >= 0 and s + n <= len(xs)]
    lev, ph, snr = (np.full(len(freqs), np.nan) for _ in range(3))
    if len(starts) < 10:
        return lev, ph, snr
    seg = np.stack([xs[s: s + n] for s in starts])  # [M, n, 2]
    tt = np.arange(n) / sr

    def proj(f):
        return np.einsum("mnc,n->mc", seg, np.exp(-2j * np.pi * f * tt) * w)

    for k, f in enumerate(freqs):
        if not np.isfinite(f) or f > 0.45 * sr:
            continue
        A = proj(f)
        r = A[:, 0] / (A[:, 1] + 1e-12)
        lev[k] = np.std(20 * np.log10(np.abs(r) + 1e-12))
        u = np.exp(1j * np.angle(r))
        ph[k] = np.degrees(np.sqrt(-2 * np.log(max(abs(np.mean(u)), 1e-6))))
        side = [proj(f + d) for d in (-0.5 * f1, 0.5 * f1)]
        snr[k] = 10 * np.log10(np.mean(np.abs(A) ** 2) / (np.mean([np.mean(np.abs(s) ** 2) for s in side]) + 1e-30))
    return lev, ph, snr


class CoherenceLoss:
    """A training term on the stereo image (docs/physics_revamp.md 14): per band of ``bands``, the inter-channel
    coherence of the render and of the recording on the same excerpts (``band_stereo``'s, in torch: ``nf``-point Hann
    frames, hop ``nf // 2``, grouped ``frames`` at a time, power-weighted over the band's bins), the mean over the
    batch's groups of render - recording, and its absolute value averaged over the bands. Pooled over the batch, since a
    group's coherence from ~10 frames is noisy on both sides (``msc_bias``): the term asks for the recordings' coherence
    on average, not frame by frame. The recording's side carries no gradient."""

    def __init__(self, sr, bands=BANDS, nf=2048, frames=10):
        self.sr, self.bands, self.nf, self.frames = sr, bands, nf, frames
        self._masks = None

    def msc(self, x):
        """The coherence ``[B, G, bands]`` per group of ``frames`` frames of each excerpt of ``x[B, 2, T]``."""
        import torch

        B, ch, T = x.shape
        win = torch.hann_window(self.nf, device=x.device)
        S = torch.stft(x.reshape(B * ch, T), self.nf, self.nf // 2, window=win, center=False, return_complex=True)
        S = S.view(B, ch, S.shape[-2], S.shape[-1])  # [B, 2, F, K]
        G = S.shape[-1] // self.frames
        S = S[..., : G * self.frames].reshape(B, ch, S.shape[-2], G, self.frames)
        P = (S.abs() ** 2).mean(-1)  # [B, 2, F, G]
        cross = (S[:, 0] * S[:, 1].conj()).mean(-1)  # [B, F, G]
        m = cross.abs() ** 2 / (P[:, 0] * P[:, 1] + 1e-20)
        if self._masks is None or self._masks.device != x.device:
            f = torch.fft.rfftfreq(self.nf, 1 / self.sr).to(x.device)
            self._masks = torch.stack([((f >= lo) & (f < hi)).float() for lo, hi in self.bands])  # [bands, F]
        w = P.sum(1)  # [B, F, G]
        num = torch.einsum("bfg,nf->bgn", m * w, self._masks)
        den = torch.einsum("bfg,nf->bgn", w, self._masks)
        return num / den.clamp(min=1e-20)

    def __call__(self, pred, target):
        import torch

        with torch.no_grad():
            t = self.msc(target.float())
        d = (self.msc(pred.float()) - t).mean((0, 1))  # [bands]
        return d.abs().mean(), d.detach()
