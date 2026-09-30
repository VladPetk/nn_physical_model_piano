"""Spectral reconstruction losses and an optional multi-resolution spectrogram discriminator."""

import math

import torch
import torch.nn.functional as F
from torch import nn


def _mag(x, n_fft, hop=None):
    window = torch.hann_window(n_fft, device=x.device)
    return torch.stft(x, n_fft, hop or n_fft // 4, window=window, return_complex=True).abs()


def highpass(x, sr, fc=20.0, width=10.0):
    """Zero-phase high-pass of ``x[..., T]`` (FFT domain, raised-cosine edge from ``fc - width/2`` to
    ``fc + width/2``). MAESTRO's recordings carry infrasonic rumble (0-20 Hz, ~5 % of their energy, the
    same with and without music): nothing a piano plays, so no loss should spend effort on it."""
    n = x.shape[-1]
    f = torch.fft.rfftfreq(n, 1 / sr).to(x.device)
    g = torch.sin(0.5 * math.pi * ((f - (fc - width / 2)) / width).clamp(0, 1)) ** 2
    return torch.fft.irfft(torch.fft.rfft(x) * g, n)


def log_f_band_masks(n_fft, sr, centers, open_low=True):
    """Raised-cosine (in log f) band masks ``[bands, n_fft//2+1]`` around ``centers``; they sum to one
    (everything above the last centre belongs to the last band, and with ``open_low`` everything below the
    first centre to the first; otherwise the first band rises from one band spacing below its centre)."""
    lf = torch.log2(torch.fft.rfftfreq(n_fft, 1 / sr).clamp(min=1.0))
    c = torch.log2(torch.as_tensor(centers, dtype=lf.dtype))
    masks = []
    for i in range(len(c)):
        m = torch.ones_like(lf)
        if i > 0 or not open_low:
            prev = c[i - 1] if i > 0 else 2 * c[0] - c[1]
            m = torch.where(lf < c[i], torch.sin(0.5 * math.pi * ((lf - prev) / (c[i] - prev)).clamp(0, 1)) ** 2, m)
        if i < len(c) - 1:
            m = torch.where(lf > c[i], torch.cos(0.5 * math.pi * ((lf - c[i]) / (c[i + 1] - c[i])).clamp(0, 1)) ** 2, m)
        masks.append(m)
    return torch.stack(masks)


class MultiResolutionSTFTLoss(nn.Module):
    """Spectral convergence + log-magnitude L1 over several resolutions (the loss of the first trial).

    Kept for continuity of the evaluation tables; training uses :class:`PianoLoss`. The spectral-convergence
    term is a least-squares fit of magnitudes, so wherever the fine structure is unpredictable (beating with
    unknown phase, the hall, noise) its optimum is ~1.8 dB too quiet, in every band and at every resolution
    (docs/plan_round2.md, 2.1).

    The log term's epsilon is the STFT magnitude of white noise at ``floor_db`` dBFS (so it scales with the
    window). Inputs are ``[..., T]``; leading dims (batch, channels) are flattened.
    """

    def __init__(self, fft_sizes=(4096, 2048, 1024, 512, 256, 128), floor_db=-80.0):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.floor_db = floor_db

    def eps(self, n):
        return 10 ** (self.floor_db / 20) * (0.375 * n) ** 0.5  # hann: sum(w^2) = 3n/8

    def forward(self, pred, target, per_resolution=False, per_example=False):
        """``per_example``: losses ``[B]`` averaged over channels instead of a scalar (inputs ``[B, ch, T]``)."""
        lead = pred.shape[:-1]
        pred, target = pred.reshape(-1, pred.shape[-1]), target.reshape(-1, target.shape[-1])
        terms = {}
        for n in self.fft_sizes:
            S, P = _mag(target, n), _mag(pred, n)
            eps = self.eps(n)
            sc = torch.linalg.norm(S - P, dim=(-2, -1)) / (torch.linalg.norm(S, dim=(-2, -1)) + eps)
            lm = (torch.log(S + eps) - torch.log(P + eps)).abs().mean((-2, -1))
            terms[n] = (sc + lm).reshape(lead).flatten(1).mean(1) if per_example else (sc + lm).mean()
        loss = sum(terms.values()) / len(self.fft_sizes)
        return (loss, terms) if per_resolution else loss


class PianoLoss(nn.Module):
    """The training loss of round 2 (docs/plan_round2.md): unbiased in level, sharp in time.

    Both signals are high-passed at 20 Hz first. Units are log10 of power (1 = 10 dB) throughout.

    * ``band`` (the backbone): L1 of log band energies in 1/6-octave raised-cosine bands from 30 Hz to 11 kHz,
      10 ms hop. The analysis window grows towards the bass (512 points above 1.6 kHz, 2048 down to 200 Hz,
      8192 below) so every band spans a few bins. Pooling bins before the log makes the optimum level the
      mean energy, not a per-bin median: on real excerpts this term's optimum is within +-0.9 dB of the
      energy match in every band, where spectral convergence sits 1.8 dB low (plan_round2.md, 2.1).
    * ``fine``: per-bin log-magnitude L1 at 4096 and 1024 points, only 20 Hz - 2 kHz, where it is unbiased
      (harmonic detail: partial levels inside a band, strike-position notches, beating).
    * ``attack``: log band energies in 1/3-octave bands from 400 Hz at a 256-point window and 2.7 ms hop,
      only in frames from 5 ms before to 40 ms after an onset.

    The floor of every log is white noise at ``floor_db`` dBFS (bins far below any recording's floor do not
    count, review 3, F1). ``forward(pred, target, onsets=None, onset_mask=None)`` with ``[B, ch, T]`` audio and
    onsets ``[B, N]`` in seconds re the first sample of ``pred``; returns ``(loss, terms)``.
    """

    GROUPS = ((8192, 0.0, 200.0), (2048, 200.0, 1600.0), (512, 1600.0, 1e9))  # (n_fft, band centres from, to)

    def __init__(self, sr, weights=(1.0, 0.25, 0.5), floor_db=-80.0, hop=240, f_lo=30.0, f_hi=11000.0,
                 fine_sizes=(4096, 1024), fine_max_hz=2000.0, hp_hz=20.0):
        super().__init__()
        self.sr, self.hop, self.floor_db, self.hp_hz = sr, hop, floor_db, hp_hz
        self.w_band, self.w_fine, self.w_attack = weights
        self.fine_sizes, self.fine_max_hz = fine_sizes, fine_max_hz
        f_hi = min(f_hi, 0.45 * sr)
        n_bands = int(math.floor(6 * math.log2(f_hi / f_lo))) + 1
        centers = f_lo * 2 ** (torch.arange(n_bands, dtype=torch.float64) / 6)
        self.register_buffer("centers", centers.float())
        self.groups = []
        for gi, (n, lo, hi) in enumerate(self.GROUPS):
            full = log_f_band_masks(n, sr, centers)  # the partition of unity over all bands, sliced per group
            sel = ((centers >= lo) & (centers < hi)).nonzero().squeeze(1)
            if sel.numel() == 0:
                continue
            m = full[sel]
            n_bins = int((m.sum(0) > 0).nonzero().max()) + 1
            self.register_buffer(f"mask{gi}", m[:, :n_bins].float())
            self.groups.append((gi, n, n_bins))
        att_c = 500.0 * 2 ** (torch.arange(int(math.floor(3 * math.log2(f_hi / 500.0))) + 1, dtype=torch.float64) / 3)
        self.register_buffer("att_mask", log_f_band_masks(256, sr, att_c, open_low=False).float())

    def _eps(self, n, mask):
        return 10 ** (self.floor_db / 10) * 0.375 * n * mask.sum(-1)[:, None]  # white floor in each band

    def _log_bands(self, x, n, mask, n_bins, hop):
        P = _mag(x, n, hop)[:, :n_bins] ** 2
        return torch.log10(torch.einsum("kf,nft->nkt", mask, P) + self._eps(n, mask))

    def terms(self, pred, target, onsets=None, onset_mask=None):
        """Per-example terms ``{name: [B]}`` (mean over channels, bands and frames)."""
        B = pred.shape[0]
        pred = highpass(pred, self.sr, self.hp_hz).reshape(-1, pred.shape[-1])
        target = highpass(target, self.sr, self.hp_hz).reshape(-1, target.shape[-1])
        per_ex = lambda v: v.reshape(B, -1).mean(1)
        out = {}
        bands = []
        for gi, n, n_bins in self.groups:
            m = getattr(self, f"mask{gi}")
            bands.append((self._log_bands(pred, n, m, n_bins, self.hop) - self._log_bands(target, n, m, n_bins, self.hop)).abs())
        out["band"] = per_ex(torch.cat(bands, 1).mean((1, 2)))
        if self.w_fine:
            fine = []
            for n in self.fine_sizes:
                k = int(self.fine_max_hz * n / self.sr) + 1
                k0 = int(math.ceil(self.hp_hz * n / self.sr))
                S, P = _mag(target, n)[:, k0:k], _mag(pred, n)[:, k0:k]
                eps = 10 ** (self.floor_db / 20) * (0.375 * n) ** 0.5
                fine.append(2 * (torch.log10(S + eps) - torch.log10(P + eps)).abs().mean((1, 2)))
            out["fine"] = per_ex(sum(fine) / len(fine))
        if self.w_attack and onsets is not None:
            n, hop = 256, 64
            d = self._log_bands(pred, n, self.att_mask, self.att_mask.shape[1], hop) - self._log_bands(
                target, n, self.att_mask, self.att_mask.shape[1], hop)  # [B*ch, bands, frames]
            t = torch.arange(d.shape[-1], device=d.device) * hop / self.sr
            rel = t[None, None, :] - onsets[..., None]  # [B, N, frames]
            near = ((rel >= -0.005) & (rel <= 0.040) & onset_mask[..., None]).any(1).float()  # [B, frames]
            ch = d.shape[0] // B
            w = near.repeat_interleave(ch, 0)[:, None, :]
            num = (d.abs() * w).sum((1, 2)).reshape(B, -1).sum(1)
            den = (w.sum((1, 2)) * d.shape[1]).reshape(B, -1).sum(1)
            out["attack"] = num / den.clamp(min=1.0)
        return out

    def forward(self, pred, target, onsets=None, onset_mask=None, per_example=False):
        t = self.terms(pred, target, onsets, onset_mask)
        w = {"band": self.w_band, "fine": self.w_fine, "attack": self.w_attack}
        total = sum(w[k] * v for k, v in t.items())
        if per_example:
            return total, t
        return total.mean(), {k: v.mean() for k, v in t.items()}


def mel_filterbank(sr, n_fft, n_mels, f_lo=30.0):
    """Triangular mel filters ``[n_mels, n_fft//2+1]``."""
    import numpy as np

    mel = lambda f: 2595 * np.log10(1 + f / 700)
    imel = lambda m: 700 * (10 ** (m / 2595) - 1)
    pts = imel(np.linspace(mel(f_lo), mel(sr / 2), n_mels + 2))
    f = np.fft.rfftfreq(n_fft, 1 / sr)
    M = np.stack([np.clip(np.minimum((f - lo) / (c - lo), (hi - f) / (hi - c)), 0, None)
                  for lo, c, hi in zip(pts[:-2], pts[1:-1], pts[2:])])
    return torch.tensor(M, dtype=torch.float32)


class LogMelLoss(nn.Module):
    """L1 distance of log mel energies (in units of 10 dB), at a long and a short window.

    Band energies do not care where exactly inside a band a partial sits, so unlike the per-bin
    log-magnitude term this cannot be lowered by making a partial that is slightly misaligned
    (beating, detune, inharmonicity) quieter: at 60 Hz a 4096-point bin is 170 cents wide, and
    the per-bin term alone let the 60-125 Hz octave drift 6 dB under the recordings. The floor
    (white noise at ``floor_db`` dBFS) matches the STFT loss.
    """

    def __init__(self, sr, fft_sizes=(4096, 1024), n_mels=64, floor_db=-80.0):
        super().__init__()
        self.fft_sizes, self.floor_db = fft_sizes, floor_db
        for n in fft_sizes:
            self.register_buffer(f"mel{n}", mel_filterbank(sr, n, n_mels))

    def forward(self, pred, target):
        pred, target = pred.reshape(-1, pred.shape[-1]), target.reshape(-1, target.shape[-1])
        loss = 0.0
        for n in self.fft_sizes:
            M = getattr(self, f"mel{n}")
            eps = 10 ** (self.floor_db / 10) * 0.375 * n * M.sum(-1)[:, None]  # white floor in each mel band
            e = lambda x: torch.log10(torch.einsum("mf,bft->bmt", M, _mag(x, n) ** 2) + eps)
            loss = loss + (e(pred) - e(target)).abs().mean()
        return loss / len(self.fft_sizes)


def band_energies(x, band_masks, n_fft=512, hop=120):
    """Frame energies per band ``[..., bands, frames]`` of ``x[..., T]`` (``band_masks[bands, n_fft//2+1]``)."""
    lead = x.shape[:-1]
    P = _mag(x.reshape(-1, x.shape[-1]), n_fft, hop) ** 2  # [N, bins, frames]
    E = torch.einsum("kf,nft->nkt", band_masks, P)
    return E.reshape(*lead, *E.shape[1:])


class SpecDiscriminator(nn.Module):
    def __init__(self, n_fft, ch=32):
        super().__init__()
        self.n_fft = n_fft
        self.convs = nn.ModuleList([
            nn.Conv2d(1, ch, (3, 9), padding=(1, 4)),
            nn.Conv2d(ch, ch, (3, 9), stride=(1, 2), padding=(1, 4)),
            nn.Conv2d(ch, ch, (3, 9), stride=(1, 2), padding=(1, 4)),
            nn.Conv2d(ch, ch, (3, 3), padding=(1, 1)),
        ])
        self.out = nn.Conv2d(ch, 1, (3, 3), padding=(1, 1))

    def forward(self, x):
        h = torch.log(_mag(x, self.n_fft) + 1e-5)[:, None]  # [B,1,F,T]
        feats = []
        for conv in self.convs:
            h = F.leaky_relu(conv(h), 0.1)
            feats.append(h)
        return self.out(h), feats


class MultiResolutionDiscriminator(nn.Module):
    """Adversarial critic on log spectrograms. Spectral losses alone average away
    the attack transients and noise texture that make a piano sound real."""

    def __init__(self, fft_sizes=(512, 1024, 2048)):
        super().__init__()
        self.discs = nn.ModuleList([SpecDiscriminator(n) for n in fft_sizes])

    def forward(self, x):
        x = x.reshape(-1, x.shape[-1])  # channels are judged separately
        return [d(x) for d in self.discs]


def discriminator_loss(disc, real, fake):
    loss = 0.0
    for (lr, _), (lf, _) in zip(disc(real), disc(fake.detach())):
        loss = loss + ((lr - 1) ** 2).mean() + (lf**2).mean()
    return loss


def generator_adv_loss(disc, real, fake):
    adv, fm = 0.0, 0.0
    with torch.no_grad():
        real_out = disc(real)
    for (lf, ff), (_, fr) in zip(disc(fake), real_out):
        adv = adv + ((lf - 1) ** 2).mean()
        fm = fm + sum((a - b).abs().mean() for a, b in zip(ff, fr)) / len(ff)
    return adv, fm
