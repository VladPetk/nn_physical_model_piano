"""Spectral reconstruction losses and an optional multi-resolution spectrogram discriminator."""

import math

import numpy as np
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


def taper(length, flat=0.0, device=None, dtype=torch.float32):
    """Hann window (``flat`` = 0), or a Tukey window whose tapers take ``1 - flat`` of the length (N6's knock window
    is ``flat`` = 0.7: the first milliseconds of an attack count)."""
    if flat <= 0:
        return torch.hann_window(length, periodic=False, device=device, dtype=dtype)
    n = torch.arange(length, device=device, dtype=dtype)
    edge = 0.5 * (1 - flat) * (length - 1)
    up = 0.5 - 0.5 * torch.cos(math.pi * (n / edge).clamp(max=1))
    down = 0.5 - 0.5 * torch.cos(math.pi * ((length - 1 - n) / edge).clamp(max=1))
    return torch.minimum(up, down)


# the recordings' sound onset (N0) re the MIDI onset, ms: a + b (pitch - 60) + c (velocity - 64); 382 evaluation notes
# of the 2018 bench (residual IQR -3.8 .. +4.2 ms); the model follows the same law (docs/tone_measures.md 12.6)
ONSET_DELAY_MS = (7.17, -0.273, -0.073)


class OnsetLoss(nn.Module):
    """The attack at every note's expected sound onset, pooled (docs/tone_measures.md 12.6, 12.8).

    Windows from 3 ms before to 33 ms after each note's expected sound onset (its MIDI onset plus ``delay_ms``),
    flat-topped as N6's, band powers in 1/3-octave bands from 100 Hz (the last band open above); notes whose expected
    onsets lie within ``merge`` s share a window (a chord is one attack). A band of a window counts where the target's
    or the prediction's window stands ``min_over_bg`` dB over its own background (330 to 30 ms before; detached). The
    counted windows' powers are summed per band before the log, so a level offset's optimum is the energy match of the
    pool; the loss is the mean over the counted bands of |log10 P_pred - log10 P_target| (1 = 10 dB), as in
    ``PianoLoss``.

    - ``relative``: the attack re the same onset's early window (``early``, 30-100 ms, Hann), both pooled: the attack's
      excess over the tone, which a level or brightness error of the tone, common to both windows, does not move. A
      cell then needs the early window over the background too (on the side that counts it).
    - ``pool_decay`` > 0 (training mode only): the pool runs across steps. Per band, both sides' counted powers are
      kept as sums decaying by ``pool_decay`` per call; the loss's value and the sign of its gradient come from the
      running sums, the gradient's size from this batch's own log ratio. The optimum is then the energy match over
      about ``1 / (1 - pool_decay)`` batches, whatever the batch size (pooled over one batch of 2 segments the median
      and the energy mean part ways by up to 1.5 dB where the spreads differ, 12.6). In eval mode the pool is the
      batch.

    It sees what ``PianoLoss`` does not: the attack below ~400 Hz (the attack term has no band there, the band term
    averages the bass over 341 ms). ``forward(pred, target, batch, t_lo)`` takes the whole rendered window
    ``[B, ch, T]`` (the background needs the warm-up) and the performance ``batch`` (``onset`` in s re the window's
    first sample, ``pitch``, ``velocity``, ``mask``); windows anchored before ``t_lo`` s do not count.
    """

    def __init__(self, sr, window=(-0.003, 0.033), flat=0.7, background=(-0.33, -0.03), n_bands=20, f_lo=100.0,
                 min_over_bg=6.0, merge=0.010, delay_ms=ONSET_DELAY_MS, relative=False, early=(0.030, 0.100),
                 pool_decay=0.0):
        super().__init__()
        self.sr, self.window, self.background, self.merge, self.delay_ms = sr, window, background, merge, delay_ms
        self.relative, self.pool_decay = relative, pool_decay
        self.over = 10 ** (min_over_bg / 10)
        centers = f_lo * 2 ** (torch.arange(n_bands, dtype=torch.float64) / 3)
        self.register_buffer("centers", centers.float())
        # running sums per band: prediction attack, prediction early, target attack, target early
        self.register_buffer("pool", torch.zeros(4, n_bands, dtype=torch.float64), persistent=False)
        self.spec = {}
        wins = [("attack", window, flat), ("bg", background, 0.0)] + ([("early", early, 0.0)] if relative else [])
        for name, (a, b), fl in wins:
            L = int(round((b - a) * sr))
            n = 1 << int(math.ceil(math.log2(L)))
            self.register_buffer(f"mask_{name}", log_f_band_masks(n, sr, centers, open_low=False).float())
            self.register_buffer(f"taper_{name}", taper(L, fl))
            self.spec[name] = (a, L, n)

    def anchors(self, batch, t_lo, t_hi):
        """``(rows, times, velocity)`` of every onset group whose window anchor (s re the window's first sample) lies
        in ``[t_lo, t_hi]``: the example, the anchor and the group's loudest velocity."""
        a, b, c = self.delay_ms
        rows, times, vels = [], [], []
        onset, pitch, vel, mask = (batch[k].detach().cpu().numpy() for k in ("onset", "pitch", "velocity", "mask"))
        for r in range(onset.shape[0]):
            m = mask[r].astype(bool)
            v = vel[r][m].astype(float)
            t = onset[r][m] + 1e-3 * np.maximum(a + b * (pitch[r][m] - 60) + c * (v - 64), 0.0)
            o = np.argsort(t, kind="stable")
            t, v = t[o], v[o]
            i = 0
            while i < len(t):
                j = i
                while j + 1 < len(t) and t[j + 1] - t[i] < self.merge:
                    j += 1
                if t_lo <= t[i] <= t_hi:
                    rows.append(r), times.append(float(t[i])), vels.append(float(v[i: j + 1].max()))
                i = j + 1
        return np.array(rows, dtype=np.int64), np.array(times), np.array(vels)

    def powers(self, x, rows, times):
        """Band powers (mean power per sample, channels summed) ``{"attack", "bg"[, "early"]}: [G, bands]`` of
        ``x[B, ch, T]`` in the windows of the onset groups ``rows``, ``times``."""
        rows_t = torch.as_tensor(rows, device=x.device)
        out = {}
        for name, (a, L, n) in self.spec.items():
            start = torch.as_tensor(np.round((times + a) * self.sr), dtype=torch.long, device=x.device)
            idx = start.clamp(0, x.shape[-1] - L)[:, None] + torch.arange(L, device=x.device)
            w = getattr(self, f"taper_{name}").to(x.dtype)
            seg = x[rows_t[:, None], :, idx].permute(0, 2, 1) * w  # [G, ch, L]
            P = (torch.fft.rfft(seg, n).abs() ** 2).sum(1)
            out[name] = 2 * P @ getattr(self, f"mask_{name}").T / (n * (w ** 2).sum())
        return out

    def cells(self, p, q):
        """1 where either side's window stands ``min_over_bg`` dB over its own background (detached): ``[G, bands]``;
        with ``relative``, its early window too."""
        def stands(d):
            c = d["attack"] >= self.over * d["bg"]
            return c & (d["early"] >= self.over * d["bg"]) if self.relative else c
        return (stands(p) | stands(q)).detach().float()

    def pooled(self, d, c):
        """Counted powers summed over the windows: ``[2, bands]`` (attack; early, or ones)."""
        a = (d["attack"] * c).sum(0)
        return torch.stack([a, (d["early"] * c).sum(0) if self.relative else torch.ones_like(a)])

    @staticmethod
    def log_ratio(s, eps=1e-12):
        """log10 of the pooled attack (re the pooled early window) from ``pooled``'s ``[2, bands]``."""
        return torch.log10(s[0] + eps) - torch.log10(s[1] + eps)

    def forward(self, pred, target, batch, t_lo):
        rows, times, _ = self.anchors(batch, t_lo, pred.shape[-1] / self.sr - self.window[1])
        if not len(rows):
            return pred.sum() * 0.0
        p = self.powers(pred, rows, times)
        with torch.no_grad():
            q = self.powers(target, rows, times)
        c = self.cells(p, q)
        sp, sq = self.pooled(p, c), self.pooled(q, c)
        mine = self.log_ratio(sp)
        if self.pool_decay > 0 and self.training:
            with torch.no_grad():
                run = self.pool * self.pool_decay + torch.cat([sp, sq]).detach().double()
                self.pool.copy_(run)
                if not self.relative:
                    run[1] = run[3] = 1.0
                d = (self.log_ratio(run[:2]) - self.log_ratio(run[2:])).to(mine.dtype)
                has = (run[0] > 0) | (run[2] > 0)
            s = torch.sign(d)
            per_band = s * mine - (s * mine).detach() + d.abs()  # value from the pool, gradient from the batch
        else:
            per_band = (mine - self.log_ratio(sq)).abs()
            has = c.sum(0) > 0
        return (per_band * has).sum() / has.sum().clamp(min=1)


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
