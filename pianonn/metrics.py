"""Evaluation metrics with a scale (review 4, sections 2-3; docs/plan_round2.md, A5).

A distance alone says little: two renders of the same model that differ only in their noise are already far
apart, and a model can score well by being quiet. So every table carries references, and the loss itself is
checked for level bias:

* :func:`level_bias`: per octave band, where each loss term's optimum gain sits relative to the energy match;
* :func:`references`: the same model with another noise seed, the recording plus the model's floor, a
  per-excerpt gain oracle, the model driven by the wrong MIDI, and stationary noise with the recording's
  long-term spectrum;
* :func:`band_levels`, :func:`level_regression`: model minus recording per band, and against dynamics and pedal.
"""

import copy
import math

import numpy as np
import torch

from .losses import LogMelLoss, MultiResolutionSTFTLoss, PianoLoss, _mag, highpass, log_f_band_masks

CENTERS = [31.5 * 2 ** k for k in range(9)]  # octave bands, 31.5 Hz ... 8 kHz
COVERS = {"new: fine": (20.0, 2000.0), "new: attack": (400.0, 12000.0)}  # frequency ranges of the partial terms


def old_terms(pred, target):
    """Per-example spectral convergence and log-magnitude L1 of the trial's MR-STFT loss, averaged over resolutions."""
    rec = MultiResolutionSTFTLoss()
    B = pred.shape[0]
    p, t = pred.reshape(-1, pred.shape[-1]), target.reshape(-1, target.shape[-1])
    sc, lm = 0.0, 0.0
    for n in rec.fft_sizes:
        S, P = _mag(t, n), _mag(p, n)
        eps = rec.eps(n)
        sc = sc + torch.linalg.norm(S - P, dim=(-2, -1)) / (torch.linalg.norm(S, dim=(-2, -1)) + eps)
        lm = lm + (torch.log(S + eps) - torch.log(P + eps)).abs().mean((-2, -1))
    k = len(rec.fft_sizes)
    return (sc / k).reshape(B, -1).mean(1), (lm / k).reshape(B, -1).mean(1)


def log_mel_db(sr, device):
    """Per-example log-mel L1 in dB (64 mel bands, 4096 and 1024 points, the -80 dBFS floor)."""
    mel = LogMelLoss(sr).to(device)
    return lambda p, t: torch.stack([10 * mel(p[i:i + 1], t[i:i + 1]) for i in range(p.shape[0])])


def make_terms(sr, device, weights=(1.0, 0.25, 0.5), level_weight=0.0):
    """``f(pred, target, onsets, onset_mask) -> {name: [B]}``: the new loss and its terms, the trial's loss and its
    two terms, and log-mel (dB). With ``level_weight`` also ``new: level`` and ``train total`` (the loss as the runs
    since phase 4 train on it: ``new: total`` + ``level_weight`` x level); ``new: total`` keeps its old meaning."""
    new = PianoLoss(sr, weights=weights, level_weight=level_weight).to(device)
    mel = log_mel_db(sr, device)

    def terms(p, t, on, om):
        total, parts = new(p, t, on, om, per_example=True)
        sc, lm = old_terms(p, t)
        out = {"new: total": total, **{f"new: {k}": v for k, v in parts.items()},
               "old: MR-STFT": sc + lm, "old: SC only": sc, "old: log-mag only": lm, "log-mel (dB)": mel(p, t)}
        if level_weight:
            out["train total"], out["new: total"] = total, total - level_weight * parts["level"]
        return out

    return terms


def clustered_mean(d, groups):
    """Mean of ``d`` (per excerpt) and its standard error with excerpts grouped (by piece): excerpts of one piece
    are not independent, so the error counts the groups, not the excerpts. Returns ``(mean, se, n_groups)``."""
    d, groups = np.asarray(d, float), np.asarray(groups)
    ids = np.unique(groups)
    m = d.mean()
    sums = np.array([(d[groups == g] - m).sum() for g in ids])
    G = len(ids)
    se = math.sqrt(G / max(G - 1, 1) * (sums ** 2).sum()) / len(d) if G > 1 else float("nan")
    return float(m), se, G


def piece_gains_loo(e_model, e_rec, groups):
    """Gain (dB) per excerpt that matches the model's level to the recording's for its piece, from the piece's
    *other* excerpts (leave one out), so an excerpt is never scored with a gain fitted on itself. ``e_model``,
    ``e_rec``: energies per excerpt. Excerpts alone in their piece get 0 dB; returns ``(gains, n_alone)``."""
    e_model, e_rec, groups = (np.asarray(x) for x in (e_model, e_rec, groups))
    gains, alone = np.zeros(len(e_model)), 0
    for i in range(len(e_model)):
        other = (groups == groups[i]) & (np.arange(len(e_model)) != i)
        if other.any():
            gains[i] = 10 * math.log10(e_rec[other].sum() / max(e_model[other].sum(), 1e-30))
        else:
            alone += 1
    return gains, alone


def band_split(x, sr, centers=CENTERS):
    M = log_f_band_masks(x.shape[-1], sr, centers).to(x.device)
    return torch.fft.irfft(torch.fft.rfft(x)[..., None, :] * M, x.shape[-1])  # [..., bands, T]


def parabolic(g, v):
    i = int(np.argmin(v))
    if 0 < i < len(v) - 1:
        den = v[i - 1] - 2 * v[i] + v[i + 1]
        if den > 0:
            return g[i] + 0.5 * (v[i - 1] - v[i + 1]) / den * (g[1] - g[0])
    return g[i]


def render(model, b, residual, seed=0):
    n, s = b["audio"].shape[-1], int(b["loss_start"][0])
    return model(b, n, residual=residual, generator=torch.Generator(device=b["audio"].device).manual_seed(seed))["audio"][..., s:]


@torch.no_grad()
def level_bias(model, batches, residual, gains=np.arange(-4.0, 4.01, 0.5)):
    """For each excerpt and octave band: the gain on the model's band that minimises each term, minus the gain that
    matches the band's energy to the recording's (dB). Returns ``({term: [excerpts, bands]}, err[excerpts, bands])``
    with ``err`` the energy error model - recording. NaN where a term does not cover the band."""
    dev = batches[0]["audio"].device
    sr = model.cfg.sample_rate
    terms = make_terms(sr, dev)
    bias, err = {}, []
    g_lin = torch.tensor(10 ** (gains / 20), device=dev, dtype=torch.float32)
    for b in batches:
        s = int(b["loss_start"][0])
        y, t = render(model, b, residual), b["audio"][..., s:]
        on, om = b["onset"] - s / sr, b["mask"]
        yb, tb = band_split(highpass(y, sr), sr), band_split(highpass(t, sr), sr)  # [B, ch, bands, T]
        for i in range(y.shape[0]):
            e_y, e_t = yb[i].pow(2).sum((0, 2)), tb[i].pow(2).sum((0, 2))
            truth = 10 * torch.log10(e_t / e_y.clamp(min=1e-20)).cpu().numpy()
            err.append(-truth)
            rows = {}
            for k in range(len(CENTERS)):
                pred = y[i][None] + (g_lin - 1)[:, None, None] * yb[i, :, k][None]  # [G, ch, T]
                vals = terms(pred, t[i][None].expand_as(pred), on[i][None].expand(len(gains), -1),
                             om[i][None].expand(len(gains), -1))
                for name, v in vals.items():
                    lo, hi = COVERS.get(name, (0.0, math.inf))
                    outside = not (lo / 2 <= CENTERS[k] <= 2 * hi)  # what the term sees there is window leakage
                    rows.setdefault(name, []).append(np.nan if outside else parabolic(gains, v.cpu().numpy()) - truth[k])
            for name, v in rows.items():
                bias.setdefault(name, []).append(v)
    return {k: np.array(v) for k, v in bias.items()}, np.array(err)


def bias_report(bias, err):
    lines = ["| term | " + " | ".join(f"{c:.0f} Hz" for c in CENTERS) + " | all | gate |",
             "|---|" + "---|" * (len(CENTERS) + 2),
             "| *model − recording, energy (dB)* | " + " | ".join(f"*{np.median(err[:, k]):+.1f}*" for k in range(len(CENTERS)))
             + " | | |"]
    for name, v in bias.items():
        med = np.nanmedian(v, 0)
        ok = abs(np.nanmedian(v)) <= 0.5 and np.all(np.abs(med[np.isfinite(med)]) <= 1.0)
        lines.append(f"| {name} | " + " | ".join("" if not np.isfinite(x) else f"{x:+.1f}" for x in med)
                     + f" | {np.nanmedian(v):+.2f} | {'pass' if ok else 'fail'} |")
    return lines


@torch.no_grad()
def inharmonicity_visibility(model, batches, residual, keys=30, factor=2.0):
    """Per term: distance(model, model with the bass B x ``factor``) / distance(model seed 0, model seed 1)."""
    dev = batches[0]["audio"].device
    sr = model.cfg.sample_rate
    terms = make_terms(sr, dev)
    pert = copy.deepcopy(model)
    raw = pert.physics.raw_log_B.data
    cur = 1.5 * torch.tanh(raw[:keys] / 1.5)
    raw[:keys] = 1.5 * torch.atanh(((cur + math.log(factor)) / 1.5).clamp(-0.999, 0.999))
    d_b, d_seed = {}, {}
    for b in batches:
        s = int(b["loss_start"][0])
        on, om = b["onset"] - s / sr, b["mask"]
        y0, y1, yb = render(model, b, residual, 0), render(model, b, residual, 1), render(pert, b, residual, 0)
        for name, v in terms(yb, y0, on, om).items():
            d_b.setdefault(name, []).append(v.cpu())
        for name, v in terms(y1, y0, on, om).items():
            d_seed.setdefault(name, []).append(v.cpu())
    return {k: float(torch.cat(d_b[k]).mean() / torch.cat(d_seed[k]).mean()) for k in d_b}


def _stationary_like(t, generator):
    """Noise with each channel's long-term power spectrum of ``t[B, ch, T]`` (random phase)."""
    n = t.shape[-1]
    P = (_mag(t.reshape(-1, n), 2048, 512) ** 2).mean(-1)  # [B*ch, 1025]
    f_src = torch.linspace(0, 1, P.shape[-1], device=t.device)
    f_dst = torch.linspace(0, 1, n // 2 + 1, device=t.device)
    idx = (f_dst * (P.shape[-1] - 1)).clamp(max=P.shape[-1] - 1.001)
    i0 = idx.floor().long()
    w = idx - i0
    psd = P[:, i0] * (1 - w) + P[:, i0 + 1] * w  # [B*ch, n//2+1]
    white = torch.randn(t.reshape(-1, n).shape, generator=generator, device=t.device)
    x = torch.fft.irfft(torch.fft.rfft(white) * psd.sqrt(), n)
    x = x * (t.reshape(-1, n).pow(2).mean(-1, keepdim=True) / x.pow(2).mean(-1, keepdim=True).clamp(min=1e-20)).sqrt()
    return x.reshape(t.shape)


@torch.no_grad()
def references(model, batches, residual, gains=np.arange(-6.0, 6.01, 0.5)):
    """Distances ``{row: {term: mean}}`` that give the numbers a scale, and the gain oracle's median gain (dB)."""
    dev = batches[0]["audio"].device
    sr = model.cfg.sample_rate
    terms = make_terms(sr, dev)
    acc, oracle_gain = {}, []
    renders, targets, ons = [], [], []
    for b in batches:
        s = int(b["loss_start"][0])
        renders.append(render(model, b, residual, 0))
        targets.append(b["audio"][..., s:])
        ons.append((b["onset"] - s / sr, b["mask"]))

    def add(row, vals):
        for k, v in vals.items():
            acc.setdefault(row, {}).setdefault(k, []).append(v.cpu())

    g = torch.Generator(device=dev).manual_seed(5)
    for i, b in enumerate(batches):
        y, t, (on, om) = renders[i], targets[i], ons[i]
        add("model", terms(y, t, on, om))
        add("same model, another noise seed (vs the model)", terms(render(model, b, residual, 1), y, on, om))
        fl = model.room.floor_noise(b["condition"], b["audio"].shape[-1], g)[..., int(b["loss_start"][0]):]
        add("recording + the model's floor (vs the recording)", terms(t + fl, t, on, om))
        add("model driven by another excerpt's MIDI", terms(torch.roll(y, 1, 0), t, on, om))  # needs batches of >= 2
        add("stationary noise, the recording's long-term spectrum", terms(_stationary_like(t, g), t, on, om))
        best = None
        for j in range(y.shape[0]):  # per-excerpt best broadband gain under the new loss
            G = torch.tensor(10 ** (gains / 20), device=dev, dtype=torch.float32)
            pred = y[j][None] * G[:, None, None]
            v = terms(pred, t[j][None].expand_as(pred), on[j][None].expand(len(gains), -1), om[j][None].expand(len(gains), -1))
            k = int(v["new: total"].argmin())
            oracle_gain.append(gains[k])
            row = {name: vv[k:k + 1] for name, vv in v.items()}
            best = row if best is None else {name: torch.cat([best[name], row[name]]) for name in row}
        add("model x best broadband gain per excerpt (oracle)", best)
    table = {row: {k: float(torch.cat(v).mean()) for k, v in d.items()} for row, d in acc.items()}
    return table, float(np.median(oracle_gain))


@torch.no_grad()
def band_levels(model, batches, residual):
    """Model minus recording per octave band (dB): per-excerpt energy ratio, ``[excerpts, bands]``, and per excerpt
    the broadband level error plus its mean velocity and pedal fractions (for :func:`level_regression`)."""
    sr, cfg = model.cfg.sample_rate, model.cfg
    per_band, rows = [], []
    for b in batches:
        s = int(b["loss_start"][0])
        y, t = highpass(render(model, b, residual), sr), highpass(b["audio"][..., s:], sr)
        yb, tb = band_split(y, sr), band_split(t, sr)
        eb = 10 * torch.log10(yb.pow(2).sum((1, 3)) / tb.pow(2).sum((1, 3)).clamp(min=1e-20))  # [B, bands]
        per_band.append(eb.cpu().numpy())
        H = int(b["hist_frames"].reshape(-1)[0]) if "hist_frames" in b else 0
        for i in range(y.shape[0]):
            on = b["onset"][i] - s / sr
            m = b["mask"][i] & (on > -1.0) & (on < t.shape[-1] / sr)
            vel = float(b["velocity"][i][m].float().mean()) if m.any() else float("nan")
            sus = b["sustain"][i][H + s // cfg.hop: H + (s + t.shape[-1]) // cfg.hop].float()
            rows.append((vel, float(((sus > 0.3) & (sus < 0.75)).float().mean()), float((sus >= 0.75).float().mean()),
                         10 * math.log10(float(y[i].pow(2).mean()) / max(1e-20, float(t[i].pow(2).mean())))))
    return np.concatenate(per_band), np.array(rows)


def level_regression(rows):
    """Least squares ``error = a + b*vel/127 + c*half_pedal + d*full_pedal`` (dB) over excerpts with notes."""
    r = rows[np.isfinite(rows[:, 0])]
    X = np.stack([np.ones(len(r)), r[:, 0] / 127, r[:, 1], r[:, 2]], 1)
    coef = np.linalg.lstsq(X, r[:, 3], rcond=None)[0]
    return coef, r
