"""Fitting on isolated notes (docs/tone_measures.md 12.1, step 2): per-note terms aligned to each note's own onset.

A ``NoteSet`` holds notes of the bench: the recording's clip, the MIDI context and N0 on the recording. Each fitting
step renders a batch of them with the model in their context (``LOOKBACK`` s of earlier notes, as the bench does),
finds N0 on the model's render (detached) and compares band levels in windows placed at each clip's own onset: the
recording's at its N0, the model's at its own. The latency between MIDI and the strike does not enter, nor does its
scatter (section 4.1: the attack term penalised timing errors at the steepest point).

The note term is the L1 distance of log band energies (1/3 octave, 100 Hz - 8 kHz) per (note, window, band), over
the cells where the recording's or the model's note stands at least ``MIN_OVER_BG`` dB over its own background in that
band (330-30 ms before its onset) and the band is above 0.7 f0 (``NoteTerm`` has the options: flat-topped windows,
the energy between the partials, levels relative to another window). Its optimum for a level offset is the median of
the paired differences: what the bench reports. ``bias_gate`` compares it with the energy match (section 4.2): the two
differ where the piano's notes vary more than the model's (N11), by about 0.115 sigma^2 dB for a log-normal spread of
sigma dB. A free level per piece (``PieceLevels``) and smooth per-key corrections (``KeyKnots``) keep the fit from
following the recordings' level per piece and single notes.
"""

import json
import math
import os

import numpy as np
import torch
from torch import nn
from torch.nn.utils import parametrize

from . import measures as M
from .config import year_to_condition
from .losses import taper

BAND_CENTERS = 100.0 * 2 ** (np.arange(20) / 3)  # 100 Hz .. 8 kHz
WINDOWS = {"early": (0.03, 0.10), "sustain": (0.10, 0.40)}  # s after the onset (N5's windows; the attack is step 2b)
BACKGROUND = (-0.33, -0.03)
MIN_OVER_BG = 6.0  # dB
WARM, PRE, POST = 1.0, 0.3, 0.5  # s of clip: room warm-up, before and after the MIDI onset
OCTAVES = (125, 250, 500, 1000, 2000, 4000, 8000)  # the reports pool the 1/3-octave bands by these


def band_masks(n_fft, sr, centers=BAND_CENTERS, device=None):
    """Raised-cosine masks ``[bands, n_fft // 2 + 1]`` in log f, one band step wide on either side of each centre:
    neighbours sum to one between the first and the last centre."""
    f = torch.fft.rfftfreq(n_fft, 1 / sr, device=device, dtype=torch.float64)
    lc = torch.log2(torch.as_tensor(np.asarray(centers), device=device, dtype=torch.float64))
    step = float(lc[1] - lc[0])
    d = (torch.log2(f.clamp(min=1.0))[None] - lc[:, None]) / step
    return (torch.cos(0.5 * math.pi * d.clamp(-1, 1)) ** 2).float()


def window_levels(x, start, length, masks, n_fft, flat=0.0, keep=None):
    """Band levels (dB of mean power per sample, summed over channels; ``taper(length, flat)``) of ``x[B, ch, T]``
    over samples ``[start_b, start_b + length)``; ``start`` is ``[B]`` (long). A sine of amplitude a in each of two
    channels reads 10 log10(a^2) in its band, whatever the window's length. ``keep[B, bins]`` weights the bins (e.g.
    only those between the partials). Returns ``[B, bands]``."""
    idx = start[:, None] + torch.arange(length, device=x.device)
    seg = x.gather(2, idx[:, None, :].expand(-1, x.shape[1], -1))
    w = taper(length, flat, x.device, x.dtype)
    P = (torch.fft.rfft(seg * w, n_fft).abs() ** 2).sum(1)
    P = P if keep is None else P * keep
    return 10 * torch.log10(2 * P @ masks.T / (n_fft * (w ** 2).sum()) + 1e-20)


class NoteSet:
    """Bench notes with their recordings' clips, MIDI contexts and N0; ``levels`` gives the recordings' band levels."""

    def __init__(self, root, notes, cfg, table, lookback=M.LOOKBACK, log=print):
        from .data import _perf_from_notes, history_frames

        self.cfg, sr = cfg, cfg.sample_rate
        self.n = int(round((WARM + PRE + POST) * sr))
        self.t_midi = WARM + PRE  # s re the clip's first sample
        with open(os.path.join(root, "index.json")) as f:
            meta = {p["id"]: p for p in json.load(f)}
        import soundfile as sf

        midi, keep = {}, []
        self.perfs, self.audio, self.t_rec = [], [], []
        for ev in notes:
            p = meta[ev["piece"]]
            if p["id"] not in midi:
                with np.load(os.path.join(root, p["midi"])) as z:
                    midi[p["id"]] = {k: z[k] for k in z.files}
            z = midi[p["id"]]
            t0 = ev["onset"] - self.t_midi
            x, _ = sf.read(os.path.join(root, p["audio"]), start=int(round(t0 * sr)), frames=self.n, dtype="float32",
                           always_2d=True)
            x = np.pad(x, ((0, self.n - len(x)), (0, 0)))
            t_on = M.onset(x.astype(np.float64), sr, self.t_midi, float(table[ev["pitch"] - 21][0]))
            if not np.isfinite(t_on):
                continue
            perf = _perf_from_notes(z["notes"], z, t0, self.n / sr, lookback, self.n // cfg.hop + 2, cfg,
                                    history_frames(lookback, cfg))
            perf["condition"] = torch.tensor(year_to_condition(p["year"]))
            self.perfs.append(perf)
            self.audio.append(torch.from_numpy(np.ascontiguousarray(x.T)))
            self.t_rec.append(t_on)
            keep.append(ev)
        self.notes = keep
        self.f0 = np.array([float(table[ev["pitch"] - 21][0]) for ev in keep])
        self.partials = np.stack([np.asarray(table[ev["pitch"] - 21], float) for ev in keep])
        self.pieces = sorted({ev["piece"] for ev in keep})
        self.piece = np.array([self.pieces.index(ev["piece"]) for ev in keep])
        log(f"note set: {len(keep)} of {len(notes)} notes (N0 found on the recording)")

    def __len__(self):
        return len(self.notes)

    def batch(self, idx, device):
        from .train import collate, to_device

        return to_device(collate([self.perfs[i] for i in idx]), device), torch.stack([self.audio[i] for i in idx]).to(device)


class NoteTerm:
    """Band levels in ``windows`` at given onsets, the recordings' cells that count, and the L1 distance.

    ``windows``: ``{name: (start, end)}`` in s re the onset, or ``(start, end, options)`` with options
    - ``flat``: a flat-topped taper (``taper``); N6's knock window is 0.7, so the first milliseconds count;
    - ``gaps``: only the energy between the note's partials (bins farther than max(70 Hz, f0 / 4) from every partial,
      as N6 knock): the knock without the partials' own onset. A band counts where at least 10 % of it is kept;
    - ``rel``: the level minus that of the named window, band by band (e.g. the attack re the early window: the
      attack's excess over the tone, which a tonal error common to both windows does not move);
    - ``all_bands``: count bands below 0.7 f0 too (the knock sits there; ``gaps`` windows always do);
    - ``partials``: K: instead of bands, the levels of the note's own partials 1..K (power within f0 / 4 of each,
      raised-cosine), so a decay that differs between neighbouring partials shows; counted where the partial stands
      ``MIN_OVER_BG`` dB over its own level in the background window.
    Cells count where the recording's or the model's window stands ``MIN_OVER_BG`` dB over the same measure of its
    own background. Selecting on the recording alone biased the paired median towards "the model is too weak" by up
    to 7 dB near the background (the 8 kHz knock between the partials: -6.4 dB selected on the recording, +0.6
    symmetric, +1.8 over every cell; 2018 evaluation notes, docs/tone_measures.md 12.5): where the recording's knock
    is weak, the model's excess was never counted."""

    def __init__(self, sr, device, windows=WINDOWS, centers=BAND_CENTERS):
        self.sr, self.windows, self.centers = sr, windows, np.asarray(centers)
        self.spec, self.opts = {}, {}
        for name, win in list(windows.items()) + [("bg", BACKGROUND)]:
            a, b, opts = (tuple(win) + ({},))[:3]
            L = int(round((b - a) * sr))
            n_fft = 1 << int(math.ceil(math.log2(L)))
            self.spec[name] = (a, L, n_fft, band_masks(n_fft, sr, centers, device), opts.get("flat", 0.0))
            self.opts[name] = opts
        self.gaps = any(o.get("gaps") for o in self.opts.values())
        self.n_partials = max([int(o.get("partials", 0)) for o in self.opts.values()] + [0])

    def _away(self, n_fft, partials, device):
        """Per-note bin masks ``[B, n_fft // 2 + 1]``: 1 farther than max(70 Hz, f0 / 4) from every partial."""
        hz = np.fft.rfftfreq(n_fft, 1 / self.sr)
        m = [~M.near_partials(hz, p, max(70.0, 0.25 * p[0])) for p in np.asarray(partials, float)]
        return torch.as_tensor(np.stack(m), dtype=torch.float32, device=device)

    def _partial_masks(self, n_fft, partials, device):
        """``[B, K, n_fft // 2 + 1]``: raised-cosine masks within f0 / 4 of each of the first K partials."""
        hz = torch.as_tensor(np.fft.rfftfreq(n_fft, 1 / self.sr), dtype=torch.float32, device=device)
        p = torch.as_tensor(np.asarray(partials, float)[:, : self.n_partials], dtype=torch.float32, device=device)
        half = 0.25 * p[:, :1, None]
        d = ((hz[None, None] - p[..., None]) / half).clamp(-1, 1)
        return torch.cos(0.5 * math.pi * d) ** 2 * (p[..., None] < 0.45 * self.sr)

    def levels(self, x, t_on, partials=None):
        """``{window: [B, bands]}`` of ``x[B, ch, T]`` with each clip's onset at ``t_on[B]`` s and partial
        frequencies ``partials[B, P]`` (Hz; needed by ``gaps`` windows); also ``"bg"`` (and ``"bg:gaps"``),
        ``"<name>:raw"`` for ``rel`` windows and ``"frac:<name>"`` (the share of each band kept) for ``gaps``."""
        out = {}
        for name, (a, L, n_fft, masks, flat) in self.spec.items():
            start = torch.as_tensor(np.round((np.asarray(t_on) + a) * self.sr), dtype=torch.long, device=x.device)
            start = start.clamp(0, x.shape[-1] - L)
            gaps = self.opts[name].get("gaps") or (name == "bg" and self.gaps)
            key = f"{name}:raw" if self.opts[name].get("rel") else name
            if self.opts[name].get("partials") or (name == "bg" and self.n_partials):
                pm = self._partial_masks(n_fft, partials, x.device)
                lv = _partial_window_levels(x, start, L, pm, n_fft, flat)
                out["bg:partials" if name == "bg" else key] = lv
                if name != "bg":
                    continue
            if name == "bg" or not self.opts[name].get("gaps"):
                out[key] = window_levels(x, start, L, masks, n_fft, flat)
            if gaps:
                away = self._away(n_fft, partials, x.device)
                gkey = "bg:gaps" if name == "bg" else key
                out[gkey] = window_levels(x, start, L, masks, n_fft, flat, away)
                out[f"frac:{name}"] = (away @ masks.T) / masks.sum(1)
        for name, opts in self.opts.items():
            if opts.get("rel"):
                out[name] = out[f"{name}:raw"] - out[opts["rel"]]
        return out

    def cells(self, rec, f0, mod=None):
        """``{window: [B, bands]}`` float weights: 1 where the recording's or (given ``mod``, detached) the model's
        window stands ``MIN_OVER_BG`` dB over its own background (for ``rel`` windows the reference window too), the
        band is above 0.7 f0 unless the window counts every band, and (``gaps``) a tenth of the band is kept."""
        above = torch.as_tensor(self.centers[None] >= 0.7 * np.asarray(f0)[:, None], device=rec["bg"].device)

        def stands(lv, w, o):
            if o.get("partials"):
                c = (lv[f"{w}:raw"] if o.get("rel") else lv[w]) >= lv["bg:partials"] + MIN_OVER_BG
                return c & (lv[o["rel"]] >= lv["bg:partials"] + MIN_OVER_BG) if o.get("rel") else c
            if o.get("gaps"):
                c = lv[w] >= lv["bg:gaps"] + MIN_OVER_BG
            else:
                c = (lv[f"{w}:raw"] if o.get("rel") else lv[w]) >= lv["bg"] + MIN_OVER_BG
            return c & (lv[o["rel"]] >= lv["bg"] + MIN_OVER_BG) if o.get("rel") else c

        out = {}
        for w in self.windows:
            o = self.opts[w]
            c = stands(rec, w, o)
            if mod is not None:
                c = c | stands({k: v.detach() for k, v in mod.items()}, w, o)
            if o.get("gaps"):
                c = c & (rec[f"frac:{w}"] >= 0.1)
            elif not o.get("partials"):
                c = c & (above | bool(o.get("all_bands")))
            out[w] = c.float()
        return out

    def loss(self, mod, rec, cells, weight=None, offset=None):
        """Mean |model + offset - recording| (dB) over the counted cells; ``weight[B]`` drops notes (e.g. N0 failed),
        ``offset[B]`` (dB) is each note's recording level re the others (its piece's, ``PieceLevels``; not added to
        ``rel`` windows, which are level differences)."""
        num, den = 0.0, 0.0
        for w in self.windows:
            c = cells[w] if weight is None else cells[w] * weight[:, None]
            d = mod[w] - rec[w] if offset is None or self.opts[w].get("rel") else mod[w] + offset[:, None] - rec[w]
            num = num + (d.abs() * c).sum()
            den = den + c.sum()
        return num / den.clamp(min=1.0)


def _partial_window_levels(x, start, length, pmask, n_fft, flat=0.0):
    """Levels (dB, as ``window_levels``) of ``x[B, ch, T]`` over ``[start_b, start_b + length)`` in per-note masks
    ``pmask[B, K, bins]``: ``[B, K]``."""
    idx = start[:, None] + torch.arange(length, device=x.device)
    seg = x.gather(2, idx[:, None, :].expand(-1, x.shape[1], -1))
    w = taper(length, flat, x.device, x.dtype)
    P = (torch.fft.rfft(seg * w, n_fft).abs() ** 2).sum(1)  # [B, bins]
    return 10 * torch.log10(2 * torch.einsum("bf,bkf->bk", P, pmask) / (n_fft * (w ** 2).sum()) + 1e-20)


def _is_partials(win):
    return len(win) > 2 and bool(win[2].get("partials"))


class PieceLevels(nn.Module):
    """A free level (dB) per piece, averaging zero: the recordings' level at equal key and velocity differs between
    pieces of one year by ~2 dB (sd of the per-piece median, 2018; docs/tone_measures.md 12.5), which the model's
    per-year parameters cannot follow. Fitted alongside, it keeps that scatter out of the per-key and velocity terms;
    the model's own level becomes the average piece's."""

    def __init__(self, n):
        super().__init__()
        self.db = nn.Parameter(torch.zeros(n))

    def forward(self, idx):
        return (self.db - self.db.mean())[torch.as_tensor(idx, device=self.db.device)]


class KeyKnots(nn.Module):
    """Parametrization of a per-key table ``[88, ...]`` as its value plus a piecewise-linear correction over the keys
    with knots every ``step`` keys: a fit on a few notes per key then moves the table smoothly instead of note by
    note."""

    def __init__(self, step, rest=(), n_keys=88):
        super().__init__()
        self.register_buffer("pos", torch.arange(n_keys, dtype=torch.float32) / step)
        self.knots = nn.Parameter(torch.zeros(int(math.ceil((n_keys - 1) / step)) + 1, *rest))

    def forward(self, x):
        i0 = self.pos.floor().long().clamp(max=len(self.knots) - 2)
        w = (self.pos - i0).reshape(-1, *([1] * (x.dim() - 1)))
        return x + self.knots[i0] * (1 - w) + self.knots[i0 + 1] * w


def smooth_per_key(module, names, step):
    """Register ``KeyKnots(step)`` on each per-key table ``names`` of ``module``; returns ``{name: knots parameter}``.
    ``parametrize.remove_parametrizations(module, name)`` bakes the correction in."""
    out = {}
    for n in names:
        t = getattr(module, n)
        parametrize.register_parametrization(module, n, KeyKnots(step, tuple(t.shape[1:])).to(t.device))
        getattr(module.parametrizations, n).original.requires_grad_(False)
        out[n] = getattr(module.parametrizations, n)[0].knots
    return out


def model_onsets(y, sr, t_midi, f0):
    """N0 on each of the model's clips ``y[B, ch, T]`` (detached), s re the clip start; NaN where it fails."""
    x = y.detach().float().cpu().numpy().astype(np.float64)
    return np.array([M.onset(x[i].T, sr, t_midi, float(f0[i])) for i in range(len(x))])


@torch.no_grad()
def evaluate(model, notes, term, device, batch=12, residual=False):
    """Band levels of the recordings and the model on every note of ``notes`` (a ``NoteSet``):
    ``{"rec": {w: [N, bands]}, "mod": ..., "cells": ..., "ok": [N]}`` as numpy (``ok``: N0 found on the model)."""
    sr = model.cfg.sample_rate
    acc = {"rec": {}, "mod": {}, "cells": {}}
    ok = []
    for s in range(0, len(notes), batch):
        idx = list(range(s, min(s + batch, len(notes))))
        b, audio = notes.batch(idx, device)
        y = model(b, notes.n, residual=residual, generator=torch.Generator(device=device).manual_seed(0))["audio"]
        t_mod = model_onsets(y, sr, notes.t_midi, notes.f0[idx])
        good = np.isfinite(t_mod)
        rec = term.levels(audio, np.array([notes.t_rec[i] for i in idx]), notes.partials[idx])
        mod = term.levels(y, np.where(good, t_mod, notes.t_midi), notes.partials[idx])
        cells = term.cells(rec, notes.f0[idx], mod)
        for key, d in (("rec", rec), ("mod", mod), ("cells", cells)):
            for w, v in d.items():
                acc[key].setdefault(w, []).append(v.cpu().numpy())
        ok.append(good)
    out = {k: {w: np.concatenate(v) for w, v in d.items()} for k, d in acc.items()}
    out["ok"] = np.concatenate(ok)
    return out


def level_and_shape(ev, notes, windows=WINDOWS):
    """Model - recording over the counted cells, split into the level (the median over pieces of each piece's
    median, dB), the piece-level scatter (sd of the per-piece medians) and the shape (mean |d - its piece's median|);
    also mean |d| and the number of cells."""
    d, pc = [], []
    for w in windows:
        c = (ev["cells"][w] > 0) & ev["ok"][:, None]
        d.append((ev["mod"][w] - ev["rec"][w])[c])
        pc.append(np.broadcast_to(notes.piece[:, None], c.shape)[c])
    d, pc = np.concatenate(d), np.concatenate(pc)
    med = {p: np.median(d[pc == p]) for p in set(pc.tolist())}
    pm = np.array([m for p, m in med.items() if (pc == p).sum() >= 20])
    return {"abs": float(np.abs(d).mean()), "level": float(np.median(pm)), "piece_sd": float(pm.std()),
            "shape": float(np.abs(d - np.array([med[p] for p in pc.tolist()])).mean()), "n": int(len(d)), "pieces": len(pm)}


def bias_gate(ev, notes, windows=WINDOWS, centers=BAND_CENTERS, min_cells=8):
    """Bias of the L1 term, per window and octave (1/3-octave bands pooled by their nearest octave, 125 Hz - 8 kHz).
    Within each stratum (register x velocity bin, where the model's notes are alike and the recordings scatter):
    the term's optimum level offset (median of model - recording over the counted cells) minus the energy match (10
    log10 of the model's summed energy over the recording's, same cells); bias = the median of that over the strata
    with ``min_cells`` cells. Pooling across strata would mix the model's errors, which differ between strata, into
    it. A term passes if the median bias over the bands is within +-0.5 dB and every band's within +-1 dB (as
    ``scripts/loss_bias.py``). Negative: the L1 optimum leaves the model's energy under the recordings'.

    Returns ``[{"window", "band", "strata", "n", "median", "energy", "bias"}]`` and whether it passes; "median" and
    "energy" are medians over the strata."""
    octs = np.array(OCTAVES)
    near = octs[np.argmin(np.abs(np.log2(np.asarray(centers)[:, None] / octs[None])), 1)]
    strata = np.array([f"{n['register']}|{n['vel_bin']}" for n in notes.notes])
    rows = []
    for w in windows:
        if _is_partials(windows[w]):
            continue
        c = (ev["cells"][w] > 0) & ev["ok"][:, None]
        for o in octs:
            med, en, n = [], [], 0
            for s in sorted(set(strata)):
                sel = c & (near == o)[None] & (strata == s)[:, None]
                if sel.sum() < min_cells:
                    continue
                med.append(np.median(ev["mod"][w][sel] - ev["rec"][w][sel]))
                en.append(10 * np.log10((10 ** (ev["mod"][w][sel] / 10)).sum() / (10 ** (ev["rec"][w][sel] / 10)).sum()))
                n += int(sel.sum())
            if len(med) >= 2:
                rows.append({"window": w, "band": int(o), "strata": len(med), "n": n, "median": float(np.median(med)),
                             "energy": float(np.median(en)), "bias": float(np.median(np.array(med) - np.array(en)))})
    b = np.array([r["bias"] for r in rows])
    return rows, bool(len(b) and abs(np.median(b)) <= 0.5 and np.all(np.abs(b) <= 1.0))


def residual_table(ev, notes, windows=WINDOWS, centers=BAND_CENTERS, key="register"):
    """Median model - recording (dB) per (``key`` value, window, octave) over counted cells: ``{(k, w, oct): (median, n)}``."""
    octs = np.array(OCTAVES)
    near = octs[np.argmin(np.abs(np.log2(np.asarray(centers)[:, None] / octs[None])), 1)]
    groups = np.array([n[key] for n in notes.notes])
    out = {}
    for g in sorted(set(groups)):
        for w in windows:
            c = (ev["cells"][w] > 0) & ev["ok"][:, None] & (groups == g)[:, None]
            if _is_partials(windows[w]):  # per partial: "p1".."pK" in place of the octaves
                for k in range(c.shape[1]):
                    sel = c[:, k]
                    if sel.sum() >= 10:
                        out[(g, w, f"p{k + 1}")] = (float(np.median(ev["mod"][w][sel, k] - ev["rec"][w][sel, k])), int(sel.sum()))
                continue
            for o in octs:
                sel = c & (near == o)[None]
                if sel.sum() >= 10:
                    out[(g, w, int(o))] = (float(np.median(ev["mod"][w][sel] - ev["rec"][w][sel])), int(sel.sum()))
    return out
