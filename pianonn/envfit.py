"""Fitting on the whole note's envelope (N13, docs/tone_measures.md 17): each partial's fade from 0.1 s to 2.5 s.

An ``EnvelopeSet`` holds N13 events (``measures.envelope_events``: notes in any texture, read while they sound
freely) with the recording's clip, the MIDI context and the recording's N13 reading (``measures.note_envelope``):
levels, local floors, which readings are clear of the other notes and the frames each reading used. Each fitting step
renders a batch in context and reads the model's partials (refined on its own render, detached) at the same frames,
differentiably (``readout``). ``EnvelopeTerm`` is the L1 distance of the fades (level at t re 0.1 s, each held at or
above its floor), over the readings where both sides are clear, both stand ``SNR`` dB over their floor at 0.1 s and at
least one at t: the N13 cells. Its optimum per cell is the median fade, what N13 reports.
"""

import json
import math
import os

import numpy as np
import torch

from . import measures as M
from .config import year_to_condition

WARM, PRE, POST = 1.0, 0.3, 3.1  # s of clip: room warm-up, before and after the MIDI onset
SNR = 6.0
T_MAX = 3.0
REGS = (("R2", 36, 46), ("R3", 47, 59), ("R4", 60, 71), ("R5", 72, 83), ("R6", 84, 88))
LENGTHS = (("<1 s", 0, 1.0), ("1-2 s", 1.0, 2.0), ("2+ s", 2.0, 99.0))


def mine(root, year, cap, seed):
    """N13's events of every piece of ``year`` (``measures.envelope_events``), at most ``cap`` per (register, velocity,
    length; 1.5 and 3 times that for 1-2 and 2+ s), drawn with ``seed``: ``(events, available per cell)``."""
    with open(os.path.join(root, "index.json")) as f:
        pieces = [p for p in json.load(f) if p["year"] == year]
    cells = {}
    for p in pieces:
        with np.load(os.path.join(root, p["midi"])) as z:
            notes, ped = z["notes"], {k: z[k] for k in z.files}
        for i, end, others in M.envelope_events(notes, ped, t_max=T_MAX):
            pi, v = int(notes[i, 0]), int(notes[i, 3])
            ev = {"piece": p["id"], "split": p["split"], "pitch": pi, "velocity": v, "onset": float(notes[i, 1]),
                  "offset": float(notes[i, 2]), "t_end": end, "register": M.bin_of(pi, REGS),
                  "vel_bin": M.bin_of(v, M.VELOCITY), "length": M.bin_of(end, LENGTHS), "others": others}
            cells.setdefault((ev["register"], ev["vel_bin"], ev["length"]), []).append(ev)
    rng = np.random.default_rng(seed)
    out, avail = [], {}
    scale = {"<1 s": 1.0, "1-2 s": 1.5, "2+ s": 3.0}  # the late readings need the long notes
    for key in sorted(cells):
        evs = cells[key]
        avail["|".join(key)] = len(evs)
        out += [evs[j] for j in sorted(rng.permutation(len(evs))[: int(cap * scale[key[2]])])]
    return out, avail


def control_freqs(fr, f0):
    """N13's control frequencies ``[K, 2]``: half-way to each partial's neighbours (as ``measures.note_envelope``)."""
    sp = np.diff(np.concatenate([[fr[0] - f0], fr, [fr[-1] + (fr[-1] - fr[-2] if len(fr) > 1 else f0)]]))
    return np.stack([fr - 0.5 * sp[:-1], fr + 0.5 * sp[1:]], 1)


class EnvelopeSet:
    """N13 events with their recordings' clips, MIDI contexts and the recordings' readings."""

    def __init__(self, root, events, cfg, table, tabs, lookback=M.LOOKBACK, log=print):
        import soundfile as sf

        from .data import _perf_from_notes, history_frames

        self.cfg, sr = cfg, cfg.sample_rate
        self.n = int(round((WARM + PRE + POST) * sr))
        self.t_on = WARM + PRE
        with open(os.path.join(root, "index.json")) as f:
            meta = {p["id"]: p for p in json.load(f)}
        midi, keep = {}, []
        self.perfs, self.audio, self.rec = [], [], []
        for j, ev in enumerate(events):
            if j and j % 250 == 0:
                log(f"reading {j} of {len(events)} notes")
            p = meta[ev["piece"]]
            if p["id"] not in midi:
                with np.load(os.path.join(root, p["midi"])) as z:
                    midi[p["id"]] = {k: z[k] for k in z.files}
            z = midi[p["id"]]
            t0 = ev["onset"] - self.t_on
            x, _ = sf.read(os.path.join(root, p["audio"]), start=int(round(t0 * sr)), frames=self.n, dtype="float32",
                           always_2d=True)
            x = np.pad(x, ((0, self.n - len(x)), (0, 0)))
            r = M.note_envelope(x.astype(np.float64), sr, self.t_on, table[ev["pitch"] - 21], ev["t_end"], ev["others"], tabs)
            up = r["L"] >= r["ctrl"] + SNR
            if not (r["clear"][1:] & r["clear"][:1] & up[:1]).any():
                continue
            perf = _perf_from_notes(z["notes"], z, t0, self.n / sr, lookback, self.n // cfg.hop + 2, cfg,
                                    history_frames(lookback, cfg))
            perf["condition"] = torch.tensor(year_to_condition(p["year"]))
            self.perfs.append(perf)
            self.audio.append(torch.from_numpy(np.ascontiguousarray(x.T)))
            Lc = np.fmax(r["L"], r["ctrl"])
            self.rec.append({"fade": Lc - Lc[:1], "up": up, "clear": r["clear"], "frames": r["frames"],
                             "win": r["win"], "n": len(r["freqs"])})
            keep.append(ev)
        self.notes = keep
        self.table = table
        log(f"envelope set: {len(keep)} of {len(events)} notes (some reading counts on the recording)")

    def __len__(self):
        return len(self.notes)

    def batch(self, idx, device):
        from .train import collate, to_device

        return to_device(collate([self.perfs[i] for i in idx]), device)


def readout(y, sr, t_on, freqs, frames, win):
    """Differentiable N13 levels of one clip ``y[ch, T]``: per time, the mean power over its ``frames`` (s re
    ``t_on``) of a Hann window of ``win`` s evaluated at ``freqs[K]`` (as ``measures.partial_tracks``), dB:
    ``[T, K]`` (NaN where a time has no frames)."""
    n = int(round(win * sr))
    w = torch.as_tensor(np.hanning(n), dtype=y.dtype, device=y.device)
    ph = 2 * np.pi * np.outer(np.arange(n) / sr, freqs)
    Ec = torch.as_tensor(np.cos(ph), dtype=y.dtype, device=y.device) * w[:, None]
    Es = torch.as_tensor(np.sin(ph), dtype=y.dtype, device=y.device) * w[:, None]
    rows = []
    for fr in frames:
        if fr is None:
            rows.append(torch.full((len(freqs),), float("nan"), dtype=y.dtype, device=y.device))
            continue
        starts = torch.as_tensor(np.round((t_on + np.asarray(fr)) * sr).astype(int) - n // 2, device=y.device)
        idx = starts[:, None] + torch.arange(n, device=y.device)
        seg = y[:, idx]  # [ch, F, n]
        P = (seg @ Ec) ** 2 + (seg @ Es) ** 2  # [ch, F, K]
        rows.append(10 * torch.log10(P.sum(0).mean(0) + 1e-30))
    return torch.stack(rows)


class EnvelopeTerm:
    """The L1 distance of N13 fades between the model's render and the recording, over the N13 cells."""

    def __init__(self, sr):
        self.sr = sr

    def note(self, y, rec, table_row, t_on, t_end):
        """``(sum |d|, count, d [T, K] detached, cells [T, K])`` for one clip ``y[ch, T]`` against the recording's
        reading ``rec`` (an ``EnvelopeSet.rec`` entry)."""
        K = rec["n"]
        tab = np.asarray(table_row, float)
        tab = tab[(tab > 0) & (tab < 6000.0)][:K]
        x = y.detach().float().cpu().numpy().T.astype(np.float64)
        fr = M.refine_partials(x, self.sr, t_on, tab, float(tab[0]), t_end)
        ctrl = control_freqs(fr, float(tab[0]))
        L = readout(y, self.sr, t_on, fr, rec["frames"], rec["win"])
        with torch.no_grad():
            C = readout(y, self.sr, t_on, ctrl.ravel(), rec["frames"], rec["win"]).reshape(len(rec["frames"]), K, 2).amax(-1)
        Lc = torch.maximum(L, C)
        fade = Lc - Lc[:1]
        up = (L.detach() >= C + SNR).cpu().numpy()
        cells = rec["clear"] & rec["clear"][:1] & rec["up"][:1] & up[:1] & (rec["up"] | up)
        cells[0] = False
        c = torch.as_tensor(cells, device=y.device)
        d = fade - torch.as_tensor(rec["fade"], dtype=fade.dtype, device=y.device)
        d = torch.where(c, d, torch.zeros_like(d))
        return d.abs().sum(), int(cells.sum()), d.detach().cpu().numpy(), cells


@torch.no_grad()
def evaluate(model, envs, term, device, batch=4, seed=0, residual=False):
    """``{"d": [N, T, K], "cells": [N, T, K]}`` (model fade - recording fade) over every note of ``envs``."""
    sr = model.cfg.sample_rate
    T, Kmax = len(M.ENV_TIMES), 40
    D = np.full((len(envs), T, Kmax), np.nan)
    Cm = np.zeros((len(envs), T, Kmax), bool)
    for s in range(0, len(envs), batch):
        idx = list(range(s, min(s + batch, len(envs))))
        y = model(envs.batch(idx, device), envs.n, residual=residual,
                  generator=torch.Generator(device=device).manual_seed(seed + s))["audio"].float()
        for j, i in enumerate(idx):
            ev = envs.notes[i]
            _, _, d, c = term.note(y[j], envs.rec[i], envs.table[ev["pitch"] - 21], envs.t_on, ev["t_end"])
            K = envs.rec[i]["n"]
            D[i, :, :K] = np.where(c, d, np.nan)
            Cm[i, :, :K] = c
    return {"d": D, "cells": Cm}


def summary(ev, envs, groups, times=M.ENV_TIMES, regs=None):
    """Median of ``d`` per (partial group, time) over the cells, optionally per register: ``{(reg, group, t): (median, n)}``."""
    out = {}
    reg = np.array([n["register"] for n in envs.notes])
    kk = np.arange(ev["d"].shape[-1])
    for r in [None] + list(regs or []):
        sel_e = np.ones(len(reg), bool) if r is None else reg == r
        for g, a, b in groups:
            for i, t in enumerate(times):
                if i == 0:
                    continue
                v = ev["d"][sel_e][:, i][:, (kk >= a) & (kk < b)]
                v = v[np.isfinite(v)]
                out[(r, g, t)] = (float(np.median(v)), len(v)) if len(v) else (float("nan"), 0)
    return out
