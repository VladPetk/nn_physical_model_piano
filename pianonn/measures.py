"""The note bench and the measures of docs/tone_measures.md (N0-N10, N3 glide, N12, E1, E3/F4, E4, P3, P4).

The *bench* is a fixed, seeded list of isolated notes from the recordings (no other onset from 0.3 s before to
0.65 s after), stratified by register, velocity and pedal and split by piece: a calibration group (training
pieces) and an evaluation group (validation and test pieces). It also holds note-offs (N10), free decays (P4)
and re-strike runs (F4).

Every measure is computed identically on a recording's clip and on a model's render of the same MIDI in its
context (12 s lookback, 1 s warm-up). Clips are ``[T, channels]`` arrays; power is summed over the channels
(averaging a spaced pair comb-filters). Each measure is checked on synthetic notes with a known answer
(``tests/test_measures.py``) and, with a model, on known parameter changes (``scripts/note_bench.py --validate``).
"""

import json
import math
import os

import numpy as np

# the instrument's own boundaries (docs/tone_measures.md, 1.3)
REGISTERS = (("R1", 21, 28), ("R2", 29, 46), ("R3", 47, 59), ("R4", 60, 71), ("R5", 72, 83), ("R6", 84, 88), ("R7", 89, 108))
VELOCITY = (("p", 0, 40), ("mp-mf", 40, 64), ("mf-f", 64, 90), ("ff", 90, 128))
# CC64: the model's lift curve has the dampers on below ~0.3 and fully off above ~0.6
PEDAL = (("up", 0, 38), ("half", 38, 76), ("down", 76, 128))
OCTAVES = (250, 500, 1000, 2000, 4000, 8000)
PRE, POST = 0.7, 0.8  # s of clip around a note's MIDI onset
REL_PRE, REL_POST = 0.4, 0.45  # s of clip around a release
DECAY_PRE, DECAY_POST = 0.4, 1.2  # s of clip around a free decay's stop
REPEAT_POST = 1.5  # s of clip after a re-strike run's first strike
WARMUP, LOOKBACK = 1.0, 12.0


def bin_of(x, table):
    for name, lo, hi in table:
        if lo <= x < hi:
            return name
    return table[-1][0]


def register_of(pitch):
    for name, lo, hi in REGISTERS:
        if lo <= pitch <= hi:
            return name
    return None


# ------------------------------------------------------------------ the bench


def _value_at(t_ev, v_ev, t):
    k = np.searchsorted(t_ev, t, side="right") - 1
    return float(v_ev[k]) if k >= 0 else 0.0


def isolated(onsets, before=0.3, clear=0.65):
    """Indices of notes with no other onset from ``before`` s before to ``clear`` s after (``onsets`` sorted)."""
    lo = np.searchsorted(onsets, onsets - before)
    hi = np.searchsorted(onsets, onsets + clear)
    return np.nonzero(hi - lo == 1)[0]


def release_candidates(notes, pedals, hold=0.4, quiet_on=(0.3, 0.5), quiet_off=(0.2, 0.4)):
    """Isolated note-offs: held >= ``hold`` s, no onset in [off - 0.3, off + 0.5], no other note-off in
    [off - 0.2, off + 0.4], sustain and sostenuto pedals off from 0.3 s before to 0.4 s after. The bench's first
    release list (59 in 2018); N10 now uses ``release_events``, which do not need the texture to be empty."""
    on, off = notes[:, 1], notes[:, 2]
    offs = np.sort(off)
    out = []
    for i in range(len(notes)):
        t = off[i]
        if t - on[i] < hold:
            continue
        if np.any((on > t - quiet_on[0]) & (on < t + quiet_on[1])):
            continue
        n_off = np.searchsorted(offs, t + quiet_off[1]) - np.searchsorted(offs, t - quiet_off[0])
        if n_off != 1:
            continue
        ok = True
        for name in ("sustain", "sostenuto"):
            ts, vs = pedals[name + "_t"], pedals[name + "_v"]
            inside = vs[(ts > t - 0.3) & (ts < t + 0.4)]
            if _value_at(ts, vs, t - 0.3) >= PEDAL[0][2] or np.any(inside >= PEDAL[0][2]):
                ok = False
        if ok:
            out.append(i)
    return out


def build_bench(root, years, cap=24, cap_release=40, cap_decay=150, cap_repeat=30, cap_off=60, seed=0, before=0.3,
                clear=0.65):
    """The bench of ``years``: ``{"notes": [...], "releases": [...], "decays": [...], "repeats": [...],
    "available...": {...}}``. Each stratum (group, register, velocity, pedal) keeps at most ``cap`` notes, drawn with
    ``seed``; releases keep at most ``cap_release`` per (group, register), free decays (``free_decays``) at most
    ``cap_decay`` per group, re-strike runs (``repeat_runs``) at most ``cap_repeat`` per (group, register), note-offs
    for N10 (``release_events``) at most ``cap_off`` per (group, register)."""
    with open(os.path.join(root, "index.json")) as f:
        pieces = sorted((p for p in json.load(f) if p["year"] in years), key=lambda p: p["id"])
    cands, rels, decs, reps, offs = {}, {}, {}, {}, {}
    for p in pieces:
        group = "calib" if p["split"] == "train" else "eval"
        with np.load(os.path.join(root, p["midi"])) as z:
            z = {k: z[k] for k in z.files}
        notes = z["notes"]
        assert np.all(np.diff(notes[:, 1]) >= 0), p["id"]
        base = {"piece": p["id"], "year": p["year"], "split": p["split"], "group": group}
        for i in isolated(notes[:, 1], before, clear):
            pitch, t, t_off, vel = notes[i]
            reg = register_of(int(pitch))
            if reg is None or t < WARMUP + PRE + 0.1 or t + POST > p["duration"]:
                continue
            nxt = notes[i + 1, 1] - t if i + 1 < len(notes) else p["duration"] - t
            prv = t - notes[i - 1, 1] if i > 0 else t
            ped = _value_at(z["sustain_t"], z["sustain_v"], t)
            rec = {**base, "index": int(i), "pitch": int(pitch), "velocity": int(vel), "onset": float(t),
                   "offset": float(t_off), "clear_before": float(prv), "clear_after": float(nxt), "pedal": ped,
                   "register": reg, "vel_bin": bin_of(vel, VELOCITY), "pedal_bin": bin_of(ped, PEDAL)}
            cands.setdefault((group, reg, rec["vel_bin"], rec["pedal_bin"]), []).append(rec)
        for t_stop, t_next in free_decays(notes, z, p["duration"]):
            decs.setdefault(group, []).append({**base, "stop": t_stop, "next": t_next})
        for run in repeat_runs(notes, z):
            reg = register_of(run["pitch"])
            if reg is None or run["times"][0] < WARMUP + PRE + 0.1 or run["times"][0] + REPEAT_POST > p["duration"]:
                continue
            reps.setdefault((group, reg), []).append({**base, **run, "onset": run["times"][0], "register": reg})
        for i, others in release_events(notes, z):
            pitch, t, t_off, vel = notes[i]
            reg = register_of(int(pitch))
            if reg is None or t_off < WARMUP + REL_PRE + 0.1 or t_off + REL_POST > p["duration"]:
                continue
            offs.setdefault((group, reg), []).append({**base, "index": int(i), "pitch": int(pitch), "velocity": int(vel),
                                                      "onset": float(t), "offset": float(t_off), "register": reg,
                                                      "others": others})
        for i in release_candidates(notes, z):
            pitch, t, t_off, vel = notes[i]
            reg = register_of(int(pitch))
            if reg is None or t_off < WARMUP + REL_PRE + 0.1 or t_off + REL_POST > p["duration"]:
                continue
            rels.setdefault((group, reg), []).append({**base, "index": int(i), "pitch": int(pitch), "velocity": int(vel),
                                                      "onset": float(t), "offset": float(t_off), "register": reg})
    rng = np.random.default_rng(seed)
    pick = lambda lst, k: [lst[j] for j in sorted(rng.permutation(len(lst))[:k])]
    return {"years": list(years), "cap": cap, "seed": seed, "before": before, "clear": clear,
            "notes": [n for key in sorted(cands) for n in pick(cands[key], cap)],
            "releases": [n for key in sorted(rels) for n in pick(rels[key], cap_release)],
            "available": {"|".join(k): len(v) for k, v in sorted(cands.items())},
            "available_releases": {"|".join(k): len(v) for k, v in sorted(rels.items())},
            # drawn after the notes and releases, so adding them left those unchanged
            "decays": [n for key in sorted(decs) for n in pick(decs[key], cap_decay)],
            "repeats": [n for key in sorted(reps) for n in pick(reps[key], cap_repeat)],
            "available_decays": {k: len(v) for k, v in sorted(decs.items())},
            "available_repeats": {"|".join(k): len(v) for k, v in sorted(reps.items())},
            "offs": [n for key in sorted(offs) for n in pick(offs[key], cap_off)],
            "available_offs": {"|".join(k): len(v) for k, v in sorted(offs.items())}}


# ------------------------------------------------------------------ recording clips and model renders


def render_clips(root, events, models, cfg, device, pre=PRE, post=POST, batch=16, at="onset"):
    """Clips around each event's ``at`` time (s): ``{"recording": [...], label: [...]}``, each ``[T, ch]`` float64
    with the event at ``pre`` s. ``models`` is a list of ``(label, model, residual)``; every model renders the
    event's MIDI in its context (``LOOKBACK`` s of earlier notes, ``WARMUP`` s of warm-up discarded)."""
    import soundfile as sf
    import torch

    from .config import year_to_condition
    from .data import _perf_from_notes, history_frames
    from .train import collate, to_device

    sr = cfg.sample_rate
    n = int(round((WARMUP + pre + post) * sr))
    s0 = int(round(WARMUP * sr))
    midi, meta = {}, {}
    with open(os.path.join(root, "index.json")) as f:
        for p in json.load(f):
            meta[p["id"]] = p
    out = {"recording": []} | {label: [] for label, _, _ in models}
    for b0 in range(0, len(events), batch):
        items = []
        for ev in events[b0: b0 + batch]:
            p = meta[ev["piece"]]
            if p["id"] not in midi:
                with np.load(os.path.join(root, p["midi"])) as z:
                    midi[p["id"]] = {k: z[k] for k in z.files}
            z = midi[p["id"]]
            t0 = ev[at] - pre - WARMUP
            x, _ = sf.read(os.path.join(root, p["audio"]), start=int(round(t0 * sr)), frames=n, dtype="float32", always_2d=True)
            x = np.pad(x, ((0, n - len(x)), (0, 0)))
            perf = _perf_from_notes(z["notes"], z, t0, n / sr, LOOKBACK, n // cfg.hop + 2, cfg, history_frames(LOOKBACK, cfg))
            perf["condition"] = torch.tensor(year_to_condition(p["year"]))
            perf["audio"] = torch.from_numpy(np.ascontiguousarray(x.T))
            perf["loss_start"] = torch.tensor(s0)
            items.append(perf)
            out["recording"].append(x[s0:].astype(np.float64))
        b = to_device(collate(items), device)
        for label, model, residual in models:
            with torch.no_grad():
                y = model(b, n, residual=residual, generator=torch.Generator(device=device).manual_seed(0))["audio"]
            y = y[..., s0:].float().cpu().numpy().astype(np.float64)
            out[label] += [yy.T for yy in y]
    return out


def partial_table(model, cond, device):
    """Partial frequencies ``[88, P]`` (Hz) of the model's prompt mode at mf: the reference comb for every clip."""
    import torch

    with torch.no_grad():
        ki = torch.arange(88, device=device)[None]
        u = torch.full(ki.shape, 0.6, device=device)
        c = torch.tensor([cond], device=device)
        return model.physics.modes(ki, u, torch.zeros_like(u), c, phantoms=False)["freq"][0, :, :, 0].cpu().numpy()


# ------------------------------------------------------------------ primitives


def _power(seg, n_fft=None, window=True):
    """Power spectrum summed over channels (Hann window unless ``window`` is False) and its frequencies scale."""
    seg = np.atleast_2d(seg.T).T if seg.ndim == 1 else seg
    n_fft = n_fft or len(seg)
    w = np.hanning(len(seg))[:, None] if window else 1.0
    return (np.abs(np.fft.rfft(seg * w, n_fft, axis=0)) ** 2).sum(1)


def _segment(x, sr, t0, t1):
    a, b = max(0, int(round(t0 * sr))), min(len(x), int(round(t1 * sr)))
    return x[a:b]


def band_mask(f, lo, hi, edge=1 / 6):
    """Amplitude mask of the band [lo, hi) Hz with raised-cosine edges ``edge`` octaves wide on either side of
    each edge (0.5 at lo and hi; adjacent bands sum to one). A brick-wall mask rings for a long time (its impulse
    response falls as 1/t): a loud part of a clip then leaks 30-50 dB down into quiet parts, before and after it,
    which caps decay fits and fakes pre-onset energy."""
    lf = np.log2(np.maximum(f, 1e-3))
    m = np.ones_like(f)
    if lo > 0:
        m *= np.sin(0.5 * np.pi * np.clip((lf - math.log2(lo) + edge) / (2 * edge), 0, 1)) ** 2
    if hi < f[-1]:
        m *= np.cos(0.5 * np.pi * np.clip((lf - math.log2(hi) + edge) / (2 * edge), 0, 1)) ** 2
    return m


def band_envelope(x, sr, lo, hi, smooth=0.002):
    """Power in [lo, hi) Hz over time (smooth-edged FFT mask, ``band_mask``), summed over channels, ``smooth`` s
    moving average."""
    X = np.fft.rfft(x, axis=0)
    f = np.fft.rfftfreq(len(x), 1 / sr)
    X *= band_mask(f, lo, hi)[:, None] if X.ndim == 2 else band_mask(f, lo, hi)
    e = np.fft.irfft(X, len(x), axis=0) ** 2
    e = e.sum(1) if e.ndim == 2 else e
    k = max(1, int(smooth * sr))
    c = np.concatenate([[0.0], np.cumsum(e)])
    i = np.arange(len(e))
    lo_i, hi_i = np.clip(i - k // 2, 0, len(e)), np.clip(i - k // 2 + k, 0, len(e))
    return (c[hi_i] - c[lo_i]) / np.maximum(hi_i - lo_i, 1)


def near_partials(hz, partials, tol):
    """Bins within ``tol`` Hz of any partial."""
    p = np.asarray(partials)
    p = p[(p > 0) & (p < hz[-1])]
    if not len(p):
        return np.zeros(len(hz), bool)
    j = np.clip(np.searchsorted(p, hz), 1, len(p) - 1)
    d = np.minimum(np.abs(hz - p[j - 1]), np.abs(hz - p[j]))
    return d <= tol


def partial_levels(x, sr, t0, t1, freqs, spacing, noise=None):
    """Noise-compensated level (dB, power summed over channels) of each partial in [t0, t1] s; NaN under 10 dB SNR.

    A Hann window over the segment, zero-padded 8x; each partial's power is the maximum within ``max(2 / T, 0.4 %)``
    of its frequency, never more than a quarter of the partial spacing."""
    seg = _segment(x, sr, t0, t1)
    n_fft = 1 << int(math.ceil(math.log2(len(seg) * 8)))
    P = _power(seg, n_fft)
    hz = np.fft.rfftfreq(n_fft, 1 / sr)
    N = np.zeros_like(P)
    if noise is not None and len(noise) >= len(seg):
        chunks = [noise[i: i + len(seg)] for i in range(0, len(noise) - len(seg) + 1, max(1, len(seg) // 2))]
        N = np.mean([_power(c, n_fft) for c in chunks], 0)
    out = []
    for f in freqs:
        tol = min(max(2.0 / (t1 - t0), 0.004 * f), 0.25 * spacing)
        band = (hz >= f - tol) & (hz <= f + tol)
        if f >= 0.45 * sr or not band.any():
            out.append(float("nan"))
            continue
        sig, noi = P[band].max(), N[band].mean()
        out.append(10 * math.log10(sig - noi) if sig > 10 * noi and sig > 0 else float("nan"))
    return np.array(out)


def slope_db_oct(levels, n):
    ok = np.isfinite(levels)
    if ok.sum() < 3:
        return float("nan")
    return float(np.polyfit(np.log2(np.asarray(n, float)[ok]), levels[ok], 1)[0])


def _db(x):
    return 10 * math.log10(max(float(x), 1e-30))


# ------------------------------------------------------------------ N0: onset


def onset(x, sr, t_ref, f0, search=0.04, min_rise_db=3.0):
    """N0: the onset (s) of the note struck near ``t_ref``.

    The note's band (above 1.5 f0 and 100 Hz) is followed at 0.5 ms, averaged over one period (at least 2 ms:
    partials beat at their spacing, so a shorter average jitters by several dB); the steepest rise over one period
    (at least 3 ms) within ``search`` s of ``t_ref`` is found, then followed back to where the rise was 10 % done
    (above the median of the 25-60 ms before it). Walking back from the rise, not forward from the search edge,
    keeps it off the ringing of earlier notes. NaN if the rise is under ``min_rise_db``: on 2018's evaluation
    notes that is ~40 % in R2 (the hall's tail of earlier notes sits a few dB under a bass note), 12 % in R3,
    under 10 % above."""
    sm = max(0.002, 1.0 / f0)
    e = band_envelope(x, sr, max(100.0, 1.5 * f0), min(8000.0, 0.45 * sr), sm)
    step = max(1, sr // 2000)
    es = e[::step]
    L = 10 * np.log10(es + 1e-30)
    t = np.arange(len(es)) * step / sr
    lag = max(1, int(max(0.003, 1.0 / f0) * sr / step))
    rise = np.full(len(L), -np.inf)
    rise[:-lag] = L[lag:] - L[:-lag]
    win = np.nonzero((t >= t_ref - search) & (t <= t_ref + search))[0]
    if not len(win):
        return float("nan")
    k = win[np.argmax(rise[win])]
    bg_idx = np.nonzero((t >= t[k] - 0.06 - sm) & (t <= t[k] - 0.025 - sm))[0]
    pk_idx = np.nonzero((t >= t[k]) & (t <= t[k] + 0.03 + sm))[0]
    if not len(bg_idx) or not len(pk_idx):
        return float("nan")
    bg, pk = np.median(es[bg_idx]), es[pk_idx].max()
    if 10 * math.log10(pk / max(bg, 1e-30)) < min_rise_db:
        return float("nan")
    thr = bg + 0.1 * (pk - bg)
    j = k
    while j > bg_idx[-1] and es[j] > thr:  # back out of the rise
        j -= 1
    m = j
    while m < pk_idx[-1] and es[m] < thr:  # then forward to the crossing
        m += 1
    # a centred average of width sm crosses 10 % of a step 0.4 sm early: add it back
    if m == j or es[m] <= es[m - 1]:
        return float(t[m] + 0.4 * sm)
    return float(t[m - 1] + (thr - es[m - 1]) / (es[m] - es[m - 1]) * step / sr + 0.4 * sm)


KNOCK_BANDS = (125,) + OCTAVES


def _note_ref(x, sr, t_on):
    """The note's energy per second at 100-400 ms, 0.2-8 kHz: the reference of N6 knock and N4 percussive."""
    sg = _segment(x, sr, t_on + 0.1, t_on + 0.4)
    nf = 1 << int(math.ceil(math.log2(len(sg) * 2)))
    Q, f = _power(sg, nf), np.fft.rfftfreq(nf, 1 / sr)
    return Q[(f >= 200) & (f < 8000)].sum() / 0.3


def percussive(x, sr, t_on, ref, bands=KNOCK_BANDS, n_win=512, hop=64):
    """N4 percussive (see ``note_measures``)."""
    from scipy.ndimage import median_filter

    out = {}
    seg = _segment(x, sr, t_on - 0.3, t_on + 0.3)
    w = np.hanning(n_win)[:, None]
    starts = np.arange(0, len(seg) - n_win, hop)
    if not len(starts) or ref <= 0:
        return out
    S = np.stack([(np.abs(np.fft.rfft(seg[s: s + n_win] * w, axis=0)) ** 2).sum(1) for s in starts])
    f = np.fft.rfftfreq(n_win, 1 / sr)
    H = median_filter(S, size=(17, 1), mode="nearest")
    P = median_filter(S, size=(1, 17), mode="nearest")
    Pp = S * P ** 2 / (H ** 2 + P ** 2 + 1e-30)
    t = (starts + n_win / 2) / sr - 0.3
    att, bg = (t >= -0.005) & (t <= 0.04), (t >= -0.15) & (t <= -0.06)
    norm = (w ** 2).sum() / n_win  # frame power -> power per sample, as the reference
    for c in bands:
        sel = (f >= c / math.sqrt(2)) & (f < min(c * math.sqrt(2), 0.45 * sr))
        ea, eb = Pp[att][:, sel].sum(1).mean(), Pp[bg][:, sel].sum(1).mean()
        out[f"N4 percussive {c}"] = _db(max(ea - eb, 1e-3 * ea) / norm * sr / n_win / ref) if ea > 0 else float("nan")
    return out


def own_partials(P, hz, table, fmax, snr_db=6.0, track=False):
    """The note's own partial frequencies: for each partial of ``table`` below ``fmax``, the strongest peak of the
    power spectrum ``P`` within min(f0/3, 2 % + 3 Hz) of it, if it stands ``snr_db`` above that window's median;
    else the table's value. A per-key B that is off by 25 % (the model's table at E2-A2 in 2018) moves partial 25
    by ~40 Hz, far more than the few Hz that separate a phantom from its neighbours. With ``track``, partial k > 1
    is searched within min(f0/4, 0.6 % + 3 Hz) of the table's value times the last found partial's ratio to the
    table (and, if not found, placed there): the recordings' stretch drifts from the table's by 20 cents and more
    by partial 30 in the tenor, and a tight window does not take a phantom 30-40 cents below a partial for it."""
    table = np.asarray(table, float)
    out = table.copy()
    f0 = table[0]
    ratio = 1.0
    for k, fk in enumerate(table):
        if fk >= fmax:
            break
        fc, tol = fk * ratio, min(f0 / 3, 0.02 * fk + 3.0)
        if track and k > 0:
            tol = min(f0 / 4, 0.006 * fk + 3.0)
        sel = np.nonzero((hz >= fc - tol) & (hz <= fc + tol))[0]
        if len(sel) < 3:
            continue
        i = sel[np.argmax(P[sel])]
        if P[i] > 10 ** (snr_db / 10) * np.median(P[sel]):
            out[k] = hz[i]
            if track:
                ratio = hz[i] / fk
        elif track:
            out[k] = fc
    return out


def phantom_levels(x, sr, t_on, partials, t0=0.03, t1=0.43, fmax=5000.0):
    """N9: the peak level near 2 f_j re partial 2j, ``N9 phantom`` (median over j), and the same at a control
    position half-way from 2 f_j down to the nearest transverse partial, ``N9 control``: what a chance peak reads.
    The partials are the note's own (``own_partials``, found near ``partials``). Only where both positions are
    clear of every transverse partial (2 f_j sits ~3 B j^3 f0 under partial 2j, so low j coincide with it and high
    j reach the next partials) and 2 f_j is below ``fmax``. The peak is searched within max(one bin of the 0.4 s
    window, 0.25 %) of the position. In the mid register the room's energy between the partials masks the model's
    phantoms (+20 dB of them read +0.05 dB on R3-R5 notes); on loud R2-R3 notes +20 dB reads +15 dB."""
    out = {}
    seg = _segment(x, sr, t_on + t0, t_on + t1)
    n_fft = 1 << int(math.ceil(math.log2(len(seg) * 8)))
    P, hz = _power(seg, n_fft), np.fft.rfftfreq(n_fft, 1 / sr)
    res = 2.0 / (t1 - t0)
    partials = own_partials(P, hz, partials, 2 * fmax + 500)

    def peak(fc, tol):
        sel = (hz >= fc - tol) & (hz <= fc + tol)
        return P[sel].max() if sel.any() and fc + tol < hz[-1] else float("nan")

    ph, ct = [], []
    for j in range(2, len(partials) // 2 + 1):
        fph = 2 * partials[j - 1]
        if fph > fmax:
            break
        tol = max(res, 0.0025 * fph)
        below = partials[partials < fph]
        fct = 0.5 * (fph + below[-1]) if len(below) else fph - 0.5 * partials[0]
        clear = lambda fc: np.min(np.abs(partials - fc)) >= 2 * tol + res
        if not (clear(fph) and clear(fct)):
            continue
        ref = peak(partials[2 * j - 1], tol)
        if not ref > 0:
            continue
        ph.append(_db(peak(fph, tol) / ref))
        ct.append(_db(peak(fct, tol) / ref))
    if ph:
        out["N9 phantom"], out["N9 control"], out["N9 n"] = float(np.median(ph)), float(np.median(ct)), float(len(ph))
    return out


def glide(x, sr, t_on, partials, t_end=0.62, early=0.01, span=0.1, n_partials=8, fmax=4000.0, hop=0.004, snr_db=20.0):
    """N3 glide (cents): the pitch of the note's own partials early re late, positive when the note starts sharp
    (tension modulation, C8). A partial's frequency is the phase advance of its DFT (Hann window max(40 ms, 4/f0),
    every ``hop`` s) between frames, summed over frames and channels; early: frame centres over ``span`` s from the
    first window that starts ``early`` s after the onset; late: over ``span`` s up to the last window that ends at
    ``t_end`` s. The partials are the note's own (``own_partials`` on 30 ms to ``t_end``): the first ``n_partials``
    below ``fmax`` that stand ``snr_db`` over the spectrum half-way to their neighbours. ``N3 glide``: the median
    over them; ``N3 n``. A double decay of detuned unison strings moves the apparent pitch too (towards the slower
    string's), with the sign of each key's detuning; tension modulation grows with the square of the amplitude, so
    read it from the velocity dependence."""
    f0 = float(partials[0])
    win = max(0.04, 4.0 / f0)
    c_early, c_late = t_on + early + win / 2, t_on + t_end - win / 2 - span
    if c_late < c_early + span:
        return {}
    seg = _segment(x, sr, t_on + 0.03, t_on + t_end)
    n_fft = 1 << int(math.ceil(math.log2(len(seg) * 4)))
    P, hz = _power(seg, n_fft), np.fft.rfftfreq(n_fft, 1 / sr)
    own = own_partials(P, hz, partials, fmax)
    at = lambda f: P[min(len(P) - 1, int(round(f * n_fft / sr)))]
    ks = [k for k in range(len(own)) if own[k] < fmax
          and at(own[k]) > 10 ** (snr_db / 10) * max(at(own[k] - f0 / 2), at(own[k] + f0 / 2))][:n_partials]
    if not ks:
        return {}
    fk = own[ks]
    n = int(round(win * sr))
    E = np.exp(-2j * np.pi * np.outer(np.arange(n) / sr, fk)) * np.hanning(n)[:, None]  # [n, K]
    hop_n = max(1, int(round(hop * sr)))

    def pitch(c0):  # Hz, from the frames centred over [c0, c0 + span]
        starts = np.arange(int(round(c0 * sr)), int(round((c0 + span) * sr)) + 1, hop_n) - n // 2
        starts = starts[(starts >= 0) & (starts + n <= len(x))]
        A = np.einsum("fnc,nk->fck", np.stack([x[s: s + n] for s in starts]), E)
        A = A * np.exp(-2j * np.pi * np.outer(starts / sr, fk))[:, None, :]  # each partial to its own baseband
        return fk + np.angle((A[1:] * np.conj(A[:-1])).sum((0, 1))) / (2 * np.pi * hop_n / sr)

    g = 1200 * np.log2(pitch(c_early) / pitch(c_late))
    return {"N3 glide": float(np.median(g)), "N3 n": float(len(ks))}


def extra_peaks(x, sr, t_on, partials, t0=0.05, t1=0.65, fmax=8000.0, prominence_db=10.0, new_db=10.0, near=25.0):
    """N12: narrow peaks of the note's spectrum that are not its partials (duplex and aliquot segments, E2; the
    undamped strings, E1; phantoms and longitudinal modes, C7; narrow body modes). Over ``t0``...``t1`` s after the
    onset (Hann, 4x zero padding), a local maximum up to ``fmax`` counts if it is the largest within one main-lobe
    half width ``res``, stands ``prominence_db`` over the median of the spectrum within ±max(25 Hz, 8 res),
    ``new_db`` over the same frequency in as long a stretch before the note (ending 20 ms before the onset),
    ``prominence_db`` over the Hann sidelobes of the nearest partial, and more than 2 res + 0.15 % (unison strings
    a few cents apart) from every partial of the note's own (``own_partials``, tracked), and not stronger than the
    nearest of them (then that partial was missed; nothing but a partial outshines one). Labels: "phantom" within
    res + 0.1 % of an f_j + f_k; else "near" within ``near`` cents of a partial (unison splitting beyond the main
    lobe, undamped strings tuned to the note's partials, duplex segments tuned near them); else "between". Returns
    ``N12 n`` and ``N12 n <label>``, ``N12 level`` and ``N12 level <label>`` (dB, the peaks' summed power re the
    partials' below ``fmax``, when there are any), and ``N12 peaks``: a list of ``(Hz, cents re the nearest
    partial, its number, dB re it, label)``. Sensitivity, from tones added to the recordings' pedal-up R3-R7 notes:
    a component 20 dB under partial 3 is found 35-55 % of the time, 30 dB under 10-25 %, 40 dB under 0-7 %. In
    music a note's surroundings hold earlier notes' partials, so most positions are not "new" (a shorter stretch
    before the note did worse: 24-41 % at -20 dB)."""
    seg = _segment(x, sr, t_on + t0, t_on + t1)
    bg = _segment(x, sr, t_on - 0.02 - (t1 - t0), t_on - 0.02)
    if len(bg) < len(seg):
        return {}
    n_fft = 1 << int(math.ceil(math.log2(len(seg) * 4)))
    P, B = _power(seg, n_fft), _power(bg, n_fft)
    hz = np.fft.rfftfreq(n_fft, 1 / sr)
    T = t1 - t0
    res = 2.0 / T
    own = own_partials(P, hz, partials, fmax, track=True)
    own = own[own < fmax]
    if not len(own):
        return {}
    df = sr / n_fft
    idx = lambda f: int(round(f / df))
    Pk = np.array([P[max(0, idx(f) - 2): idx(f) + 3].max() for f in own])
    lobe = max(1, int(round(res / df)))
    pairs = (own[:, None] + own[None, :])[np.triu_indices(len(own))]
    lo, hi = idx(max(own[0] / 2, 20.0)), min(idx(fmax), len(P) - lobe - 1)
    top = np.lib.stride_tricks.sliding_window_view(P, 2 * lobe + 1).max(1)  # top[i] = max of P[i : i + 2 lobe + 1]
    peaks = []
    for i in lo + np.nonzero(P[lo:hi] >= top[lo - lobe: hi - lobe])[0]:
        f = hz[i]
        k = int(np.argmin(np.abs(own - f)))
        d = abs(f - own[k])
        if d <= 2 * res + 0.0015 * own[k]:
            continue
        w = max(25.0, 8 * res)
        floor = np.median(P[idx(f - w): idx(f + w) + 1])
        side = Pk[k] * 10 ** ((-31.5 - 60 * math.log10(max(d * T, 2.4) / 2.4)) / 10)  # Hann sidelobe envelope
        thr = 10 ** (prominence_db / 10)
        if P[i] < thr * floor or P[i] < thr * side or P[i] < 10 ** (new_db / 10) * B[i - lobe: i + lobe + 1].max():
            continue
        if P[i] > Pk[k]:  # stronger than its partial: the partial was not found (the tenor's high partials), skip
            continue
        cents = 1200 * math.log2(f / own[k])
        if np.min(np.abs(pairs - f)) <= res + 0.001 * f:
            label = "phantom"
        elif abs(cents) <= near:
            label = "near"
        else:
            label = "between"
        peaks.append((float(f), cents, k + 1, _db(P[i] / Pk[k]), label, P[i]))
    out = {"N12 n": float(len(peaks)), "N12 peaks": [p[:5] for p in peaks]}
    ref = Pk.sum()
    for label in ("phantom", "near", "between"):
        sel = [p[5] for p in peaks if p[4] == label]
        out[f"N12 n {label}"] = float(len(sel))
        if sel:
            out[f"N12 level {label}"] = _db(sum(sel) / ref)
    if peaks:
        out["N12 level"] = _db(sum(p[5] for p in peaks) / ref)
    return out


# ------------------------------------------------------------------ N1-N8: one isolated note


def note_measures(x, sr, t_midi, partials, clear_before=1.0, clear_after=0.65, t_on=None):
    """N0-N8 of one clip ``x`` ([T, ch]) whose note has its MIDI onset at ``t_midi`` s and partial frequencies
    ``partials`` (Hz). Returns a flat dict; ``failed`` is set when N0 finds no onset."""
    f0 = float(partials[0])
    fmax = 0.45 * sr
    ps = np.asarray(partials, float)
    ps = ps[ps < min(fmax, 11000.0)]
    t_on = onset(x, sr, t_midi, f0) if t_on is None else t_on
    out = {"onset_ms": 1000 * (t_on - t_midi)}
    if not np.isfinite(t_on):
        out["failed"] = True
        return out
    noise = _segment(x, sr, t_on - min(clear_before, 0.35) + 0.02, t_on - 0.03) if clear_before > 0.15 else None

    # N1 level: energy of the note's partials and broadband, 0-300 ms
    seg = _segment(x, sr, t_on, t_on + 0.3)
    n_fft = 1 << int(math.ceil(math.log2(len(seg) * 2)))
    P, hz = _power(seg, n_fft), np.fft.rfftfreq(n_fft, 1 / sr)
    comb = near_partials(hz, ps, max(2 * sr / n_fft, 0.25 * f0))
    out["N1 level"] = _db(P[comb].sum())
    out["N1 level broadband"] = _db(P[(hz >= 30) & (hz < 11000)].sum())

    # N2 spectrum: partials 1-16 at 30-130 ms
    p16 = np.asarray(partials, float)[:16]
    lv = partial_levels(x, sr, t_on + 0.03, t_on + 0.13, p16, f0, noise)
    ok = np.isfinite(lv)
    top = lv[ok].max() if ok.any() else np.nan
    for k in range(len(p16)):
        out[f"N2 partial {k + 1}"] = float(lv[k] - top) if ok[k] else float("nan")
    out["N2 slope"] = slope_db_oct(lv, np.arange(1, len(p16) + 1))
    w = 10 ** (lv[ok] / 10)
    out["N2 centroid"] = 1200 * math.log2((p16[ok] * w).sum() / w.sum() / f0) if ok.any() else float("nan")  # cents re f0

    # N4 rise: 10 -> 90 % of the peak (first 150 ms) of the note's power averaged over one period (a sum of
    # partials beats at the partial spacing, so nothing finer is defined). Resolution ~1/f0: only for f0 >= 196 Hz.
    if f0 >= 196.0:
        e = band_envelope(_segment(x, sr, t_on - 0.08, t_on + 0.16), sr, max(100.0, 0.75 * f0), min(8000.0, fmax), max(0.001, 1 / f0))
        i0 = int(0.08 * sr)
        bg = np.median(e[: int(0.05 * sr)])
        s = e[i0 - int(0.01 * sr): i0 + int(0.15 * sr)] - bg
        pk = s.max()
        out["N4 rise"] = 1000 * (np.argmax(s >= 0.9 * pk) - np.argmax(s >= 0.1 * pk)) / sr if pk > 3 * max(bg, 1e-30) else float("nan")

    # N5 attack spectrum: octave bands in 0-30, 30-100, 100-400 ms, each re the note's own 100-400 ms (0.2-8 kHz)
    def bands(t0, t1):
        sg = _segment(x, sr, t_on + t0, t_on + t1)
        nf = 1 << int(math.ceil(math.log2(len(sg) * 2)))
        Q, f = _power(sg, nf), np.fft.rfftfreq(nf, 1 / sr)
        return Q, f

    Qs, fs = bands(0.1, 0.4)
    ref = Qs[(fs >= 200) & (fs < 8000)].sum()
    for tag, (t0, t1) in (("attack", (0.0, 0.03)), ("early", (0.03, 0.1)), ("sustain", (0.1, 0.4))):
        Q, f = bands(t0, t1)
        scale = (0.3 / (t1 - t0))  # per unit time, so windows of different length compare
        for c in OCTAVES:
            sel = (f >= c / math.sqrt(2)) & (f < min(c * math.sqrt(2), fmax))
            out[f"N5 {tag} {c}"] = _db(Q[sel].sum() * scale / ref) if sel.any() else float("nan")

    # N6 knock: energy between the partials in -3..33 ms (a flat-topped window, so the first milliseconds count),
    # minus the same window's background at -83..-47 ms, per band re the note's 100-400 ms energy. Below f0 it is
    # all knock (and background). Bins within max(70 Hz, f0/4) of a partial are left out, so it needs f0 above
    # ~140 Hz (R3 and up) except at the top of the spectrum, where inharmonicity spreads the partials. A brighter string leaks a little into it (contact time x0.7: +1-5 dB at 1-4 kHz);
    # +20 dB of knock noise reads +10-30 dB (scripts/note_bench.py --validate). The earlier "N6 attack" (a Hann
    # window from -5 ms, background kept) read +3 dB for +20 dB of knock.
    from scipy.signal.windows import tukey

    ref = _note_ref(x, sr, t_on)
    spec = {}
    for tag, (t0, t1) in (("att", (-0.003, 0.033)), ("bg", (-0.083, -0.047))):
        sg = _segment(x, sr, t_on + t0, t_on + t1)
        nf = 1 << int(math.ceil(math.log2(len(sg) * 4)))
        spec[tag] = ((np.abs(np.fft.rfft(sg * tukey(len(sg), 0.3)[:, None], nf, axis=0)) ** 2).sum(1), np.fft.rfftfreq(nf, 1 / sr))
    (Q, f), (Qb, _) = spec["att"], spec["bg"]
    away = ~near_partials(f, ps, max(70.0, 0.25 * f0))
    for c in KNOCK_BANDS:
        sel = (f >= c / math.sqrt(2)) & (f < min(c * math.sqrt(2), fmax)) & away
        if sel.sum() >= 3 and ref > 0:
            e = max(Q[sel].sum() - Qb[sel].sum(), 1e-3 * Q[sel].sum())
            out[f"N6 knock {c}"] = _db(e / 0.036 / ref)

    # N4 percussive: the attack's sharpness per band. Median-filter harmonic/percussive separation (soft masks) of
    # 21 ms frames every 2.7 ms; percussive energy in -5..40 ms minus its background at -150..-60 ms, re the note's
    # 100-400 ms energy. It reads the knock and the partials' own onset alike (contact time x0.7: +5-7 dB at 2-4 kHz).
    out.update(percussive(x, sr, t_on, ref))

    # N6 sustain: energy between the partials at 100-400 ms (the pedal halo, E1), per band and 0.2-8 kHz
    sg = _segment(x, sr, t_on + 0.1, t_on + 0.4)
    nf = 1 << int(math.ceil(math.log2(len(sg) * 4)))
    Q, f = _power(sg, nf), np.fft.rfftfreq(nf, 1 / sr)
    away = ~near_partials(f, ps, max(15.0, 0.05 * f0))
    for c in OCTAVES:
        sel = (f >= c / math.sqrt(2)) & (f < min(c * math.sqrt(2), fmax))
        ok = sel & away
        out[f"N6 sustain {c}"] = _db(Q[ok].sum() / Q[sel].sum()) if ok.sum() >= 3 and Q[sel].sum() > 0 else float("nan")
    sel = (f >= 200) & (f < 8000)
    out["N6 sustain all"] = _db(Q[sel & away].sum() / Q[sel].sum()) if (sel & away).sum() >= 3 else float("nan")

    # N9 phantoms: the level at 2 f_j re partial 2j (tens of Hz above it), 30-430 ms, where the two are resolved
    out.update(phantom_levels(x, sr, t_on, np.asarray(partials, float)))

    # N7 pre-onset: excess over the background's trend, in 10 ms frames, -200..-8 ms (needs 0.65 s clear before)
    if clear_before >= 0.65:
        for name, lo, hi in (("low", 30, 1000), ("high", 1000, 8000)):
            e = band_envelope(_segment(x, sr, t_on - 0.62, t_on), sr, lo, hi, 0.01)
            fr = int(0.01 * sr)
            Lf = 10 * np.log10(e[: len(e) // fr * fr].reshape(-1, fr).mean(1) + 1e-30)
            tf = -0.62 + (np.arange(len(Lf)) + 0.5) * 0.01
            bgm = (tf >= -0.6) & (tf <= -0.25)
            prm = (tf >= -0.2) & (tf <= -0.008)
            a, b = np.polyfit(tf[bgm], Lf[bgm], 1)
            excess = np.convolve(Lf - (a * tf + b), np.ones(3) / 3, "same")  # 30 ms: a burst, not one frame
            out[f"N7 pre-onset {name}"] = float(excess[prm].max())

    # N8 early decay of partials 1-8: level at 30-100 ms minus level at 415-485 ms
    if clear_after >= 0.5:
        l1 = partial_levels(x, sr, t_on + 0.03, t_on + 0.1, ps[:8], f0, noise)
        l2 = partial_levels(x, sr, t_on + 0.415, t_on + 0.485, ps[:8], f0, noise)
        for k in range(min(8, len(ps))):
            out[f"N8 drop {k + 1}"] = float(l1[k] - l2[k])
        d = l1 - l2
        out["N8 drop 1-4"] = float(np.nanmedian(d[:4])) if np.isfinite(d[:4]).any() else float("nan")
        out["N8 drop 5-8"] = float(np.nanmedian(d[4:8])) if len(d) > 4 and np.isfinite(d[4:8]).any() else float("nan")
    return out


# ------------------------------------------------------------------ N10: release


def release_events(notes, pedals, hold=0.3, quiet_on=(0.1, 0.3), lo=48, undamped=89):
    """N10 events in any texture: note-offs of pitch >= ``lo`` (f0 >= 130 Hz: partials far enough apart to track
    one at a time) held >= ``hold`` s, no onset in [off - 0.1, off + 0.3], sustain and sostenuto pedals below the
    damper threshold from 0.1 s before to 0.4 s after. Other notes may sound: ``others`` lists the pitches ringing
    in [off - 0.05, off + 0.3] (their keys down, or undamped keys struck in the 5 s before), and the measure uses
    only the released note's partials clear of theirs. ``[(row, others)]``."""
    on, off = notes[:, 1], notes[:, 2]
    thr = PEDAL[0][2]
    out = []
    for i in range(len(notes)):
        t = off[i]
        if notes[i, 0] < lo or notes[i, 0] >= undamped or t - on[i] < hold:
            continue
        if np.any((on > t - quiet_on[0]) & (on < t + quiet_on[1])):
            continue
        ok = True
        for name in ("sustain", "sostenuto"):
            ts, vs = pedals[name + "_t"], pedals[name + "_v"]
            if _value_at(ts, vs, t - 0.1) >= thr or np.any(vs[(ts > t - 0.1) & (ts < t + 0.4)] >= thr):
                ok = False
        if not ok:
            continue
        # keys down around the note-off (notes released earlier are damped already), and undamped keys still ringing
        ring = ((on < t + 0.3) & (off > t - 0.05)) | ((notes[:, 0] >= undamped) & (on > t - 5.0) & (on < t + 0.3))
        ring[i] = False
        out.append((i, sorted({int(p) for p in notes[ring, 0]})))
    return out


def partial_tracks(x, sr, t0, t1, freqs, win, hop=0.002):
    """Level (dB, power summed over channels) of each frequency in ``freqs`` over time: a Hann window of ``win`` s
    centred every ``hop`` s from ``t0`` to ``t1``, the DFT evaluated at each frequency exactly. ``(t, L[t, k])``."""
    n = int(round(win * sr))
    w = np.hanning(n)
    E = np.exp(-2j * np.pi * np.outer(np.arange(n) / sr, freqs)) * w[:, None]  # [n, K]
    centres = np.arange(t0, t1, hop)
    starts = np.round(centres * sr).astype(int) - n // 2
    ok = (starts >= 0) & (starts + n <= len(x))
    centres, starts = centres[ok], starts[ok]
    frames = np.stack([x[s: s + n] for s in starts])  # [F, n, ch]
    A = np.einsum("fnc,nk->fck", frames, E)
    return centres, 10 * np.log10((np.abs(A) ** 2).sum(1) + 1e-30)


def release_tracks(x, sr, t_off, partials, others=(), n_partials=8, fmax=4000.0, search=(-0.05, 0.25)):
    """N10 from the released note's own partials (up to ``n_partials`` below ``fmax`` that are clear, by 1.5
    main-lobe half widths, of every partial of the ``others``' partial lists). Each partial's level track (window
    max(50 ms, 8/f0), every 2 ms, 0.35 s before to 0.3 s after the note-off) is fitted by two lines with a step
    between them, the breakpoint searched over ``search`` s re the note-off: at the microphones the direct sound
    is only a few dB above the reverberant field, so the dampers show as a step down and a steeper slope (the hall's
    decay instead of the string's), not as a deep drop. The breakpoint falls where the direct sound has fallen
    to the reverberant field's level, so it follows the damper's contact plus part of its fall (+18 ms for a
    synthetic 400 dB/s fall): read it as a difference between model and recording. Medians over partials: ``N10 delay`` (ms, breakpoint re
    the note-off), ``N10 step`` (dB), ``N10 slope after`` and ``N10 slope before`` (dB/s), ``N10 n`` (partials).
    A partial counts only if the two-line fit beats one line by 3 dB of residual power and steps or steepens by
    at least 1.5 dB or 15 dB/s; ``N10 damped`` is the share of the partials that do."""
    f0 = float(partials[0])
    win = max(0.05, 8.0 / f0)
    res = 2.0 / win  # the Hann window's main-lobe half width (Hz)
    others = [np.asarray(o, float) for o in others]
    ks = []
    for k in range(len(partials)):
        fk = float(partials[k])
        if fk >= fmax or len(ks) >= n_partials:
            break
        if all(np.min(np.abs(o - fk)) > 1.5 * res for o in others if len(o)):
            ks.append(k)
    out = {"N10 n": 0.0}
    if not ks:
        return out
    t, L = partial_tracks(x, sr, t_off - 0.35, t_off + 0.3, np.asarray(partials, float)[ks], win)
    fits = []
    for j in range(len(ks)):
        y = L[:, j]
        one = np.polyfit(t, y, 1)
        sse1 = ((y - np.polyval(one, t)) ** 2).sum()
        best = None
        for tb in np.arange(t_off + search[0], t_off + search[1], 0.002):
            a, b = t < tb, t >= tb
            if a.sum() < 20 or b.sum() < 15:
                continue
            p1, p2 = np.polyfit(t[a] - tb, y[a], 1), np.polyfit(t[b] - tb, y[b], 1)
            sse = ((y[a] - np.polyval(p1, t[a] - tb)) ** 2).sum() + ((y[b] - np.polyval(p2, t[b] - tb)) ** 2).sum()
            if best is None or sse < best[0]:
                best = (sse, tb, p2[1] - p1[1], p1[0], p2[0])
        if best is None:
            continue
        sse, tb, step, s1, s2 = best
        fits.append((sse < 0.5 * sse1 and (step < -1.5 or s2 < s1 - 15.0), tb, step, s1, s2))
    if not fits:
        return out
    good = [f for f in fits if f[0]]
    out.update({"N10 n": float(len(fits)), "N10 damped": len(good) / len(fits)})
    if good:
        g = np.array([f[1:] for f in good])
        out.update({"N10 delay": 1000 * (float(np.median(g[:, 0])) - t_off), "N10 step": float(np.median(g[:, 1])),
                    "N10 slope before": float(np.median(g[:, 2])), "N10 slope after": float(np.median(g[:, 3]))})
    return out


# ------------------------------------------------------------------ P4: room and image


def free_decays(notes, pedals, duration, gap=0.7, hold=0.2, undamped=89, undamped_clear=4.0, pre=0.4):
    """Times ``(t_stop, t_next)`` (s) when everything stops: the last damper falls (every key up, sustain and
    sostenuto below the damper threshold) and no onset follows for ``gap`` s. The last note sounded at least
    ``hold`` s; no undamped key (>= ``undamped``) in the ``undamped_clear`` s before; the pedals stay down-free
    through the gap. What rings then is the room (and a little of the strings, until the dampers have acted)."""
    on, off = notes[:, 1], notes[:, 2]
    thr = PEDAL[0][2]
    st, sv = pedals["sustain_t"], pedals["sustain_v"]

    def pedal_release_after(t):  # first time at or after t with the sustain below the threshold
        if _value_at(st, sv, t) < thr:
            return t
        k = np.searchsorted(st, t, side="right")
        below = np.nonzero(sv[k:] < thr)[0]
        return float(st[k + below[0]]) if len(below) else float("inf")

    damp = np.array([pedal_release_after(t) for t in off])
    order = np.argsort(on, kind="stable")
    out, last = [], -np.inf
    for n, i in enumerate(order):
        last = max(last, damp[i])
        t_next = on[order[n + 1]] if n + 1 < len(order) else duration
        if not np.isfinite(last) or t_next - last < gap or last < pre + WARMUP + 0.1 or last + gap > duration:
            continue
        sounding = (on < last - hold) & (damp >= last - 1e-6)
        if not sounding.any():
            continue
        if np.any((notes[:, 0] >= undamped) & (on > last - undamped_clear) & (on <= last)):
            continue
        ok = True
        for name in ("sustain", "sostenuto"):
            ts, vs = pedals[name + "_t"], pedals[name + "_v"]
            if np.any(vs[(ts > last) & (ts < last + gap)] >= thr) or (name == "sostenuto" and _value_at(ts, vs, last - 1.0) >= thr):
                ok = False
        if ok:
            out.append((float(last), float(t_next)))
    return out


def band_floor(floor, sr, bands=OCTAVES):
    """Power per octave band (the ``band_envelope`` scale) of a floor-noise recording ``[T, ch]``."""
    return {c: float(np.mean(band_envelope(floor, sr, c / math.sqrt(2), min(c * math.sqrt(2), 0.45 * sr), 0.01)))
            for c in bands}


def room_measures(x, sr, t_stop, t_end, floor=None, bands=(125,) + OCTAVES, smooth=0.01):
    """P4 decay of one clip whose sound stops at ``t_stop`` s (the MIDI's last damper) with nothing new until
    ``t_end``. Per octave band: ``P4 T60`` (s) from a line fitted to the level from ``t_stop + a`` (0.4 s below
    500 Hz, where the bass dampers are slow; 0.25 s to 1 kHz; 0.15 s above, where the room decays fast and a long
    extrapolation back to the stop costs dB) while it is 10 dB above ``floor``
    (per band power); ``P4 early`` (dB): the level over the 50 ms before ``t_stop`` over that line extrapolated to
    ``t_stop``, i.e. the direct sound and early reflections of what was sounding, above its reverberant field.
    Both are anchored on the MIDI, not on a detected knee: model and recording are then read at the same times.
    ``P4 knee ms``: where the broadband level (0.1-8 kHz) begins its steepest 20 ms descent (at half that slope,
    within -50..+250 ms), re ``t_stop``; the 10 ms smoothing puts it ~13 ms early on an instant stop."""
    out = {}
    L = 10 * np.log10(band_envelope(x, sr, 100.0, min(8000.0, 0.45 * sr), smooth) + 1e-30)
    step = max(1, sr // 1000)
    Ls, t = L[::step], np.arange(0, len(L), step) / sr
    lag = max(1, int(0.02 * sr / step))
    slope = np.full(len(Ls), np.inf)
    slope[:-lag] = (Ls[lag:] - Ls[:-lag]) / (lag * step / sr)
    win = np.nonzero((t >= t_stop - 0.05) & (t <= t_stop + 0.25))[0]
    if not len(win):
        return out
    k = win[np.argmin(slope[win])]
    if slope[k] > -30:  # no descent at all: not a stop
        return out
    j = k
    while j > win[0] and slope[j - 1] < 0.5 * slope[k]:
        j -= 1
    out["P4 knee ms"] = 1000 * (float(t[j]) - t_stop)
    t_all = np.arange(len(x)) / sr
    for c in bands:
        e = band_envelope(x, sr, c / math.sqrt(2), min(c * math.sqrt(2), 0.45 * sr), smooth)
        fl = floor.get(c, 0.0) if floor else 0.0
        Lb = 10 * np.log10(np.maximum(e - fl, 1e-30))
        a = 0.4 if c < 500 else 0.25 if c < 2000 else 0.15  # the high bands decay fast: a long lever arm costs dB
        fit = (t_all >= t_stop + a) & (t_all <= t_end - 0.02) & (e > 10 * fl)
        idx = np.nonzero(fit)[0]
        if len(idx) < int(0.2 * sr) or idx[-1] - idx[0] < int(0.2 * sr):
            continue
        idx = idx[:: max(1, sr // 1000)]
        s, b = np.polyfit(t_all[idx], Lb[idx], 1)
        if s >= -5:  # not decaying
            continue
        # a line fitted to log power sits under the mean power (by ~2 dB at 125 Hz, where a 10 ms envelope has few
        # degrees of freedom): move it onto the mean
        b += 10 * math.log10(np.mean(np.maximum(e[idx] - fl, 1e-30) / 10 ** ((s * t_all[idx] + b) / 10)))
        out[f"P4 T60 {c}"] = -60.0 / s
        near = (t_all >= t_stop - 0.05) & (t_all <= t_stop)
        out[f"P4 early {c}"] = float(10 * np.log10(np.mean(np.maximum(e[near] - fl, 1e-30))) - (s * t_stop + b))
    return out


def partial_profile(x, sr, t_on, freqs, t_end, win=0.04, hop=0.005, bg=(-0.45, -0.1), min_snr_db=6.0):
    """T1/T2 on one note: each partial's level over the note and what it does (docs/tone_measures.md 15).

    ``freqs[K]`` (Hz, the note's own partials), tracked with ``partial_tracks`` (a ``win`` s Hann window, so level
    changes up to ~12 Hz are followed) from 30 ms before ``t_on`` to ``t_on + t_end`` (s re the clip). A point counts
    where it stands ``min_snr_db`` over the partial's level in the background window ``bg`` (s re ``t_on``); a value
    needs 60 % of its window counted, else NaN. Returns ``t`` [F] (s re ``t_on``), ``L`` [F, K] dB, ``bg`` [K] dB and,
    per partial [K]:
    - ``peak`` (dB) and ``t_peak`` (s) over -10..150 ms;
    - ``early`` and ``late``: the decay (dB/s, a line fitted to the counted points) over 50-350 ms and 500 ms to
      ``t_end``: a two-stage decay shows as ``early`` steeper than ``late``;
    - ``fluct`` (dB rms) and ``beat`` (Hz): the track from 80 ms to ``t_end`` minus a cubic in time (which takes the
      smooth decay, the two-stage knee included), its rms and its strongest periodicity between 1.2 and 12 Hz
      (slower beating than about one cycle in the window reads as part of the decay); ``periodic``: the share of the
      residual's power within 0.6 Hz of that rate (one steady beat: near 1; irregular fluctuation: low).
    The decays are fitted jointly with a sinusoid at ``beat`` where the window holds a full cycle: a unison's strings
    start in phase, so a line alone reads the same part of the beat's cycle every time. A beat slower than one cycle
    per window (3.3 Hz for ``early``) still biases it."""
    t, L = partial_tracks(x, sr, t_on - 0.03, t_on + t_end, freqs, win, hop)
    t = t - t_on
    _, Lb = partial_tracks(x, sr, t_on + bg[0], t_on + bg[1], freqs, win, 0.02)
    bgl = 10 * np.log10(np.mean(10 ** (Lb / 10), 0) + 1e-30) if len(Lb) else np.full(len(freqs), -300.0)
    valid = L >= bgl[None] + min_snr_db
    K = len(freqs)
    out = {"t": t, "L": L, "bg": bgl}
    for key in ("peak", "t_peak", "early", "late", "fluct", "beat", "periodic"):
        out[key] = np.full(K, np.nan)

    def fit(k, a, b):
        """The decay over [a, b]: a line, jointly with a sinusoid at the partial's beat rate when the window holds a
        full cycle of a beat that shows (a line alone reads a 6 Hz beat of +-2.3 dB on -10 dB/s as -18)."""
        sel = (t >= a) & (t <= b)
        ok = sel & valid[:, k]
        if sel.sum() < 4 or ok.sum() < 0.6 * sel.sum():
            return np.nan
        tt = t[ok]
        X = [np.ones_like(tt), tt]
        f = out["beat"][k]
        if np.isfinite(f) and out["fluct"][k] > 0.3 and f * (b - a) >= 1.0:
            X += [np.sin(2 * np.pi * f * tt), np.cos(2 * np.pi * f * tt)]
        coef = np.linalg.lstsq(np.stack(X, 1), L[ok, k], rcond=None)[0]
        return float(coef[1])

    for k in range(K):
        pk = (t >= -0.01) & (t <= 0.15)
        if pk.any() and valid[pk, k].any():
            i = np.argmax(np.where(valid[pk, k], L[pk, k], -np.inf))
            out["peak"][k], out["t_peak"][k] = L[pk, k][i], t[pk][i]
        sel = (t >= 0.08) & (t <= t_end - win / 2)
        if sel.sum() >= 40 and valid[sel, k].mean() >= 0.9:
            tt, ll = t[sel], L[sel, k]
            r = ll - np.polyval(np.polyfit(tt, ll, 3), tt)
            out["fluct"][k] = float(np.sqrt(np.mean(r ** 2)))
            n_fft = 1 << int(math.ceil(math.log2(len(r) * 16)))
            S = np.abs(np.fft.rfft(r * np.hanning(len(r)), n_fft))
            hz = np.fft.rfftfreq(n_fft, hop)
            band = (hz >= 1.2) & (hz <= 12.0)
            j = np.argmax(np.where(band, S, -1.0))
            out["beat"][k] = float(hz[j])
            near = np.abs(hz - hz[j]) <= 0.6  # the main lobe of the Hann window over the residual (~0.8 s): +-0.5 Hz
            out["periodic"][k] = float((S[near] ** 2).sum() / max((S[hz <= 25.0] ** 2).sum(), 1e-30))
        out["early"][k] = fit(k, 0.05, 0.35)
        if t_end >= 0.75:
            out["late"][k] = fit(k, 0.5, t_end - win / 2)
    return out


ONSET_WINDOWS = ((0.0, 0.005), (0.005, 0.01), (0.01, 0.02), (0.02, 0.04))


def onset_profile(x, sr, t_on, f0, bands=KNOCK_BANDS, span=(-0.02, 0.12), ref=(0.05, 0.1), bg=(-0.08, -0.03),
                  smooth=None, step=0.0005, windows=ONSET_WINDOWS):
    """The first tens of ms of one note, per octave band (docs/tone_measures.md 16, item 1).

    Each band's power (``band_envelope``: a zero-phase band filter, summed over channels) minus its background (the
    mean over ``bg``, s re ``t_on``), divided by the band's own mean over ``ref``: the band's envelope re its level
    once the note has settled, so a band that arrives late or overshoots shows whatever the spectral balance. The
    envelope is averaged over ``smooth`` s (default per band: one period of f0 where the band holds two partials or
    more, since they beat at their spacing, else the band's own resolution 1 / (0.7 c); at least 1 ms) and sampled
    every ``step`` s over ``span``. Returns ``t`` [F] (s re ``t_on``), ``P`` [F, B] (linear, re ``ref``),
    ``ref_db`` [B] (each band's ``ref`` level re the note's total at 0.2-8 kHz in ``ref``, dB) and per band [B]:
    - ``peak`` (dB re ``ref``) and ``t_peak`` (s) over -5..60 ms: the attack's prominence over the settled note;
    - ``arrival`` (s): where the envelope first reaches 10 % of that peak, from -10 ms;
    - ``rise`` (s): from 10 % to 90 % of the peak;
    - ``win`` [W, B]: the mean level (dB re ``ref``) in each window of ``windows`` (s re ``t_on``), unsmoothed.
    A band whose peak is not 6 dB over its background, or whose ``ref`` level is not, reads NaN."""
    t_lo = min(span[0], bg[0]) - 0.03
    t_hi = max(span[1], ref[1]) + 0.03
    a = int(round((t_on + t_lo) * sr))
    seg = x[max(a, 0): max(a, 0) + int(round((t_hi - t_lo) * sr))]
    t_seg = (np.arange(len(seg)) + max(a, 0)) / sr - t_on
    t = np.arange(span[0], span[1] + step / 2, step)
    B = len(bands)
    out = {"t": t, "P": np.full((len(t), B), np.nan), "ref_db": np.full(B, np.nan)}
    for key in ("peak", "t_peak", "arrival", "rise"):
        out[key] = np.full(B, np.nan)
    out["win"] = np.full((len(windows), B), np.nan)
    if len(seg) < sr * 0.05:
        return out

    def mean_in(e, lo, hi):
        m = (t_seg >= lo) & (t_seg < hi)
        return float(e[m].mean()) if m.any() else float("nan")

    tot = mean_in(band_envelope(seg, sr, 200.0, min(8000.0, 0.45 * sr), 1.0 / sr), *ref)
    for j, c in enumerate(bands):
        lo, hi = c / math.sqrt(2), min(c * math.sqrt(2), 0.45 * sr)
        if lo >= hi:
            continue
        sm = smooth if smooth is not None else max(0.001, 1.0 / f0 if hi > 2 * f0 else 1.0 / (0.7 * c))
        raw = band_envelope(seg, sr, lo, hi, 1.0 / sr)
        e = band_envelope(seg, sr, lo, hi, sm)
        b, r = mean_in(raw, *bg), mean_in(raw, *ref)
        if not r > 4 * b or not r > 0:
            continue
        out["ref_db"][j] = _db(r / tot) if tot > 0 else float("nan")
        P = (np.interp(t, t_seg, e) - b) / (r - b)
        out["P"][:, j] = P
        for w, (w0, w1) in enumerate(windows):
            v = (mean_in(raw, w0, w1) - b) / (r - b)
            out["win"][w, j] = _db(v) if v > 0 else float("nan")
        pk = (t >= -0.005) & (t <= 0.06)
        i = np.argmax(np.where(pk, P, -np.inf))
        if P[i] * (r - b) < 3 * b:  # the peak not 6 dB over the background
            continue
        out["peak"][j], out["t_peak"][j] = _db(P[i]), t[i]
        k0 = np.nonzero(t >= -0.01)[0][0]
        up10 = np.nonzero(P[k0: i + 1] >= 0.1 * P[i])[0]
        up90 = np.nonzero(P[k0: i + 1] >= 0.9 * P[i])[0]
        if len(up10) and len(up90):
            out["arrival"][j] = t[k0 + up10[0]]
            out["rise"][j] = t[k0 + up90[0]] - t[k0 + up10[0]]
    return out


def non_tonal(x, sr, t_on, partials, windows=((-0.003, 0.04), (0.1, 0.4), (0.5, 0.95)), bands=(125,) + OCTAVES):
    """The energy away from the note's partials (bins farther than max(70 Hz, f0 / 4) from every partial, as N6), per
    octave band and window (s re ``t_on``), in dB re the note's whole energy in the same window: the knock, the
    noise and (pedal down) the halo against the tone. ``{(window, band): dB}``; NaN where a band keeps no bins."""
    out = {}
    f0 = float(partials[0])
    for a, b in windows:
        seg = _segment(x, sr, t_on + a, t_on + b)
        n_fft = 1 << int(math.ceil(math.log2(max(len(seg), 2) * 2)))
        P, hz = _power(seg, n_fft), np.fft.rfftfreq(n_fft, 1 / sr)
        away = ~near_partials(hz, partials, max(70.0, 0.25 * f0))
        tot = P[(hz >= 50) & (hz < 0.45 * sr)].sum()
        for c in bands:
            m = band_mask(hz, c / math.sqrt(2), c * math.sqrt(2)) * away
            out[((a, b), c)] = 10 * math.log10(max((P * m).sum(), 1e-30) / max(tot, 1e-30)) if m.sum() > 2 else float("nan")
    return out


def channel_measures(x, sr, t_on, bands=OCTAVES, max_lag=0.003):
    """P4 image of one note's clip (onset at ``t_on`` s), per octave band: ``P4 iacc early/late`` the largest
    normalised cross-correlation of the two channels within ``max_lag`` s, over 0-30 ms (direct sound and first
    reflections: close to 1) and over 150-400 ms (the reverberant field: low); ``P4 lr`` the level of channel 0
    over channel 1 (dB) over 30-300 ms."""
    out = {}
    if x.ndim < 2 or x.shape[1] < 2:
        return out
    X = np.fft.rfft(x, axis=0)
    f = np.fft.rfftfreq(len(x), 1 / sr)
    m = int(max_lag * sr)
    for c in bands:
        Y = X.copy()
        Y[(f < c / math.sqrt(2)) | (f >= min(c * math.sqrt(2), 0.45 * sr))] = 0
        y = np.fft.irfft(Y, len(x), axis=0)
        for tag, (a, b) in (("early", (0.0, 0.03)), ("late", (0.15, 0.4))):
            i0, i1 = int((t_on + a) * sr), int((t_on + b) * sr)
            if i0 - m < 0 or i1 + m > len(y):
                continue
            l = y[i0:i1, 0]
            norm = math.sqrt((l ** 2).sum() * (y[i0:i1, 1] ** 2).sum())
            if norm <= 0:
                continue
            r = [abs((l * y[i0 + d: i1 + d, 1]).sum()) / norm for d in range(-m, m + 1)]
            out[f"P4 iacc {tag} {c}"] = float(max(r))
        i0, i1 = int((t_on + 0.03) * sr), int((t_on + 0.3) * sr)
        out[f"P4 lr {c}"] = _db((y[i0:i1, 0] ** 2).sum() / max((y[i0:i1, 1] ** 2).sum(), 1e-30))
    return out


# ------------------------------------------------------------------ F4: re-strike


def repeat_runs(notes, pedals, gap=(0.08, 0.6), span=1.2, clear=2.0):
    """F4: runs of repeated strikes of one key while the sustain pedal holds it: each strike 0.08-0.6 s after the
    key's previous one, the pedal down (>= 76) from the run's first strike to its last, no strike of the key in the
    ``clear`` s before the run; strikes within ``span`` s of the first. ``[{"pitch", "times", "velocities",
    "index"}]`` (``index``: the first strike's row)."""
    down = PEDAL[-1][1]
    st, sv = pedals["sustain_t"], pedals["sustain_v"]
    out = []
    for pitch in np.unique(notes[:, 0]):
        rows = np.nonzero(notes[:, 0] == pitch)[0]
        rows = rows[np.argsort(notes[rows, 1], kind="stable")]
        t = notes[rows, 1]
        k = 0
        while k < len(rows) - 1:
            if k > 0 and t[k] - t[k - 1] < clear:
                k += 1
                continue
            m = k
            while m + 1 < len(rows) and gap[0] <= t[m + 1] - t[m] <= gap[1] and t[m + 1] - t[k] <= span:
                m += 1
            if m > k:
                inside = sv[(st > t[k]) & (st < t[m])]
                if _value_at(st, sv, t[k]) >= down and np.all(inside >= down):
                    out.append({"pitch": int(pitch), "times": [float(v) for v in t[k: m + 1]],
                                "velocities": [int(notes[r, 3]) for r in rows[k: m + 1]], "index": int(rows[k])})
            k = m + 1
    return sorted(out, key=lambda r: r["times"][0])


def restrike_levels(x, sr, t_first, times, partials, window=(0.02, 0.1)):
    """F4: the level (dB) of the key's partials (below 8 kHz) in ``window`` after each strike of a run; ``times``
    are the strikes (s, absolute) and ``t_first`` the clip time of the first. Compared as model - recording at
    strike k minus the same at the first strike: a model whose re-strikes leave too much of the ringing string
    builds up across the run."""
    f0 = float(partials[0])
    ps = np.asarray(partials, float)
    ps = ps[ps < 8000]
    out = []
    for tk in times:
        a = t_first + tk - times[0]
        seg = _segment(x, sr, a + window[0], a + window[1])
        n_fft = 1 << int(math.ceil(math.log2(len(seg) * 2)))
        P, hz = _power(seg, n_fft), np.fft.rfftfreq(n_fft, 1 / sr)
        out.append(_db(P[near_partials(hz, ps, max(2 * sr / n_fft, 0.25 * f0))].sum()))
    return out


# ------------------------------------------------------------------ E4 and P3: music


def onset_flux(x, sr, onsets, bands=OCTAVES, n_win=256, hop=64, span=(-0.005, 0.04)):
    """E4: positive spectral flux (dB) per octave band, summed over the frames within ``span`` of each onset (s)."""
    w = np.hanning(n_win)[:, None]
    f = np.fft.rfftfreq(n_win, 1 / sr)
    starts = np.arange(0, len(x) - n_win, hop)
    P = np.stack([(np.abs(np.fft.rfft(x[s: s + n_win] * w, axis=0)) ** 2).sum(1) for s in starts])
    B = np.stack([P[:, (f >= c / math.sqrt(2)) & (f < c * math.sqrt(2))].sum(1) for c in bands], 1)
    L = 10 * np.log10(B + 1e-20)
    flux = np.vstack([np.zeros((1, len(bands))), np.clip(np.diff(L, axis=0), 0, None)])
    t = (starts + n_win / 2) / sr
    return np.array([flux[(t >= o + span[0]) & (t <= o + span[1])].sum(0) for o in onsets]).reshape(-1, len(bands))


def texture_stats(x, sr, bands=(125, 250, 500, 1000, 2000, 4000, 8000), frame=0.005, mod_edges=(0.5, 1, 2, 4, 8, 16, 32, 64)):
    """P3: per octave band, statistics of the compressed envelope (amplitude^0.3 of 5 ms band power): mean, CV,
    skewness, the share of modulation power in each octave of ``mod_edges`` (Hz), and the correlation of adjacent
    bands' envelopes (McDermott & Simoncelli 2011)."""
    fr = int(frame * sr)
    envs = []
    for c in bands:
        e = band_envelope(x, sr, c / math.sqrt(2), min(c * math.sqrt(2), 0.45 * sr), frame)
        envs.append(e[: len(e) // fr * fr].reshape(-1, fr).mean(1) ** 0.15)  # power^0.15 = amplitude^0.3
    E = np.stack(envs)
    out = {}
    fm = np.fft.rfftfreq(E.shape[1], frame)
    for i, c in enumerate(bands):
        e = E[i]
        m, sd = e.mean(), e.std()
        out[f"P3 mean {c}"] = _db(m ** (2 / 0.3))
        out[f"P3 cv {c}"] = float(sd / m) if m > 0 else float("nan")
        out[f"P3 skew {c}"] = float(((e - m) ** 3).mean() / sd ** 3) if sd > 0 else float("nan")
        M = np.abs(np.fft.rfft(e - m)) ** 2
        tot = M[(fm >= mod_edges[0]) & (fm < mod_edges[-1])].sum()
        for lo, hi in zip(mod_edges[:-1], mod_edges[1:]):
            out[f"P3 mod {c} {lo:g}-{hi:g}"] = _db(M[(fm >= lo) & (fm < hi)].sum() / tot) if tot > 0 else float("nan")
        if i + 1 < len(bands):
            out[f"P3 corr {c}-{bands[i + 1]}"] = float(np.corrcoef(E[i], E[i + 1])[0, 1])
    return out


# ------------------------------------------------------------------ reporting

JND = {"N1": 1.0, "N2 slope": 1.0, "N2 partial": 1.0, "N2 centroid": 85.0, "N4": 2.0, "N4 percussive": 1.0, "N5": 1.0, "N6": 1.0,
       "N7": 1.0, "N8": 1.0, "N9": 1.0, "P4 T60": 0.1, "P4 early": 1.0, "P4 iacc": 0.075, "P4 lr": 1.0, "F4": 1.0,
       "N10 delay": 5.0, "N10 step": 1.0, "N10 slope": 5.0, "onset_ms": 1.0}


def jnd_of(key):
    best = 0.0
    for k, v in JND.items():
        if key.startswith(k):
            best = v
    return best


def paired_summary(rows, label, key, group_of, min_n=5):
    """Per group: n, recording median and IQR, model median, paired median difference, flag (|diff| > max(JND,
    IQR/2)) and spread ratio (model/recording IQR of the residuals after a linear fit on pitch and velocity)."""
    out = {}
    groups = {}
    for r in rows:
        if label in r and key in r["recording"] and key in r[label]:
            a, b = r["recording"][key], r[label][key]
            if np.isfinite(a) and np.isfinite(b):
                groups.setdefault(group_of(r), []).append((a, b, r["pitch"], r["velocity"]))
    for g, v in groups.items():
        if len(v) < min_n:
            continue
        v = np.array(v)
        a, b = v[:, 0], v[:, 1]
        q = lambda z: np.percentile(z, 75) - np.percentile(z, 25)
        X = np.stack([np.ones(len(v)), v[:, 2], v[:, 3]], 1)
        res = lambda z: z - X @ np.linalg.lstsq(X, z, rcond=None)[0]
        iq = q(a)
        d = float(np.median(b - a))
        out[g] = {"n": len(v), "rec": float(np.median(a)), "rec_iqr": float(iq), "model": float(np.median(b)), "diff": d,
                  "flag": abs(d) > max(jnd_of(key), iq / 2), "spread": float(q(res(b)) / max(q(res(a)), 1e-9))}
    return out
