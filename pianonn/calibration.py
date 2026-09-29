"""Measure piano physics from recordings of isolated notes (and from the model, identically).

    python -m pianonn.calibration data/iowa --json data/iowa_analysis.json --out docs/calibration_iowa.md

Recordings may be multichannel. Every spectrum and envelope sums the channels' *powers*, so
microphone positions never comb-filter each other as they would in a mono sum.

Per note (key, dynamic):
- inharmonicity B and tuning, from a stiff-string comb search (B constrained to x/4 of the
  Rigaud et al. curve) and a regression f_n^2 / n^2 = f0^2 + f0^2 B n^2 over the tracked partials;
- the decay profile: level (dB re peak) of the power-summed decay partials at fixed times.
  The partials are chosen by number, with the same rule for recordings and model
  (``decay_partials``: n = 2..5 below C3, where recordings often lack the fundamental; else 1..4);
- a two-segment fit of that envelope with a free breakpoint: prompt T60, aftersound T60, and
  the knee (the aftersound line extrapolated to the onset);
- early partial levels, the spectral slope (dB/oct), a partial-based centroid, and the rise time.
"""

import argparse
import glob
import json
import math
import os
import re

import numpy as np
from scipy.signal import fftconvolve

PITCH_CLASS = {"C": 0, "Db": 1, "D": 2, "Eb": 3, "E": 4, "F": 5, "Gb": 6, "G": 7, "Ab": 8, "A": 9, "Bb": 10, "B": 11}
LANDMARKS = {"A0": 21, "C1": 24, "C2": 36, "C3": 48, "C4": 60, "A4": 69, "C5": 72, "C6": 84, "C7": 96, "C8": 108}
PROFILE_TIMES = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)


def note_to_midi(name):
    m = re.fullmatch(r"([A-G]b?)(-?\d)", name)
    return 12 * (int(m.group(2)) + 1) + PITCH_CLASS[m.group(1)]


def midi_name(p):
    names = [k for k, v in sorted(PITCH_CLASS.items(), key=lambda kv: kv[1])]
    return f"{names[p % 12]}{p // 12 - 1}"


def rigaud_B(pitch):
    """Rigaud, David & Daudet (DAFx 2011) two-asymptote inharmonicity curve (m = MIDI pitch)."""
    return math.exp(0.0926 * pitch - 13.64) + math.exp(-0.0847 * pitch - 5.82)


def decay_partials(pitch):
    """Partial numbers whose summed power defines a note's decay (same rule for recordings and model)."""
    return (2, 3, 4, 5) if pitch < 48 else (1, 2, 3, 4)


def _as_2d(x):
    x = np.asarray(x, dtype=np.float64)
    return x[:, None] if x.ndim == 1 else x


def _moving_average(x, k):
    """Centred moving average of length ``k`` in O(N) (np.convolve is O(N k))."""
    c = np.concatenate([[0.0], np.cumsum(x)])
    lo = np.clip(np.arange(len(x)) - k // 2, 0, len(x))
    hi = np.clip(lo + k, 0, len(x))
    return (c[hi] - c[lo]) / k


def _rms(x, sr, win=0.005):
    return np.sqrt(np.clip(_moving_average((_as_2d(x) ** 2).sum(1), max(1, int(win * sr))), 0, None))


def find_onset(x, sr):
    r = _rms(x, sr)
    return int(np.argmax(r > 0.05 * r.max()))


def _power_spectrum(seg, n_fft, window):
    return (np.abs(np.fft.rfft(seg * window[:, None], n_fft, axis=0)) ** 2).sum(1)


def _peak(spec_db, hz, f, tol):
    band = np.nonzero((hz > f - tol) & (hz < f + tol))[0]
    if len(band) < 3:
        return None
    i = band[np.argmax(spec_db[band])]
    if i in (band[0], band[-1]):
        return None  # edge maximum: no real peak inside the search band
    a, b, c = spec_db[i - 1], spec_db[i], spec_db[i + 1]
    d = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0.0
    local = spec_db[max(0, i - 3 * len(band)): i + 3 * len(band)]
    return hz[i] + d * (hz[1] - hz[0]), b - 0.25 * (a - c) * d, b - np.median(local)


def comb_search(spec_db, hz, f_nominal, max_freq, B_range, n_max=40, cents=100.0, decimate=4):
    """Coarse (f0, B) that best explains the spectrum as a stiff-string comb.

    Scores every (f0, B) on a grid by the prominence (dB above a moving average) of the peaks at
    the predicted partials, counting only clear peaks (> 10 dB), weighting low partials more
    (1/sqrt(n)) and max-pooling over +-0.6 % of each partial's frequency. B is searched only
    within ``B_range``: otherwise near-harmonic components (distortion, phantom partials) that
    dominate treble recordings win with B ~ 0.
    """
    from scipy.ndimage import maximum_filter1d, uniform_filter1d

    spec_db, hz = spec_db[::decimate], hz[::decimate]
    df = hz[1] - hz[0]
    prom = np.clip(spec_db - uniform_filter1d(spec_db, max(3, int(f_nominal / df))) - 10, 0, 30)
    tols = [1, 2, 4, 8, 16, 32, 64, 128]
    pooled = np.stack([maximum_filter1d(prom, 2 * t + 1) for t in tols])
    f0s = f_nominal * 2 ** (np.linspace(-cents, cents, 81) / 1200)
    Bs = np.logspace(math.log10(B_range[0]), math.log10(B_range[1]), 40)
    n = np.arange(1, n_max + 1)
    n = n[: max(2, min(n_max, int(max_freq / (1.02 * f_nominal))))]
    w = 1 / np.sqrt(n)
    f = f0s[:, None, None] * (n * np.sqrt(1 + Bs[:, None] * n * n))[None]  # [f0, B, n]
    ok = f < max_freq
    bins = np.clip(np.round(f / df).astype(int), 0, len(prom) - 1)
    tol_idx = np.clip(np.searchsorted(tols, 0.006 * f / df), 0, len(tols) - 1)
    score = (pooled[tol_idx, bins] * ok * w).sum(-1) / w.sum()
    i, j = np.unravel_index(np.argmax(score), score.shape)
    return f0s[i], Bs[j]


def track_partials(x, sr, onset, f_nominal, B_range, seconds=4.0, max_freq=11000.0, snr_db=15.0):
    """Partial frequencies and levels; returns n, f, level_db, f0, B and the fit residual."""
    x = _as_2d(x)
    seg = x[onset + int(0.05 * sr): onset + int(seconds * sr)]
    n_fft = 1 << int(math.ceil(math.log2(len(seg) * 4)))
    spec = 10 * np.log10(_power_spectrum(seg, n_fft, np.blackman(len(seg))) + 1e-24)
    hz = np.fft.rfftfreq(n_fft, 1 / sr)
    f0, B = comb_search(spec, hz, f_nominal, min(max_freq, 0.45 * sr), B_range)

    found = {}
    for n in range(1, 7):  # seed from the best stiff-string comb, tight tolerance
        p = _peak(spec, hz, n * f0 * math.sqrt(1 + B * n * n), 0.05 * f0)
        if p and p[2] >= snr_db:
            found[n] = p

    def fit(ps):
        n = np.array(sorted(ps), dtype=float)
        f = np.array([ps[k][0] for k in sorted(ps)])
        if len(n) < 3:
            return f[0] / n[0], B
        slope, icpt = np.polyfit(n**2, (f / n) ** 2, 1)
        return math.sqrt(max(icpt, 1e-6)), max(slope / max(icpt, 1e-6), 0.0)

    def reject_outliers(ps):
        """Drop partials more than 5 cents (or 3 MADs) off the stiff-string fit, then refit."""
        for _ in range(3):
            f0, B = fit(ps)
            n = np.array(sorted(ps), dtype=float)
            res = 1200 * np.log2(np.array([ps[k][0] for k in sorted(ps)]) / (n * f0 * np.sqrt(1 + B * n * n)))
            mad = np.median(np.abs(res - np.median(res))) + 1e-9
            bad = [int(k) for k, r in zip(n, res) if abs(r) > max(5.0, 3 * mad)]
            if not bad or len(ps) - len(bad) < 2:
                break
            for k in bad:
                ps.pop(k)
        return fit(ps)

    if len(found) < 2:
        return None
    f0, B = reject_outliers(found)
    misses = 0
    for n in range(2, 200):
        f_pred = n * f0 * math.sqrt(1 + B * n * n)
        if f_pred > min(max_freq, 0.45 * sr):
            break
        if n in found:
            continue
        p = _peak(spec, hz, f_pred, 0.2 * f0)
        if p and p[2] >= snr_db:
            found[n] = p
            misses = 0
            f0, B = fit(found)
        else:
            misses += 1
            if misses >= 4 and n > 10:
                break
    f0, B = reject_outliers(found)
    ns = sorted(found)
    f = np.array([found[k][0] for k in ns])
    res = 1200 * np.log2(f / (np.array(ns) * f0 * np.sqrt(1 + B * np.array(ns) ** 2)))
    return {"n": ns, "f": list(f), "level": [found[k][1] for k in ns], "f0": f0, "B": B,
            "fit_rms_cents": float(np.sqrt(np.mean(res**2)))}


def partial_power(x, sr, f, f_spacing):
    """Power envelope of the partial at ``f`` (heterodyne + Hann low-pass narrower than the partial
    spacing), summed over channels."""
    x = _as_2d(x)
    t = np.arange(len(x)) / sr
    k = int(np.clip(2.5 / f_spacing, 0.02, 0.2) * sr)
    w = np.hanning(k)
    w /= w.sum()
    z = fftconvolve(x * np.exp(-2j * np.pi * f * t)[:, None], w[:, None], mode="same", axes=0)
    return (np.abs(z) ** 2).sum(1)


def decay_envelope(x, sr, onset, partials, f_spacing, pre):
    """Noise-subtracted power of the given partials, 0.1 s smoothing, decimated to 100 Hz: (t, dB re peak, floor dB)."""
    P = sum(partial_power(x, sr, f, f_spacing) for f in partials)
    noise = 0.0
    if len(pre) > int(0.1 * sr):
        noise = float(np.median(sum(partial_power(pre, sr, f, f_spacing) for f in partials)[int(0.05 * sr):]))
    else:
        noise = float(np.percentile(P[-int(sr):], 10))
    step = sr // 100
    Ps = _moving_average(P, int(0.1 * sr))[onset::step]
    peak = Ps[: 30].max()
    db = 10 * np.log10(np.clip(Ps - noise, 1e-30, None) / peak)
    floor = 10 * math.log10(max(noise, 1e-30) / peak)
    return np.arange(len(db)) / 100.0, db, floor


def decay_profile(t, db, floor, times=PROFILE_TIMES, margin=6.0):
    """Level at fixed times (nan where the note is within ``margin`` dB of the noise floor)."""
    out = []
    for tt in times:
        i = int(round(tt * 100))
        out.append(float(db[i]) if i < len(db) and db[i] > floor + margin else float("nan"))
    return out


def two_segment(t, db, floor, t_start=0.05, margin=10.0):
    """Continuous two-segment fit with a free breakpoint: prompt T60, aftersound T60, breakpoint, knee.

    The knee is the aftersound line extrapolated back to the onset (dB re peak). With no clear
    second stage the two slopes coincide.
    """
    live = np.nonzero(db > floor + margin)[0]
    if len(live) == 0:
        return {}
    i0, i1 = int(t_start * 100), live[-1] + 1
    tt, yy = t[i0:i1], db[i0:i1]
    if len(tt) < 30:
        return {}
    best = None
    for tb in np.linspace(tt[0] + 0.1, tt[-1] - 0.2, 60):
        A = np.stack([np.ones_like(tt), tt, np.maximum(0, tt - tb)], 1)
        coef, res, *_ = np.linalg.lstsq(A, yy, rcond=None)
        err = float(np.sum((A @ coef - yy) ** 2))
        if best is None or err < best[0]:
            best = (err, tb, coef)
    _, tb, (a, s1, ds) = best
    s2 = s1 + ds
    t60 = lambda s: -60 / s if s < 0 else float("inf")
    return {"t60_prompt": t60(s1), "t60_after": t60(s2), "t_knee": float(tb), "knee_db": float(a - ds * tb),
            "fit_until_s": float(tt[-1])}


def early_partials(x, sr, onset, freqs, noise_seg, t0=0.01, t1=0.2):
    """Noise-compensated levels (dB) of the partials at ``freqs`` in the early window after the onset
    (nan below 10 dB SNR)."""
    x, noise_seg = _as_2d(x), _as_2d(noise_seg)
    seg = x[onset + int(t0 * sr): onset + int(t1 * sr)]
    n_fft = 1 << int(math.ceil(math.log2(len(seg) * 8)))
    win = np.hanning(len(seg))
    P = _power_spectrum(seg, n_fft, win)
    hz = np.fft.rfftfreq(n_fft, 1 / sr)
    if len(noise_seg) >= len(seg):
        chunks = [noise_seg[i: i + len(seg)] for i in range(0, len(noise_seg) - len(seg) + 1, len(seg))]
        N = np.mean([_power_spectrum(c, n_fft, win) for c in chunks], 0)
    else:
        N = np.zeros_like(P)
    tol = max(2, int(0.5 / (t1 - t0) / (hz[1] - hz[0])))
    out = []
    for f in freqs:
        i = int(round(f / (hz[1] - hz[0])))
        lo, hi = max(0, i - tol), i + tol + 1
        sig = P[lo:hi].max() - N[lo:hi].mean()
        out.append(10 * math.log10(sig) if sig > 10 * N[lo:hi].mean() + 1e-30 else float("nan"))
    return np.array(out)


def spectral_slope(levels, n):
    """Least-squares slope (dB/oct) of partial level against log2(n), over the partials that were measured."""
    ok = np.isfinite(levels)
    if ok.sum() < 3:
        return float("nan")
    return float(np.polyfit(np.log2(np.asarray(n, dtype=float)[ok]), levels[ok], 1)[0])


def rise_time(x, sr, onset):
    r = _rms(x, sr, 0.002)
    seg = r[max(0, onset - int(0.02 * sr)): onset + int(0.2 * sr)]
    pk = seg.max()
    return (np.argmax(seg >= 0.9 * pk) - np.argmax(seg >= 0.1 * pk)) / sr


def analyze_note(x, sr, pitch):
    """All measurements for one isolated note ``x`` ([T] or [T, channels]) of MIDI ``pitch``."""
    x = _as_2d(x)
    onset = find_onset(x, sr)
    pre = x[: max(0, onset - int(0.05 * sr))]
    f_nom = 440.0 * 2 ** ((pitch - 69) / 12)
    Br = rigaud_B(pitch)
    tr = track_partials(x, sr, onset, f_nom, (Br / 4, Br * 4))
    out = {"pitch": pitch, "onset_s": onset / sr, "peak_db": 20 * math.log10(np.abs(x).max() + 1e-12),
           "rise_ms": 1000 * rise_time(x, sr, onset)}
    if tr is None:
        return out
    f1 = tr["f"][0] if tr["n"][0] == 1 else tr["f0"] * math.sqrt(1 + tr["B"])
    cents = 1200 * math.log2(f1 / f_nom)
    out.update(B=tr["B"], B_reliable=bool(len(tr["n"]) >= 8 and tr["fit_rms_cents"] < 3.0), f1=f1, cents=cents,
               suspect=bool(abs(cents) > 50), fit_rms_cents=tr["fit_rms_cents"], n_partials=len(tr["n"]))

    lv = early_partials(x, sr, onset, tr["f"][:12], pre)
    n12 = tr["n"][:12]
    ok = np.isfinite(lv)
    if ok.any():
        amp = 10 ** (lv[ok] / 20)
        out["harmonic_centroid_hz"] = float((np.array(tr["f"][:12])[ok] * amp).sum() / amp.sum())
    out["slope_db_oct"] = spectral_slope(lv, n12)
    ref = lv[0] if n12[0] == 1 and np.isfinite(lv[0]) else None
    out["early_partials"] = {int(k): float(v - ref) for k, v in zip(n12, lv)} if ref is not None else {}

    want = decay_partials(pitch)
    freqs = [f for n, f in zip(tr["n"], tr["f"]) if n in want]
    if freqs:
        t, db, floor = decay_envelope(x, sr, onset, freqs, tr["f0"], pre)
        out["decay_partials"] = [n for n in tr["n"] if n in want]
        out["profile_db"] = decay_profile(t, db, floor)
        out["floor_db"] = floor
        out.update(two_segment(t, db, floor))
    return out


def _analyze_file(path):
    import soundfile as sf

    _, dyn, note = os.path.basename(path).rsplit(".", 1)[0].split(".")
    x, sr = sf.read(path, always_2d=True)
    r = analyze_note(x, sr, note_to_midi(note))
    r.update(file=os.path.basename(path), dynamic=dyn)
    return r


def analyze_dir(root, workers=None):
    from multiprocessing import Pool

    paths = sorted(glob.glob(os.path.join(root, "*.aif*")))
    with Pool(workers or os.cpu_count()) as pool:
        results = []
        for r in pool.imap(_analyze_file, paths):
            results.append(r)
            print(f"{r['file']:24s} B={r.get('B', float('nan')):.2e}{'' if r.get('B_reliable') else '?'} "
                  f"cents={r.get('cents', float('nan')):+.1f}{' SUSPECT' if r.get('suspect') else ''} "
                  f"T60p={r.get('t60_prompt', float('nan')):.1f} T60a={r.get('t60_after', float('nan')):.1f} "
                  f"knee={r.get('knee_db', float('nan')):.0f}", flush=True)
    return results


PARTIAL_TIMES = (0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 16.0)


def partial_table(x, sr, pitch, times=PARTIAL_TIMES, max_freq=10000.0, n_max=60):
    """Per-partial early level and decay of one note: for n = 1.. (frequencies from the stiff-string fit),
    noise from the silence before the onset, or from the last 2 s of the file,
    ``early_db`` is the peak power in the first 0.15 s (dB, noise-compensated; nan below 10 dB SNR) and
    ``level_db[i]`` the mean power over +-0.1 s around ``times[i]`` re that peak (nan below 3 dB SNR);
    ``noise_db`` is the noise power in the partial's band re the loudest partial's early peak."""
    x = _as_2d(x)
    onset = find_onset(x, sr)
    pre = x[: max(0, onset - int(0.05 * sr))]
    f_nom = 440.0 * 2 ** ((pitch - 69) / 12)
    Br = rigaud_B(pitch)
    tr = track_partials(x, sr, onset, f_nom, (Br / 4, Br * 4))
    if tr is None:
        return None
    if len(pre) < int(0.1 * sr):  # noise from the end of the file instead (the note has been released by then)
        pre = x[-int(2 * sr):]
    f0, B = tr["f0"], tr["B"]
    seg = x[: onset + int((max(times) + 0.5) * sr)]
    step = sr // 100
    out = {"pitch": pitch, "f0": f0, "B": B, "n": [], "f": [], "early_db": [], "level_db": [], "noise_db": []}
    for n in range(1, n_max + 1):
        f = n * f0 * math.sqrt(1 + B * n * n)
        if f > min(max_freq, 0.45 * sr):
            break
        P = partial_power(seg, sr, f, f0)[onset::step]
        noise = float(np.median(partial_power(pre, sr, f, f0)[len(pre) // 4: -len(pre) // 4 or None]))
        t = np.arange(len(P)) / 100
        e = float(P[t < 0.15].max())
        out["n"].append(n)
        out["f"].append(f)
        out["early_db"].append(10 * math.log10(e - noise) if e > 10 * noise else float("nan"))
        lv = []
        for tt in times:
            v = float(P[(t > tt - 0.1) & (t < tt + 0.1)].mean()) - noise
            lv.append(10 * math.log10(v / e) if v > noise and e > 10 * noise else float("nan"))
        out["level_db"].append(lv)
        out["noise_db"].append(10 * math.log10(max(noise, 1e-30)))
    peak = np.nanmax(out["early_db"]) if np.isfinite(out["early_db"]).any() else 0.0
    out["noise_db"] = [v - peak for v in out["noise_db"]]
    return out


def _partial_file(path):
    import soundfile as sf

    _, dyn, note = os.path.basename(path).rsplit(".", 1)[0].split(".")
    x, sr = sf.read(path, always_2d=True)
    r = partial_table(x, sr, note_to_midi(note))
    if r is not None:
        r.update(file=os.path.basename(path), dynamic=dyn)
    return r


def partial_tables(root, workers=None):
    from multiprocessing import Pool

    paths = sorted(glob.glob(os.path.join(root, "*.aif*")))
    with Pool(workers or os.cpu_count()) as pool:
        return [r for r in pool.map(_partial_file, paths) if r is not None]


def render_note(model, pitch, velocity, seconds, body=False, snr_db=60.0, seed=0):
    """Render one isolated note like a recording: 0.3 s of silence first, dry strings (or strings through
    the soundboard body, no hall), and a white noise floor ``snr_db`` below the peak so that low-level
    partials are censored as they are in the recordings."""
    import torch

    from .diagnostics import _perf, _variant

    m = _variant(model, use_noise=False, use_sympathetic=False, use_room=body)
    sr = model.cfg.sample_rate
    pre = int(0.3 * sr)
    n = int(seconds * sr) + pre
    with torch.no_grad():
        if body:
            saved = m.room.log_gain.clone()
            m.room.log_gain.fill_(-30.0)
        x = m(_perf(model, n, [(pitch, pre / sr, n / sr, velocity)]), n)["audio"][0].double().numpy()
        if body:
            m.room.log_gain.copy_(saved)
    rng = np.random.default_rng(seed + pitch)
    return x + np.abs(x).max() * 10 ** (-snr_db / 20) * rng.standard_normal(len(x))


def analyze_model(model, dynamics=None, keys=None, seconds=45.0):
    """Run the same analysis on the model: decays on the dry strings, spectra through the body."""
    dynamics = dynamics or {"mf": 64, "ff": 110}
    keys = keys or list(LANDMARKS.values())
    sr = model.cfg.sample_rate
    results = []
    for dyn, vel in dynamics.items():
        for p in keys:
            r = analyze_note(render_note(model, p, vel, seconds), sr, p)
            spec = analyze_note(render_note(model, p, vel, 1.5, body=True), sr, p)
            for k in ("slope_db_oct", "early_partials", "harmonic_centroid_hz", "rise_ms"):
                r[k] = spec.get(k, float("nan"))
            r.update(file=f"model.{dyn}.{midi_name(p)}", dynamic=dyn)
            results.append(r)
    return results


# ---------------------------------------------------------------- summaries

def _median(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else float("nan")


def _key_values(results, extract, dynamics):
    """Per-key medians of ``extract(note)`` over the given dynamics, skipping suspect notes."""
    by_key = {}
    for r in results:
        if r["dynamic"] in dynamics and not r.get("suspect"):
            by_key.setdefault(r["pitch"], []).append(extract(r))
    return {p: _median(v) for p, v in by_key.items()}


def _at_landmarks(per_key, half_width=3):
    return {name: _median([v for q, v in per_key.items() if abs(q - p) <= half_width]) for name, p in LANDMARKS.items()}


def landmark_samples(results, extract, dynamics, half_width=3):
    """All note-level values feeding each landmark (for counts and bootstrap intervals)."""
    out = {}
    for name, p in LANDMARKS.items():
        out[name] = [v for r in results if r["dynamic"] in dynamics and not r.get("suspect") and abs(r["pitch"] - p) <= half_width
                     for v in [extract(r)] if v is not None and np.isfinite(v)]
    return out


def bootstrap_ci(values, n=2000, q=(5, 95), seed=0):
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    meds = [np.median(rng.choice(values, len(values))) for _ in range(n)]
    return tuple(float(np.percentile(meds, qq)) for qq in q)


def _pair(results, extract, a="ff", b="mf"):
    notes = {(r["pitch"], r["dynamic"]): r for r in results if not r.get("suspect")}
    return {p: extract(notes[(p, a)]) - extract(notes[(p, b)]) for (p, d) in notes if d == a and (p, b) in notes}


def _profile(i):
    return lambda r: (r.get("profile_db") or [float("nan")] * len(PROFILE_TIMES))[i]


SUMMARY_ROWS = [
    ("inharmonicity B (reliable fits)", lambda r: r.get("B") if r.get("B_reliable") else None, ("pp", "mf", "ff"), "{:.2e}"),
    ("tuning, cents re ET", lambda r: r.get("cents"), ("pp", "mf", "ff"), "{:+.1f}"),
] + [(f"decay profile at {t:g} s (dB re peak)", _profile(i), ("mf", "ff"), "{:.0f}") for i, t in enumerate(PROFILE_TIMES)] + [
    ("prompt T60, two-segment fit (s)", lambda r: r.get("t60_prompt"), ("mf", "ff"), "{:.1f}"),
    ("aftersound T60, two-segment fit (s)", lambda r: r.get("t60_after"), ("mf", "ff"), "{:.0f}"),
    ("knee, two-segment fit (dB re peak)", lambda r: r.get("knee_db"), ("mf", "ff"), "{:.0f}"),
    ("early spectral slope, mf (dB/oct)", lambda r: r.get("slope_db_oct"), ("mf",), "{:.0f}"),
    ("rise time 10-90 %, mf (ms)", lambda r: r.get("rise_ms"), ("mf",), "{:.0f}"),
]
PAIR_ROWS = [
    ("slope change mf -> ff (dB/oct)", lambda r: r.get("slope_db_oct", float("nan")), "{:+.1f}"),
    ("peak level ff - mf (dB)", lambda r: r["peak_db"], "{:+.0f}"),
    ("harmonic centroid ff / mf", lambda r: math.log(r.get("harmonic_centroid_hz", float("nan"))), "x{:.2f}"),
]


def summarize(results):
    table = {}
    for label, extract, dyns, fmt in SUMMARY_ROWS:
        table[label] = (_at_landmarks(_key_values(results, extract, dyns)), fmt)
    for label, extract, fmt in PAIR_ROWS:
        vals = _at_landmarks(_pair(results, extract))
        if fmt.startswith("x"):
            vals = {k: math.exp(v) for k, v in vals.items()}
        table[label] = (vals, fmt)
    return table


def report(rec, mod=None):
    lines = ["| quantity | source | " + " | ".join(LANDMARKS) + " |", "|---|---|" + "---|" * len(LANDMARKS)]
    srec, smod = summarize(rec), summarize(mod) if mod else None
    for label in srec:
        for tag, t in (("recording", srec), ("model", smod)):
            if t is None:
                continue
            v, fmt = t[label]
            lines.append(f"| {label} | {tag} | " + " | ".join(fmt.format(v[k]) if np.isfinite(v[k]) else "-" for k in LANDMARKS) + " |")
    counts = landmark_samples(rec, lambda r: r.get("t60_prompt"), ("mf", "ff"))
    lines.append("| notes behind each decay value | recording | " + " | ".join(str(len(counts[k])) for k in LANDMARKS) + " |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", help="directory of isolated-note recordings named <x>.<dyn>.<note>.aiff (Iowa MIS naming)")
    ap.add_argument("--json", help="write per-note measurements here")
    ap.add_argument("--out", help="write the markdown comparison report here")
    ap.add_argument("--no-model", action="store_true", help="skip analysing the model prior")
    args = ap.parse_args(argv)
    rec = analyze_dir(args.root)
    mod = None
    if not args.no_model:
        from .render import load_model

        mod = analyze_model(load_model())
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"recordings": rec, "model": mod}, f, indent=1, default=float)
    text = report(rec, mod)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
