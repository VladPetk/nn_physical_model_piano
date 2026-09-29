"""Measure piano physics from recordings of isolated notes (and from the model, identically).

    python -m pianonn.calibration data/iowa --json data/iowa_analysis.json --out docs/calibration_iowa.md

Per note (key, dynamic): inharmonicity B and tuning (from a regression over the tracked
partials, f_n^2 / n^2 = f0^2 + f0^2 B n^2), prompt T60 of the lowest partials (energy decay
curve, -3 to -13 dB: the same convention as ``pianonn.diagnostics``), aftersound T60 and the
level of the knee, partial levels, spectral centroid and rise time. The report compares the
recordings with the model prior analysed by the same code.
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


def note_to_midi(name):
    m = re.fullmatch(r"([A-G]b?)(-?\d)", name)
    return 12 * (int(m.group(2)) + 1) + PITCH_CLASS[m.group(1)]


def midi_name(p):
    names = [k for k, v in sorted(PITCH_CLASS.items(), key=lambda kv: kv[1])]
    return f"{names[p % 12]}{p // 12 - 1}"


def _moving_average(x, k):
    """Centred moving average of length ``k`` in O(N) (np.convolve is O(N k))."""
    c = np.concatenate([[0.0], np.cumsum(x)])
    lo = np.clip(np.arange(len(x)) - k // 2, 0, len(x))
    hi = np.clip(lo + k, 0, len(x))
    return (c[hi] - c[lo]) / k


def _rms(x, sr, win=0.005):
    return np.sqrt(np.clip(_moving_average(x**2, max(1, int(win * sr))), 0, None))


def find_onset(x, sr):
    r = _rms(x, sr)
    return int(np.argmax(r > 0.05 * r.max()))


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


def comb_search(spec_db, hz, f_nominal, max_freq, n_max=40, cents=100.0, decimate=4):
    """Coarse (f0, B) that best explains the spectrum as a stiff-string comb.

    Scores every (f0, B) on a grid by the prominence (dB above a moving average) of the peaks at
    the predicted partials n = 1..N, counting only clear peaks (> 10 dB), weighting low partials
    more (1/sqrt(n)) and max-pooling over +-0.6 % of each partial's frequency so the coarse grid
    still lands on it. Robust to a weak fundamental and stray peaks, which fool greedy peak picking.
    """
    from scipy.ndimage import maximum_filter1d, uniform_filter1d

    spec_db, hz = spec_db[::decimate], hz[::decimate]
    df = hz[1] - hz[0]
    prom = np.clip(spec_db - uniform_filter1d(spec_db, max(3, int(f_nominal / df))) - 10, 0, 30)
    tols = [1, 2, 4, 8, 16, 32, 64, 128]
    pooled = np.stack([maximum_filter1d(prom, 2 * t + 1) for t in tols])
    f0s = f_nominal * 2 ** (np.linspace(-cents, cents, 81) / 1200)
    Bs = np.concatenate([[0.0], np.logspace(-5, -1, 60)])
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


def track_partials(x, sr, onset, f_nominal, seconds=4.0, max_freq=11000.0, snr_db=15.0):
    """Partial frequencies and levels; returns (n, f, level_db), f0 and B."""
    seg = x[onset + int(0.05 * sr): onset + int(seconds * sr)]
    n_fft = 1 << int(math.ceil(math.log2(len(seg) * 4)))
    spec = 20 * np.log10(np.abs(np.fft.rfft(seg * np.blackman(len(seg)), n_fft)) + 1e-12)
    hz = np.fft.rfftfreq(n_fft, 1 / sr)

    f0, B = comb_search(spec, hz, f_nominal, max_freq=min(max_freq, 0.45 * sr))
    found = {}
    for n in range(1, 7):  # seed from the best stiff-string comb, tight tolerance
        p = _peak(spec, hz, n * f0 * math.sqrt(1 + B * n * n), 0.05 * f0)
        if p and p[2] >= snr_db:
            found[n] = p

    def fit(ps):
        n = np.array(sorted(ps), dtype=float)
        f = np.array([ps[k][0] for k in sorted(ps)])
        if len(n) < 3:
            return f[0] / n[0], 0.0
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
    """Power envelope of the partial at ``f`` (heterodyne + Hann low-pass narrower than the partial spacing)."""
    t = np.arange(len(x)) / sr
    k = int(np.clip(2.5 / f_spacing, 0.02, 0.2) * sr)
    w = np.hanning(k)
    w /= w.sum()
    z = fftconvolve(x * np.exp(-2j * np.pi * f * t), w, mode="same")
    return np.abs(z) ** 2


def edc_t60(p, sr, start, noise, lo=-3.0, hi=-13.0):
    """Prompt T60 from the noise-compensated energy decay curve, fitted from ``lo`` to ``hi`` dB."""
    q = np.clip(p[start:] - noise, 0, None)
    edc = np.cumsum(q[::-1])[::-1]
    if edc[0] <= 0:
        return float("nan")
    edc = 10 * np.log10(edc / edc[0] + 1e-30)
    sel = np.nonzero((edc <= lo) & (edc >= hi))[0]
    if len(sel) < 10:
        return float("nan")
    slope = np.polyfit(sel / sr, edc[sel], 1)[0]
    return -60 / slope if slope < 0 else float("inf")


def aftersound(p, sr, start, noise, knee_db=-25.0, margin_db=10.0, min_span=1.5):
    """Late-decay T60 and the level of the aftersound line extrapolated back to the onset (dB re peak)."""
    k = int(0.5 * sr)
    ps = _moving_average(p[start:], k)
    db = 10 * np.log10(ps + 1e-30)
    peak = db[: int(0.3 * sr)].max()
    floor = 10 * math.log10(noise + 1e-30)
    below = np.nonzero(db < peak + knee_db)[0]
    if len(below) == 0:
        return float("nan"), float("nan")
    i0 = below[0]
    live = np.nonzero(db[i0:] > floor + margin_db)[0]
    if len(live) == 0:
        return float("nan"), float("nan")
    i1 = i0 + live[-1] - k // 2
    if (i1 - i0) / sr < min_span:
        return float("nan"), float("nan")
    t = np.arange(i0, i1) / sr
    slope, icpt = np.polyfit(t, db[i0:i1], 1)
    return (-60 / slope if slope < 0 else float("inf")), icpt - peak


def early_partials(x, sr, onset, freqs, noise_seg, t0=0.01, t1=0.2):
    """Noise-compensated levels (dB) of the partials at ``freqs`` in the early window after the onset.

    The noise power at each frequency is estimated from ``noise_seg`` (the silence before the note)
    and subtracted, so quiet pp notes are not mistaken for bright ones by the recording hiss.
    """
    seg = x[onset + int(t0 * sr): onset + int(t1 * sr)]
    n_fft = 1 << int(math.ceil(math.log2(len(seg) * 8)))
    win = np.hanning(len(seg))
    P = np.abs(np.fft.rfft(seg * win, n_fft)) ** 2
    hz = np.fft.rfftfreq(n_fft, 1 / sr)
    if len(noise_seg) >= len(seg):
        chunks = [noise_seg[i: i + len(seg)] for i in range(0, len(noise_seg) - len(seg) + 1, len(seg))]
        N = np.mean([np.abs(np.fft.rfft(c * win, n_fft)) ** 2 for c in chunks], 0)
    else:
        N = np.zeros_like(P)
    tol = max(2, int(0.5 / (t1 - t0) / (hz[1] - hz[0])))  # half the window's main lobe
    out = []
    for f in freqs:
        i = int(round(f / (hz[1] - hz[0])))
        lo, hi = max(0, i - tol), i + tol + 1
        sig = P[lo:hi].max() - N[lo:hi].mean()
        out.append(10 * math.log10(sig) if sig > 10 * N[lo:hi].mean() + 1e-30 else float("nan"))  # SNR > 10 dB
    return np.array(out)


def spectral_slope(levels, n):
    """Least-squares slope (dB/oct) of partial level against log2(n), over the partials that were measured."""
    ok = np.isfinite(levels)
    if ok.sum() < 3:
        return float("nan")
    return float(np.polyfit(np.log2(np.asarray(n, dtype=float)[ok]), levels[ok], 1)[0])


def centroid(x, sr, onset, t0=0.02, t1=0.12):
    frame = x[onset + int(t0 * sr): onset + int(t1 * sr)]
    s = np.abs(np.fft.rfft(frame * np.hanning(len(frame))))
    return float((s * np.fft.rfftfreq(len(frame), 1 / sr)).sum() / (s.sum() + 1e-15))


def rise_time(x, sr, onset):
    r = _rms(x, sr, 0.002)
    seg = r[max(0, onset - int(0.02 * sr)): onset + int(0.2 * sr)]
    pk = seg.max()
    return (np.argmax(seg >= 0.9 * pk) - np.argmax(seg >= 0.1 * pk)) / sr


def analyze_note(x, sr, pitch, n_decay=8):
    """All measurements for one isolated note ``x`` (mono) of MIDI ``pitch``."""
    onset = find_onset(x, sr)
    pre = x[: max(0, onset - int(0.05 * sr))]
    f_nom = 440.0 * 2 ** ((pitch - 69) / 12)
    tr = track_partials(x, sr, onset, f_nom)
    out = {"pitch": pitch, "onset_s": onset / sr, "peak_db": 20 * math.log10(np.abs(x).max() + 1e-12),
           "centroid_hz": centroid(x, sr, onset), "rise_ms": 1000 * rise_time(x, sr, onset)}
    if tr is None:
        return out
    f1 = tr["f"][0] if tr["n"][0] == 1 else tr["f0"] * math.sqrt(1 + tr["B"])
    cents = 1200 * math.log2(f1 / f_nom)
    # quality: B needs many partials and a clean stiff-string fit (treble recordings carry exact
    # integer harmonics from distortion that mimic B ~ 0); a note > 50 cents off is mislabelled
    out.update(B_reliable=bool(len(tr["n"]) >= 8 and tr["fit_rms_cents"] < 3.0), suspect=bool(abs(cents) > 50),
               fit_rms_cents=tr["fit_rms_cents"])
    lv = early_partials(x, sr, onset, tr["f"][:12], pre)
    n12 = tr["n"][:12]
    ok = np.isfinite(lv)
    if ok.any():
        amp = 10 ** (lv[ok] / 20)
        out["harmonic_centroid_hz"] = float((np.array(tr["f"][:12])[ok] * amp).sum() / amp.sum())
    out["slope_db_oct"] = spectral_slope(lv, n12)
    out["early_partials"] = {int(k): float(v - lv[0]) for k, v in zip(n12, lv)} if n12[0] == 1 and np.isfinite(lv[0]) else {}
    out.update(B=tr["B"], f1=f1, cents=cents, n_partials=len(tr["n"]),
               partials=dict(zip(tr["n"], [lv - tr["level"][0] if tr["n"][0] == 1 else float("nan") for lv in tr["level"]])))
    prompt, after, after_level = [], [], []
    for n, f in list(zip(tr["n"], tr["f"]))[:n_decay]:
        p = partial_power(x, sr, f, tr["f0"])
        if len(pre) > int(0.05 * sr):
            noise = float(np.median(partial_power(pre, sr, f, tr["f0"])[int(0.02 * sr):]))
        else:
            noise = float(np.percentile(p[-int(1.0 * sr):], 10))
        start = onset + int(0.05 * sr)
        prompt.append(edc_t60(p, sr, start, noise))
        a, lvl = aftersound(p, sr, start, noise)
        after.append(a)
        after_level.append(lvl)
    out.update(t60_prompt=prompt, t60_after=after, after_level_db=after_level, decay_partials=tr["n"][:n_decay])
    return out


def _analyze_file(path):
    import soundfile as sf

    _, dyn, note = os.path.basename(path).rsplit(".", 1)[0].split(".")
    x, sr = sf.read(path, always_2d=True)
    r = analyze_note(x.mean(1), sr, note_to_midi(note))
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
                  f"T60p={_first(r.get('t60_prompt')):.1f} T60a={_first(r.get('t60_after')):.1f}", flush=True)
    return results


def _first(v):
    return v[0] if v else float("nan")


def analyze_model(model, dynamics=None, keys=None, seconds=45.0):
    """Run the same analysis on the model's dry string output."""
    import torch

    from .diagnostics import _perf, _variant

    dynamics = dynamics or {"pp": 30, "mf": 64, "ff": 110}
    keys = keys or list(LANDMARKS.values())
    dry = _variant(model, use_noise=False, use_sympathetic=False, use_room=False)
    sr = model.cfg.sample_rate
    pre = int(0.3 * sr)  # silence before the onset so the analysis can estimate the noise floor
    results = []
    for dyn, vel in dynamics.items():
        for p in keys:
            n = int(seconds * sr) + pre
            with torch.no_grad():
                x = dry(_perf(model, n, [(p, pre / sr, n / sr, vel)]), n)["audio"][0].double().numpy()
            x = x + 1e-7 * np.random.default_rng(p).standard_normal(len(x))  # a noise floor, as in a recording
            r = analyze_note(x, sr, p)
            r.update(file=f"model.{dyn}.{midi_name(p)}", dynamic=dyn)
            results.append(r)
    return results


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


def _pair(results, extract, a="ff", b="mf"):
    """Per-key difference extract(a) - extract(b) for keys recorded at both dynamics."""
    notes = {(r["pitch"], r["dynamic"]): r for r in results if not r.get("suspect")}
    return {p: extract(notes[(p, a)]) - extract(notes[(p, b)]) for (p, d) in notes if d == a and (p, b) in notes}


def _lst(r, key, lo, hi):
    return _median((r.get(key) or [])[lo:hi])


SUMMARY_ROWS = [
    ("inharmonicity B (reliable fits)", lambda r: r.get("B") if r.get("B_reliable") else None, ("pp", "mf", "ff"), "{:.2e}"),
    ("tuning, cents re ET", lambda r: r.get("cents"), ("pp", "mf", "ff"), "{:+.1f}"),
    ("prompt T60, median partials 1-4 (s)", lambda r: _lst(r, "t60_prompt", 0, 4), ("mf", "ff"), "{:.1f}"),
    ("aftersound T60, median partials 1-4 (s)", lambda r: _lst(r, "t60_after", 0, 4), ("mf", "ff"), "{:.0f}"),
    ("aftersound knee, partials 1-4 (dB re peak)", lambda r: _lst(r, "after_level_db", 0, 4), ("mf", "ff"), "{:.0f}"),
    ("aftersound knee, partials 5-8 (dB re peak)", lambda r: _lst(r, "after_level_db", 4, 8), ("mf", "ff"), "{:.0f}"),
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
    for label, (vals, fmt) in srec.items():
        for tag, t in (("recording", srec), ("model", smod)):
            if t is None:
                continue
            v, _ = t[label]
            lines.append(f"| {label} | {tag} | " + " | ".join(fmt.format(v[k]) if np.isfinite(v[k]) else "-" for k in LANDMARKS) + " |")
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
