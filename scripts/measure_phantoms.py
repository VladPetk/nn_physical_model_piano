"""Are there phantom partials in the recordings? (review 4, section 6; docs/plan_round2.md, 2.3 and A8)

    python scripts/measure_phantoms.py data/maestro24k --years 2018 --ckpt runs/round1_trial/main/best.pt --out runs/measurements/phantoms_2018.md

A phantom partial at 2f_j comes from the longitudinal coupling of transverse partial j with itself: its
amplitude goes as a_j^2. On notes of MIDI 51-87 with a clear first 0.65 s (no other strike 0.3 s before or
0.65 s after), each note's own f0 and B are tracked (``pianonn.calibration.track_partials``), and the
spectrum of 30-630 ms is searched near 2f_j, which sits tens of Hz below transverse partial 2j. Three tests
separate a phantom from anything else that could sit there:

1. **frequency**: a phantom sits at exactly 2f_j. The same detector is run at two control positions (half a
   gap below 2f_j, and half-way to partial 2j): detections there, and the peak offsets there, show what
   chance peaks look like;
2. **velocity**: regressed on partial j's level across notes, a phantom's level has slope ~2 (a_j^2), while
   partial 2j has slope ~1 plus the hammer's brightening;
3. **sympathy**: detections within 15 cents of an undamped string (MIDI >= 90) or on notes with the pedal down
   are counted apart.

Decay is reported too, but it is weak evidence here: in the fitted model a phantom decays only 1.1-1.5x as fast
as partial 2j once the aftersound dominates (docs/plan_round2.md, 2.3). With ``--ckpt`` the model's phantom
levels re partial 2j (velocity 64) are printed for comparison.
"""

import argparse
import json
import math
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

J = range(3, 9)
POSITIONS = ("phantom", "control below", "control between")


def _isolated(notes, before, clear, lo, hi):
    on = notes[:, 1]
    out = []
    for i in range(len(notes)):
        if not lo <= notes[i, 0] <= hi:
            continue
        others = np.ones(len(notes), bool)
        others[i] = False
        if not np.any(others & (on > on[i] - before) & (on < on[i] + clear)):
            out.append(i)
    return out


def _peak(P, f, fc, tol):
    """(level dB, offset in cents from fc) of the largest bin within fc +- tol, parabolic in log power."""
    b = np.nonzero((f > fc - tol) & (f < fc + tol))[0]
    if len(b) < 3:
        return None
    i = b[np.argmax(P[b])]
    L = 10 * np.log10(P[i - 1: i + 2] + 1e-30)
    den = L[0] - 2 * L[1] + L[2]
    d = 0.5 * (L[0] - L[2]) / den if den < 0 else 0.0
    fp = f[i] + d * (f[1] - f[0])
    return L[1] - 0.25 * (L[0] - L[2]) * d, 1200 * math.log2(fp / fc)


def _spectrum(x, sr, t0, t1):
    seg = x[int(t0 * sr): int(t1 * sr)]
    n = 4 * len(seg)
    P = (np.abs(np.fft.rfft(seg * np.hanning(len(seg))[:, None], n, axis=0)) ** 2).sum(1)
    return P, np.fft.rfftfreq(n, 1 / sr)


def _f(r, n):
    """Partial n of a measured note (its tracked f0 and B)."""
    return n * r["f0"] * math.sqrt(1 + r["B"] * n * n)


def _undamped(stretch_cents=15.0):
    """Frequencies of the undamped strings' first two partials (MIDI 90-108, equal temperament + a treble stretch)."""
    f = []
    for m in range(90, 109):
        f1 = 440 * 2 ** ((m - 69) / 12) * 2 ** ((stretch_cents + (m - 90)) / 1200)
        f += [f1, 2 * f1]
    return np.array(f)


def _measure_piece(job):
    import soundfile as sf

    from pianonn.calibration import rigaud_B, track_partials

    root, piece, before, clear = job
    with np.load(os.path.join(root, piece["midi"])) as z:
        notes, sus_t, sus_v = z["notes"], z["sustain_t"], z["sustain_v"]
    path = os.path.join(root, piece["audio"])
    sr = sf.info(path).samplerate
    und = _undamped()
    rows = []
    for i in _isolated(notes, before, clear, 51, 87):
        pitch, t0, _, vel = notes[i]
        start = int((t0 - 0.3) * sr)
        if start < 0:
            continue
        x, _ = sf.read(path, start=start, frames=int((0.35 + clear) * sr), dtype="float64", always_2d=True)
        f_nom = 440.0 * 2 ** ((pitch - 69) / 12)
        Br = rigaud_B(pitch)
        tr = track_partials(x, sr, int(0.3 * sr), f_nom, (Br / 4, Br * 4), seconds=clear)
        if tr is None or len(tr["n"]) < 6 or tr["fit_rms_cents"] > 3.0:
            continue
        f0, B = tr["f0"], tr["B"]
        k = np.searchsorted(sus_t, t0, side="right") - 1
        pedal = bool(k >= 0 and sus_v[k] >= 64)
        P, f = _spectrum(x, sr, 0.33, 0.93)
        Pa, fa = _spectrum(x, sr, 0.33, 0.63)
        Pb, fb = _spectrum(x, sr, 0.63, 0.93)
        fk = lambda n: n * f0 * math.sqrt(1 + B * n * n)
        per_j = {}
        for j in J:
            fph, ftr = 2 * fk(j), fk(2 * j)
            gap = ftr - fph
            if ftr > 10000 or gap < 12:
                continue
            tol = 0.25 * gap
            loc = (f > fph - 2 * gap) & (f < ftr + gap)
            floor = 10 * np.log10(np.median(P[loc]) + 1e-30)
            pj, p2j = _peak(P, f, fk(j), 0.25 * f0), _peak(P, f, ftr, tol)
            if pj is None or p2j is None:
                continue
            res = {"L_j": pj[0], "L_2j": p2j[0], "floor": floor,
                   "sympathetic": bool(np.min(np.abs(1200 * np.log2(und / fph))) < 15.0)}
            for name, fc in zip(POSITIONS, (fph, fph - 0.5 * gap, fph + 0.5 * gap)):
                pk = _peak(P, f, fc, tol)
                if pk is not None:
                    res[name] = {"L": pk[0], "offset_cents": pk[1], "detected": bool(pk[0] - floor > 8.0)}
            a, b = _peak(Pa, fa, fph, tol), _peak(Pb, fb, fph, tol)
            a2, b2 = _peak(Pa, fa, ftr, tol), _peak(Pb, fb, ftr, tol)
            if a and b and a2 and b2:
                res["decay_db_per_s"] = ((a[0] - b[0]) / 0.3, (a2[0] - b2[0]) / 0.3)
            per_j[j] = res
        rows.append({"piece": piece["id"], "pitch": int(pitch), "velocity": int(vel), "pedal": pedal, "B": B, "f0": f0,
                     "j": per_j})
    return rows


def model_levels(ckpt, cond=9):
    import torch

    from pianonn.render import load_model

    m = load_model(ckpt) if ckpt else load_model()
    ki = torch.arange(51 - 21, 88 - 21)[None]
    u = torch.full(ki.shape, 64 / 127)
    with torch.no_grad():
        md = m.physics.modes(ki, u, torch.zeros_like(u), torch.tensor([cond]))
    amp = md["amp"][0, :, :, 0].abs()  # [K, P] prompt mode
    ph = md["ph_amp"][0].reshape(ki.shape[1], -1, 2, 2)[:, :, 0, 0].abs()  # doubling series, prompt x prompt
    rel = lambda j: 20 * torch.log10(ph[:, j - 1] / amp[:, 2 * j - 1])  # inf/nan where partial 2j is above Nyquist
    return {j: float(rel(j)[torch.isfinite(rel(j))].median()) for j in J}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data")
    ap.add_argument("--years", type=int, nargs="*")
    ap.add_argument("--split", default="train")
    ap.add_argument("--before", type=float, default=0.3)
    ap.add_argument("--clear", type=float, default=0.65)
    ap.add_argument("--ckpt", help="also print this model's phantom levels (and the untrained prior's)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True, help="markdown report; the per-note measurements go next to it as .json")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    with open(os.path.join(args.data, "index.json")) as f:
        pieces = [p for p in json.load(f) if p["split"] == args.split and (not args.years or p["year"] in args.years)]
    rows = []
    with Pool(args.workers) as pool:
        for r in pool.imap_unordered(_measure_piece, [(args.data, p, args.before, args.clear) for p in pieces]):
            rows += r
    with open(os.path.splitext(args.out)[0] + ".json", "w") as f:
        json.dump(rows, f)

    vel = np.array([r["velocity"] for r in rows])
    lines = [f"# Phantom partials in MAESTRO {args.years}, {args.split} split", "",
             f"{len(rows)} notes of MIDI 51-87 with a clear first {args.clear} s; velocity 10/50/90 % "
             f"{np.percentile(vel, 10):.0f} / {np.percentile(vel, 50):.0f} / {np.percentile(vel, 90):.0f}; "
             f"pedal down on {100 * np.mean([r['pedal'] for r in rows]):.0f} %.", "",
             "## 1. Detection and frequency", "",
             "Detected: a peak 8 dB above the local median within ±¼ gap of the position. Offset: the detected peak's "
             "distance from the position (cents); chance peaks spread over the whole window (± a quarter of the gap, "
             "tens of cents), a line at 2f_j sits within a few cents.", "",
             "| j | notes | detected at 2f_j | control below | control between | median abs offset at 2f_j (cents) | at the controls | half-window (cents) |",
             "|---|---|---|---|---|---|---|---|"]
    clean = lambda r, j: not r["pedal"] and not r["j"][j]["sympathetic"]
    for j in J:
        rs = [r for r in rows if j in r["j"] and clean(r, j)]
        if not rs:
            continue
        det = {p: [r["j"][j][p] for r in rs if p in r["j"][j] and r["j"][j][p]["detected"]] for p in POSITIONS}
        off = lambda p: np.median([abs(d["offset_cents"]) for d in det[p]]) if det[p] else float("nan")
        half = np.median([1200 * math.log2(1 + 0.25 * (_f(r, 2 * j) - 2 * _f(r, j)) / (2 * _f(r, j))) for r in rs])
        ctrl = [d for p in POSITIONS[1:] for d in det[p]]
        lines.append(f"| {j} | {len(rs)} | {100 * len(det['phantom']) / len(rs):.0f} % | {100 * len(det['control below']) / len(rs):.0f} % | "
                     f"{100 * len(det['control between']) / len(rs):.0f} % | {off('phantom'):.1f} | "
                     f"{np.median([abs(d['offset_cents']) for d in ctrl]) if ctrl else float('nan'):.1f} | {half:.1f} |")

    lines += ["", "## 2. Velocity", "",
              "Across notes (pedal up, not near an undamped string), levels regressed on partial j's level with pitch as a "
              "covariate: a phantom's slope should be ~2, partial 2j's ~1 plus the hammer's brightening. Detected notes only "
              "for the phantom, so its slope is biased low (at low partial levels only the strongest components pass the "
              "detector); the levels at 2f_j re partial 2j are medians.", "",
              "| j | notes | slope, level at 2f_j | slope, partial 2j | 2f_j re partial 2j (dB) |", "|---|---|---|---|---|"]
    for j in J:
        rs = [r for r in rows if j in r["j"] and clean(r, j) and "phantom" in r["j"][j] and r["j"][j]["phantom"]["detected"]]
        if len(rs) < 8:
            continue
        X = np.array([[1.0, r["j"][j]["L_j"], r["pitch"]] for r in rs])
        s_ph = np.linalg.lstsq(X, np.array([r["j"][j]["phantom"]["L"] for r in rs]), rcond=None)[0][1]
        s_2j = np.linalg.lstsq(X, np.array([r["j"][j]["L_2j"] for r in rs]), rcond=None)[0][1]
        rel = np.median([r["j"][j]["phantom"]["L"] - r["j"][j]["L_2j"] for r in rs])
        lines.append(f"| {j} | {len(rs)} | {s_ph:.2f} | {s_2j:.2f} | {rel:+.1f} |")

    lines += ["", "## 3. Sympathy and pedal", "",
              "| j | detected at 2f_j: pedal up, not near an undamped string | near an undamped string | pedal down |", "|---|---|---|---|"]
    for j in J:
        groups = ([r for r in rows if j in r["j"] and clean(r, j)],
                  [r for r in rows if j in r["j"] and not r["pedal"] and r["j"][j]["sympathetic"]],
                  [r for r in rows if j in r["j"] and r["pedal"]])
        rate = lambda g: (f"{100 * np.mean([('phantom' in r['j'][j]) and r['j'][j]['phantom']['detected'] for r in g]):.0f} % "
                          f"of {len(g)}") if g else "-"
        lines.append(f"| {j} | " + " | ".join(rate(g) for g in groups) + " |")

    lines += ["", "## 4. Decay (weak evidence)", "", "Level change from 30-330 ms to 330-630 ms after the onset, "
              "dB/s, detected notes; the ratio phantom / partial 2j (the fitted model: 1.1-1.5 in the aftersound).", "",
              "| j | notes | phantom | partial 2j | ratio |", "|---|---|---|---|---|"]
    for j in J:
        d = [r["j"][j]["decay_db_per_s"] for r in rows if j in r["j"] and clean(r, j) and "decay_db_per_s" in r["j"][j]
             and "phantom" in r["j"][j] and r["j"][j]["phantom"]["detected"]]
        if len(d) >= 8:
            a, b = np.median([x[0] for x in d]), np.median([x[1] for x in d])
            lines.append(f"| {j} | {len(d)} | {a:.1f} | {b:.1f} | {a / b if b > 0 else float('nan'):.2f} |")

    if args.ckpt:
        prior, fitted = model_levels(None), model_levels(args.ckpt)
        lines += ["", "## Model (velocity 64, median over MIDI 51-87): 2f_j re partial 2j (dB)", "",
                  "| | " + " | ".join(f"j={j}" for j in J) + " |", "|---|" + "---|" * len(J),
                  "| prior | " + " | ".join(f"{prior[j]:+.1f}" for j in J) + " |",
                  f"| `{args.ckpt}` | " + " | ".join(f"{fitted[j]:+.1f}" for j in J) + " |"]
    text = "\n".join(lines) + "\n"
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
