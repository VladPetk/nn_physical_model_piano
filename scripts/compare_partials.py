"""Compare per-partial spectra and decays of the model with measured isolated notes.

    python scripts/compare_partials.py data/iowa_partials.json [--dynamics mf ff] [--save model_partials.json]

The recordings' tables come from ``pianonn.calibration.partial_tables`` (early level and the level at
0.5/1/2/3/5 s of every partial below 10 kHz). The model renders the same keys and dynamics like a
recording (through the soundboard body, no hall, 60 dB noise floor) and is measured with the same code.
Results are medians over the notes of a register, binned by partial frequency (octaves) and by number.
"""

import argparse
import json
import math

import numpy as np

from pianonn.calibration import PARTIAL_TIMES, partial_table, render_note

VELOCITY = {"pp": 30, "mf": 64, "ff": 110}
REGISTERS = [("A0-B1", 21, 35), ("C2-B2", 36, 47), ("C3-B3", 48, 59), ("C4-B4", 60, 71), ("C5-B5", 72, 83), ("C6-C8", 84, 108)]
BANDS = [55 * 2**k for k in range(9)]  # 55 Hz .. 14 kHz octave edges


def model_tables(model, notes):
    sr = model.cfg.sample_rate
    out = []
    for pitch, dyn in notes:
        r = partial_table(render_note(model, pitch, VELOCITY[dyn], max(PARTIAL_TIMES) + 0.6, body=True), sr, pitch)
        if r is not None:
            r["dynamic"] = dyn
            out.append(r)
    return out


def rows(tables):
    """(register, band index, n, early re loudest, levels...) per partial."""
    out = []
    for r in tables:
        e = np.array(r["early_db"], dtype=float)
        if not np.isfinite(e).any():
            continue
        e = e - np.nanmax(e)
        for n, f, ee, lv in zip(r["n"], r["f"], e, r["level_db"]):
            b = int(np.searchsorted(BANDS, f)) - 1
            out.append((r["pitch"], b, n, ee, *[float(v) for v in lv]))
    return np.array(out, dtype=float)


def summary(rec, mod):
    R, M = rows(rec), rows(mod)
    times = PARTIAL_TIMES
    lines = []
    for name, lo, hi in REGISTERS:
        r, m = R[(R[:, 0] >= lo) & (R[:, 0] <= hi)], M[(M[:, 0] >= lo) & (M[:, 0] <= hi)]
        if not len(r) or not len(m):
            continue
        lines.append(f"\n{name}: level re the loudest partial at the attack, then level at "
                     + "/".join(f"{t:g}" for t in times) + " s re its own attack   [recording | model]  (count)")
        for b in range(len(BANDS) - 1):
            rb, mb = r[r[:, 1] == b], m[m[:, 1] == b]
            if len(rb) < 5 or len(mb) < 5:
                continue
            fr = lambda a: " ".join(f"{np.nanmedian(a[:, j]):6.1f}" if np.isfinite(a[:, j]).sum() >= 5 else "     -" for j in range(3, 4 + len(times)))
            lines.append(f"  {BANDS[b]:5.0f}-{BANDS[b + 1]:5.0f} Hz  {fr(rb)} | {fr(mb)}   ({len(rb)}/{len(mb)})")
    lines.append("\nspread of the 1 s level over partials 1-16 within a note (median IQR, dB)   [recording | model]")
    for name, lo, hi in REGISTERS:
        def iqr(tables):
            v = []
            for r in tables:
                if lo <= r["pitch"] <= hi:
                    lv = np.array([x[1] for x in r["level_db"][:16]], dtype=float)
                    if np.isfinite(lv).sum() >= 6:
                        v.append(np.subtract(*np.nanpercentile(lv, [75, 25])))
            return np.median(v) if v else float("nan")
        lines.append(f"  {name}: {iqr(rec):5.1f} | {iqr(mod):5.1f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("recordings")
    ap.add_argument("--dynamics", nargs="+", default=["mf", "ff"])
    ap.add_argument("--keys", default="21-108")
    ap.add_argument("--save")
    ap.add_argument("--load")
    args = ap.parse_args()
    lo, hi = map(int, args.keys.split("-"))
    rec = [r for r in json.load(open(args.recordings)) if r["dynamic"] in args.dynamics and lo <= r["pitch"] <= hi]
    if args.load:
        mod = json.load(open(args.load))
    else:
        from pianonn.render import load_model

        mod = model_tables(load_model(), [(r["pitch"], r["dynamic"]) for r in rec])
    if args.save:
        json.dump(mod, open(args.save, "w"), default=float)
    print(summary(rec, mod))


if __name__ == "__main__":
    main()
