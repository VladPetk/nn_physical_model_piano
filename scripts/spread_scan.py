"""Per-strike, per-partial spreads (docs/tone_measures.md 16, item 3): how much the partials' decays and fluctuation
differ from note to note, recordings against models rendered with different spreads.

    python scripts/spread_scan.py runs/phase5/spread/base runs/phase5/spread/d04 ... --out runs/phase5/spread/report.md

Each argument is a ``scripts/note_profile.py`` output folder (same notes; its ``profile.npz``). Per feature (early
and late decay, the two-stage contrast, the fluctuation), the spread across notes (IQR) of each partial 1-12, and the
median over the partials of log2(model IQR / recording IQR): 0 where the model's notes vary as the piano's, -1 where
half as much. The spreads to keep are those nearest 0 on all four.
"""

import argparse
import json
import os

import numpy as np

FEATS = ["early decay (dB/s, 50-350 ms)", "late decay (dB/s, 0.5-1 s)", "two-stage: early − late (dB/s)", "fluctuation (dB rms)",
         "peak time (ms)"]
SOUNDING = ("late", "two-stage", "fluct")


def iqr(v):
    v = v[np.isfinite(v)]
    return float(np.subtract(*np.percentile(v, [75, 25]))) if len(v) >= 5 else np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows, rec_iqr = [], None
    for run in args.runs:
        d = np.load(os.path.join(run, "profile.npz"))
        label = [k.split("|")[0] for k in d.files if "|" in k and not k.startswith("recording")][0]
        snd = d["sounding"]
        ratios, cells = [], []
        for f in FEATS:
            sel = snd if f.startswith(SOUNDING) else np.ones(len(snd), bool)
            r = np.array([iqr(d[f"recording|{f}"][sel][:, k]) for k in range(d[f"recording|{f}"].shape[1])])
            m = np.array([iqr(d[f"{label}|{f}"][sel][:, k]) for k in range(d[f"{label}|{f}"].shape[1])])
            ok = np.isfinite(r) & np.isfinite(m) & (r > 0) & (m > 0)
            ratios.append(float(np.median(np.log2(m[ok] / r[ok]))))
            cells.append(f"{np.median(m[ok]):.1f} / {np.median(r[ok]):.1f}")
        with open(os.path.join(run, "events.json")) as fh:
            n = len(json.load(fh))
        rows.append((label, run, ratios, cells, n))
    L = ["# Per-partial spreads: model against recordings", "",
         "Median over partials 1-12 of log2(model IQR / recording IQR) across notes; in brackets the median IQRs "
         "(model / recording). 0: the model's notes differ from one another as much as the piano's.", "",
         "| model | " + " | ".join(FEATS) + " | rms of the first four |", "|---|" + "---|" * (len(FEATS) + 1)]
    for label, run, ratios, cells, n in rows:
        rms = float(np.sqrt(np.mean(np.square(ratios[:4]))))
        L.append(f"| {label} | " + " | ".join(f"{x:+.2f} ({c})" for x, c in zip(ratios, cells)) + f" | {rms:.2f} |")
    text = "\n".join(L) + "\n"
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
