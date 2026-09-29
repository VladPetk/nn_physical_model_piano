"""Mine effectively isolated notes from prepared MAESTRO recordings and measure them (review 3, section 4.6).

Performances almost never leave a note alone for long (of 727k notes in the 2018 training split, 24 have
nothing else sounding for a second and no pedal). What the spectrum needs is a clear first second: a note
qualifies when no other key is struck from ``--before`` s before it to ``--clear`` s after it, so every
other string is either damped or has been decaying for a while and sits well below the fresh strike.
Its first second (both channels in power) goes through the Iowa calibration's partial tracker
(``pianonn.calibration.track_partials``, B constrained to x/4 of the Rigaud curve): inharmonicity B and
tuning (cents re equal temperament at A440).

The per-key medians are the measured side of the identifiability check: after fitting, the
model's B and stretch should agree with them. They are also per-year priors.

    python scripts/mine_notes.py data/maestro24k --years 2018 --out runs/mined_2018.json
"""

import argparse
import json
import os
from multiprocessing import Pool

import numpy as np


def _isolated(notes, before, clear, max_per_piece):
    on = notes[:, 1]
    out = []
    for i in range(len(notes)):
        others = np.ones(len(notes), bool)
        others[i] = False
        if not np.any(others & (on > on[i] - before) & (on < on[i] + clear)):
            out.append(i)
    rng = np.random.default_rng(0)
    return list(rng.choice(out, min(len(out), max_per_piece), replace=False)) if len(out) > max_per_piece else out


def _mine_piece(job):
    import math

    import soundfile as sf

    from pianonn.calibration import rigaud_B, track_partials

    root, piece, before, clear, max_per_piece = job
    with np.load(os.path.join(root, piece["midi"])) as z:
        notes = z["notes"]
    path = os.path.join(root, piece["audio"])
    sr = sf.info(path).samplerate
    res = []
    for i in _isolated(notes, before, clear, max_per_piece):
        pitch, t0, t1, vel = notes[i]
        start = int((t0 - 0.3) * sr)
        if start < 0:
            continue
        x, _ = sf.read(path, start=start, frames=int((0.3 + clear) * sr), dtype="float64", always_2d=True)
        f_nom = 440.0 * 2 ** ((pitch - 69) / 12)
        Br = rigaud_B(pitch)
        tr = track_partials(x, sr, int(0.3 * sr), f_nom, (Br / 4, Br * 4), seconds=clear)
        if tr is None:
            continue
        f1 = tr["f"][0] if tr["n"][0] == 1 else tr["f0"] * math.sqrt(1 + tr["B"])
        cents = 1200 * math.log2(f1 / f_nom)
        res.append({"piece": piece["id"], "pitch": int(pitch), "velocity": int(vel), "onset": float(t0),
                    "B": tr["B"], "B_reliable": bool(len(tr["n"]) >= 8 and tr["fit_rms_cents"] < 3.0),
                    "cents": cents, "suspect": bool(abs(cents) > 50), "n_partials": len(tr["n"]),
                    "fit_rms_cents": tr["fit_rms_cents"]})
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data")
    ap.add_argument("--years", type=int, nargs="*")
    ap.add_argument("--split", default="train")
    ap.add_argument("--before", type=float, default=1.0, help="seconds with no other strike before the note")
    ap.add_argument("--clear", type=float, default=1.0, help="seconds with no other strike after the note (analysis window)")
    ap.add_argument("--max-per-piece", type=int, default=60)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    with open(os.path.join(args.data, "index.json")) as f:
        pieces = [p for p in json.load(f) if p["split"] == args.split and (not args.years or p["year"] in args.years)]
    jobs = [(args.data, p, args.before, args.clear, args.max_per_piece) for p in pieces]
    notes = []
    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_mine_piece, jobs), 1):
            notes += r
            print(f"[{i}/{len(jobs)}] {len(notes)} notes", flush=True)
    per_key = {}
    for n in notes:
        if not n["suspect"]:
            per_key.setdefault(n["pitch"], []).append(n)
    summary = {}
    for p, ns in sorted(per_key.items()):
        rel = [n["B"] for n in ns if n["B_reliable"]]
        summary[p] = {"n": len(ns), "cents": float(np.median([n["cents"] for n in ns])),
                      "B": float(np.median(rel)) if rel else None, "n_B": len(rel)}
    with open(args.out, "w") as f:
        json.dump({"notes": notes, "per_key": summary}, f, indent=1)
    print(f"{len(notes)} isolated notes on {len(summary)} keys -> {args.out}")


if __name__ == "__main__":
    main()
