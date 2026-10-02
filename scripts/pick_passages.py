"""Pick the listening passages of review 6, step 2: held-out windows of kinds of texture, not of guessed faults.

    python scripts/pick_passages.py data/maestro24k --out runs/fitted_passages/passages/passages.json

Scans the validation and test pieces of the year in windows of ``--seconds`` (every 2.5 s) and describes each from
its MIDI: onsets per second, mean velocity, the sustain pedal's share of the time (CC >= 64), the share of onsets at
or above C5 (MIDI 72), the share struck together with another (within 30 ms) and the longest gap between onsets.
Then one window per kind, each from another piece:

1. ``note66``: excerpt 0 of the phase-4/5/6 listening sets around note 66 (a D3 that rang too long; tone_measures
   15, 16.7). Fixed, not searched.
2. ``dense``: the most onsets per second at mf (mean velocity 50-85), at most half of them in chords.
3. ``soft_pedal``: soft and sparse with the pedal down (<= 3 onsets/s, mean velocity <= 50, pedal >= 70 %), the
   longest gap.
4. ``loud_chords``: the loudest of the windows where most onsets come in chords (>= 50 %).
5. ``treble``: a melody over an accompaniment: 35-65 % of the onsets at C5 or above, 3-8 onsets/s, the highest share.

``start_s`` is where the rendered window starts; the scored part begins after the 1 s warm-up.
"""

import argparse
import json
import os

import numpy as np

NOTE66 = {"kind": "note66", "piece": "MIDI-Unprocessed_Recital13-15_MID--AUDIO_15_R1_2018_wav--1", "start_s": 76.0,
          "why": "listen_long excerpt 0: the slow B1-B2 bass under a texture and note 66 (D3, velocity 87, struck at "
                 "81.8 s, released at 82.4 s under the pedal; heard ringing too long with a metallic tone at 83-84 s)"}


def describe(z, a, b):
    n = z["notes"]
    on = n[(n[:, 1] >= a) & (n[:, 1] < b)]
    if len(on) == 0:
        return None
    t = np.sort(on[:, 1])
    chord = np.zeros(len(t), bool)
    d = np.diff(t) < 0.03
    chord[1:] |= d
    chord[:-1] |= d
    grid = np.arange(a, b, 0.01)
    st, sv = z["sustain_t"], z["sustain_v"]
    k = np.searchsorted(st, grid, side="right") - 1
    ped = np.where(k >= 0, sv[np.clip(k, 0, None)] if len(sv) else 0.0, 0.0) >= 64
    gaps = np.diff(np.concatenate([[a], t, [b]]))
    return {"onsets_per_s": round(len(t) / (b - a), 2), "velocity": round(float(on[:, 3].mean()), 1),
            "pedal": round(float(ped.mean()), 2), "treble": round(float((on[:, 0] >= 72).mean()), 2),
            "chords": round(float(chord.mean()), 2), "longest_gap_s": round(float(gaps.max()), 2),
            "lowest": int(on[:, 0].min()), "highest": int(on[:, 0].max())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data")
    ap.add_argument("--year", type=int, default=2018)
    ap.add_argument("--seconds", type=float, default=10.0, help="scored length (the 1 s warm-up comes on top)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    with open(os.path.join(args.data, "index.json")) as f:
        pieces = [p for p in json.load(f) if p["year"] == args.year and p["split"] in ("validation", "test")]
    rows = []
    for p in pieces:
        z = dict(np.load(os.path.join(args.data, p["midi"])))
        for s in np.arange(0.0, p["duration"] - args.seconds - 1.0, 2.5):
            d = describe(z, s + 1.0, s + 1.0 + args.seconds)
            if d:
                rows.append({"piece": p["id"], "split": p["split"], "start_s": float(s), **d})
    used = {NOTE66["piece"]}
    z66 = dict(np.load(os.path.join(args.data, next(p for p in pieces if p["id"] == NOTE66["piece"])["midi"])))
    chosen = [{**NOTE66, "split": "test", **describe(z66, NOTE66["start_s"] + 1, NOTE66["start_s"] + 1 + args.seconds)}]

    def pick(kind, ok, key, why):
        cand = [r for r in rows if r["piece"] not in used and ok(r)]
        if not cand:
            print("no window for", kind)
            return
        r = max(cand, key=key)
        used.add(r["piece"])
        chosen.append({"kind": kind, **r, "why": why})

    pick("dense", lambda r: 50 <= r["velocity"] <= 85 and r["chords"] <= 0.5, lambda r: r["onsets_per_s"],
         "the most onsets per second at mf, mostly single notes (runs, not repeated chords)")
    pick("soft_pedal", lambda r: r["onsets_per_s"] <= 3 and r["velocity"] <= 50 and r["pedal"] >= 0.7 and
         r["onsets_per_s"] >= 0.8, lambda r: r["longest_gap_s"], "soft and sparse, pedal down: tails exposed")
    pick("loud_chords", lambda r: r["chords"] >= 0.5, lambda r: r["velocity"], "loud chords: attack and knock")
    pick("treble", lambda r: 0.35 <= r["treble"] <= 0.65 and 3 <= r["onsets_per_s"] <= 8, lambda r: r["treble"],
         "treble melody over an accompaniment")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"seconds": args.seconds, "warmup": 1.0, "passages": chosen}, f, indent=1)
    for c in chosen:
        print(c["kind"], c["piece"], c["start_s"], {k: c[k] for k in ("onsets_per_s", "velocity", "pedal", "treble", "chords")})


if __name__ == "__main__":
    main()
