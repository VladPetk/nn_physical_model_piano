"""N12 (extra peaks) on notes that follow silence, in the recordings of every MAESTRO year (docs/tone_measures.md,
section 11). In music a note's surroundings hold earlier notes' partials, so N12 sees only components within
~20 dB of the partials; after a second of silence the only background is the room's floor.

    python scripts/survey_after_silence.py data/maestro24k --model runs/round2/b_control/last.pt --out runs/survey/after_silence

A note qualifies when everything has been still for ``--silence`` s before it (every key up, pedal up, no undamped
key for 4 s: ``M.free_decays``), it is struck alone (no other onset within 30 ms), nothing follows for 0.7 s, the
pedal stays up through the window and it is held 0.45 s (any length for the undamped keys). The model's 2018
partial table is only the starting comb: N12 tracks each note's own partials.

After silence the floor between the partials is the recording's own noise (it matches the stretch before the
note), so what N12 can see depends on the note's level: each note's *limit* is the median, over partials 1-6, of
the floor half-way to the next partial re the partial, + 10 dB (N12's prominence). Results are grouped by it, and
checked on the same clips: steady tones added half-way between partials 3 and 4, and 2 % and 4 % above partial 3.
"""

import argparse
import collections
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

from pianonn import measures as M  # noqa: E402
from pianonn.config import year_to_condition  # noqa: E402
from pianonn.render import load_model  # noqa: E402

UNDAMPED = 89
LABELS = ("phantom", "near", "between")
LIMITS = (("limit ≤ −40 dB", -200, -40), ("−40 … −30 dB", -40, -30), ("above −30 dB", -30, 200))


def candidates(root, years, silence=1.0, window=0.65, clear=0.7, hold=0.45):
    with open(os.path.join(root, "index.json")) as f:
        pieces = sorted((p for p in json.load(f) if p["year"] in years), key=lambda p: p["id"])
    out = []
    for p in pieces:
        with np.load(os.path.join(root, p["midi"])) as z:
            z = {k: z[k] for k in z.files}
        notes = z["notes"]
        on = notes[:, 1]
        st, sv = z["sustain_t"], z["sustain_v"]
        for _, t in M.free_decays(notes, z, p["duration"], gap=silence):
            idx = np.nonzero(np.abs(on - t) < 0.03)[0]
            if len(idx) != 1 or np.any((on > t + 0.03) & (on < t + clear)) or t + M.POST > p["duration"] or t < M.PRE + 0.1:
                continue
            ped = [M._value_at(st, sv, t)] + [v for tt, v in zip(st, sv) if t <= tt < t + window]
            if max(ped) >= M.PEDAL[0][2]:
                continue
            pitch, _, t_off, vel = notes[idx[0]]
            reg = M.register_of(int(pitch))
            if reg is None or (t_off - t < hold and pitch < UNDAMPED):
                continue
            out.append({"piece": p["id"], "year": p["year"], "audio": p["audio"], "pitch": int(pitch), "velocity": int(vel),
                        "onset": float(t), "offset": float(t_off), "register": reg, "vel_bin": M.bin_of(vel, M.VELOCITY)})
    return out


def clip(root, n, sr):
    x, _ = sf.read(os.path.join(root, n["audio"]), start=int(round((n["onset"] - M.PRE) * sr)),
                   frames=int(round((M.PRE + M.POST) * sr)), dtype="float64", always_2d=True)
    return x


def with_tone(x, sr, t_on, f, level_db, ref_bin, n_fft, seg):
    """``x`` plus a steady tone at ``f`` from ``t_on``, ``level_db`` re the peak power at ``ref_bin`` (per channel)."""
    N = len(seg)
    X = np.abs(np.fft.rfft(seg * np.hanning(N)[:, None], n_fft, axis=0)) ** 2
    pk = X[ref_bin - 2: ref_bin + 3].max(0)
    a = 2 * np.sqrt(10 ** (level_db / 10) * pk) / np.hanning(N).sum()
    t = np.arange(len(x)) / sr
    return x + np.clip((t - t_on) / 0.005, 0, 1)[:, None] * a[None, :] * np.sin(2 * np.pi * f * t + 0.7)[:, None]


def med(v):
    return f"{np.median(v):+.1f}" if len(v) else "–"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data")
    ap.add_argument("--model", required=True, help="checkpoint whose 2018 partial table seeds the partial search")
    ap.add_argument("--years", type=int, nargs="*", default=[2004, 2006, 2008, 2009, 2011, 2013, 2014, 2015, 2017, 2018])
    ap.add_argument("--silence", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(args.out, exist_ok=True)
    model = load_model(args.model, device="cpu")
    sr = model.cfg.sample_rate
    table = M.partial_table(model, year_to_condition(2018), torch.device("cpu"))
    notes = candidates(args.data, args.years, args.silence)
    print(f"{len(notes)} notes after {args.silence} s of silence", flush=True)

    rows, sens = [], collections.defaultdict(list)
    for i, n in enumerate(notes):
        x = clip(args.data, n, sr)
        ps = table[n["pitch"] - 21]
        t_on = M.onset(x, sr, M.PRE, float(ps[0]))
        t_on = M.PRE if not np.isfinite(t_on) else t_on
        t1 = 0.65 if n["pitch"] >= UNDAMPED else min(0.65, n["offset"] - n["onset"] - 0.03)
        m = M.extra_peaks(x, sr, t_on, ps, t1=t1)
        if "N12 n" not in m:
            continue
        seg = M._segment(x, sr, t_on + 0.05, t_on + t1)
        n_fft = 1 << int(math.ceil(math.log2(len(seg) * 4)))
        hz = np.fft.rfftfreq(n_fft, 1 / sr)
        P = M._power(seg, n_fft)
        own = M.own_partials(P, hz, ps, 8000.0, track=True)
        own = own[own < 8000]
        if len(own) < 2:
            continue
        df, w = sr / n_fft, int(round(26.7 * n_fft / sr))
        gaps = []
        for k in range(min(6, len(own) - 1)):
            pk = P[int(round(own[k] / df)) - 2: int(round(own[k] / df)) + 3].max()
            c = int(round(0.5 * (own[k] + own[k + 1]) / df))
            gaps.append(10 * math.log10(np.median(P[c - w: c + w + 1]) / pk))
        limit = float(np.median(gaps)) + 10.0
        group = next(g for g, lo, hi in LIMITS if lo < limit <= hi)
        rows.append({**{k: n[k] for k in ("piece", "year", "pitch", "velocity", "register", "vel_bin")}, "limit": limit, "N12": m})
        if len(own) < 4:
            continue
        res = 2.0 / (t1 - 0.05)
        ref = int(round(own[2] * n_fft / sr))
        for where, f in (("half-way 3-4", 0.5 * (own[2] + own[3])), ("4 % above 3", 1.04 * own[2]), ("2 % above 3", 1.02 * own[2])):
            if abs(f - own[2]) <= 2 * res + 0.0015 * own[2] or f > 0.5 * (own[2] + own[3]) + 1:
                continue
            for db in (-30.0, -40.0, -50.0, -60.0):
                got = M.extra_peaks(with_tone(x, sr, t_on, f, db, ref, n_fft, seg), sr, t_on, ps, t1=t1).get("N12 peaks", [])
                sens[(group, where, db)].append(any(abs(p[0] - f) <= res + 1 for p in got))
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(notes)}", flush=True)

    regs = [r for r, _, _ in M.REGISTERS]
    L = [f"# N12 on notes after {args.silence} s of silence: recordings of {', '.join(map(str, args.years))}", "",
         f"{len(rows)} single notes, pedal up, nothing else for 0.7 s (`scripts/survey_after_silence.py`).", "",
         "## Sensitivity: steady tones added to these clips, re partial 3, by the note's limit", "",
         "| limit | position | " + " | ".join(f"{db:+.0f} dB" for db in (-30, -40, -50, -60)) + " |", "|---|---|---|---|---|---|"]
    for g, _, _ in LIMITS:
        for where in ("half-way 3-4", "4 % above 3", "2 % above 3"):
            if not sens[(g, where, -30.0)]:
                continue
            L.append(f"| {g} | {where} | " + " | ".join(
                f"{100 * np.mean(sens[(g, where, db)]):.0f} % ({len(sens[(g, where, db)])})" for db in (-30.0, -40.0, -50.0, -60.0)) + " |")
    L += ["", "## Extra peaks by register and limit", "",
          "| register | limit | notes | per note: phantom / near / between | notes with near or between | dB re nearest partial: phantom / near / between (median) |",
          "|---|---|---|---|---|---|"]
    for reg in regs:
        for g, lo, hi in LIMITS:
            rr = [r["N12"] for r in rows if r["register"] == reg and lo < r["limit"] <= hi]
            if not rr:
                continue
            per = " / ".join(f"{np.mean([m[f'N12 n {k}'] for m in rr]):.2f}" for k in LABELS)
            share = np.mean([m["N12 n near"] + m["N12 n between"] > 0 for m in rr])
            lv = " / ".join(med([p[3] for m in rr for p in m["N12 peaks"] if p[4] == k]) for k in LABELS)
            L.append(f"| {reg} | {g} | {len(rr)} | {per} | {100 * share:.0f} % | {lv} |")
    L += ["", f"Near peaks below 1397 Hz on damped keys whose limit is ≤ −40 dB (no undamped string answers there):", "",
          "| register | notes | near peaks below 1397 Hz per note | notes with one | dB re partial (median) |", "|---|---|---|---|---|"]
    for reg in ("R3", "R4", "R5", "R6"):
        rr = [r["N12"] for r in rows if r["register"] == reg and r["pitch"] < UNDAMPED and r["limit"] <= -40]
        if rr:
            v = [[p for p in m["N12 peaks"] if p[4] == "near" and p[0] < 1397] for m in rr]
            L.append(f"| {reg} | {len(rr)} | {np.mean([len(a) for a in v]):.2f} | {100 * np.mean([len(a) > 0 for a in v]):.0f} % | "
                     f"{med([p[3] for a in v for p in a])} |")
    pk = [p for r in rows for p in r["N12"]["N12 peaks"] if p[4] != "phantom"]
    L += ["", "Near and between peaks by cents from the nearest partial (all notes):", "", "| cents | peaks | dB re that partial (median) |", "|---|---|---|"]
    edges = (-1200, -100, -25, 0, 25, 100, 1200)
    for a, b in zip(edges[:-1], edges[1:]):
        s = [p for p in pk if a <= p[1] < b]
        L.append(f"| {a} … {b} | {len(s)} | {med([p[3] for p in s])} |")
    L += ["", "By year (a piano or hall resonance recurs within a year):", "",
          "| year | notes | per note: phantom / near / between | peaks at a frequency seen in ≥ 3 notes of ≥ 2 keys |", "|---|---|---|---|"]
    for y in sorted({r["year"] for r in rows}):
        rr = [r for r in rows if r["year"] == y]
        per = " / ".join(f"{np.mean([r['N12'][f'N12 n {k}'] for r in rr]):.2f}" for k in LABELS)
        fs = sorted((p[0], j, r["pitch"]) for j, r in enumerate(rr) for p in r["N12"]["N12 peaks"] if p[4] != "phantom")
        rec = set()
        for a in range(len(fs)):
            grp = [g for g in fs if abs(g[0] - fs[a][0]) <= 1.5]
            if len({g[1] for g in grp}) >= 3 and len({g[2] for g in grp}) >= 2:
                rec.add(round(float(np.median([g[0] for g in grp]))))
        L.append(f"| {y} | {len(rr)} | {per} | {', '.join(f'{f} Hz' for f in sorted(rec)) or '–'} |")
    text = "\n".join(L) + "\n"
    with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8") as f:
        f.write(text)
    with open(os.path.join(args.out, "rows.json"), "w") as f:
        json.dump({"rows": rows, "sensitivity": {f"{g} | {w} | {db}": v for (g, w, db), v in sens.items()}}, f)
    print(text)


if __name__ == "__main__":
    main()
