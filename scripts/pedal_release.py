"""E6: what a part-pressed sustain pedal does to a released note, recording against models (docs/tone_measures.md 16).

    python scripts/pedal_release.py data/maestro24k --out runs/phase5/pedal_release \\
        --model phase5=runs/phase5/train_run/train/last.pt:physics \\
        --model theta79=runs/phase5/train_run/train/last.pt:physics:pedal_theta=79

Events (``measures.pedal_releases``): note-offs of MIDI 48-88 with no onset 0.1 s before to 0.6 s after and the
sustain pedal steady (within 12 CC) from 0.3 s before to 0.6 s after, at any depth, in every piece of ``--year``
(any split: this measures the sound, not a fit), at most ``--cap`` per (pedal depth, register). Each is rendered by
every model in its context. Per partial of the released note clear of the other notes that may ring
(``measures.pedal_release_decay``): the drop 0.4-0.6 s after the note-off (dB) and the extra decay the dampers add
(dB/s, the slope after minus before). Reported by pedal depth (CC at the note-off) and partial group: recording,
model and paired (model - recording, median over the partials of the events where both count). Fully down (112+)
the dampers stay off; below 40 they fall: the depths between show where this piano's dampers lift, and how much
more they take from the upper partials than from the lower.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from pianonn import measures as M  # noqa: E402
from pianonn.config import year_to_condition  # noqa: E402
from pianonn.render import load_variant  # noqa: E402

PRE, POST = 0.45, 0.75
GROUPS = (("1", 0, 1), ("2-3", 1, 3), ("4-6", 3, 6), ("7-10", 6, 10), ("11-16", 10, 16))
REGS = (("R3", 48, 59), ("R4", 60, 71), ("R5-6", 72, 88))
FEATS = (("drop", "drop 0.4-0.6 s after the note-off (dB)"), ("extra", "extra decay after the note-off (dB/s)"))


def mine(root, year, cap, seed):
    with open(os.path.join(root, "index.json")) as f:
        pieces = [p for p in json.load(f) if p["year"] == year]
    cells = {}
    for p in pieces:
        with np.load(os.path.join(root, p["midi"])) as z:
            notes, ped = z["notes"], {k: z[k] for k in z.files}
        for i, cc, others in M.pedal_releases(notes, ped):
            pi = int(notes[i, 0])
            depth = M.bin_of(cc, M.PEDAL_DEPTHS)
            reg = M.bin_of(pi, REGS)
            ev = {"piece": p["id"], "split": p["split"], "pitch": pi, "velocity": int(notes[i, 3]),
                  "onset": float(notes[i, 1]), "offset": float(notes[i, 2]), "cc": float(cc), "depth": depth,
                  "register": reg, "others": others}
            cells.setdefault((depth, reg), []).append(ev)
    rng = np.random.default_rng(seed)
    out, avail = [], {}
    for key in sorted(cells):
        evs = cells[key]
        avail["|".join(key)] = len(evs)
        idx = rng.permutation(len(evs))[:cap]
        out += [evs[j] for j in sorted(idx)]
    return out, avail


def med(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return (float(np.median(a)), len(a)) if len(a) else (float("nan"), 0)


def cell(v, p=1):
    m, n = med(v)
    return "" if not n else f"{m:+.{p}f} ({n})"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data")
    ap.add_argument("--model", action="append", required=True,
                    help="label=checkpoint:physics[:options] (pianonn.render.load_variant); the first gives the partial table")
    ap.add_argument("--year", type=int, default=2018)
    ap.add_argument("--cap", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=48)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.6)

    events, avail = mine(args.data, args.year, args.cap, args.seed)
    models = [load_variant(m, device=dev) for m in args.model]
    labels = [m[0] for m in models]
    cfg, sr = models[0][1].cfg, models[0][1].cfg.sample_rate
    table = M.partial_table(models[0][1], year_to_condition(args.year), dev)
    print(f"{len(events)} note-offs; available per (depth, register): {avail}", flush=True)
    srcs = ["recording"] + labels
    res = {s: {f: [] for f, _ in FEATS} for s in srcs}
    for s0 in range(0, len(events), args.chunk):
        part = events[s0: s0 + args.chunk]
        clips = M.render_clips(args.data, part, models, cfg, dev, pre=PRE, post=POST, batch=8, at="offset")
        for i, ev in enumerate(part):
            others = [table[p - 21] for p in ev["others"] if 21 <= p <= 108]
            for s in srcs:
                r = M.pedal_release_decay(clips[s][i], sr, PRE, table[ev["pitch"] - 21], others)
                for f, _ in FEATS:
                    res[s][f].append(r[f])
        print(f"{min(s0 + args.chunk, len(events))}/{len(events)}", flush=True)
    A = {s: {f: np.array(res[s][f]) for f, _ in FEATS} for s in srcs}  # [E, 16]
    depth = np.array([ev["depth"] for ev in events])
    reg = np.array([ev["register"] for ev in events])

    L = [f"# E6: release under a part-pressed pedal ({args.year})", "",
         f"Note-offs of MIDI 48-88, no onset 0.1 s before to 0.6 s after, the sustain steady (12 CC) from 0.3 s before "
         f"to 0.6 s after: {len(events)} (at most {args.cap} per pedal depth and register, of "
         f"{sum(avail.values())}). Models: " + "; ".join(f"`{m}`" for m in args.model) + ". Cells: median over the "
         "partials of the events (in brackets the partials counted); paired = median of model - recording where both "
         "count.", "", "Events per depth: " + ", ".join(f"{d} {int((depth == d).sum())}" for d, _, _ in M.PEDAL_DEPTHS), ""]
    for f, title in FEATS:
        L += [f"## {title}", ""]
        for s in srcs:
            hdr = s if s == "recording" else f"{s}, paired re recording"
            L += [f"### {hdr}", "", "| pedal CC | " + " | ".join(f"partials {g}" for g, _, _ in GROUPS) + " | all |",
                  "|---|" + "---|" * (len(GROUPS) + 1)]
            for d, _, _ in M.PEDAL_DEPTHS:
                sel = depth == d
                row = []
                for _, a, b in GROUPS + (("all", 0, 16),):
                    v = A[s][f][sel][:, a:b] if s == "recording" else (A[s][f] - A["recording"][f])[sel][:, a:b]
                    row.append(cell(v))
                L.append(f"| {d} | " + " | ".join(row) + " |")
            L.append("")
        L += [f"### by register, all partials ({title})", "",
              "| pedal CC | register | recording | " + " | ".join(f"{s} paired" for s in labels) + " |",
              "|---|---|---|" + "---|" * len(labels)]
        for d, _, _ in M.PEDAL_DEPTHS:
            for r, _, _ in REGS:
                sel = (depth == d) & (reg == r)
                if sel.any():
                    L.append(f"| {d} | {r} | {cell(A['recording'][f][sel])} | "
                             + " | ".join(cell((A[s][f] - A['recording'][f])[sel]) for s in labels) + " |")
        L.append("")
    text = "\n".join(L) + "\n"
    with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    with open(os.path.join(args.out, "events.json"), "w") as fh:
        json.dump(events, fh, indent=1)
    np.savez_compressed(os.path.join(args.out, "release.npz"), depth=depth, register=reg,
                        cc=np.array([ev["cc"] for ev in events]),
                        **{f"{s}|{f}": A[s][f] for s in srcs for f, _ in FEATS})
    print(text)


if __name__ == "__main__":
    main()
