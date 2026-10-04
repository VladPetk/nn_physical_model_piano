"""When is the model duller than the piano? The brightness gap frame by frame against the musical context
(docs/physics_revamp.md 13).

    python scripts/brightness_context.py data/maestro24k --out runs/physics_revamp/attack/context \\
        --model main=runs/physics_revamp/main/residual/last.pt:residual

``--clips`` excerpts of ``--seconds`` s drawn at random from the year's pieces (seeded), rendered by each model in their
MIDI context (``measures.render_clips``). Per 50 ms frame, on both sides: the level (dB) of each octave band (centres
``BANDS``) and the brightness, the level of 0.7-5.6 kHz re 88-700 Hz. Per frame from the MIDI: the sustain pedal, the
notes sounding (key held, or released under the pedal within the last 4 s), the onsets in the last second, the time
since the last onset, the loudest velocity struck in the last 0.3 s and the mean pitch of the notes sounding.

Writes ``report.md`` (the gap, model − recording, by context: medians over frames with a 95 % interval from a
bootstrap over clips), ``frames.npz``.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from pianonn import measures as M  # noqa: E402
from pianonn.render import load_variant  # noqa: E402

BANDS = (125, 250, 500, 1000, 2000, 4000, 8000)
HOP = 0.05


def band_levels(x, sr):
    """``[F, B]`` dB per 50 ms frame (100 ms Hann windows) and octave band."""
    n = int(0.1 * sr)
    h = int(HOP * sr)
    w = np.hanning(n)[:, None]
    f = np.fft.rfftfreq(n, 1 / sr)
    out = []
    for s in range(0, len(x) - n, h):
        P = (np.abs(np.fft.rfft(x[s: s + n] * w, axis=0)) ** 2).sum(1)
        out.append([P[(f >= c / np.sqrt(2)) & (f < c * np.sqrt(2))].sum() for c in BANDS])
    return 10 * np.log10(np.array(out) + 1e-20), (np.arange(len(out)) * h + n / 2) / sr


def context(notes, st, sv, t):
    """MIDI context at absolute time ``t`` (s)."""
    k = np.searchsorted(st, t, side="right") - 1
    ped = sv[k] if k >= 0 else 0.0
    on, off = notes[:, 1], notes[:, 2]
    struck = on <= t
    held = struck & (off > t)
    # released under the pedal: the pedal down (>= 64) at the release and ever since, within the last 4 s
    rel = struck & (off <= t) & (on > t - 4.0)
    under = np.zeros(len(notes), bool)
    for i in np.nonzero(rel)[0]:
        sel = (st > off[i]) & (st <= t)
        k0 = np.searchsorted(st, off[i], side="right") - 1
        v0 = sv[k0] if k0 >= 0 else 0.0
        under[i] = v0 >= 64 and np.all(sv[sel] >= 64)
    sounding = held | under
    recent = (on > t - 1.0) & struck
    last = on[struck].max() if struck.any() else t - 10
    hard = notes[(on > t - 0.3) & struck, 3]
    return [ped, sounding.sum(), recent.sum(), t - last, hard.max() if len(hard) else 0,
            notes[sounding, 0].mean() if sounding.any() else np.nan]


def boot(d, clip, n=500, seed=0):
    ok = np.isfinite(d)
    d, clip = d[ok], clip[ok]
    if len(d) < 20:
        return np.nan, np.nan, np.nan, len(d)
    u = np.unique(clip)
    idx = {c: np.nonzero(clip == c)[0] for c in u}
    rng = np.random.default_rng(seed)
    meds = [np.median(d[np.concatenate([idx[c] for c in rng.choice(u, len(u))])]) for _ in range(n)]
    return float(np.median(d)), *np.percentile(meds, [2.5, 97.5]), len(d)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data")
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--year", type=int, default=2018)
    ap.add_argument("--clips", type=int, default=48)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", action="store_true", help="add the recording cut by 6 dB above 1 kHz as a side")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.6)
    with open(os.path.join(args.data, "index.json")) as f:
        pieces = [p for p in json.load(f) if p["year"] == args.year]
    rng = np.random.default_rng(args.seed)
    events = []
    for i in rng.choice(len(pieces), args.clips, replace=len(pieces) < args.clips):
        p = pieces[i]
        t0 = float(rng.uniform(M.WARMUP + 1.0, p["duration"] - args.seconds - 1.0))
        events.append({"piece": p["id"], "onset": t0, "split": p["split"]})
    models = [load_variant(s, device=dev) for s in args.model]
    labels = [m[0] for m in models]
    bsrc = labels[-1]  # the model whose per-band gap the tables show
    cfg = models[0][1].cfg
    sr = cfg.sample_rate
    clips = M.render_clips(args.data, events, models, cfg, dev, pre=0.0, post=args.seconds, batch=2)
    if args.check:
        from tenor_attack import shelf
        clips["recording-6dB>1k"] = [shelf(x, sr, 1000.0, -6.0) for x in clips["recording"]]
        labels = labels + ["recording-6dB>1k"]
    midi = {p["id"]: p for p in pieces}
    rows = {s: [] for s in ["recording"] + labels}
    ctx, clip_id = [], []
    for i, ev in enumerate(events):
        with np.load(os.path.join(args.data, midi[ev["piece"]]["midi"])) as z:
            notes, st, sv = z["notes"], z["sustain_t"], z["sustain_v"]
        L = {s: band_levels(clips[s][i], sr) for s in rows}
        t = L["recording"][1]
        g = {s: np.median(L["recording"][0] - L[s][0]) for s in labels}  # level match per clip, over bands and frames
        for s in rows:
            rows[s].append(L[s][0] + (g[s] if s != "recording" else 0.0))
        ctx += [context(notes, st, sv, ev["onset"] + tt) for tt in t]
        clip_id += [i] * len(t)
    X = {s: np.concatenate(v) for s, v in rows.items()}
    C = np.array(ctx, float)
    clip_id = np.array(clip_id)
    lo = np.log10(np.sum(10 ** (X["recording"][:, :3] / 10), 1)) * 10
    hi = np.log10(np.sum(10 ** (X["recording"][:, 3:6] / 10), 1)) * 10
    bright = {s: 10 * np.log10(np.sum(10 ** (X[s][:, 3:6] / 10), 1)) - 10 * np.log10(np.sum(10 ** (X[s][:, :3] / 10), 1))
              for s in X}
    tot = 10 * np.log10(np.sum(10 ** (X["recording"] / 10), 1))
    live = tot > np.percentile(tot, 10)  # leave out the quietest frames (silence, the floor)
    np.savez_compressed(os.path.join(args.out, "frames.npz"), clip=clip_id, ctx=C, live=live,
                        **{f"{s}|bands": v for s, v in X.items()}, events=np.array(json.dumps(events)))
    names = ["pedal", "sounding", "onsets in 1 s", "since onset", "velocity (0.3 s)", "pitch sounding"]
    L = [f"# Brightness against context ({args.year}, {len(events)} clips of {args.seconds:g} s)", "",
         f"Models: " + ", ".join(f"`{s}`" for s in args.model) + ". Brightness: 0.7-5.6 kHz re 88-700 Hz (dB). Gap: "
         "model − recording per 50 ms frame (each model's level matched to the recording per clip); medians over frames, "
         "95 % interval from a bootstrap over clips, (n frames). The quietest 10 % of frames left out. Recording's "
         "brightness in brackets.", ""]

    def table(title, groups):
        nonlocal L
        L += [f"## {title}", "", "| context | recording | " + " | ".join(f"{s} − rec" for s in labels) +
              " | " + " | ".join(f"{s} − rec, {b} Hz" for s in [bsrc] for b in BANDS) + " |",
              "|---|---|" + "---|" * (len(labels) + len(BANDS))]
        for name, m in groups:
            m = m & live
            if m.sum() < 20:
                continue
            cells = []
            for s in labels:
                md, a, b, n = boot(np.where(m, bright[s] - bright["recording"], np.nan), clip_id)
                cells.append(f"{md:+.1f} [{a:+.1f}, {b:+.1f}] ({n})")
            band = [f"{np.median((X[bsrc][m, j] - X['recording'][m, j])):+.1f}" for j in range(len(BANDS))]
            L.append(f"| {name} | {np.median(bright['recording'][m]):+.1f} | " + " | ".join(cells) + " | " + " | ".join(band) + " |")
        L.append("")

    ped = C[:, 0] >= 64
    table("All frames", [("all", np.ones(len(C), bool))])
    table("Pedal", [("pedal up (< 64)", ~ped), ("pedal down (>= 64)", ped)])
    table("Notes sounding", [(f"{a}-{b}", (C[:, 1] >= a) & (C[:, 1] <= b)) for a, b in ((0, 2), (3, 5), (6, 10), (11, 20), (21, 999))])
    table("Pedal x notes sounding", [(f"pedal {'down' if p else 'up'}, {a}-{b} sounding", (ped == p) & (C[:, 1] >= a) & (C[:, 1] <= b))
                                     for p in (False, True) for a, b in ((0, 3), (4, 10), (11, 999))])
    table("Time since the last onset", [(f"{a}-{b} s", (C[:, 3] >= a) & (C[:, 3] < b)) for a, b in ((0, 0.1), (0.1, 0.3), (0.3, 1.0), (1.0, 99))])
    table("Loudest velocity struck in the last 0.3 s", [("none", C[:, 4] == 0)] + [(f"{a}-{b - 1}", (C[:, 4] >= a) & (C[:, 4] < b)) for a, b in ((1, 45), (45, 60), (60, 75), (75, 90), (90, 128))])
    table("Mean pitch sounding", [(f"MIDI {a}-{b - 1}", (C[:, 5] >= a) & (C[:, 5] < b)) for a, b in ((21, 48), (48, 60), (60, 72), (72, 109))])
    # a regression of the gap on the context, standardised
    m = live & np.all(np.isfinite(C), 1)
    Z = np.column_stack([np.ones(m.sum()), ped[m], np.log1p(C[m, 1]), np.log1p(C[m, 2]), np.log(C[m, 3] + 0.05), C[m, 4], C[m, 5]])
    nm = ["pedal down", "log(1 + sounding)", "log(1 + onsets in 1 s)", "log(time since onset)", "velocity (0.3 s)", "pitch sounding"]
    sd = Z.std(0)
    sd[0] = 1
    L += ["## Regression of the gap on the context (dB per sd; pedal: down − up)", "", "| model | " + " | ".join(nm) + " | R2 |",
          "|---|" + "---|" * (len(nm) + 1)]
    for s in labels:
        y = (bright[s] - bright["recording"])[m]
        Zs = Z / sd
        Zs[:, 1] = Z[:, 1]
        c = np.linalg.lstsq(Zs, y, rcond=None)[0]
        r2 = 1 - np.var(y - Zs @ c) / np.var(y)
        L.append(f"| {s} | " + " | ".join(f"{v:+.2f}" for v in c[1:]) + f" | {r2:.2f} |")
    with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
