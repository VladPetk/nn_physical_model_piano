"""Piano roll and note list of a listening excerpt (``scripts/ab_render.py`` output), to point at single notes.

    python scripts/excerpt_roll.py samples/phase5/listen_long --excerpt 0

Writes ``<k>_roll.png`` (the notes, numbered, over the spectrograms of every clip, on the clip's own 0-20 s time axis)
and ``<k>_notes.md`` (each note: number, time in the clip and in ``<k>_ab.wav``, name, velocity, length). Notes struck
before the clip that still sound (their key or the pedal down) are listed with negative times.
"""

import argparse
import json
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pianonn.config import PianoConfig  # noqa: E402
from pianonn.data import MaestroSegments  # noqa: E402

NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def name(p):
    return f"{NAMES[p % 12]}{p // 12 - 1}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--excerpt", type=int, default=0)
    ap.add_argument("--data", default="data/maestro24k")
    ap.add_argument("--seed", type=int, default=12, help="ab_render's --seed")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(os.path.join(args.folder, "manifest.json")) as f:
        man = json.load(f)
    ex = man["excerpts"][args.excerpt]
    sr, secs, order = man["sample_rate"], man["seconds"], man["order"]
    # the same window ab_render drew: its dataset, same seed and length; with --pick-from the index differs, so find
    # it by piece and start
    cfg = PianoConfig()
    ds = MaestroSegments(args.data, man["split"], cfg, secs - 1.0, 1.0, 12.0, length=64, deterministic=True,
                         years=[2018], seed=args.seed)
    item = next(it for it in (ds[i] for i in range(64))
                if ds.pieces[int(it["piece"])]["id"] == ex["piece"] and abs(int(it["start"]) / sr - ex["start_s"]) < 0.06)
    on, off = item["onset"].numpy(), item["offset"].numpy()
    pitch, vel = item["pitch"].numpy(), item["velocity"].numpy()
    H = int(item["hist_frames"])
    sus = item["sustain"].numpy()[H:]
    ts = np.arange(len(sus)) * cfg.hop / sr
    # a note still sounds at the clip start if its key is held or the pedal was down since its release
    pedal_at = lambda t: sus[min(len(sus) - 1, max(0, int(t * sr / cfg.hop)))] >= 0.5  # noqa: E731
    keep = (on >= 0) | (off > 0) | ((on > -4) & pedal_at(0.0))
    sel = np.where(keep & (on < secs))[0]
    sel = sel[np.lexsort((pitch[sel], on[sel]))]

    clips = [sf.read(os.path.join(args.folder, f"{args.excerpt}_{lab}.{man['format']}"), dtype="float32")[0].mean(1)
             for lab in order]
    fig, axes = plt.subplots(1 + len(clips), 1, figsize=(22, 6 + 2.6 * len(clips)), sharex=True,
                             gridspec_kw={"height_ratios": [3] + [1] * len(clips)})
    ax = axes[0]
    ax.fill_between(ts, 20, 109, where=sus >= 0.5, color="0.9", step="post", label="sustain pedal down")
    cmap = plt.get_cmap("viridis")
    lo, hi = pitch[sel].min() - 2, pitch[sel].max() + 2
    rows = []
    for k, i in enumerate(sel, 1):
        a, b = max(on[i], -0.3), min(off[i], secs)
        ax.add_patch(plt.Rectangle((a, pitch[i] - 0.4), b - a, 0.8, color=cmap(vel[i] / 127), ec="k", lw=0.4))
        ax.text(a + 0.02, pitch[i] + 0.45, f"{k}", fontsize=7, va="bottom")
        rows.append((k, i))
    yt = [p for p in range(lo, hi + 1) if p % 12 in (0, 4, 7)]
    ax.set_yticks(yt, [name(p) for p in yt])
    ax.set_ylim(lo, hi)
    ax.grid(axis="x", alpha=0.4)
    ax.set_title(f"excerpt {args.excerpt}: {ex['piece']} at {ex['start_s']} s - number = row in "
                 f"{args.excerpt}_notes.md, colour = velocity, grey = pedal down")
    ax.legend(loc="upper right")
    for axc, c, lab in zip(axes[1:], clips, order):
        axc.specgram(c, NFFT=2048, Fs=sr, noverlap=1792, cmap="magma", vmin=-130, vmax=-30)
        axc.set_yscale("symlog", linthresh=500)
        axc.set_ylim(50, 8000)
        axc.set_ylabel(f"{lab}\nHz")
    off_ab = {lab: j * (secs + 0.5) for j, lab in enumerate(order)}
    axes[-1].set_xlabel("time in the clip (s); in " + f"{args.excerpt}_ab.wav add "
                        + ", ".join(f"{v:g} s for {lab}" for lab, v in off_ab.items()))
    axes[-1].set_xlim(-0.3, secs)
    axes[-1].set_xticks(np.arange(0, secs + 0.01, 1.0))
    fig.tight_layout()
    fig.savefig(os.path.join(args.folder, f"{args.excerpt}_roll.png"), dpi=90)

    L = [f"# Excerpt {args.excerpt}: the notes", "",
         f"`{ex['piece']}` from {ex['start_s']} s. Times in the clip; in `{args.excerpt}_ab.wav` add "
         + ", ".join(f"{v:g} s for {lab}" for lab, v in off_ab.items())
         + ". Negative times: struck before the clip, still sounding. The numbers match `"
         + f"{args.excerpt}_roll.png`.", "",
         "| # | onset (s) | " + " | ".join(f"in ab: {lab}" for lab in order) + " | note | MIDI | velocity | key held (s) | pedal at onset |",
         "|---|---|" + "---|" * len(order) + "---|---|---|---|---|"]
    for k, i in rows:
        L.append(f"| {k} | {on[i]:.2f} | " + " | ".join(f"{on[i] + v:.2f}" if on[i] >= 0 else "" for v in off_ab.values())
                 + f" | {name(int(pitch[i]))} | {int(pitch[i])} | {int(vel[i])} | {off[i] - on[i]:.2f} | "
                 + ("down" if pedal_at(on[i]) else "up") + " |")
    with open(os.path.join(args.folder, f"{args.excerpt}_notes.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"{len(rows)} notes")


if __name__ == "__main__":
    main()
