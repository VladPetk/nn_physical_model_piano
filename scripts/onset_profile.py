"""The first tens of ms of a note, recordings against models (docs/tone_measures.md 16, item 1).

    python scripts/onset_profile.py data/maestro24k --out runs/phase5/onset \\
        --model phase4=runs/phase4/train_run/train/last.pt:physics \\
        --model nojitter=runs/phase4/train_run/train/last.pt:physics:strike_onset_ms=0

The bench's isolated notes (``runs/measurements/bench_2018.json``, both groups by default), each side aligned at its
own N0. Per register: ``measures.onset_profile`` per octave band (the level in 0-5, 5-10, 10-20, 20-40 ms re the
band's own 50-100 ms, the arrival and rise re N0, the peak), the spectral balance at 50-100 ms, the brightness over
the windows; the bands' mean envelopes over the notes (beating averages out across notes, so 1 ms smoothing) and
waveform examples as figures. With a model labelled ``nojitter`` (the per-strike onset jitter off), the per-strike
spread of N0 re MIDI against the recordings': the jitter that matches is the root of the difference of the squares
(review 5, 3.1).
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
from pianonn.config import year_to_condition  # noqa: E402
from pianonn.render import load_variant  # noqa: E402
from strike_spread import components, residuals  # noqa: E402

BANDS = M.KNOCK_BANDS
WIN = M.ONSET_WINDOWS
REGS = [r for r, _, _ in M.REGISTERS]


def measure(x, sr, f0):
    t_on = M.onset(x, sr, M.PRE, f0)
    if not np.isfinite(t_on):
        return None
    pr = M.onset_profile(x, sr, t_on, f0)
    fine = M.onset_profile(x, sr, t_on, f0, smooth=0.001)
    mono = x.mean(1)
    a = int(round((t_on - 0.005) * sr))
    wave = mono[a: a + int(0.045 * sr)]
    ref = M._segment(mono, sr, t_on + 0.05, t_on + 0.1)
    return {"onset_ms": 1000 * (t_on - M.PRE), "pr": pr, "P_fine": fine["P"], "wave": wave / (np.sqrt(np.mean(ref ** 2)) + 1e-12)}


def centroid(levels_db):
    """Brightness: the power-weighted mean of log2(band centre) over 250 Hz-8 kHz, in octaves re 1 kHz."""
    c = np.log2(np.array(BANDS) / 1000.0)
    m = (np.array(BANDS) >= 250) & np.isfinite(levels_db)
    if m.sum() < 4:
        return np.nan
    w = 10 ** (levels_db[m] / 10)
    return float((c[m] * w).sum() / w.sum())


def med(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return (float(np.median(v)), len(v)) if len(v) else (np.nan, 0)


def fmt(v, n=None, p=1):
    return "" if not np.isfinite(v) else (f"{v:+.{p}f}" if n is None else f"{v:+.{p}f} ({n})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data")
    ap.add_argument("--model", action="append", required=True, help="label=checkpoint:physics[:options]")
    ap.add_argument("--bench", default="runs/measurements/bench_2018.json")
    ap.add_argument("--group", default="all", choices=("all", "eval", "calib"))
    ap.add_argument("--registers", default="", help="comma list, e.g. R3,R4,R5 (default every register)")
    ap.add_argument("--max-notes", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.6)

    with open(args.bench) as f:
        bench = json.load(f)
    notes = [n for n in bench["notes"] if (args.group == "all" or n["group"] == args.group)
             and (not args.registers or n["register"] in args.registers.split(","))]
    if args.max_notes:
        notes = notes[:: max(1, len(notes) // args.max_notes)][: args.max_notes]
    models = [load_variant(s, device=dev) for s in args.model]
    labels = [m[0] for m in models]
    cfg, sr = models[0][1].cfg, models[0][1].cfg.sample_rate
    table = M.partial_table(models[0][1], year_to_condition(bench["years"][0]), dev)
    srcs = ["recording"] + labels
    res = {s: [] for s in srcs}
    for s0 in range(0, len(notes), 48):
        part = notes[s0: s0 + 48]
        clips = M.render_clips(args.data, part, models, cfg, dev, batch=8)
        for i, n in enumerate(part):
            f0 = float(table[n["pitch"] - 21][0])
            for s in srcs:
                res[s].append(measure(clips[s][i], sr, f0))
        print(f"notes {min(s0 + 48, len(notes))}/{len(notes)}", flush=True)
    reg = np.array([n["register"] for n in notes])
    proto = next(r for r in res["recording"] if r)

    def feat(s, fn):
        blank = np.full(np.shape(fn(proto)), np.nan)
        return np.array([fn(r) if r is not None else blank for r in res[s]], float)

    np.savez_compressed(os.path.join(args.out, "profile.npz"), bands=np.array(BANDS), windows=np.array(WIN),
                        **{f"{s}_{k}": feat(s, lambda r, k=k: r["pr"][k]) for s in srcs
                           for k in ("peak", "arrival", "rise", "win", "ref_db")},
                        **{f"{s}_onset_ms": feat(s, lambda r: r["onset_ms"]) for s in srcs})
    with open(os.path.join(args.out, "events.json"), "w") as f:
        json.dump(notes, f)

    L = [f"# The first tens of ms: recordings against {', '.join(f'`{a}`' for a in args.model)}", "",
         f"Notes: the bench's isolated notes ({args.group}; {len(notes)}), each side aligned at its own N0. N0 found: "
         + ", ".join(f"{s} {sum(r is not None for r in res[s])}" for s in srcs)
         + ". Medians over notes, in brackets the notes counted; Δ = the median of model − recording over the notes "
           "where both count. Levels in dB re the band's own level at 50-100 ms after N0; times in ms re N0. Band "
           "envelopes are averaged over one period of f0 where a band holds several partials (they beat at their "
           "spacing), so rise times shorter than ~1/f0 read as ~1/f0 on both sides.", ""]
    regs = [r for r in REGS if (reg == r).sum() >= 5]
    L += ["Notes per register: " + ", ".join(f"{r} {(reg == r).sum()}" for r in regs), ""]

    def table_of(title, fn, p=1, unit=""):
        out = [f"### {title}", "", "| register | source | " + " | ".join(str(c) for c in BANDS) + " |",
               "|---|---|" + "---|" * len(BANDS)]
        vals = {s: feat(s, fn) for s in srcs}
        for r in regs:
            m = reg == r
            for s in srcs:
                row = [fmt(*med(vals[s][m, j]), p=p) for j in range(len(BANDS))]
                out.append(f"| {r} | {s} | " + " | ".join(row) + " |")
            for s in labels:
                d = vals[s][m] - vals["recording"][m]
                out.append(f"| {r} | Δ {s} | " + " | ".join((lambda v: f"**{v}**" if v else "")(fmt(*med(d[:, j]), p=p)) for j in range(len(BANDS))) + " |")
        return out + [""]

    L += ["## Each band's level in the first 40 ms, re its own 50-100 ms (dB)", "",
          "A band that arrives late, or rises slowly, reads low in the first windows; a transient on top of the tone "
          "(the knock, a precursor) reads high.", ""]
    for w, (a, b) in enumerate(WIN):
        L += table_of(f"{1000 * a:.0f}-{1000 * b:.0f} ms", lambda r, w=w: r["pr"]["win"][w])
    L += ["## Arrival and rise", ""]
    L += table_of("arrival: 10 % of the band's peak (ms re N0)", lambda r: 1000 * r["pr"]["arrival"])
    L += table_of("rise: 10 → 90 % of the peak (ms)", lambda r: 1000 * r["pr"]["rise"])
    L += table_of("peak over -5..60 ms (dB re 50-100 ms)", lambda r: r["pr"]["peak"])
    L += ["## Spectral balance once settled", ""]
    L += table_of("each band at 50-100 ms re the note's 0.2-8 kHz (dB)", lambda r: r["pr"]["ref_db"])

    # brightness per window: the bands' absolute levels (window re band ref + band ref re total)
    L += ["## Brightness over the onset", "", "Power-weighted mean of log2(band centre / 1 kHz) over 250 Hz-8 kHz, "
          "octaves; the last column at 50-100 ms.", "",
          "| register | source | " + " | ".join(f"{1000 * a:.0f}-{1000 * b:.0f} ms" for a, b in WIN) + " | 50-100 ms |",
          "|---|---|" + "---|" * (len(WIN) + 1)]
    bright = {s: feat(s, lambda r: np.array([centroid(r["pr"]["win"][w] + r["pr"]["ref_db"]) for w in range(len(WIN))]
                                            + [centroid(r["pr"]["ref_db"])])) for s in srcs}
    for r in regs:
        m = reg == r
        for s in srcs:
            L.append(f"| {r} | {s} | " + " | ".join(fmt(*med(bright[s][m, w]), p=2) for w in range(len(WIN) + 1)) + " |")
        for s in labels:
            d = bright[s][m] - bright["recording"][m]
            L.append(f"| {r} | Δ {s} | " + " | ".join((lambda v: f"**{v}**" if v else "")(fmt(*med(d[:, w]), p=2)) for w in range(len(WIN) + 1)) + " |")
    L.append("")

    # onset timing: per-strike spread of N0 re MIDI (strike_spread's residuals and key pairs)
    pitch = np.array([n["pitch"] for n in notes], float)
    vel = np.array([n["velocity"] for n in notes], float)
    piece = np.array([n["piece"] for n in notes])
    on = {s: feat(s, lambda r: r["onset_ms"]) for s in srcs}
    L += ["## Onset timing: N0 re the MIDI onset", "",
          "Per register: the median, and the per-strike spread (sd from pairs of notes of the same key after a linear fit "
          "on pitch and velocity and each piece's median, as `scripts/strike_spread.py`), ms.", "",
          "| register | " + " | ".join(f"{s} median | {s} per-strike sd" for s in srcs) + " |", "|---|" + "---|---|" * len(srcs)]
    sds = {}
    for r in regs:
        row = []
        for s in srcs:
            m = (reg == r) & np.isfinite(on[s])
            if m.sum() < 8:
                row += ["", ""]
                continue
            rr = residuals(on[s][m], pitch[m], vel[m], piece[m])
            _, sd, _ = components(rr, pitch[m])
            sds[(r, s)] = sd
            row += [f"{np.median(on[s][m]):+.1f} ({m.sum()})", f"{sd:.2f}"]
        L.append(f"| {r} | " + " | ".join(row) + " |")
    if "nojitter" in labels:
        L += ["", "The jitter that matches (sqrt(recording² − nojitter²), ms): " + ", ".join(
            f"{r} {np.sqrt(max(sds[(r, 'recording')] ** 2 - sds[(r, 'nojitter')] ** 2, 0)):.2f}"
            for r in regs if (r, "recording") in sds and (r, "nojitter") in sds)]
    L.append("")
    with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n".join(L))
    plots(args.out, res, srcs, reg, regs, sr)


def plots(out, res, srcs, reg, regs, sr):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib missing: no figures")
        return
    t = next(r for r in res["recording"] if r)["pr"]["t"] * 1000
    colors = {"recording": "k"} | {s: c for s, c in zip(srcs[1:], ("tab:red", "tab:blue", "tab:green", "tab:orange", "tab:purple", "tab:brown", "tab:pink"))}
    fig, axes = plt.subplots(len(regs), len(BANDS), figsize=(2.2 * len(BANDS), 1.8 * len(regs)), sharex=True, squeeze=False)
    for i, r in enumerate(regs):
        for j, c in enumerate(BANDS):
            ax = axes[i, j]
            for s in srcs:
                P = [q["P_fine"][:, j] for q, rg in zip(res[s], reg) if q is not None and rg == r and np.isfinite(q["P_fine"][:, j]).all()]
                if len(P) >= 3:
                    ax.plot(t, 10 * np.log10(np.clip(np.mean(P, 0), 1e-3, None)), color=colors[s], lw=1, label=f"{s} ({len(P)})")
            ax.axvline(0, color="0.7", lw=0.5)
            ax.set_ylim(-25, 15)
            ax.grid(alpha=0.3)
            if i == 0:
                ax.set_title(f"{c} Hz", fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{r}\ndB re 50-100 ms", fontsize=8)
            if i == len(regs) - 1:
                ax.set_xlabel("ms re N0", fontsize=8)
            ax.tick_params(labelsize=7)
        axes[i, -1].legend(fontsize=6)
    fig.suptitle("Mean band envelope over the notes (power, re each band's 50-100 ms), aligned at N0")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "envelopes.png"), dpi=100)
    plt.close(fig)

    fig, axes = plt.subplots(len(regs), 3, figsize=(13, 1.9 * len(regs)), squeeze=False)
    tw = np.arange(int(0.045 * sr)) / sr * 1000 - 5
    for i, r in enumerate(regs):
        idx = [k for k, rg in enumerate(reg) if rg == r and all(res[s][k] is not None for s in srcs)]
        for j, k in enumerate(idx[:: max(1, len(idx) // 3)][:3]):
            ax = axes[i, j]
            for s in srcs:
                w = res[s][k]["wave"]
                ax.plot(tw[: len(w)], w, color=colors[s], lw=0.6, alpha=0.8, label=s)
            ax.axvline(0, color="0.7", lw=0.5)
            ax.set_title(f"{r}, note {k}", fontsize=8)
            ax.tick_params(labelsize=7)
            if j == 0:
                ax.legend(fontsize=6)
    fig.suptitle("Waveforms, mono, re each side's 50-100 ms rms, from 5 ms before N0")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "waveforms.png"), dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    main()
