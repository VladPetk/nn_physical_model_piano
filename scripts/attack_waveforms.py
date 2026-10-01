"""Attacks of bass and tenor notes as waveforms: when the strings' first pulse arrives after the strike.

With the strike point x0 measured from the agraffe, the transverse pulse reaches the bridge (1 - x0) T/2 after the
strike; the round-1/2 models put it at x0 T/2 (their comb is the agraffe end's force, config ``bridge_end_comb``).
The partial levels are the same either way, so no magnitude measure or loss sees it. Per note, on the two channels'
summed power:
- the first arrival t_a: the 1.5-8 kHz envelope (0.5 ms) crossing 10 dB over its level 80-30 ms before the MIDI
  onset, within -20..+40 ms of it (knock, precursor or string, whichever comes first); NaN (a failure, reported)
  unless it peaks 20 dB over that level within 20 ms;
- the string delay: from t_a to where the 150-2000 Hz envelope (0.5 ms) first reaches half its maximum over
  [t_a, t_a + T], as a fraction of the period T. The agraffe-end comb puts it near x0/2 ~ 0.06, the bridge end near
  (1 - x0)/2 ~ 0.44.
Validation: one model rendered both ways must read the two. A projection of the envelope on the period (its phase)
was tried first and dropped: the models' body filter puts its 150-2000 Hz energy ~30 ms later than its 1.5-8 kHz
energy (the narrow modes of docs/tone_measures.md 11.3), which turns that phase by more than a period in the bass.
The table of that delay is kept in the report.

    python scripts/attack_waveforms.py data/maestro24k --out runs/phase3/step0/attack_waveforms \\
        --model control=runs/round2/b_control/best.pt:physics \\
        --model bridge=runs/round2/b_control/best.pt:physics:bridge_end_comb=1
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
from pianonn.dsp import bounded  # noqa: E402
from pianonn.render import load_variant  # noqa: E402

PRE, POST = 0.2, 0.3
HI, MID = (1500.0, 8000.0), (150.0, 2000.0)


def first_arrival(x, sr, t_midi):
    e = M.band_envelope(x, sr, *HI, smooth=0.0005)
    t = np.arange(len(e)) / sr
    bg = np.median(e[(t >= t_midi - 0.08) & (t <= t_midi - 0.03)]) + 1e-20
    win = np.nonzero((t >= t_midi - 0.02) & (t <= t_midi + 0.04))[0]
    above = win[e[win] > 10 * bg]
    if not len(above):
        return float("nan")
    k = above[0]
    if e[k: k + int(0.02 * sr)].max() < 100 * bg:
        return float("nan")
    return k / sr


def string_delay(x, sr, t_a, f0):
    """Fraction of the period from ``t_a`` to where the 150-2000 Hz envelope first reaches half its maximum over
    [t_a, t_a + T]."""
    T = 1.0 / f0
    e = M.band_envelope(x, sr, *MID, smooth=0.0005)
    seg = e[int(round(t_a * sr)): int(round((t_a + T) * sr))]
    if len(seg) < 3:
        return float("nan")
    return float(np.argmax(seg >= 0.5 * seg.max()) / sr / T)


def band_delay(h, sr, band):
    """Energy centroid (ms) of the band-passed impulse response ``h[T]``."""
    X = np.fft.rfft(h, 4 * len(h))
    f = np.fft.rfftfreq(4 * len(h), 1 / sr)
    y = np.fft.irfft(X * M.band_mask(f, *band), 4 * len(h))[: len(h)] ** 2
    return float((np.arange(len(y)) * y).sum() / (y.sum() + 1e-30) / sr * 1000)


def svg_waves(rows, path, sr, width=560, row_h=150):
    """Channel-0 waveforms from t_a - 5 ms to t_a + T + 3 ms, each trace scaled to its own peak there; t_a (solid),
    x0 T/2 (dotted) and (1 - x0) T/2 (dashed) after it."""
    colours = {"recording": "#222222"}
    palette = ["#1f6fb4", "#d9480f", "#2b8a3e", "#862e9c"]
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{row_h * len(rows) + 30}" '
           f'font-family="sans-serif" font-size="11"><rect width="100%" height="100%" fill="white"/>']
    for r_i, r in enumerate(rows):
        y0 = 20 + r_i * row_h
        span = 1 / r["f0"] + 0.008
        out.append(f'<text x="5" y="{y0 + 12}">MIDI {r["pitch"]} v{r["velocity"]} pedal {r["pedal_bin"]} '
                   f'(T {1000 / r["f0"]:.1f} ms)</text>')
        labels = list(r["waves"])
        h = (row_h - 20) / len(labels)
        for j, lab in enumerate(labels):
            w, t_a = r["waves"][lab]
            col = colours.get(lab, palette[(j - 1) % len(palette)])
            yc = y0 + 20 + (j + 0.5) * h
            if not np.isfinite(t_a):
                out.append(f'<text x="70" y="{yc:.1f}" fill="{col}">{lab}: no arrival</text>')
                continue
            a, b = int((t_a - 0.005) * sr), int((t_a - 0.005 + span) * sr)
            seg = w[max(0, a): b]
            seg = seg / (np.abs(seg).max() + 1e-30)
            xs = 70 + np.arange(len(seg)) / (span * sr) * (width - 80)
            pts = " ".join(f"{x:.1f},{yc - 0.45 * h * v:.1f}" for x, v in zip(xs, seg))
            out.append(f'<polyline fill="none" stroke="{col}" stroke-width="0.8" points="{pts}"/>')
            out.append(f'<text x="5" y="{yc + 4:.1f}" fill="{col}">{lab}</text>')
        for frac, dash in ((0.0, ""), (r["x0"] / 2, "2,2"), ((1 - r["x0"]) / 2, "6,3")):
            x = 70 + (0.005 + frac / r["f0"]) / span * (width - 80)
            out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{y0 + 16}" y2="{y0 + row_h - 4}" stroke="#999" '
                       f'stroke-dasharray="{dash}"/>')
    out.append("</svg>")
    with open(path, "w") as f:
        f.write("\n".join(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data")
    ap.add_argument("--years", type=int, nargs="*", default=[2018])
    ap.add_argument("--bench")
    ap.add_argument("--model", action="append", required=True, help="label=checkpoint:physics|residual[:options]")
    ap.add_argument("--max-pitch", type=int, default=59)
    ap.add_argument("--min-velocity", type=int, default=64)
    ap.add_argument("--per-register", type=int, default=80)
    ap.add_argument("--plot-per-register", type=int, default=3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.6)
    os.makedirs(args.out, exist_ok=True)
    with open(args.bench or f"runs/measurements/bench_{'_'.join(map(str, args.years))}.json") as f:
        bench = json.load(f)
    notes, count = [], {}
    for n in bench["notes"]:  # both groups: nothing is fitted here
        reg = n["register"]
        if n["pitch"] <= args.max_pitch and n["velocity"] >= args.min_velocity and count.get(reg, 0) < args.per_register:
            count[reg] = count.get(reg, 0) + 1
            notes.append(n)
    models = [load_variant(s, device=dev) for s in args.model]
    cfg = models[0][1].cfg
    sr = cfg.sample_rate
    cond = year_to_condition(args.years[0])
    table = M.partial_table(models[0][1], cond, dev)
    ph = models[0][1].physics
    with torch.no_grad():
        x0 = (ph.prior_strike * torch.exp(bounded(ph.raw_strike, 0.5))).cpu().numpy()
    lines = ["# Attack waveforms: when the strings' first pulse arrives after the strike\n",
             f"{len(notes)} bench notes (both groups), MIDI 21-{args.max_pitch}, velocity >= {args.min_velocity}, "
             f"{PRE + POST:.1f} s clips in context. String delay: fraction of the period from the first 1.5-8 kHz "
             "arrival to where the 150-2000 Hz envelope first reaches half its maximum over the next period.\n",
             "## Body filter delay (energy centroid of the body FIR alone, ms), per model and channel\n",
             "| model | ch | 150-2000 Hz | 1.5-8 kHz | difference |", "|---|---|---|---|---|"]
    for label, m, _ in models:
        ir = m.room.body[cond].detach().float().cpu().numpy()
        for ch in range(ir.shape[0]):
            dm, dh = band_delay(ir[ch], sr, MID), band_delay(ir[ch], sr, HI)
            lines.append(f"| {label} | {ch} | {dm:.2f} | {dh:.2f} | {dm - dh:+.2f} |")
    rows = []
    for s in range(0, len(notes), 16):
        part = notes[s: s + 16]
        clips = M.render_clips(args.data, part, models, cfg, dev, pre=PRE, post=POST)
        for i, n in enumerate(part):
            f0 = float(table[n["pitch"] - 21, 0])
            r = {k: n[k] for k in ("piece", "pitch", "velocity", "register", "pedal_bin")}
            r.update(f0=f0, x0=float(x0[n["pitch"] - 21]), res={}, waves={})
            for src, xs in clips.items():
                x = xs[i]
                t_a = first_arrival(x, sr, PRE)
                d = string_delay(x, sr, t_a, f0) if np.isfinite(t_a) else float("nan")
                r["res"][src] = {"t_a_ms": (t_a - PRE) * 1000 if np.isfinite(t_a) else float("nan"), "delay": d}
                r["waves"][src] = (x[:, 0], t_a)
            rows.append(r)
        print(f"notes {min(s + 16, len(notes))}/{len(notes)}", flush=True)
    srcs = ["recording"] + [label for label, _, _ in models]

    def cell(v):
        v = np.asarray([u for u in v if np.isfinite(u)])
        return f"{np.median(v):.2f} [{np.percentile(v, 25):.2f}, {np.percentile(v, 75):.2f}], {len(v)}" if len(v) else "-"

    lines += ["\n## String delay by register (fraction of the period: median [IQR], n)\n",
              "| register | notes | x0/2 (agraffe end) | (1-x0)/2 (bridge end) | arrival failures (rec) | "
              + " | ".join(srcs) + " | recording: share nearer the bridge end |",
              "|---|---|---|---|---|" + "---|" * len(srcs) + "---|"]
    for reg in sorted({r["register"] for r in rows}):
        rs = [r for r in rows if r["register"] == reg]
        xs = np.median([r["x0"] for r in rs])
        fails = sum(not np.isfinite(r["res"]["recording"]["t_a_ms"]) for r in rs)
        rec = np.asarray([r["res"]["recording"]["delay"] for r in rs])
        rec = rec[np.isfinite(rec)]
        near = f"{np.mean(np.abs(rec - (1 - xs) / 2) < np.abs(rec - xs / 2)):.2f} of {len(rec)}" if len(rec) else "-"
        lines.append(f"| {reg} | {len(rs)} | {xs / 2:.2f} | {(1 - xs) / 2:.2f} | {fails} | "
                     + " | ".join(cell([r["res"][s]["delay"] for r in rs]) for s in srcs) + f" | {near} |")
    lines.append("\n## First arrival re the MIDI onset\n")
    for s in srcs:
        ta = [r["res"][s]["t_a_ms"] for r in rows]
        lines.append(f"- {s}: median {np.nanmedian(ta):+.1f} ms, IQR {np.nanpercentile(ta, 75) - np.nanpercentile(ta, 25):.1f} ms, "
                     f"failures {sum(not np.isfinite(v) for v in ta)} of {len(ta)}")
    text = "\n".join(lines) + "\n"
    with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8") as f:
        f.write(text)
    with open(os.path.join(args.out, "rows.json"), "w") as f:
        json.dump([{k: v for k, v in r.items() if k != "waves"} for r in rows], f)
    picks = []
    for reg in sorted({r["register"] for r in rows}):
        rs = sorted((r for r in rows if r["register"] == reg and np.isfinite(r["res"]["recording"]["t_a_ms"])),
                    key=lambda r: r["pitch"])
        picks += [rs[int(i)] for i in np.linspace(0, len(rs) - 1, min(args.plot_per_register, len(rs)))] if rs else []
    svg_waves(picks, os.path.join(args.out, "waves.svg"), sr)
    print(text)


if __name__ == "__main__":
    main()
