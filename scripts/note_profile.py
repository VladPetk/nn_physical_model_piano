"""Profile isolated notes, recording against model: each partial over the first second (docs/tone_measures.md 15).

    python scripts/note_profile.py data/maestro24k --model phase4=runs/phase4/train_run/train/last.pt:physics \\
        --out runs/phase4/r3_profile --listen samples/r3_notes

Notes: every note of the year in ``--pitch`` and ``--velocity`` with no other onset ``--before`` s before and
``--after`` s after it (any split: this measures the model's sound, not a fit). Each is rendered by the model in its
context (``measures.render_clips``: 12 s of MIDI before, the same pedals) with the per-strike variation as configured.
On both sides, at each side's own onset (N0): ``measures.partial_profile`` on the note's partials (found near the
model's partial table on each side separately) and ``measures.non_tonal``. A note counts as sounding to 1 s when its
key is held or the pedal is down; released notes count up to their release.

Writes ``report.md`` (medians over notes, recording, model and paired model - recording, per partial number),
``tracks.png`` (median partial tracks), ``features.png`` (per-partial features), ``examples.png`` (three notes'
tracks), ``profile.npz`` and, with ``--listen``, A/B files: each note's recording and model (the model's level
matched to the recording's over the first 0.5 s), and ``ab.wav`` with every pair in turn.
"""

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from pianonn import measures as M  # noqa: E402
from pianonn.config import year_to_condition  # noqa: E402
from pianonn.losses import ONSET_DELAY_MS  # noqa: E402
from pianonn.render import load_variant  # noqa: E402

PRE, POST = 0.5, 1.15  # s of clip around the MIDI onset
N_PARTIALS, F_MAX = 12, 5000.0
TIMES = (0.05, 0.3, 0.9)  # s re N0: the spectral envelope


def mine(root, year, pitch, vel, before, after):
    """Isolated notes: ``[{piece, pitch, velocity, onset, offset, t_end, state}]`` (``t_end``: s re the onset to
    which the note sounds undamped, at most ``after``)."""
    with open(os.path.join(root, "index.json")) as f:
        pieces = [p for p in json.load(f) if p["year"] == year]
    out = []
    for p in pieces:
        with np.load(os.path.join(root, p["midi"])) as z:
            n, st, sv = z["notes"], z["sustain_t"], z["sustain_v"]
        on = np.sort(n[:, 1])
        for pi, t, off, v in n:
            if not (pitch[0] <= pi <= pitch[1] and vel[0] <= v <= vel[1]):
                continue
            prev, nxt = on[on < t - 1e-4], on[on > t + 1e-4]
            if (np.abs(on - t) < 1e-4).sum() != 1 or (len(prev) and t - prev[-1] < before) or (len(nxt) and nxt[0] - t < after):
                continue
            k = np.searchsorted(st, t, side="right") - 1
            ped = np.array([sv[k] if k >= 0 else 0.0] + list(sv[(st > t) & (st < t + after)]))
            held = off - t
            if ped.min() >= M.PEDAL[2][1]:
                state = "pedal down"
            elif held >= after - 0.05:
                state = "held"
            else:
                state = "released"
            t_end = after - 0.05 if state != "released" else max(0.2, min(after - 0.05, held))
            out.append({"piece": p["id"], "split": p["split"], "year": year, "pitch": int(pi), "velocity": int(v),
                        "onset": float(t), "offset": float(off), "t_end": float(t_end), "state": state,
                        "pedal_cc": float(ped.max())})
    return out


def refine(x, sr, t_on, table, f0, t_end):
    """The note's own partial frequencies: the spectral peak within min(1.5 %, f0 / 4) of each table frequency, over
    50 ms to ``min(0.6, t_end)`` s after the onset."""
    seg = M._segment(x, sr, t_on + 0.05, t_on + min(0.6, t_end))
    n_fft = 1 << int(math.ceil(math.log2(len(seg) * 8)))
    P, hz = M._power(seg, n_fft), np.fft.rfftfreq(n_fft, 1 / sr)
    out = []
    for f in table:
        tol = min(0.015 * f, 0.25 * f0)
        band = (hz >= f - tol) & (hz <= f + tol)
        out.append(float(hz[band][np.argmax(P[band])]) if band.any() else f)
    return np.array(out)


def measure(x, sr, t_ref, table, f0, t_end):
    t_on = M.onset(x, sr, t_ref, f0)
    found = np.isfinite(t_on)
    t_on = t_on if found else t_ref
    freqs = refine(x, sr, t_on, table, f0, t_end)
    pr = M.partial_profile(x, sr, t_on, freqs, t_end)
    nt = M.non_tonal(x, sr, t_on, M_partials(freqs, f0), windows=((-0.003, 0.04), (0.1, 0.4), (0.5, 0.95)))
    env = {}
    for tau in TIMES:
        sel = np.abs(pr["t"] - tau) <= 0.02
        if tau <= t_end and sel.any():
            lv = 10 * np.log10(np.mean(10 ** (pr["L"][sel] / 10), 0))
            env[tau] = lv - 10 * np.log10(np.sum(10 ** (lv / 10)))
        else:
            env[tau] = np.full(len(freqs), np.nan)
    sel = (pr["t"] >= 0.0) & (pr["t"] <= 0.5)
    total = 10 * np.log10(np.sum(10 ** (pr["L"][sel] / 10)) + 1e-30)
    return {"t_on": t_on, "found": found, "freqs": freqs, "profile": pr, "non_tonal": nt, "env": env, "total": total}


def M_partials(freqs, f0, f_max=8000.0):
    """The partial comb for ``non_tonal``: the measured ones, continued with their last spacing up to ``f_max``."""
    fr = list(freqs)
    step = fr[-1] - fr[-2] if len(fr) > 1 else f0
    while fr[-1] + step < f_max:
        fr.append(fr[-1] + step)
        step *= 1.0 + 1e-3
    return np.array(fr)


def med(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return (float(np.median(a)), len(a)) if len(a) else (float("nan"), 0)


def fmt(v, n=None, p=1):
    return "" if not np.isfinite(v) else (f"{v:+.{p}f}" if n is None else f"{v:+.{p}f} ({n})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data")
    ap.add_argument("--model", required=True, help="label=checkpoint:physics[:options] (pianonn.render.load_variant)")
    ap.add_argument("--year", type=int, default=2018)
    ap.add_argument("--pitch", type=int, nargs=2, default=(47, 59))
    ap.add_argument("--velocity", type=int, nargs=2, default=(50, 127))
    ap.add_argument("--before", type=float, default=0.3)
    ap.add_argument("--after", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--listen", help="directory for the A/B listening files")
    ap.add_argument("--n-listen", type=int, default=12)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.6)

    events = mine(args.data, args.year, args.pitch, args.velocity, args.before, args.after)
    label, model, residual = load_variant(args.model, device=dev)
    cfg, sr = model.cfg, model.cfg.sample_rate
    print(f"{len(events)} notes; rendering with {label}")
    clips = M.render_clips(args.data, events, [(label, model, residual)], cfg, dev, pre=PRE, post=POST, batch=8)
    table = M.partial_table(model, year_to_condition(args.year), dev)
    a, b, c = ONSET_DELAY_MS

    res = {"recording": [], label: []}
    for i, ev in enumerate(events):
        tab = table[ev["pitch"] - 21]
        tab = tab[(tab > 0) & (tab < F_MAX)][:N_PARTIALS]
        f0 = float(tab[0])
        t_ref = PRE + 1e-3 * max(a + b * (ev["pitch"] - 60) + c * (ev["velocity"] - 64), 0.0)
        for src in res:
            res[src].append(measure(clips[src][i], sr, t_ref, tab, f0, ev["t_end"]))
    K = N_PARTIALS
    srcs = ["recording", label]

    def per_note(src, fn):
        return np.array([np.pad(np.asarray(fn(r), float), (0, max(0, K - len(np.atleast_1d(fn(r))))),
                                constant_values=np.nan)[:K] for r in res[src]])

    feats = {
        "peak time (ms)": lambda r: 1000 * r["profile"]["t_peak"],
        "early decay (dB/s, 50-350 ms)": lambda r: r["profile"]["early"],
        "late decay (dB/s, 0.5-1 s)": lambda r: r["profile"]["late"],
        "two-stage: early − late (dB/s)": lambda r: r["profile"]["early"] - r["profile"]["late"],
        "fluctuation (dB rms)": lambda r: r["profile"]["fluct"],
        "periodicity of the fluctuation (share at one rate)": lambda r: np.where(r["profile"]["fluct"] > 0.5, r["profile"]["periodic"], np.nan),
        **{f"level re all partials at {int(tau * 1000)} ms (dB)": (lambda r, tau=tau: r["env"][tau]) for tau in TIMES},
    }
    sounding = np.array([ev["state"] != "released" for ev in events])
    F = {name: {s: per_note(s, fn) for s in srcs} for name, fn in feats.items()}
    beat = {s: per_note(s, lambda r: np.where(r["profile"]["fluct"] > 0.5, r["profile"]["beat"], np.nan)) for s in srcs}

    # ---- report
    states = {s: sum(ev["state"] == s for ev in events) for s in ("held", "pedal down", "released")}
    splits = ", ".join(f"{s} {sum(e['split'] == s for e in events)}" for s in ("train", "validation", "test"))
    L = [f"# Isolated notes, recording against `{label}` ({args.year})", "",
         f"Notes: MIDI {args.pitch[0]}–{args.pitch[1]}, velocity {args.velocity[0]}–{args.velocity[1]}, no other onset "
         f"{args.before} s before or {args.after} s after: {len(events)} ({', '.join(f'{k} {v}' for k, v in states.items())}; "
         f"splits {splits}). "
         f"Model: `{args.model}`. N0 found: recording {sum(r['found'] for r in res['recording'])}, model "
         f"{sum(r['found'] for r in res[label])}. Medians over notes; in brackets the notes counted; paired = the median "
         "of model − recording over the notes where both count. Late decay, fluctuation and beat: notes sounding to 1 s "
         "(held or pedal down) only.", ""]
    tot = np.array([r["total"] for r in res[label]]) - np.array([r["total"] for r in res["recording"]])
    L += [f"Level (all partials, 0–0.5 s), model − recording: median {np.median(tot):+.1f} dB, IQR "
          f"{np.percentile(tot, 25):+.1f} … {np.percentile(tot, 75):+.1f} (the recordings' level differs by piece by ~2 dB, 12.5).", ""]
    for name in feats:
        L += [f"## {name}", "", "| partial | " + " | ".join(str(k + 1) for k in range(K)) + " |", "|---|" + "---|" * K]
        only = sounding if name.startswith(("late", "two-stage", "fluct", "period")) or "900 ms" in name else np.ones(len(events), bool)
        rec, mod = F[name]["recording"][only], F[name][label][only]
        L.append("| recording | " + " | ".join(fmt(*med(rec[:, k])) for k in range(K)) + " |")
        L.append(f"| {label} | " + " | ".join(fmt(*med(mod[:, k])) for k in range(K)) + " |")
        L.append("| paired | " + " | ".join(fmt(*med(mod[:, k] - rec[:, k])) for k in range(K)) + " |")
        L.append("")
    L += ["## Beat rate (Hz; partials whose fluctuation exceeds 0.5 dB)", "",
          "| partial | " + " | ".join(str(k + 1) for k in range(K)) + " |", "|---|" + "---|" * K]
    for s in srcs:
        L.append(f"| {s} | " + " | ".join(fmt(*med(beat[s][sounding][:, k])) for k in range(K)) + " |")
    share = {s: np.mean(np.isfinite(beat[s][sounding]), 0) for s in srcs}
    L += ["", "Share of partials beating (fluctuation > 0.5 dB): " + "; ".join(
        f"{s} " + ", ".join(f"{100 * v:.0f}" for v in share[s]) + " %" for s in srcs), ""]
    L += ["## Spread across notes (IQR over notes, per partial; recording / model)", "",
          "How much the notes differ from one another at equal register and similar velocity: medians alone do not show "
          "whether the model's notes vary as the piano's do.", "",
          "| feature | " + " | ".join(str(k + 1) for k in range(K)) + " |", "|---|" + "---|" * K]
    iqr = lambda v: float(np.subtract(*np.nanpercentile(v, [75, 25]))) if np.isfinite(v).sum() >= 5 else float("nan")
    for name in feats:
        only = sounding if name.startswith(("late", "two-stage", "fluct", "period")) or "900 ms" in name else np.ones(len(events), bool)
        cells = [f"{iqr(F[name]['recording'][only][:, k]):.1f} / {iqr(F[name][label][only][:, k]):.1f}" for k in range(K)]
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    L += ["", "## Energy away from the partials (dB re the note's energy in the window)", "",
          "| window | band | recording | model | paired |", "|---|---|---|---|---|"]
    for key in res["recording"][0]["non_tonal"]:
        r = np.array([x["non_tonal"][key] for x in res["recording"]])
        m = np.array([x["non_tonal"][key] for x in res[label]])
        if np.isfinite(r).sum() >= 5:
            (w0, w1), band = key
            L.append(f"| {w0 * 1000:.0f}–{w1 * 1000:.0f} ms | {band} | {fmt(*med(r))} | {fmt(*med(m))} | {fmt(*med(m - r))} |")
    with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.join(args.out, "events.json"), "w") as f:
        json.dump(events, f, indent=1)
    np.savez_compressed(os.path.join(args.out, "profile.npz"),
                        **{f"{s}|{n}": F[n][s] for n in feats for s in srcs}, sounding=sounding)
    print("\n".join(L))
    plots(args.out, res, srcs, events, sounding, F, feats)
    if args.listen:
        listen(args.listen, events, clips, res, srcs, sr, args.n_listen, args.model)


def plots(out, res, srcs, events, sounding, F, feats):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {srcs[0]: "black", srcs[1]: "tab:red"}
    grid = np.arange(-0.02, 0.96, 0.005)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True, sharey=True)
    for ax, k in zip(axes.flat, (0, 1, 2, 3, 5, 7, 9, 11)):
        for s in srcs:
            tr = []
            for r, snd in zip(res[s], sounding):
                pr = r["profile"]
                if not snd or k >= pr["L"].shape[1]:
                    continue
                ref = np.nanmax(pr["peak"])
                tr.append(np.interp(grid, pr["t"], pr["L"][:, k] - ref, left=np.nan, right=np.nan))
            tr = np.array(tr)
            q = np.nanpercentile(tr, [25, 50, 75], 0)
            ax.plot(grid, q[1], color=colors[s], label=s)
            ax.fill_between(grid, q[0], q[2], color=colors[s], alpha=0.15)
        ax.set_title(f"partial {k + 1}")
        ax.grid(alpha=0.3)
        ax.set_ylim(-70, 5)
    axes[0, 0].legend()
    for ax in axes[1]:
        ax.set_xlabel("s after the onset")
    for ax in axes[:, 0]:
        ax.set_ylabel("dB re the note's strongest partial")
    fig.suptitle(f"Partial levels over time: median and IQR over {int(sounding.sum())} notes sounding to 1 s")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "tracks.png"), dpi=110)
    plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    k = np.arange(1, F[next(iter(F))][srcs[0]].shape[1] + 1)
    for ax, name in zip(axes.flat, feats):
        only = sounding if name.startswith(("late", "two-stage", "fluct", "period")) or "900 ms" in name else np.ones(len(events), bool)
        for s in srcs:
            v = F[name][s][only]
            q = np.array([np.nanpercentile(v[:, j], [25, 50, 75]) if np.isfinite(v[:, j]).sum() else [np.nan] * 3 for j in range(len(k))])
            ax.plot(k, q[:, 1], "o-", color=colors[s], label=s)
            ax.fill_between(k, q[:, 0], q[:, 2], color=colors[s], alpha=0.15)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("partial")
        ax.grid(alpha=0.3)
    axes[0, 0].legend()
    for ax in axes.flat[len(feats):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "features.png"), dpi=110)
    plt.close(fig)

    idx = [i for i in np.argsort([-e["velocity"] for e in events]) if sounding[i]][:3]
    fig, axes = plt.subplots(len(idx), 2, figsize=(14, 3.6 * len(idx)), sharex=True, squeeze=False)
    for row, i in enumerate(idx):
        for col, s in enumerate(srcs):
            pr = res[s][i]["profile"]
            ref = np.nanmax(pr["peak"])
            for k in range(min(8, pr["L"].shape[1])):
                axes[row, col].plot(pr["t"], pr["L"][:, k] - ref, lw=1, label=f"{k + 1}")
            axes[row, col].set_ylim(-70, 5)
            axes[row, col].grid(alpha=0.3)
            e = events[i]
            axes[row, col].set_title(f"{s}: MIDI {e['pitch']}, velocity {e['velocity']}, {e['state']}", fontsize=10)
        axes[row, 0].set_ylabel("dB re strongest")
    axes[0, 1].legend(title="partial", fontsize=8, ncol=2)
    for ax in axes[-1]:
        ax.set_xlabel("s after the onset")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "examples.png"), dpi=110)
    plt.close(fig)


def listen(out, events, clips, res, srcs, sr, n, spec):
    import soundfile as sf

    os.makedirs(out, exist_ok=True)
    cand = [i for i, e in enumerate(events) if e["state"] != "released"]
    cand = sorted(cand, key=lambda i: (events[i]["velocity"], events[i]["pitch"]))
    pick = [cand[j] for j in np.unique(np.linspace(0, len(cand) - 1, min(n, len(cand))).round().astype(int))]
    fade = lambda x: x * np.minimum(1, np.minimum(np.arange(len(x)), np.arange(len(x))[::-1]) / (0.02 * sr))[:, None]
    ab, lines = [], []
    for j, i in enumerate(pick):
        e = events[i]
        parts = []
        for s in srcs:
            t_on = res[s][i]["t_on"]
            x = M._segment(clips[s][i], sr, t_on - 0.3, t_on + e["t_end"] + 0.05)
            parts.append(x)
        g = 10 ** ((res[srcs[0]][i]["total"] - res[srcs[1]][i]["total"]) / 20)
        parts[1] = parts[1] * g
        for s, x in zip(("recording", "model"), parts):
            sf.write(os.path.join(out, f"{j:02d}_{s}.wav"), fade(x).astype(np.float32), sr)
        gap = np.zeros((int(0.4 * sr), parts[0].shape[1]))
        ab += [fade(parts[0]), gap, fade(parts[1]), np.zeros((int(1.0 * sr), parts[0].shape[1]))]
        lines.append(f"{j:02d}. MIDI {e['pitch']}, velocity {e['velocity']}, {e['state']}: `{e['piece']}` at {e['onset']:.1f} s "
                     f"({e['split']}); model level matched by {20 * math.log10(g):+.1f} dB")
    sf.write(os.path.join(out, "ab.wav"), np.concatenate(ab).astype(np.float32), sr)
    with open(os.path.join(out, "README.md"), "w", encoding="utf-8") as f:
        f.write(f"# Isolated tenor notes (R3, MIDI 47–59, mf–f): recording and model\n\n"
                f"Model: `{spec}` (phase 4, per-strike variation on), rendering each note in its MIDI context. Each clip "
                f"runs from 0.3 s before the note to the end of its clear second. The model's level is matched to the "
                f"recording's over the first 0.5 s, so the comparison is of timbre, attack and decay, not level. "
                f"`ab.wav` plays each pair in turn (recording, then model); `NN_recording.wav` and `NN_model.wav` are the "
                f"single notes. Measurements: `docs/tone_measures.md` 15.\n\n" + "\n".join(f"- {s}" for s in lines) + "\n")


if __name__ == "__main__":
    main()
