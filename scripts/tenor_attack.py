"""The attack of lower-tenor notes against velocity, recording against models (docs/physics_revamp.md 13).

    python scripts/tenor_attack.py data/maestro24k --out runs/physics_revamp/attack \\
        --model B_comp=runs/loss_compare/B_comp/train/last.pt:physics \\
        --model main=runs/physics_revamp/main/residual/last.pt:residual

The owner hears mid-soft lower-tenor notes struck "dull", without the ring of the piano, and not at ff. Notes: every
note of the year in ``--pitch`` with no other onset ``--before`` s before or ``--after`` s after it, up to
``--per-bin`` per velocity bin (drawn at random, the same for every model). Each is rendered by every model in its
MIDI context (``measures.render_clips``; per-strike variation as the checkpoint has it). On every side at its own
onset (N0):

- each partial's level (``measures.partial_profile``, a 40 ms window every 5 ms, up to ``--partials`` partials below
  ``--f-max``), averaged over the windows of ``WINDOWS``, re the power sum of partials 1-6 in the same window: the
  spectral shape of the tone as it unfolds; a value counts where the partial stands 6 dB over its own level before the
  onset;
- each partial's drop from the first window to the last (dB, absolute): how the partial rings on;
- the first 40 ms per octave band (``measures.onset_profile``): level in 0-5, 5-10, 10-20, 20-40 ms re 50-100 ms,
  the rise, and the band's level re the note at 50-100 ms;
- the energy away from the partials (``measures.non_tonal``).

``--check``: the recording with a known change (+6 dB above ``SHELF_HZ``, smooth over an octave) as one more side,
which the partial levels must read as +6 dB above the shelf and 0 below it.

Writes ``report.md`` (per velocity bin and partial group: recording and model medians, the paired model - recording
median with a 95 % interval from a bootstrap over pieces), ``notes.npz`` (every feature per note and side),
``events.json``, ``shape.png`` and, with ``--listen``, A/B files of the notes with the largest and smallest deficit.
"""

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from note_profile import mine, refine  # noqa: E402
from pianonn import measures as M  # noqa: E402
from pianonn.config import year_to_condition  # noqa: E402
from pianonn.losses import ONSET_DELAY_MS  # noqa: E402
from pianonn.render import load_variant  # noqa: E402

PRE, POST = 0.5, 0.6  # s of clip around the MIDI onset
WINDOWS = ((0.015, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.33))  # s re N0, the centres of the 40 ms frames
GROUPS = ((1, 1), (2, 3), (4, 6), (7, 10), (11, 16), (17, 24), (25, 32), (33, 48))
NORM = 6  # levels re the power sum of partials 1..NORM
SHELF_HZ = 2000.0


def shelf(x, sr, f0=SHELF_HZ, db=6.0):
    """``x`` with a gain of ``db`` above ``f0``, rising smoothly over the octave below it (zero phase)."""
    X = np.fft.rfft(x, axis=0)
    f = np.fft.rfftfreq(len(x), 1 / sr)
    s = np.clip(np.log2(np.maximum(f, 1e-3) / (f0 / 2)), 0, 1)
    g = 10 ** (db * np.sin(0.5 * np.pi * s) ** 2 / 20)
    return np.fft.irfft(X * g[:, None], len(x), axis=0)


def measure(x, sr, t_ref, table, f0, t_end):
    t_on = M.onset(x, sr, t_ref, f0)
    found = bool(np.isfinite(t_on))
    t_on = t_on if found else t_ref
    freqs = refine(x, sr, t_on, table, f0, t_end)
    pr = M.partial_profile(x, sr, t_on, freqs, t_end)
    valid = pr["L"] >= pr["bg"][None] + 3.0
    K = len(freqs)
    lv = np.full((len(WINDOWS), K), np.nan)  # level re partials 1..NORM
    ab = np.full((len(WINDOWS), K), np.nan)  # absolute level (dB), counted points only
    for w, (a, b) in enumerate(WINDOWS):
        sel = (pr["t"] >= a) & (pr["t"] < b)
        if b > t_end + 1e-6 or not sel.any():
            continue
        P = 10 ** (pr["L"][sel] / 10)
        norm = 10 * np.log10(P[:, :NORM].sum(1).mean() + 1e-30)
        ok = valid[sel].mean(0) >= 0.6
        lvl = 10 * np.log10(P.mean(0) + 1e-30)
        ab[w] = np.where(ok, lvl, np.nan)
        lv[w] = np.where(ok, lvl - norm, np.nan)
    op = M.onset_profile(x, sr, t_on, f0)
    nt = M.non_tonal(x, sr, t_on, np.array(list(freqs) + [freqs[-1] + f0 * j for j in range(1, 200)
                                                          if freqs[-1] + f0 * j < 8000]),
                     windows=((-0.003, 0.04), (0.1, 0.3)))
    return {"t_on": t_on, "found": found, "lv": lv, "drop": ab[-1] - ab[0], "t_peak": pr["t_peak"],
            "early": pr["early"], "on_win": op["win"], "on_rise": op["rise"], "on_ref": op["ref_db"],
            "nt": nt, "total": 10 * np.log10(np.sum(10 ** (ab[1][np.isfinite(ab[1])] / 10)) + 1e-30)}


def boot_median(d, pieces, n=1000, seed=0):
    """Median of ``d`` over notes and a 95 % interval from resampling pieces."""
    ok = np.isfinite(d)
    d, pieces = d[ok], np.asarray(pieces)[ok]
    if len(d) < 5:
        return float("nan"), float("nan"), float("nan"), len(d)
    up = np.unique(pieces)
    idx = {p: np.nonzero(pieces == p)[0] for p in up}
    rng = np.random.default_rng(seed)
    meds = [np.median(d[np.concatenate([idx[p] for p in rng.choice(up, len(up))])]) for _ in range(n)]
    lo, hi = np.percentile(meds, [2.5, 97.5])
    return float(np.median(d)), float(lo), float(hi), len(d)


def group_mean(a, g):
    """Mean over the partials of group ``g`` (1-based, inclusive) along the last axis, NaN-aware."""
    lo, hi = g
    v = a[..., lo - 1: hi]
    if v.shape[-1] == 0:
        return np.full(a.shape[:-1], np.nan)
    with np.errstate(all="ignore"):
        n = np.isfinite(v).sum(-1)
        m = np.nansum(v, -1) / np.maximum(n, 1)
    return np.where(n >= max(1, (hi - lo + 1) // 2), m, np.nan)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data")
    ap.add_argument("--model", action="append", required=True, help="label=checkpoint:physics|residual[:options]")
    ap.add_argument("--year", type=int, default=2018)
    ap.add_argument("--pitch", type=int, nargs=2, default=(45, 59))
    ap.add_argument("--bins", type=int, nargs="+", default=(1, 40, 55, 70, 85, 128))
    ap.add_argument("--per-bin", type=int, default=100)
    ap.add_argument("--before", type=float, default=0.25)
    ap.add_argument("--after", type=float, default=0.4)
    ap.add_argument("--partials", type=int, default=32)
    ap.add_argument("--f-max", type=float, default=7000.0)
    ap.add_argument("--check", action="store_true", help="add the recording with a known +6 dB shelf as a side")
    ap.add_argument("--listen", help="directory for A/B files")
    ap.add_argument("--n-listen", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.6)

    allev = mine(args.data, args.year, args.pitch, (1, 127), args.before, args.after)
    rng = np.random.default_rng(args.seed)
    bins = list(zip(args.bins[:-1], args.bins[1:]))
    events = []
    for lo, hi in bins:
        cand = [e for e in allev if lo <= e["velocity"] < hi]
        pick = rng.permutation(len(cand))[: args.per_bin]
        events += [cand[i] for i in sorted(pick)]
    models = [load_variant(s, device=dev) for s in args.model]
    labels = [m[0] for m in models]
    cfg, sr = models[0][1].cfg, models[0][1].cfg.sample_rate
    counts = ", ".join(f"{lo}-{hi - 1}: {sum(lo <= e['velocity'] < hi for e in events)}" for lo, hi in bins)
    print(f"{len(allev)} notes qualify; measuring {len(events)} ({counts})", flush=True)
    clips = M.render_clips(args.data, events, models, cfg, dev, pre=PRE, post=POST, batch=8)
    if args.check:
        clips["recording+shelf"] = [shelf(x, sr) for x in clips["recording"]]
    table = M.partial_table(models[0][1], year_to_condition(args.year), dev)
    a, b, c = ONSET_DELAY_MS
    srcs = ["recording"] + labels + (["recording+shelf"] if args.check else [])
    K = args.partials
    res = {s: [] for s in srcs}
    for i, ev in enumerate(events):
        tab = table[ev["pitch"] - 21]
        tab = tab[(tab > 0) & (tab < args.f_max)][:K]
        f0 = float(tab[0])
        t_ref = PRE + 1e-3 * max(a + b * (ev["pitch"] - 60) + c * (ev["velocity"] - 64), 0.0)
        t_end = min(ev["t_end"], 0.35)
        for s in srcs:
            res[s].append(measure(clips[s][i], sr, t_ref, tab, f0, t_end))

    def stack(s, key, shape):
        out = np.full((len(events),) + shape, np.nan)
        for i, r in enumerate(res[s]):
            v = np.asarray(r[key], float)
            sl = tuple(slice(0, min(n, m)) for n, m in zip(shape, v.shape))
            out[(i,) + sl] = v[sl]
        return out

    W = len(WINDOWS)
    D = {s: {"lv": stack(s, "lv", (W, K)), "drop": stack(s, "drop", (K,)), "t_peak": stack(s, "t_peak", (K,)),
             "early": stack(s, "early", (K,)), "on_win": stack(s, "on_win", (4, len(M.KNOCK_BANDS))),
             "on_rise": stack(s, "on_rise", (len(M.KNOCK_BANDS),)), "on_ref": stack(s, "on_ref", (len(M.KNOCK_BANDS),)),
             "total": np.array([r["total"] for r in res[s]]), "found": np.array([r["found"] for r in res[s]])}
         for s in srcs}
    nt_keys = list(res["recording"][0]["nt"].keys())
    for s in srcs:
        D[s]["nt"] = np.array([[r["nt"][k] for k in nt_keys] for r in res[s]])
    vel = np.array([e["velocity"] for e in events])
    pitch = np.array([e["pitch"] for e in events])
    pieces = np.array([e["piece"] for e in events])
    vbin = np.digitize(vel, args.bins[1:-1])
    np.savez_compressed(os.path.join(args.out, "notes.npz"), velocity=vel, pitch=pitch, pieces=pieces,
                        state=np.array([e["state"] for e in events]),
                        **{f"{s}|{k}": v for s in srcs for k, v in D[s].items()})
    with open(os.path.join(args.out, "events.json"), "w") as f:
        json.dump(events, f, indent=1)

    # ---- report
    def fmtc(m, lo, hi, n):
        return "" if not np.isfinite(m) else f"{m:+.1f} [{lo:+.1f}, {hi:+.1f}] ({n})"

    def fmt(v):
        v = v[np.isfinite(v)]
        return f"{np.median(v):+.1f} ({len(v)})" if len(v) else ""

    bin_names = [f"{lo}-{hi - 1}" for lo, hi in bins]
    L = [f"# The attack of MIDI {args.pitch[0]}-{args.pitch[1]} against velocity ({args.year})", "",
         f"{len(allev)} notes qualify (no other onset {args.before} s before or {args.after} s after); measured "
         f"{len(events)}: " + ", ".join(f"velocity {n}: {(vbin == j).sum()}" for j, n in enumerate(bin_names)) + ". "
         f"Models: " + ", ".join(f"`{s}`" for s in args.model) + ". N0 found: " +
         ", ".join(f"{s} {int(D[s]['found'].sum())}" for s in srcs) + ". Medians over notes, (n); paired: model − "
         "recording per note, its median and a 95 % interval from a bootstrap over pieces.", ""]
    for w, (wa, wb) in enumerate(WINDOWS):
        L += [f"## Partial levels re partials 1-{NORM}, {wa * 1000:.0f}-{wb * 1000:.0f} ms after N0 (dB)", ""]
        hdr = "| velocity | side | " + " | ".join(f"{g[0]}" if g[0] == g[1] else f"{g[0]}-{g[1]}" for g in GROUPS) + " |"
        L += [hdr, "|---|---|" + "---|" * len(GROUPS)]
        for j, n in enumerate(bin_names):
            sel = vbin == j
            G = {s: np.stack([group_mean(D[s]["lv"][:, w], g) for g in GROUPS], 1) for s in srcs}
            L.append(f"| {n} | recording | " + " | ".join(fmt(G["recording"][sel, q]) for q in range(len(GROUPS))) + " |")
            for s in srcs[1:]:
                L.append(f"| {n} | {s} − rec | " + " | ".join(
                    fmtc(*boot_median(G[s][sel, q] - G["recording"][sel, q], pieces[sel])) for q in range(len(GROUPS))) + " |")
        L.append("")
    L += [f"## Each partial's drop from {WINDOWS[0][0] * 1000:.0f}-{WINDOWS[0][1] * 1000:.0f} to "
          f"{WINDOWS[-1][0] * 1000:.0f}-{WINDOWS[-1][1] * 1000:.0f} ms (dB; notes sounding that long)", ""]
    L += ["| velocity | side | " + " | ".join(f"{g[0]}" if g[0] == g[1] else f"{g[0]}-{g[1]}" for g in GROUPS) + " |",
          "|---|---|" + "---|" * len(GROUPS)]
    for j, n in enumerate(bin_names):
        sel = vbin == j
        G = {s: np.stack([group_mean(D[s]["drop"], g) for g in GROUPS], 1) for s in srcs}
        L.append(f"| {n} | recording | " + " | ".join(fmt(G["recording"][sel, q]) for q in range(len(GROUPS))) + " |")
        for s in srcs[1:]:
            L.append(f"| {n} | {s} − rec | " + " | ".join(
                fmtc(*boot_median(G[s][sel, q] - G["recording"][sel, q], pieces[sel])) for q in range(len(GROUPS))) + " |")
    L += ["", "## Peak time per partial (ms re N0, -10..150 ms)", "",
          "| velocity | side | " + " | ".join(f"{g[0]}" if g[0] == g[1] else f"{g[0]}-{g[1]}" for g in GROUPS) + " |",
          "|---|---|" + "---|" * len(GROUPS)]
    for j, n in enumerate(bin_names):
        sel = vbin == j
        G = {s: np.stack([group_mean(1000 * D[s]["t_peak"], g) for g in GROUPS], 1) for s in srcs}
        L.append(f"| {n} | recording | " + " | ".join(fmt(G["recording"][sel, q]) for q in range(len(GROUPS))) + " |")
        for s in srcs[1:]:
            L.append(f"| {n} | {s} − rec | " + " | ".join(
                fmtc(*boot_median(G[s][sel, q] - G["recording"][sel, q], pieces[sel])) for q in range(len(GROUPS))) + " |")
    bands = M.KNOCK_BANDS
    ow = ("0-5", "5-10", "10-20", "20-40")
    L += ["", "## The first 40 ms per octave band (dB re the band's level at 50-100 ms)", ""]
    for wi, wn in enumerate(ow):
        L += [f"### {wn} ms", "", "| velocity | side | " + " | ".join(f"{b} Hz" for b in bands) + " |",
              "|---|---|" + "---|" * len(bands)]
        for j, n in enumerate(bin_names):
            sel = vbin == j
            L.append(f"| {n} | recording | " + " | ".join(fmt(D["recording"]["on_win"][sel, wi, q]) for q in range(len(bands))) + " |")
            for s in srcs[1:]:
                L.append(f"| {n} | {s} − rec | " + " | ".join(fmtc(*boot_median(
                    D[s]["on_win"][sel, wi, q] - D["recording"]["on_win"][sel, wi, q], pieces[sel])) for q in range(len(bands))) + " |")
        L.append("")
    L += ["### Band level re the note at 50-100 ms (dB)", "", "| velocity | side | " + " | ".join(f"{b} Hz" for b in bands) + " |",
          "|---|---|" + "---|" * len(bands)]
    for j, n in enumerate(bin_names):
        sel = vbin == j
        L.append(f"| {n} | recording | " + " | ".join(fmt(D["recording"]["on_ref"][sel, q]) for q in range(len(bands))) + " |")
        for s in srcs[1:]:
            L.append(f"| {n} | {s} − rec | " + " | ".join(fmtc(*boot_median(
                D[s]["on_ref"][sel, q] - D["recording"]["on_ref"][sel, q], pieces[sel])) for q in range(len(bands))) + " |")
    L += ["", "### Rise, 10 → 90 % (ms)", "", "| velocity | side | " + " | ".join(f"{b} Hz" for b in bands) + " |",
          "|---|---|" + "---|" * len(bands)]
    for j, n in enumerate(bin_names):
        sel = vbin == j
        L.append(f"| {n} | recording | " + " | ".join(fmt(1000 * D["recording"]["on_rise"][sel, q]) for q in range(len(bands))) + " |")
        for s in srcs[1:]:
            L.append(f"| {n} | {s} − rec | " + " | ".join(fmtc(*boot_median(
                1000 * (D[s]["on_rise"][sel, q] - D["recording"]["on_rise"][sel, q]), pieces[sel])) for q in range(len(bands))) + " |")
    L += ["", "## Energy away from the partials (dB re the note's energy in the window)", "",
          "| window | band | velocity | recording | " + " | ".join(f"{s} − rec" for s in srcs[1:]) + " |",
          "|---|---|---|---|" + "---|" * (len(srcs) - 1)]
    for q, ((w0, w1), band) in enumerate(nt_keys):
        if band < 1000:
            continue
        for j, n in enumerate(bin_names):
            sel = vbin == j
            L.append(f"| {w0 * 1000:.0f}-{w1 * 1000:.0f} ms | {band} | {n} | {fmt(D['recording']['nt'][sel, q])} | " + " | ".join(
                fmtc(*boot_median(D[s]["nt"][sel, q] - D["recording"]["nt"][sel, q], pieces[sel])) for s in srcs[1:]) + " |")
    with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    plot(args.out, D, srcs, vbin, bin_names, K)
    if args.listen:
        listen(args.listen, events, clips, res, D, srcs, labels, sr, vbin, args.n_listen)


def brilliance(D, s, w=0):
    """Per note: the mean level of partials 7-24 re partials 1-6 in window ``w`` (dB)."""
    return group_mean(D[s]["lv"][:, w], (7, 24))


def plot(out, D, srcs, vbin, bin_names, K):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k = np.arange(1, K + 1)
    fig, axes = plt.subplots(len(WINDOWS), len(bin_names), figsize=(4 * len(bin_names), 3.2 * len(WINDOWS)),
                             sharex=True, sharey=True, squeeze=False)
    colors = ["black", "tab:blue", "tab:red", "tab:green", "tab:orange", "tab:purple"]
    for w, (a, b) in enumerate(WINDOWS):
        for j, n in enumerate(bin_names):
            ax = axes[w, j]
            sel = vbin == j
            for s, col in zip(srcs, colors):
                v = D[s]["lv"][sel, w]
                with np.errstate(all="ignore"):
                    m = np.array([np.nanmedian(v[:, q]) if np.isfinite(v[:, q]).sum() >= 5 else np.nan for q in range(K)])
                ax.plot(k, m, "-", color=col, label=s, lw=1.2)
            ax.grid(alpha=0.3)
            if w == 0:
                ax.set_title(f"velocity {n} (n {sel.sum()})")
            if j == 0:
                ax.set_ylabel(f"{a * 1000:.0f}-{b * 1000:.0f} ms\ndB re partials 1-{NORM}")
    axes[0, 0].legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("partial")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "shape.png"), dpi=100)
    plt.close(fig)


def listen(out, events, clips, res, D, srcs, labels, sr, vbin, n):
    """The mid-velocity notes (bins 1-2) where the last model's partials 7-24 fall furthest under the recording's in
    the first 40 ms, and those where they fall least: recording, then every model, level matched over 0-0.33 s."""
    import soundfile as sf

    os.makedirs(out, exist_ok=True)
    d = brilliance(D, labels[-1]) - brilliance(D, "recording")
    cand = np.nonzero(np.isfinite(d) & np.isin(vbin, (1, 2)) & np.array([e["state"] != "released" for e in events]))[0]
    order = cand[np.argsort(d[cand])]
    pick = list(order[: n // 2]) + list(order[::-1][: n - n // 2])
    fade = lambda x: x * np.minimum(1, np.minimum(np.arange(len(x)), np.arange(len(x))[::-1]) / (0.02 * sr))[:, None]  # noqa: E731
    ab, lines = [], []
    sides = ["recording"] + labels
    for j, i in enumerate(pick):
        e = events[i]
        parts = []
        for s in sides:
            t_on = res[s][i]["t_on"]
            x = M._segment(clips[s][i], sr, t_on - 0.3, t_on + 0.45)
            g = 10 ** ((res["recording"][i]["total"] - res[s][i]["total"]) / 20) if s != "recording" else 1.0
            parts.append(fade(x * (g if np.isfinite(g) else 1.0)))
        gap = np.zeros((int(0.3 * sr), parts[0].shape[1]))
        for x in parts:
            ab += [x, gap]
        ab.append(np.zeros((int(0.8 * sr), parts[0].shape[1])))
        lines.append(f"{j:02d}. MIDI {e['pitch']}, velocity {e['velocity']}, {e['state']}: `{e['piece']}` at "
                     f"{e['onset']:.2f} s ({e['split']}); partials 7-24 in 0-40 ms, {labels[-1]} − recording "
                     f"{d[i]:+.1f} dB")
    sf.write(os.path.join(out, "ab.wav"), np.concatenate(ab).astype(np.float32), sr)
    with open(os.path.join(out, "README.md"), "w", encoding="utf-8") as f:
        f.write("# Lower-tenor attacks at mid velocity: recording and models\n\n"
                f"`ab.wav`: per note, {', '.join(sides)} in turn (0.75 s each, from 0.3 s before the onset; each model's "
                "level matched to the recording's partials at 40-100 ms). The first half: the notes where the "
                f"last model's partials 7-24 fall furthest under the recording's in the first 40 ms; the second half: "
                "where they fall least (`scripts/tenor_attack.py`).\n\n" + "\n".join(f"- {s}" for s in lines) + "\n")


if __name__ == "__main__":
    main()
