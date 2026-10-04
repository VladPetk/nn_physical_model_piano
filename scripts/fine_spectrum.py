"""The long-term spectrum at 1/12 octave, recording against model (docs/physics_revamp.md 13): does the model have the
octave balance right but the spectral envelope's finer shape (body resonances, formants) wrong?

    python scripts/fine_spectrum.py data/maestro24k --out runs/physics_revamp/attack/fine \\
        --model main=runs/physics_revamp/main/residual/last.pt:residual [--excerpt samples/physics_revamp/keys:3:0:2]

The clips of ``scripts/brightness_context.py`` (same draw). Per clip and side: the mean power spectrum (summed over channels; ``--mono``: of their mean, which a spaced pair's
inter-channel delays comb-filter), 4096-point
Hann frames, hop 1024) in 1/12-octave bands from 62.5 Hz to 8 kHz, in dB re the clip's total over the same range (so a
level difference is gone). Model − recording per band: median over clips, 95 % interval from a bootstrap over clips.
Each clip's notes put energy at their own partials, which both sides share, so what remains is the instrument's and the
room's spectral envelope. Writes ``report.md`` and ``fine.png``. ``--excerpt folder:index:t0:t1`` adds a listening
excerpt's window (s) of ``ab_render`` output.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402

EDGES = 100.0 * 2 ** (np.arange(0, 80) / 12)  # 100 Hz .. 9.5 kHz (below 100 Hz a 4096-point frame has too few bins)


def bands(x, sr, mono=False):
    """Per 1/12-octave band: the power summed over channels (``mono``: of the channels' mean), dB re the total."""
    x = x if x.ndim == 2 else x[:, None]
    if mono:
        x = x.mean(1, keepdims=True)
    n, h = 4096, 1024
    w = np.hanning(n)[:, None]
    P = np.mean([(np.abs(np.fft.rfft(x[i: i + n] * w, axis=0)) ** 2).sum(1) for i in range(0, len(x) - n, h)], 0)
    f = np.fft.rfftfreq(n, 1 / sr)
    B = np.array([P[(f >= a) & (f < b)].sum() for a, b in zip(EDGES[:-1], EDGES[1:])])
    return 10 * np.log10(B + 1e-20) - 10 * np.log10(B.sum() + 1e-20)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data")
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--year", type=int, default=2018)
    ap.add_argument("--clips", type=int, default=64)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--excerpt", action="append", default=[])
    ap.add_argument("--mono", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    import torch

    from pianonn import measures as M
    from pianonn.render import load_variant

    os.makedirs(args.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.6)
    with open(os.path.join(args.data, "index.json")) as f:
        pieces = [p for p in json.load(f) if p["year"] == args.year]
    rng = np.random.default_rng(args.seed)
    events = []
    for i in rng.choice(len(pieces), args.clips, replace=len(pieces) < args.clips):
        p = pieces[i]
        t0 = float(rng.uniform(M.WARMUP + 1.0, p["duration"] - args.seconds - 1.0))
        events.append({"piece": p["id"], "onset": t0})
    models = [load_variant(s, device=dev) for s in args.model]
    labels = [m[0] for m in models]
    sr = models[0][1].cfg.sample_rate
    clips = M.render_clips(args.data, events, models, models[0][1].cfg, dev, pre=0.0, post=args.seconds, batch=2)
    B = {s: np.array([bands(x, sr, args.mono) for x in clips[s]]) for s in clips}
    centres = np.sqrt(EDGES[:-1] * EDGES[1:])
    np.savez_compressed(os.path.join(args.out, "fine.npz"), centres=centres, **B)
    L = [f"# Long-term spectrum at 1/12 octave ({args.year}, {len(events)} clips of {args.seconds:g} s)", "",
         "dB re each clip's total (62.5 Hz-8 kHz); model − recording, median over clips [95 % over clips]; the "
         "recording's own median level in the band; then the detail: model − recording minus its 1-octave moving "
         "average (what a smooth correction would leave).", "",
         "| band (Hz) | recording | " + " | ".join(f"{s} − rec | detail" for s in labels) + " |", "|---|---|" + "---|---|" * len(labels)]
    rngb = np.random.default_rng(0)
    det = {}
    for s in labels:
        d = B[s] - B["recording"]
        med = np.median(d, 0)
        sm = np.convolve(np.pad(med, 6, mode="edge"), np.ones(13) / 13, mode="same")[6:-6]
        det[s] = (med, med - sm)
    for j, c in enumerate(centres):
        cells = []
        for s in labels:
            d = B[s][:, j] - B["recording"][:, j]
            bs = [np.median(d[rngb.integers(0, len(d), len(d))]) for _ in range(300)]
            cells.append(f"{np.median(d):+.1f} [{np.percentile(bs, 2.5):+.1f}, {np.percentile(bs, 97.5):+.1f}] | {det[s][1][j]:+.1f}")
        L.append(f"| {c:.0f} | {np.median(B['recording'][:, j]):+.1f} | " + " | ".join(cells) + " |")
    L.append("")
    for s in labels:
        L.append(f"rms of the detail ({s}): {np.sqrt(np.mean(det[s][1] ** 2)):.2f} dB; of the whole difference {np.sqrt(np.mean(det[s][0] ** 2)):.2f} dB")
    exc = {}
    for spec in args.excerpt:
        import soundfile as sf
        folder, k, a, b = spec.split(":")
        with open(os.path.join(folder, "manifest.json")) as f:
            order = json.load(f)["order"]
        E = {}
        for s in order:
            x, srr = sf.read(os.path.join(folder, f"{k}_{s}.wav"))
            E[s] = bands(x[int(float(a) * srr): int(float(b) * srr)], srr, args.mono)
        exc[spec] = E
        L += ["", f"## Excerpt `{folder}` {k}, {a}-{b} s: model − recording per 1/12 octave", "",
              "| band (Hz) | " + " | ".join(f"{s} − rec" for s in order[1:]) + " |", "|---|" + "---|" * (len(order) - 1)]
        for j, c in enumerate(centres):
            L.append(f"| {c:.0f} | " + " | ".join(f"{E[s][j] - E[order[0]][j]:+.1f}" for s in order[1:]) + " |")
    with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L[:12] + L[-20:]))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    for s in labels:
        d = B[s] - B["recording"]
        q = np.percentile(d, [25, 50, 75], 0)
        axes[0].plot(centres, q[1], label=f"{s} − recording (median over clips)")
        axes[0].fill_between(centres, q[0], q[2], alpha=0.2)
    for spec, E in exc.items():
        order = list(E)
        for s in order[1:]:
            axes[0].plot(centres, E[s] - E[order[0]], "--", label=f"{spec.split(':')[1]} {s} − rec ({spec.split(':', 2)[2]} s)")
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].set_ylabel("dB")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].plot(centres, np.median(B["recording"], 0), "k", label="recording")
    for s in labels:
        axes[1].plot(centres, np.median(B[s], 0), label=s)
    axes[1].set_ylabel("dB re total")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Hz")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fine.png"), dpi=90)


if __name__ == "__main__":
    main()
