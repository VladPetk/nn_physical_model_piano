"""What tells the recording from the model, beyond the band levels? (docs/physics_revamp.md 13)

    python scripts/texture_tells.py data/maestro24k --out runs/physics_revamp/attack/tells \\
        --model main=runs/physics_revamp/main/residual/last.pt:residual [--excerpt samples/physics_revamp/main:3]

The clips of ``scripts/brightness_context.py`` (same draw: ``--clips``, ``--seconds``, ``--seed``), rendered again. Per
0.5 s segment (hop 0.25 s) and band (``BANDS``), on every side:
- ``level``: the band's level (dB), re the clip's broadband level;
- ``flatness``: the spectral flatness (geometric / arithmetic mean of the power, 8192-point frames), dB: near 0 for
  noise, very negative for a few clean partials;
- ``peak share``: the share of the band's power in its strongest 5 % of bins (dB): the partials against the floor;
- ``shimmer``: the rms (dB) of the band level over 10 ms frames after taking out its 0.25 s moving average: fast
  fluctuation, 4-50 Hz;
- ``coherence``: the inter-channel magnitude-squared coherence, power-weighted over the band (0-1): low for a wide,
  diffuse image.
Per feature: model − recording on the same segments (median and a 95 % interval from a bootstrap over clips), and the
separation, the paired median over the paired IQR. Known change: ``--check`` adds the recording with uncorrelated
noise at −15 dB re each band (flatness up, peak share and coherence down).
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from pianonn import measures as M  # noqa: E402
from pianonn.render import load_variant  # noqa: E402

BANDS = ((500, 1000), (1000, 2000), (2000, 4000), (4000, 8000))
SEG, HOP = 0.5, 0.25
FEATS = ("level", "flatness", "peak share", "shimmer", "coherence")


def features(x, sr):
    """``[S, B, F]`` per segment, band and feature."""
    n_seg, h_seg = int(SEG * sr), int(HOP * sr)
    nf = 2048
    win = np.hanning(nf)
    f = np.fft.rfftfreq(nf, 1 / sr)
    nfr, hfr = int(0.01 * sr), int(0.005 * sr)
    ff = np.fft.rfftfreq(nfr * 2, 1 / sr)
    tot = 10 * np.log10(np.mean(x ** 2) + 1e-20)
    out = []
    for s0 in range(0, len(x) - n_seg, h_seg):
        seg = x[s0: s0 + n_seg]
        # spectra over 8192-point frames, two channels, hop half a frame
        specs = []
        for a in range(0, len(seg) - nf + 1, nf // 2):
            specs.append(np.fft.rfft(seg[a: a + nf] * win[:, None], axis=0))
        S = np.array(specs)  # [T, F, ch]
        P = (np.abs(S) ** 2).mean(0)  # [F, ch]
        Pm = P.sum(1)
        cross = (S[..., 0] * np.conj(S[..., 1])).mean(0)
        coh = np.abs(cross) ** 2 / (P[:, 0] * P[:, 1] + 1e-30)
        # short frames for the fast fluctuation
        fr = np.array([np.abs(np.fft.rfft(seg[a: a + 2 * nfr].sum(1) * np.hanning(2 * nfr))) ** 2
                       for a in range(0, len(seg) - 2 * nfr, hfr)])
        row = []
        for lo, hi in BANDS:
            b = (f >= lo) & (f < hi)
            pb = Pm[b] + 1e-30
            level = 10 * np.log10(pb.sum() / nf) - tot
            flat = 10 * np.log10(np.exp(np.mean(np.log(pb))) / np.mean(pb))
            top = np.sort(pb)[::-1][: max(1, int(0.05 * len(pb)))]
            peak = 10 * np.log10(top.sum() / pb.sum())
            cohw = float((coh[b] * Pm[b]).sum() / Pm[b].sum())
            bb = (ff >= lo) & (ff < hi)
            env = 10 * np.log10(fr[:, bb].sum(1) + 1e-30)
            k = max(1, int(0.25 / 0.005))
            smooth = np.convolve(env, np.ones(k) / k, mode="same")
            core = slice(k // 2, len(env) - k // 2)
            shim = float(np.sqrt(np.mean((env - smooth)[core] ** 2)))
            row.append([level, flat, peak, shim, cohw])
        out.append(row)
    return np.array(out)


def boot(d, clip, n=500, seed=0):
    ok = np.isfinite(d)
    d, clip = d[ok], clip[ok]
    u = np.unique(clip)
    idx = {c: np.nonzero(clip == c)[0] for c in u}
    rng = np.random.default_rng(seed)
    meds = [np.median(d[np.concatenate([idx[c] for c in rng.choice(u, len(u))])]) for _ in range(n)]
    return float(np.median(d)), *np.percentile(meds, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data")
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--year", type=int, default=2018)
    ap.add_argument("--clips", type=int, default=64)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--excerpt", action="append", default=[], help="folder:index of an ab_render listening excerpt to report on its own")
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
    rng = np.random.default_rng(args.seed)  # the draw of brightness_context.py
    events = []
    for i in rng.choice(len(pieces), args.clips, replace=len(pieces) < args.clips):
        p = pieces[i]
        t0 = float(rng.uniform(M.WARMUP + 1.0, p["duration"] - args.seconds - 1.0))
        events.append({"piece": p["id"], "onset": t0, "split": p["split"]})
    models = [load_variant(s, device=dev) for s in args.model]
    labels = [m[0] for m in models]
    cfg = models[0][1].cfg
    sr = cfg.sample_rate
    clips = M.render_clips(args.data, events, models, cfg, dev, pre=0.0, post=args.seconds, batch=2)
    if args.check:
        nz = np.random.default_rng(1)
        chk = []
        for x in clips["recording"]:
            y = x.copy()
            for lo, hi in BANDS:  # uncorrelated noise per channel at -30 dB re the band's power in the clip
                X = np.fft.rfft(x, axis=0)
                f = np.fft.rfftfreq(len(x), 1 / sr)
                pb = np.mean(np.abs(X[(f >= lo) & (f < hi)]) ** 2)
                N = np.fft.rfft(nz.standard_normal(x.shape), axis=0)
                N[(f < lo) | (f >= hi)] = 0
                N *= np.sqrt(pb / np.mean(np.abs(N[(f >= lo) & (f < hi)]) ** 2)) * 10 ** (-15 / 20)
                y = y + np.fft.irfft(N, len(x), axis=0)
            chk.append(y)
        clips["recording+noise-15"] = chk
        labels = labels + ["recording+noise-15"]
    srcs = ["recording"] + labels
    F = {s: [] for s in srcs}
    clip_id = []
    for i in range(len(events)):
        for s in srcs:
            F[s].append(features(clips[s][i], sr))
        clip_id += [i] * len(F["recording"][-1])
    F = {s: np.concatenate(v) for s, v in F.items()}
    clip_id = np.array(clip_id)
    np.savez_compressed(os.path.join(args.out, "features.npz"), clip=clip_id, **{s: v for s, v in F.items()},
                        events=np.array(json.dumps(events)))
    L = [f"# What tells the recording from the model ({args.year}, {len(events)} clips of {args.seconds:g} s, "
         f"{len(clip_id)} segments of {SEG} s)", "",
         "Per feature and band: the recording's median; model − recording (median over segments, 95 % interval "
         "from a bootstrap over clips); separation = paired median / paired IQR.", ""]
    for j, fn in enumerate(FEATS):
        L += [f"## {fn}", "", "| band | recording | " + " | ".join(f"{s} − rec | sep" for s in labels) + " |",
              "|---|---|" + "---|---|" * len(labels)]
        for b, (lo, hi) in enumerate(BANDS):
            r = F["recording"][:, b, j]
            cells = []
            for s in labels:
                d = F[s][:, b, j] - r
                md, a, c = boot(d, clip_id)
                iqr = np.subtract(*np.nanpercentile(d, [75, 25]))
                cells.append(f"{md:+.2f} [{a:+.2f}, {c:+.2f}] | {md / iqr if iqr > 0 else np.nan:+.2f}")
            L.append(f"| {lo}-{hi} Hz | {np.nanmedian(r):+.2f} | " + " | ".join(cells) + " |")
        L.append("")
    for spec in args.excerpt:
        folder, k = spec.rsplit(":", 1)
        import soundfile as sf
        with open(os.path.join(folder, "manifest.json")) as f:
            order = [s for s in json.load(f)["order"]]
        G = {s: features(sf.read(os.path.join(folder, f"{k}_{s}.wav"))[0], sr) for s in order}
        L += [f"## Listening excerpt `{folder}` {k}: model − recording per 0.5 s segment (first 3 s, then the rest)", ""]
        for j, fn in enumerate(FEATS):
            L += [f"### {fn}", "", "| side | band | " + " | ".join(f"{t * HOP:.2f}" for t in range(min(11, len(G[order[0]])))) + " | rest |",
                  "|---|---|" + "---|" * (min(11, len(G[order[0]])) + 1)]
            for s in order[1:]:
                for b, (lo, hi) in enumerate(BANDS):
                    d = G[s][:, b, j] - G[order[0]][:, b, j]
                    L.append(f"| {s} | {lo}-{hi} | " + " | ".join(f"{v:+.1f}" for v in d[:11]) + f" | {np.median(d[11:]):+.1f} |")
            L.append("")
    with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
