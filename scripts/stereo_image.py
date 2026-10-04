"""The stereo image, recording against models (docs/physics_revamp.md 14): ``pianonn.stereo``'s measures on music and
on isolated lower-tenor notes.

    python scripts/stereo_image.py data/maestro24k --out runs/physics_revamp/stereo/measure \\
        --model main=runs/physics_revamp/main/residual/last.pt:residual [--variant aligned=main:aligned] [--check]

Music: the clips of ``scripts/brightness_context.py`` (same draw: ``--clips`` x ``--seconds``, ``--seed``). Per 0.5 s
segment and band: the coherence, the coherent part's phase, the level difference; per clip the GCC-PHAT lag and peak.
Notes: MIDI 45-60, velocity 30-85, isolated as in ``scripts/note_realism.py`` (up to ``--notes``); per partial 1-9 over
0.1-0.6 s after the side's own onset: the coherence (7 frames) and the motion of the left/right relation (40 ms frames),
with the partial's power over its surroundings; the motion is also given for partials 25 dB or more over them (noise in
the bin moves it too: ~2 dB sd at 11 dB, 0.2 at 31 dB, tested on a stationary partial).
Report: per measure the recordings' median (music: and 95 % over clips) and each side's; notes: the recordings' 10 / 50 /
90 % points and each side's median and share under 10 % / over 90 %.

``--variant label=base:mod[,mod]`` renders a modified copy of model ``base``: ``aligned`` (both microphones get the left
one's body, in-plane body and ring-up kernel, no delays between them: the direct sound identical in both channels),
``hallN`` (the hall's level +N dB, e.g. ``hall6``), ``room0`` (no hall). ``--check``: known changes on the recordings: the
right channel delayed 0.5 ms (lag -0.5 ms, coherence unchanged), uncorrelated noise at -10 dB per channel and band
(coherence down by a known amount), the channels' mean in both (coherence 1, no motion).
"""

import argparse
import copy
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from pianonn import measures as M  # noqa: E402
from pianonn import stereo as S  # noqa: E402
from pianonn.render import load_variant  # noqa: E402

K = 9
GROUPS = ((1, 1), (2, 3), (4, 6), (7, 9))
PRE, POST = 0.5, 0.75


def make_variant(model, mods):
    m = copy.deepcopy(model)
    R = m.room
    with torch.no_grad():
        for mod in mods:
            if mod == "aligned":
                for name in ("body", "body_h"):
                    p = getattr(R, name, None)
                    if p is not None:
                        p[:, 1] = p[:, 0]
                if getattr(R, "raw_delay", None) is not None:
                    R.raw_delay.zero_()
                if R.ring_on:
                    R.ring_kernel[1] = R.ring_kernel[0]
            elif mod.startswith("hall"):
                R.log_gain += float(mod[4:]) * math.log(10) / 20
            elif mod == "room0":
                R.log_gain -= 30.0
            else:
                raise ValueError(mod)
    return m


def checks(clips, sr):
    rng = np.random.default_rng(5)
    out = {}
    d = []
    for x in clips:
        X = np.fft.rfft(x[:, 1])
        f = np.fft.rfftfreq(len(x), 1 / sr)
        d.append(np.stack([x[:, 0], np.fft.irfft(X * np.exp(-2j * np.pi * f * 5e-4), len(x))], 1))
    out["check: R delayed 0.5 ms"] = d
    nz = []
    for x in clips:  # -10 dB per band and channel, independent: msc x (1/1.1)^2 for a fully coherent band
        y = x.copy()
        X = np.fft.rfft(x, axis=0)
        f = np.fft.rfftfreq(len(x), 1 / sr)
        for lo, hi in S.BANDS:
            b = (f >= lo) & (f < hi)
            N = np.fft.rfft(rng.standard_normal(x.shape), axis=0)
            N[~b] = 0
            for c in range(2):
                N[:, c] *= np.sqrt(np.mean(np.abs(X[b, c]) ** 2) / np.mean(np.abs(N[b, c]) ** 2)) * 10 ** (-10 / 20)
            y = y + np.fft.irfft(N, len(x), axis=0)
        nz.append(y)
    out["check: noise -10 dB"] = nz
    out["check: mono"] = [np.repeat(x.mean(1, keepdims=True), 2, 1) for x in clips]
    return out


def circ_mean(deg, w):
    return float(np.degrees(np.angle(np.sum(w * np.exp(1j * np.radians(deg))))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data")
    ap.add_argument("--model", action="append", default=[])
    ap.add_argument("--variant", action="append", default=[])
    ap.add_argument("--year", type=int, default=2018)
    ap.add_argument("--clips", type=int, default=64)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--notes", type=int, default=300)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    from note_profile import mine
    from pianonn.config import year_to_condition
    from pianonn.losses import ONSET_DELAY_MS

    os.makedirs(args.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.6)
    models = [load_variant(s, device=dev) for s in args.model]
    by = {lab: (m, r) for lab, m, r in models}
    for spec in args.variant:
        lab, rest = spec.split("=")
        base, mods = rest.split(":")
        models.append((lab, make_variant(by[base][0], mods.split(",")), by[base][1]))
    labels = [m[0] for m in models]
    cfg = models[0][1].cfg
    sr = cfg.sample_rate

    # music
    with open(os.path.join(args.data, "index.json")) as f:
        pieces = [p for p in json.load(f) if p["year"] == args.year]
    rng = np.random.default_rng(args.seed)  # the draw of brightness_context.py
    events = []
    for i in rng.choice(len(pieces), args.clips, replace=len(pieces) < args.clips):
        p = pieces[i]
        t0 = float(rng.uniform(M.WARMUP + 1.0, p["duration"] - args.seconds - 1.0))
        events.append({"piece": p["id"], "onset": t0})
    clips = M.render_clips(args.data, events, models, cfg, dev, pre=0.0, post=args.seconds, batch=2)
    if args.check:
        clips |= checks(clips["recording"], sr)
    sides = ["recording"] + [s for s in clips if s != "recording"]
    music = {}
    for s in sides:
        per = [S.band_stereo(x, sr) for x in clips[s]]
        g = [S.gcc_phat(x, sr) for x in clips[s]]
        music[s] = {"msc": np.array([np.median(p["msc"], 0) for p in per]),  # [clips, bands]
                    "phase": np.array([[circ_mean(p["phase"][:, b], p["msc"][:, b] * p["power"][:, b]) for b in range(len(S.BANDS))]
                                       for p in per]),
                    "ild": np.array([np.median(p["ild"], 0) for p in per]),
                    "lag": np.array([v[0] for v in g]), "peak": np.array([v[1] for v in g])}
    np.savez_compressed(os.path.join(args.out, "music.npz"), **{f"{s}|{k}": v for s, d in music.items() for k, v in d.items()})
    rb = np.random.default_rng(0)

    def ci(v):
        bs = [np.median(v[rb.integers(0, len(v), len(v))]) for _ in range(500)]
        return f"{np.median(v):.2f} [{np.percentile(bs, 2.5):.2f}, {np.percentile(bs, 97.5):.2f}]"

    L = [f"# The stereo image ({args.year}): {len(events)} clips of {args.seconds:g} s; isolated MIDI 45-60 notes", "",
         f"Coherence read for independent channels in 0.5 s segments: {S.msc_bias(sr):.2f}.", "",
         "## Music: per clip the median over its 0.5 s segments; median over clips [95 % over clips]", "",
         "### Coherence per band", "", "| side | " + " | ".join(f"{lo}-{hi} Hz" for lo, hi in S.BANDS) + " |",
         "|---|" + "---|" * len(S.BANDS)]
    for s in sides:
        L.append(f"| {s} | " + " | ".join(ci(music[s]["msc"][:, b]) for b in range(len(S.BANDS))) + " |")
    L += ["", "### The coherent part's phase per band (deg; per clip weighted by coherence x power; median over clips, "
          "and the share of clips within +-45 deg)", "", "| side | " + " | ".join(f"{lo}-{hi} Hz" for lo, hi in S.BANDS) + " |",
          "|---|" + "---|" * len(S.BANDS)]
    for s in sides:
        L.append(f"| {s} | " + " | ".join(f"{np.median(music[s]['phase'][:, b]):+.0f} ({100 * np.mean(np.abs(music[s]['phase'][:, b]) < 45):.0f} %)"
                                          for b in range(len(S.BANDS))) + " |")
    L += ["", "### GCC-PHAT per clip: lag (ms; + left lags) and peak height (1 = identical channels)", "",
          "| side | lag 10 / 50 / 90 % | share within +-0.25 ms | peak median |", "|---|---|---|---|"]
    for s in sides:
        lg = music[s]["lag"]
        L.append(f"| {s} | {np.percentile(lg, 10):+.2f} / {np.median(lg):+.2f} / {np.percentile(lg, 90):+.2f} | "
                 f"{100 * np.mean(np.abs(lg) <= 0.25):.0f} % | {np.median(music[s]['peak']):.3f} |")

    # notes
    if args.notes > 0:
        ev = [e for e in mine(args.data, args.year, (45, 60), (30, 85), 0.25, 0.65) if e["state"] != "released"]
        rn = np.random.default_rng(0)
        ev = [ev[i] for i in sorted(rn.permutation(len(ev))[: args.notes])]
        nclips = M.render_clips(args.data, ev, models, cfg, dev, pre=PRE, post=POST, batch=8)
        table = M.partial_table(models[0][1], year_to_condition(args.year), dev)
        a, b, c = ONSET_DELAY_MS
        nsides = ["recording"] + labels
        F = {s: {k: [] for k in ("coh", "lev", "ph", "snr")} for s in nsides}
        for i, e in enumerate(ev):
            tab = table[e["pitch"] - 21]
            f0 = float(tab[0])
            t_ref = PRE + 1e-3 * max(a + b * (e["pitch"] - 60) + c * (e["velocity"] - 64), 0.0)
            for s in nsides:
                xs = nclips[s][i]
                t_on = M.onset(xs.mean(1), sr, t_ref, f0)
                t_on = t_on if np.isfinite(t_on) else t_ref
                fr = np.asarray(tab[:K], float)
                F[s]["coh"].append(S.partial_coherence(xs, sr, t_on + 0.1, t_on + 0.6, fr))
                lev, ph, snr = S.lr_motion(xs, sr, t_on + 0.08, t_on + 0.6, fr, f1=f0)
                F[s]["lev"].append(lev)
                F[s]["ph"].append(ph)
                F[s]["snr"].append(snr)
        F = {s: {k: np.array(v) for k, v in d.items()} for s, d in F.items()}
        np.savez_compressed(os.path.join(args.out, "notes.npz"), **{f"{s}|{k}": v for s, d in F.items() for k, v in d.items()},
                            pitch=np.array([e["pitch"] for e in ev]))
        L += ["", f"## Isolated notes ({len(ev)}): recordings 10 / 50 / 90 %; each side's median and % under 10 % / over 90 % "
              "(20 % if alike)", "",
              "| measure | recordings | " + " | ".join(f"{s}" for s in labels) + " |", "|---|---|" + "---|" * len(labels)]

        def row(name, key, g, snr_min=None):
            sl = slice(g[0] - 1, g[1])

            def vals(s):
                v = F[s][key][:, sl].copy()
                if snr_min is not None:
                    v[F[s]["snr"][:, sl] < snr_min] = np.nan
                with np.errstate(all="ignore"):
                    v = np.nanmean(v, 1)
                return v[np.isfinite(v)]

            r = vals("recording")
            if len(r) < 20:
                return
            q = np.percentile(r, [10, 50, 90])
            cells = []
            for s in labels:
                m = vals(s)
                cells.append(f"{np.median(m):.3g} ({100 * np.mean(m < q[0]):.0f} / {100 * np.mean(m > q[2]):.0f})")
            L.append(f"| {name}, partials {g[0]}-{g[1]} | {q[0]:.3g} / {q[1]:.3g} / {q[2]:.3g} | " + " | ".join(cells) + " |")

        for key, nm in (("coh", "coherence"), ("lev", "L/R level sd (dB)"), ("ph", "L/R phase circ. sd (deg)"),
                        ("snr", "power over surroundings (dB)")):
            for g in GROUPS:
                row(nm, key, g)
        for key, nm in (("lev", "L/R level sd (dB), >= 25 dB over surroundings"), ("ph", "L/R phase sd (deg), >= 25 dB")):
            for g in GROUPS:
                row(nm, key, g, snr_min=25.0)
    with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
