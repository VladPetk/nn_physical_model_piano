"""Does a re-strike depend on where the ringing string was in its cycle? (docs/physics_revamp.md 13)

    python scripts/restrike_phase.py data/maestro24k --out runs/physics_revamp/attack/phase \\
        --model main=runs/physics_revamp/main/residual/last.pt:residual [--pitch 36 84]

The bench's re-strike runs (each strike 0.08-0.6 s after the key's last, the pedal down), recording and every model
(rendered in context, ``measures.render_clips``). Per strike, on the mono sum, the fundamental's complex amplitude on
the clip's own time axis, ``C = 2 sum x w exp(-i w t) / sum w`` (Hann ``w``; a stationary sinusoid gives the same C in
any window), at the key's fundamental refined on the strike's after window:
- ``C_old`` over -70..-5 ms re the MIDI strike: the ringing vibration;
- ``C_after`` over 20 ms .. min(100 ms, next strike - 10 ms): old and new together;
- the blow's own part ``N = C_after - k C_old`` (``k``: the share of the old vibration the blow leaves, ``--k``), and
  ``delta = arg N - arg C_old``: where the old vibration was against the blow's push (0: in phase, adding; pi: against).
Both parts pass through the same body and room, so their phase shifts cancel in ``delta``.

Tests: (1) the spread of ``delta`` (uniform for random timing); (2) the highs' change at the strike (partials 7-24,
over -60..-5 ms to the after window) and (3) the blow's own fundamental level |N|, each after a regression on velocity,
previous velocity, log gap and pitch, regressed on cos(delta) and sin(delta): the amplitude of a dependence on the
phase, with a 95 % interval from a bootstrap over runs. ``--selftest`` checks the estimator on synthetic strikes with a
known delta and a known dependence of the highs on it.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402

PRE, POST = 0.5, 1.6


def camp(x, sr, f, a, b):
    """Complex amplitude of ``f`` Hz over [a, b] s (clip time) of mono ``x``, on the clip's time axis."""
    i0, i1 = int(round(a * sr)), int(round(b * sr))
    seg = x[i0:i1]
    t = np.arange(i0, i0 + len(seg)) / sr
    w = np.hanning(len(seg))
    return 2 * np.sum(seg * w * np.exp(-2j * np.pi * f * t)) / np.sum(w)


def refine_f(x, sr, f0, a, b, span=0.006, n=121):
    """The frequency within +-span (relative) of ``f0`` with the largest |C| over [a, b]."""
    fs = f0 * (1 + np.linspace(-span, span, n))
    m = [abs(camp(x, sr, f, a, b)) for f in fs]
    return fs[int(np.argmax(m))]


def band_power(x, sr, a, b, freqs):
    i0, i1 = int(round(a * sr)), int(round(b * sr))
    seg = x[i0:i1]
    nf = 1 << int(np.ceil(np.log2(len(seg) * 4)))
    P = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), nf)) ** 2 / len(seg)
    f = np.fft.rfftfreq(nf, 1 / sr)
    tot = 0.0
    for fk in freqs:
        band = (f > fk * 0.985) & (f < fk * 1.015)
        if band.any():
            tot += P[band].max()
    return 10 * np.log10(tot + 1e-30)


def strike(x, sr, t, nxt, f1, partials, k):
    """Per strike at clip time ``t`` (s): delta, |C_old|, |N|, |C_after| (dB), the highs' change (dB)."""
    b = min(0.1, nxt - 0.01)
    if b < 0.05:
        return None
    f = refine_f(x, sr, f1, t + 0.02, t + b)
    c_old = camp(x, sr, f, t - 0.07, t - 0.005)
    c_aft = camp(x, sr, f, t + 0.02, t + b)
    new = c_aft - k * c_old
    hi = [p for p in partials[6:24] if p < 7000]
    highs = band_power(x, sr, t + 0.02, t + b, hi) - band_power(x, sr, t - 0.06, t - 0.005, hi) if hi else np.nan
    db = lambda c: 20 * np.log10(abs(c) + 1e-12)  # noqa: E731
    return {"delta": float(np.angle(new / c_old)) if abs(c_old) > 0 else np.nan, "old": db(c_old), "new": db(new),
            "after": db(c_aft), "highs": highs, "f": f}


def selftest(sr=24000, n=400, seed=0):
    """Synthetic re-strikes: an old partial (random phase, decaying), a new one of known size at a fixed phase re the
    blow, the highs' level following cos(delta) with amplitude 3 dB; the estimator should find delta and 3 dB."""
    rng = np.random.default_rng(seed)
    f1 = 207.65
    t = np.arange(int(1.0 * sr)) / sr
    ts = 0.5
    errs, rows = [], []
    for _ in range(n):
        phi_old = rng.uniform(-np.pi, np.pi)
        a_old, a_new = 10 ** rng.uniform(-1, 0), 10 ** rng.uniform(-0.5, 0.3)
        old = a_old * np.cos(2 * np.pi * f1 * t + phi_old) * np.exp(-1.5 * t)
        phi_new = -2 * np.pi * f1 * ts + rng.uniform(-0.1, 0.1)  # the blow's phase fixed re its time
        on = t >= ts
        newp = np.where(on, a_new * np.cos(2 * np.pi * f1 * t + phi_new) * np.exp(-3 * (t - ts)), 0)
        true_delta = np.angle(np.exp(1j * (phi_new - phi_old)))
        hi_db = 3 * np.cos(true_delta) + rng.normal(0, 1)
        highs = np.zeros_like(t)
        for kk in range(7, 20):
            amp = 0.01 * np.where(on, 10 ** (hi_db / 20), 0.3)
            highs += amp * np.cos(2 * np.pi * f1 * kk * t + rng.uniform(0, 6.28))
        x = old + newp + highs + rng.normal(0, 1e-4, len(t))
        r = strike(x, sr, ts, ts + 0.3, f1, f1 * np.arange(1, 25), 1.0)
        errs.append(np.angle(np.exp(1j * (r["delta"] - true_delta))))
        rows.append((r["delta"], r["highs"]))
    errs = np.array(errs)
    d, h = np.array(rows).T
    X = np.column_stack([np.ones(n), np.cos(d), np.sin(d)])
    c = np.linalg.lstsq(X, h, rcond=None)[0]
    print(f"selftest: delta error median {np.median(np.abs(errs)):.3f} rad, 90 % {np.percentile(np.abs(errs), 90):.3f}; "
          f"highs ~ cos(delta): {c[1]:+.2f} dB (true +3.00), sin {c[2]:+.2f} (true 0)")


def phase_fit(y, d, runs, n_boot=500, seed=0):
    """y ~ a + b cos d + c sin d; returns (b, c, amplitude, 95 % interval of the amplitude over runs)."""
    ok = np.isfinite(y) & np.isfinite(d)
    y, d, runs = y[ok], d[ok], runs[ok]
    X = np.column_stack([np.ones(len(y)), np.cos(d), np.sin(d)])
    c = np.linalg.lstsq(X, y, rcond=None)[0]
    u = np.unique(runs)
    idx = {r: np.nonzero(runs == r)[0] for r in u}
    rng = np.random.default_rng(seed)
    amps = []
    for _ in range(n_boot):
        pick = np.concatenate([idx[r] for r in rng.choice(u, len(u))])
        cc = np.linalg.lstsq(X[pick], y[pick], rcond=None)[0]
        amps.append(np.hypot(cc[1], cc[2]))
    # the null: the amplitude with delta shuffled across strikes (a fit always finds some amplitude)
    null = []
    for _ in range(n_boot):
        dd = rng.permutation(d)
        Xn = np.column_stack([np.ones(len(y)), np.cos(dd), np.sin(dd)])
        cc = np.linalg.lstsq(Xn, y, rcond=None)[0]
        null.append(np.hypot(cc[1], cc[2]))
    return c[1], c[2], float(np.hypot(c[1], c[2])), np.percentile(amps, [2.5, 97.5]), float(np.percentile(null, 95)), len(y)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", nargs="?")
    ap.add_argument("--model", action="append", default=[])
    ap.add_argument("--pitch", type=int, nargs=2, default=(36, 84))
    ap.add_argument("--k", type=float, nargs="+", default=(1.0, 0.5))
    ap.add_argument("--bench", default="runs/measurements/bench_2018.json")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if args.selftest:
        selftest()
        if not args.model:
            return
    import torch

    from pianonn import measures as M
    from pianonn.config import year_to_condition
    from pianonn.render import load_variant

    os.makedirs(args.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.6)
    bench = json.load(open(args.bench))
    runs = [r for r in bench["repeats"] if args.pitch[0] <= r["pitch"] <= args.pitch[1]]
    models = [load_variant(s, device=dev) for s in args.model]
    cfg = models[0][1].cfg
    sr = cfg.sample_rate
    events = [{"piece": r["piece"], "onset": r["times"][0], "pitch": r["pitch"]} for r in runs]
    clips = M.render_clips(args.data, events, models, cfg, dev, pre=PRE, post=POST, batch=8)
    table = M.partial_table(models[0][1], year_to_condition(2018), dev)
    rows = []
    for i, r in enumerate(runs):
        tab = table[r["pitch"] - 21]
        tab = tab[tab > 0]
        for s in clips:
            x = clips[s][i].mean(1)
            for k_i, tk in enumerate(r["times"]):
                if k_i == 0:
                    continue
                t = PRE + tk - r["times"][0]
                nxt = (r["times"][k_i + 1] - tk) if k_i + 1 < len(r["times"]) else 0.6
                for k in args.k:
                    m = strike(x, sr, t, nxt, float(tab[0]), tab, k)
                    if m is None:
                        continue
                    rows.append({"run": i, "k_i": k_i, "src": s, "k": k, "pitch": r["pitch"], "vel": r["velocities"][k_i],
                                 "vprev": r["velocities"][k_i - 1], "gap": tk - r["times"][k_i - 1], **m})
    with open(os.path.join(args.out, "strikes.json"), "w") as f:
        json.dump(rows, f)
    srcs = list(clips)
    L = [f"# Re-strikes against the phase of the ringing string ({len(runs)} runs, MIDI {args.pitch[0]}-{args.pitch[1]})", "",
         "delta: where the old vibration was against the blow's own fundamental (0 adding, ±180° against). Dependences: "
         "the amplitude (dB) of a + b cos(delta) + c sin(delta) fitted to each quantity's residual after velocity, "
         "previous velocity, log gap and pitch; 95 % interval over runs; 'null 95 %': the amplitude a fit finds with "
         "delta shuffled (anything under it is no dependence).", ""]
    for k in args.k:
        L += [f"## k = {k} (the share of the old vibration the blow leaves)", "",
              "| side | n | delta: mean resultant length (0 = uniform) | mean direction (°) | highs change ~ delta: amplitude [95 %] (null 95 %), peak at (°) | "
              "blow's own fundamental ~ delta | fundamental change ~ delta |", "|---|---|---|---|---|---|---|"]
        for s in srcs:
            R = [x for x in rows if x["src"] == s and x["k"] == k and np.isfinite(x["delta"]) and np.isfinite(x["highs"])]
            if len(R) < 20:
                continue
            d = np.array([x["delta"] for x in R])
            Z = np.column_stack([np.ones(len(R)), [x["vel"] for x in R], [x["vprev"] for x in R],
                                 np.log([x["gap"] for x in R]), [x["pitch"] for x in R]]).astype(float)
            res = lambda y: y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]  # noqa: E731
            run_id = np.array([x["run"] for x in R])
            out = []
            for key in ("highs", "new", "chg"):
                y = np.array([x["after"] - x["old"] if key == "chg" else x[key] for x in R])
                b, c, amp, ci, null, n = phase_fit(res(y), d, run_id)
                out.append(f"{amp:.2f} [{ci[0]:.2f}, {ci[1]:.2f}] ({null:.2f}), {np.degrees(np.arctan2(c, b)):+.0f}°")
            mrl = abs(np.mean(np.exp(1j * d)))
            L.append(f"| {s} | {len(R)} | {mrl:.2f} | {np.degrees(np.angle(np.mean(np.exp(1j * d)))):+.0f} | " + " | ".join(out) + " |")
        L.append("")
        # histogram of delta
        L += ["| side | " + " | ".join(f"{a}..{a + 45}°" for a in range(-180, 180, 45)) + " |", "|---|" + "---|" * 8]
        for s in srcs:
            d = np.degrees([x["delta"] for x in rows if x["src"] == s and x["k"] == k and np.isfinite(x["delta"])])
            h, _ = np.histogram(d, np.arange(-180, 181, 45))
            L.append(f"| {s} | " + " | ".join(str(v) for v in h) + " |")
        L.append("")
    with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
