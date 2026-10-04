"""Where do the model's notes fall outside what real notes do? Fine structure of lower-tenor notes, recording against
model as distributions, not note by note (docs/physics_revamp.md 13).

    python scripts/note_realism.py data/maestro24k --out runs/physics_revamp/attack/realism \\
        --model main=runs/physics_revamp/main/residual/last.pt:residual

Notes: MIDI ``--pitch``, velocity ``--velocity``, no other onset ``--before`` s before or ``--after`` s after, sounding
(key held or pedal down) to ``--after``; up to ``--max-notes`` drawn at random. Rendered in context
(``measures.render_clips``). Per note and side, at the side's own onset N0:
- ``cycle r``: the correlation of each period of the waveform with the next, 5-40 ms (mono; raw, and high-passed at 800
  Hz): a tone that repeats exactly reads 1;
- per partial 1-``K`` over 0.1-0.6 s (one Hann window, zero-padded x16): its frequency's deviation from k f1 (cents),
  the peak's width at -6 dB re a single stationary line's in the same window (a cluster of close lines, or a line that
  moves, is wider), and the inter-channel coherence (8 frames);
- per partial, the fluctuation (dB rms) around a quadratic over 0.08-0.6 s, 20 ms frames;
- per partial, how its left/right relation moves within the note (40 ms frames every 10 ms over 0.08-0.6 s): the sd of
  the inter-channel level difference and the circular sd of the inter-channel phase difference. A partial whose
  components reach both microphones by one fixed path keeps both constant; components reaching them in different
  proportions, beating, move them.
Writes ``report.md`` (per feature and partial group: the recordings' 10/50/90 % points, the model's median and the
share of model notes outside the recordings' 10-90 % range, 20 % expected), ``notes.npz``.
``--selftest``: the cycle and width measures on synthetic notes with known answers.
"""

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np  # noqa: E402

PRE, POST = 0.5, 0.75
K = 12
GROUPS = ((1, 1), (2, 3), (4, 6), (7, 9), (10, 12))


def cycle_r(x, sr, t_on, f0, a=0.005, b=0.04, hp=None):
    """Mean correlation of consecutive periods of mono ``x`` over [t_on + a, t_on + b]."""
    y = x
    if hp:
        X = np.fft.rfft(y)
        f = np.fft.rfftfreq(len(y), 1 / sr)
        X[f < hp] = 0
        y = np.fft.irfft(X, len(y))
    P = sr / f0
    out = []
    t = t_on + a
    while t + 2 / f0 <= t_on + b:
        i0 = int(round(t * sr))
        n = int(round(P))
        u, v = y[i0: i0 + n], y[i0 + int(round(P)): i0 + int(round(P)) + n]
        # allow +-2 samples of drift
        best = max(np.corrcoef(u, y[i0 + int(round(P)) + s: i0 + int(round(P)) + s + n])[0, 1] for s in range(-2, 3))
        out.append(best)
        t += 1 / f0
    return float(np.mean(out)) if out else np.nan


def partial_fine(xs, sr, t_on, f0, B=2.5e-4, a=0.1, b=0.6):
    """Per partial: frequency (Hz), deviation from k f1 (cents), -6 dB width re a stationary line's, coherence."""
    x = xs.mean(1)
    i0, i1 = int(round((t_on + a) * sr)), int(round((t_on + b) * sr))
    seg = x[i0:i1]
    w = np.hanning(len(seg))
    nf = 1 << int(math.ceil(math.log2(len(seg) * 16)))
    P = np.abs(np.fft.rfft(seg * w, nf)) ** 2
    f = np.fft.rfftfreq(nf, 1 / sr)
    # the width of a single stationary line in this window: -6 dB full width of the Hann lobe
    ref = np.abs(np.fft.rfft(np.cos(2 * np.pi * 1000.0 * np.arange(len(seg)) / sr) * w, nf)) ** 2
    j = np.argmax(ref)
    lw = np.sum(ref[max(0, j - 400): j + 400] >= ref[j] / 4) * (f[1] - f[0])
    # coherence: 8 frames over the window
    nfr = len(seg) // 4
    frames = [(i0 + q * nfr // 2, i0 + q * nfr // 2 + nfr) for q in range(7)]
    fr_f = np.fft.rfftfreq(nfr * 4, 1 / sr)
    S = [np.fft.rfft(xs[s:e] * np.hanning(nfr)[:, None], nfr * 4, axis=0) for s, e in frames]
    freq, cents, width, coh = (np.full(K, np.nan) for _ in range(4))
    for k in range(1, K + 1):
        fk = k * f0 * math.sqrt(1 + B * k * k)
        band = (f > fk * 0.985) & (f < fk * 1.015)
        if not band.any() or fk > 0.45 * sr:
            continue
        idx = np.nonzero(band)[0]
        jj = idx[np.argmax(P[idx])]
        pk = P[jj]
        lo = jj
        while lo > idx[0] and P[lo] >= pk / 4:
            lo -= 1
        hi = jj
        while hi < idx[-1] and P[hi] >= pk / 4:
            hi += 1
        freq[k - 1] = f[jj]
        width[k - 1] = (hi - lo) * (f[1] - f[0]) / lw
        bb = np.argmin(np.abs(fr_f - f[jj]))
        L = np.array([s[bb, 0] for s in S])
        R = np.array([s[bb, 1] for s in S])
        coh[k - 1] = np.abs(np.sum(L * np.conj(R))) ** 2 / (np.sum(np.abs(L) ** 2) * np.sum(np.abs(R) ** 2) + 1e-30)
    cents = 1200 * np.log2(freq / (np.arange(1, K + 1) * freq[0]))
    return freq, cents, width, coh


def fluct(x, sr, t_on, freqs, a=0.08, b=0.6):
    from pianonn import measures as M

    t, L = M.partial_tracks(x, sr, t_on + a, t_on + b, freqs, 0.02, 0.005)
    out = np.full(K, np.nan)
    for k in range(min(K, L.shape[1])):
        if not np.all(np.isfinite(L[:, k])):
            continue
        r = L[:, k] - np.polyval(np.polyfit(t, L[:, k], 2), t)
        out[k] = np.sqrt(np.mean(r ** 2))
    return out


def lr_motion(xs, sr, t_on, freqs, a=0.08, b=0.6, win=0.04, hop=0.01):
    """Per partial: how much its left/right relation moves within the note: the sd over 40 ms frames of the
    inter-channel level difference (dB) and the circular sd of the inter-channel phase difference (degrees)."""
    n = int(round(win * sr))
    w = np.hanning(n)
    centres = np.arange(t_on + a, t_on + b, hop)
    lev, ph = np.full(K, np.nan), np.full(K, np.nan)
    for k in range(min(K, len(freqs))):
        f = freqs[k]
        if not np.isfinite(f):
            continue
        E = np.exp(-2j * np.pi * f * np.arange(n) / sr) * w
        A = np.array([[np.sum(xs[int(round(c * sr)) - n // 2: int(round(c * sr)) - n // 2 + n, ch] * E) for ch in (0, 1)]
                      for c in centres if int(round(c * sr)) - n // 2 >= 0 and int(round(c * sr)) + n // 2 < len(xs)])
        if len(A) < 10:
            continue
        r = A[:, 0] / (A[:, 1] + 1e-12)
        lev[k] = np.std(20 * np.log10(np.abs(r) + 1e-12))
        u = np.exp(1j * np.angle(r))
        ph[k] = np.degrees(np.sqrt(-2 * np.log(max(abs(np.mean(u)), 1e-6))))
    return lev, ph


def selftest(sr=24000):
    t = np.arange(int(1.0 * sr)) / sr
    f0 = 196.0
    on = 0.3
    env = np.where(t >= on, np.exp(-2 * (t - on)), 0)
    clean = sum(np.cos(2 * np.pi * k * f0 * t) / k for k in range(1, 12)) * env
    rng = np.random.default_rng(0)
    jitter = clean * (1 + 0.3 * np.interp(t, np.linspace(0, 1, 400), rng.normal(0, 1, 400)))
    print(f"selftest cycle r: steady tone {cycle_r(clean, sr, on, f0):.3f} (1), amplitude-jittered {cycle_r(jitter, sr, on, f0):.3f} (< 1)")
    one = np.stack([np.cos(2 * np.pi * 3 * f0 * t) * env] * 2, 1)
    two = np.stack([(np.cos(2 * np.pi * 3 * f0 * t) + np.cos(2 * np.pi * (3 * f0 + 3.0) * t)) * env] * 2, 1)
    w1 = partial_fine(one, sr, on, f0, B=0)[2][2]
    w2 = partial_fine(two, sr, on, f0, B=0)[2][2]
    print(f"selftest width: one line {w1:.2f} (1), two lines 3 Hz apart {w2:.2f} (> 1)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", nargs="?")
    ap.add_argument("--model", action="append", default=[])
    ap.add_argument("--year", type=int, default=2018)
    ap.add_argument("--pitch", type=int, nargs=2, default=(45, 60))
    ap.add_argument("--velocity", type=int, nargs=2, default=(30, 85))
    ap.add_argument("--before", type=float, default=0.25)
    ap.add_argument("--after", type=float, default=0.65)
    ap.add_argument("--max-notes", type=int, default=600)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if args.selftest:
        selftest()
        if not args.model:
            return
    import torch

    from note_profile import mine
    from pianonn import measures as M
    from pianonn.config import year_to_condition
    from pianonn.losses import ONSET_DELAY_MS
    from pianonn.render import load_variant

    os.makedirs(args.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.6)
    ev = [e for e in mine(args.data, args.year, args.pitch, args.velocity, args.before, args.after) if e["state"] != "released"]
    rng = np.random.default_rng(0)
    ev = [ev[i] for i in sorted(rng.permutation(len(ev))[: args.max_notes])]
    models = [load_variant(s, device=dev) for s in args.model]
    labels = [m[0] for m in models]
    cfg = models[0][1].cfg
    sr = cfg.sample_rate
    print(f"{len(ev)} notes", flush=True)
    clips = M.render_clips(args.data, ev, models, cfg, dev, pre=PRE, post=POST, batch=8)
    table = M.partial_table(models[0][1], year_to_condition(args.year), dev)
    a, b, c = ONSET_DELAY_MS
    srcs = ["recording"] + labels
    F = {s: {k: [] for k in ("cyc", "cyc_hp", "cents", "width", "coh", "fluct", "lr_level", "lr_phase")} for s in srcs}
    for i, e in enumerate(ev):
        tab = table[e["pitch"] - 21]
        f0 = float(tab[0])
        t_ref = PRE + 1e-3 * max(a + b * (e["pitch"] - 60) + c * (e["velocity"] - 64), 0.0)
        for s in srcs:
            xs = clips[s][i]
            x = xs.mean(1)
            t_on = M.onset(x, sr, t_ref, f0)
            t_on = t_on if np.isfinite(t_on) else t_ref
            fr, cents, width, coh = partial_fine(xs, sr, t_on, f0)
            F[s]["cyc"].append(cycle_r(x, sr, t_on, f0))
            F[s]["cyc_hp"].append(cycle_r(x, sr, t_on, f0, hp=800.0))
            F[s]["cents"].append(cents)
            F[s]["width"].append(width)
            F[s]["coh"].append(coh)
            F[s]["fluct"].append(fluct(xs, sr, t_on, np.where(np.isfinite(fr), fr, tab[:K])))
            lv_, ph_ = lr_motion(xs, sr, t_on, np.where(np.isfinite(fr), fr, tab[:K]))
            F[s]["lr_level"].append(lv_)
            F[s]["lr_phase"].append(ph_)
    F = {s: {k: np.array(v) for k, v in d.items()} for s, d in F.items()}
    np.savez_compressed(os.path.join(args.out, "notes.npz"), **{f"{s}|{k}": v for s, d in F.items() for k, v in d.items()},
                        pitch=np.array([e["pitch"] for e in ev]), velocity=np.array([e["velocity"] for e in ev]))
    with open(os.path.join(args.out, "events.json"), "w") as f:
        json.dump(ev, f)
    L = [f"# Fine structure of MIDI {args.pitch[0]}-{args.pitch[1]} notes, velocity {args.velocity[0]}-{args.velocity[1]} "
         f"({args.year}, {len(ev)} notes)", "",
         "Recordings: 10 / 50 / 90 % points over notes; each model: its median and the share of its notes outside the "
         "recordings' 10-90 % range (20 % expected if alike), with the side it falls out on (low / high).", ""]

    def row(name, r, ms):
        r = r[np.isfinite(r)]
        if len(r) < 20:
            return
        q10, q50, q90 = np.percentile(r, [10, 50, 90])
        cells = []
        for s, m in ms:
            m = m[np.isfinite(m)]
            cells.append(f"{np.median(m):+.3g} | {100 * np.mean(m < q10):.0f} / {100 * np.mean(m > q90):.0f}")
        L.append(f"| {name} | {q10:+.3g} / {q50:+.3g} / {q90:+.3g} | " + " | ".join(cells) + " |")

    hdr = "| feature | recordings 10 / 50 / 90 % | " + " | ".join(f"{s} median | % low / high" for s in labels) + " |"
    sep = "|---|---|" + "---|---|" * len(labels)
    L += [hdr, sep]
    row("cycle r, 5-40 ms (raw)", F["recording"]["cyc"], [(s, F[s]["cyc"]) for s in labels])
    row("cycle r, 5-40 ms (> 800 Hz)", F["recording"]["cyc_hp"], [(s, F[s]["cyc_hp"]) for s in labels])
    for key, nm in (("cents", "deviation from k f1 (cents)"), ("width", "peak width re one line"), ("coh", "coherence L/R"),
                    ("fluct", "fluctuation (dB rms)"), ("lr_level", "L/R level difference, sd within the note (dB)"),
                    ("lr_phase", "L/R phase difference, circular sd within the note (deg)")):
        for g in GROUPS:
            if key == "cents" and g == (1, 1):
                continue
            sl = slice(g[0] - 1, g[1])
            with np.errstate(all="ignore"):
                r = np.nanmean(F["recording"][key][:, sl], 1)
                ms = [(s, np.nanmean(F[s][key][:, sl], 1)) for s in labels]
            row(f"{nm}, partials {g[0]}-{g[1]}", r, ms)
    with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
