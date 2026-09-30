"""Attacks of isolated notes, recording against model.

    python scripts/measure_attack.py data/maestro24k --years 2018 --split test validation --out runs/round2/attack.md \\
        --model control=runs/round2/b_control/last.pt:physics --model residual=runs/round2/a_residual/last.pt:residual

Round 2's listening: the attack is off in the middle and upper registers, and the string sounds plucked. Notes
of MIDI 36-87 with no other strike 0.3 s before or 0.65 s after (as ``scripts/measure_phantoms.py``) are
rendered by each model in their context (12 s lookback, 1 s warm-up). Each signal is cut at its own detected
onset. Per note, channels averaged, high-passed at 20 Hz:

1. **Shape**: octave-band energy in the attack (-2..28 ms), early (30..100 ms) and sustain (100..400 ms)
   windows, relative to the same signal's sustain energy (0.2-8 kHz). The per-note level drops out, so
   model - recording is the error in the attack's spectrum and decay alone.
2. **Between the partials**: the share of energy more than a tolerance from every partial, in the attack
   (40 ms window; 1-8 kHz) and the sustain (300 ms window; 0.2-8 kHz). Knock, noise and sympathetic
   resonance sit there; the string does not.
3. **Level**: sustain energy, model - recording.

Medians over notes by register; pedal-up and pedal-down notes apart.
"""

import argparse
import math
import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.6,max_split_size_mb:256")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json  # noqa: E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

from pianonn.config import year_to_condition  # noqa: E402
from pianonn.data import _perf_from_notes, history_frames  # noqa: E402
from pianonn.render import load_model  # noqa: E402
from pianonn.train import collate, to_device  # noqa: E402

REGISTERS = ((36, 50), (51, 62), (63, 74), (75, 87))
BANDS = (500, 1000, 2000, 4000, 8000)  # octave bands
WINDOWS = {"attack": (-0.002, 0.028), "early": (0.030, 0.100), "sustain": (0.100, 0.400)}
WARMUP, BEFORE, AFTER = 1.0, 0.3, 0.7  # s: model warm-up, then the analysed clip around the MIDI onset


def isolated(notes, before=0.3, clear=0.65, lo=36, hi=87):
    on = notes[:, 1]
    keep = []
    for i in range(len(notes)):
        if lo <= notes[i, 0] <= hi and on[i] > WARMUP + BEFORE + 0.1:
            near = (on > on[i] - before) & (on < on[i] + clear)
            if near.sum() == 1:
                keep.append(i)
    return keep


def detect_onset(x, sr, t_midi):
    """Onset (s) of the mono clip ``x``: first 1 ms frame above 30 % of the way from the level before the note to
    the peak, above 500 Hz, searched from 40 ms before to 60 ms after the MIDI onset."""
    X = np.fft.rfft(x)
    X[np.fft.rfftfreq(len(x), 1 / sr) < 500] = 0
    y = np.fft.irfft(X, len(x))
    hop = sr // 1000
    env = np.sqrt(np.mean(y[: len(y) // hop * hop].reshape(-1, hop) ** 2, 1))
    i0, i1 = int((t_midi - 0.04) * 1000), int((t_midi + 0.06) * 1000)
    base = np.median(env[max(0, i0 - 110): max(1, i0 - 10)])
    peak = env[i0:i1].max()
    k = np.nonzero(env[i0:i1] > base + 0.3 * (peak - base))[0]
    return (i0 + (k[0] if len(k) else 0)) / 1000


def band_energy(x, sr, t0, t1, lo, hi):
    seg = x[int(t0 * sr): int(t1 * sr)]
    P = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
    f = np.fft.rfftfreq(len(seg), 1 / sr)
    return float(P[(f >= lo) & (f < hi)].sum()) + 1e-30


def between_share(x, sr, t0, t1, lo, hi, partials, tol):
    seg = x[int(t0 * sr): int(t1 * sr)]
    n = 4 * len(seg)
    P = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), n)) ** 2
    f = np.fft.rfftfreq(n, 1 / sr)
    sel = (f >= lo) & (f < hi)
    d = np.min(np.abs(f[sel][:, None] - partials[None, :]), 1) if len(partials) else np.full(sel.sum(), np.inf)
    return 10 * math.log10((P[sel][d > tol].sum() + 1e-30) / (P[sel].sum() + 1e-30))


def features(x, sr, t_on, partials, f0):
    out = {}
    ref = band_energy(x, sr, t_on + 0.100, t_on + 0.400, 200, 8000)
    for w, (a, b) in WINDOWS.items():
        for c in BANDS:
            out[f"{w} {c}"] = 10 * math.log10(band_energy(x, sr, t_on + a, t_on + b, c / math.sqrt(2), c * math.sqrt(2)) / ref)
    out["between attack"] = between_share(x, sr, t_on - 0.005, t_on + 0.035, 1000, 8000, partials, max(55.0, 0.2 * f0))
    out["between sustain"] = between_share(x, sr, t_on + 0.100, t_on + 0.400, 200, 8000, partials, max(15.0, 0.05 * f0))
    out["level"] = 10 * math.log10(ref)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data")
    ap.add_argument("--years", type=int, nargs="*", default=[2018])
    ap.add_argument("--split", nargs="*", default=["test"])
    ap.add_argument("--model", action="append", required=True, help="label=checkpoint:physics|residual")
    ap.add_argument("--max-notes", type=int, default=600)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.6)
    specs = [(m.split("=", 1)[0], *m.split("=", 1)[1].rsplit(":", 1)) for m in args.model]
    models = {label: load_model(ckpt, device=dev) for label, ckpt, _ in specs}
    cfg = next(iter(models.values())).cfg
    sr, lookback = cfg.sample_rate, 12.0

    with open(os.path.join(args.data, "index.json")) as f:
        pieces = [p for p in json.load(f) if p["split"] in args.split and p["year"] in args.years]
    jobs = []
    for p in pieces:
        z = dict(np.load(os.path.join(args.data, p["midi"])))
        for i in isolated(z["notes"]):
            jobs.append((p, z, i))
    rng = np.random.default_rng(0)
    if len(jobs) > args.max_notes:
        jobs = [jobs[k] for k in sorted(rng.choice(len(jobs), args.max_notes, replace=False))]
    print(len(jobs), "isolated notes from", len(pieces), "pieces", flush=True)

    # partial frequencies per key (frozen at the measurements, the same in every round-2 model)
    m0 = next(iter(models.values()))
    with torch.no_grad():
        ki = torch.arange(88, device=dev)[None]
        half = torch.full(ki.shape, 0.6, device=dev)
        cond = torch.tensor([year_to_condition(args.years[0])], device=dev)
        freq = m0.physics.modes(ki, half, torch.zeros_like(half), cond, phantoms=False)["freq"][0, :, :, 0].cpu().numpy()

    n_win = WARMUP + BEFORE + AFTER
    rows = []
    for s in range(0, len(jobs), 16):
        chunk = jobs[s: s + 16]
        items, clips = [], []
        for p, z, i in chunk:
            t_note = float(z["notes"][i, 1])
            t0 = t_note - BEFORE - WARMUP
            n = int(round(n_win * sr))
            x, _ = sf.read(os.path.join(args.data, p["audio"]), start=int(round(t0 * sr)), frames=n, dtype="float32", always_2d=True)
            x = np.pad(x, ((0, n - len(x)), (0, 0))).T
            perf = _perf_from_notes(z["notes"], z, t0, n / sr, lookback, n // cfg.hop + 2, cfg, history_frames(lookback, cfg))
            perf["condition"] = torch.tensor(year_to_condition(p["year"]))
            perf["audio"] = torch.from_numpy(np.ascontiguousarray(x))
            perf["loss_start"] = torch.tensor(int(WARMUP * sr))
            items.append(perf)
            k = np.searchsorted(z["sustain_t"], t_note, side="right") - 1
            clips.append({"pitch": int(z["notes"][i, 0]), "velocity": int(z["notes"][i, 3]),
                          "pedal": bool(k >= 0 and z["sustain_v"][k] >= 64)})
        b = to_device(collate(items), dev)
        n = b["audio"].shape[-1]
        s0 = int(WARMUP * sr)
        sigs = {"recording": b["audio"][..., s0:].mean(1).cpu().numpy()}
        for label, _, mode in specs:
            with torch.no_grad():
                y = models[label](b, n, residual=mode == "residual", generator=torch.Generator(device=dev).manual_seed(0))["audio"]
            sigs[label] = y[..., s0:].mean(1).float().cpu().numpy()
        for j, c in enumerate(clips):
            pk = freq[c["pitch"] - 21]
            f0 = float(pk[0])
            r = dict(c)
            for name, X in sigs.items():
                x = X[j] - np.convolve(X[j], np.ones(int(sr / 20)) / int(sr / 20), "same")  # ~20 Hz high-pass
                t_on = detect_onset(x, sr, BEFORE)
                r[name] = features(x, sr, t_on, pk[pk < 9000], f0)
                r[name]["onset_ms"] = 1000 * (t_on - BEFORE)
            rows.append(r)
        print(f"{min(s + 16, len(jobs))}/{len(jobs)}", flush=True)

    labels = [label for label, _, _ in specs]
    lines = [f"# Attacks of isolated notes: recording vs model ({len(rows)} notes, {', '.join(args.split)}, "
             f"{', '.join(map(str, args.years))})", "", "Models: " + "; ".join(f"{l} = `{c}` ({m})" for l, c, m in specs), "",
             "Shape: band energy re the same signal's sustain (0.2-8 kHz, 100-400 ms), model − recording, median dB. "
             "Negative: the model has less there, relative to its own sustain.", ""]
    for pedal in (False, True):
        sub = [r for r in rows if r["pedal"] == pedal]
        lines += [f"## Pedal {'down' if pedal else 'up'} ({len(sub)} notes)", ""]
        for label in labels:
            lines += [f"### {label}", "", "| register | notes | " + " | ".join(f"{w} {c // 1000 if c >= 1000 else c}{'k' if c >= 1000 else ''}"
                      for w in ("attack", "early") for c in BANDS) + " | sustain level |", "|---|---|" + "---|" * (2 * len(BANDS) + 1)]
            for lo, hi in REGISTERS:
                rs = [r for r in sub if lo <= r["pitch"] <= hi]
                if len(rs) < 5:
                    continue
                d = lambda k: np.median([r[label][k] - r["recording"][k] for r in rs])
                lines.append(f"| {lo}-{hi} | {len(rs)} | " + " | ".join(f"{d(f'{w} {c}'):+.1f}" for w in ("attack", "early") for c in BANDS)
                             + f" | {d('level'):+.1f} |")
            lines.append("")
        lines += ["Energy between the partials, share of the band (dB), median: recording / "
                  + " / ".join(labels), "", "| register | attack, 1-8 kHz | sustain, 0.2-8 kHz | onset offset ms (" + ", ".join(labels) + ") |",
                  "|---|---|---|---|"]
        for lo, hi in REGISTERS:
            rs = [r for r in sub if lo <= r["pitch"] <= hi]
            if len(rs) < 5:
                continue
            med = lambda name, k: np.median([r[name][k] for r in rs])
            lines.append(f"| {lo}-{hi} | " + " / ".join(f"{med(n, 'between attack'):+.1f}" for n in ["recording"] + labels)
                         + " | " + " / ".join(f"{med(n, 'between sustain'):+.1f}" for n in ["recording"] + labels)
                         + " | " + ", ".join(f"{np.median([r[n]['onset_ms'] - r['recording']['onset_ms'] for r in rs]):+.1f}" for n in labels) + " |")
        lines.append("")
    text = "\n".join(lines) + "\n"
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)
    with open(os.path.splitext(args.out)[0] + ".json", "w") as f:
        json.dump(rows, f)
    print(text)


if __name__ == "__main__":
    main()
