"""Measure what the model actually does on isolated notes, for comparison with published piano data.

    python -m pianonn.diagnostics [--ckpt runs/x/last.pt] [--out report.md]

Per key/velocity: measured partial frequencies (as effective inharmonicity), partial
levels relative to the fundamental, early/late decay of the fundamental (double decay),
T60 of low partials, spectral centroid, peak level, and the time for a released
(undamped -> damped) note to drop 60 dB.
"""

import argparse

import numpy as np
import torch

from .render import load_model

KEYS = {"A0": 21, "C2": 36, "C4": 60, "A4": 69, "C6": 84, "C8": 108}
VELOCITIES = (40, 80, 120)


def _perf(model, n, pitch, velocity, offset):
    F = model.n_frames(n)
    z = torch.zeros(1, F)
    return {"pitch": torch.tensor([[pitch]]), "onset": torch.zeros(1, 1), "offset": torch.tensor([[offset]]),
            "velocity": torch.tensor([[float(velocity)]]), "mask": torch.ones(1, 1, dtype=torch.bool),
            "condition": torch.tensor([9]), "sustain": z, "soft": z, "sostenuto": z}


def _partial_track(x, sr, f, t0, t1, win=0.1):
    """Level (dB) of a narrow band around f over time via a heterodyne + moving average."""
    t = np.arange(len(x)) / sr
    z = x * np.exp(-2j * np.pi * f * t)
    k = max(1, int(win * sr))
    env = np.abs(np.convolve(z, np.ones(k) / k, mode="same"))
    sel = (t >= t0) & (t < t1)
    return t[sel], 20 * np.log10(env[sel] + 1e-12)


def _slope_db_per_s(t, db):
    return np.polyfit(t, db, 1)[0] if len(t) > 2 else float("nan")


@torch.no_grad()
def measure_note(model, pitch, velocity, seconds=6.0, release=None):
    cfg = model.cfg
    sr = cfg.sample_rate
    n = int(seconds * sr)
    out = model(_perf(model, n, pitch, velocity, release if release is not None else seconds), n,
                generator=torch.Generator().manual_seed(0))
    x = out["audio"][0].numpy().astype(np.float64)
    strings = out["strings"][0].numpy().astype(np.float64)

    ki = torch.tensor([[pitch - 21]])
    modes = model.physics.modes(ki, torch.tensor([[velocity / 127]]), torch.zeros(1, 1), torch.tensor([9]))
    f_model = modes["freq"][0, 0, :, 0].numpy()

    # measured partial frequencies from a long FFT of the string signal
    seg = strings[int(0.05 * sr): int(min(seconds, 3.0) * sr)]
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), 8 * len(seg)))
    hz = np.fft.rfftfreq(8 * len(seg), 1 / sr)
    meas, levels = [], []
    for fk in f_model[:10]:
        if fk > 0.45 * sr:
            break
        band = (hz > fk * 0.985) & (hz < fk * 1.015)
        i = np.argmax(spec[band])
        meas.append(hz[band][i])
        levels.append(20 * np.log10(spec[band][i] + 1e-12))
    meas, levels = np.array(meas), np.array(levels) - levels[0]
    f1 = meas[0]
    B_eff = np.nan
    if len(meas) >= 5:  # fit f_n = n f0 sqrt(1 + B n^2)
        nn = np.arange(1, len(meas) + 1)
        y = (meas / (nn * f1)) ** 2
        B_eff = max(0.0, np.polyfit(nn**2 - 1, y - 1, 1)[0])

    t_e, db_e = _partial_track(x, sr, f1, 0.05, 0.6)
    t_l, db_l = _partial_track(x, sr, f1, 2.0, seconds - 0.2)
    early, late = _slope_db_per_s(t_e, db_e), _slope_db_per_s(t_l, db_l)

    frame = x[int(0.02 * sr): int(0.12 * sr)]
    s = np.abs(np.fft.rfft(frame * np.hanning(len(frame))))
    centroid = float((s * np.fft.rfftfreq(len(frame), 1 / sr)).sum() / (s.sum() + 1e-12))
    return {"f1": f1, "B_eff": B_eff, "levels": levels, "early_db_s": early, "late_db_s": late,
            "t60_early": -60 / early if early < 0 else np.inf, "t60_late": -60 / late if late < 0 else np.inf,
            "centroid": centroid, "peak_db": 20 * np.log10(np.abs(x).max() + 1e-12)}


@torch.no_grad()
def release_time(model, pitch, velocity=80, hold=1.0, seconds=3.0):
    """Seconds after key release (no pedal) for the note's RMS to fall 60 dB below its level at release."""
    sr = model.cfg.sample_rate
    n = int(seconds * sr)
    x = model(_perf(model, n, pitch, velocity, hold), n, generator=torch.Generator().manual_seed(0))["strings"][0]
    x = x.numpy()
    k = int(0.02 * sr)
    rms = np.sqrt(np.convolve(x**2, np.ones(k) / k, mode="same")) + 1e-12
    i0 = int(hold * sr)
    ref = rms[i0 - k]
    below = np.nonzero(rms[i0:] < ref * 1e-3)[0]
    return below[0] / sr if len(below) else np.inf


def report(model):
    lines = ["| key | vel | f1 Hz | B_eff | partials 2..6 dB re f1 | T60 early s | T60 late s | centroid Hz | peak dBFS |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, pitch in KEYS.items():
        for v in VELOCITIES:
            m = measure_note(model, pitch, v)
            lv = " ".join(f"{d:+.0f}" for d in m["levels"][1:6])
            lines.append(f"| {name} | {v} | {m['f1']:.2f} | {m['B_eff']:.2e} | {lv} | {m['t60_early']:.1f} | "
                         f"{m['t60_late']:.1f} | {m['centroid']:.0f} | {m['peak_db']:.1f} |")
    lines += ["", "| key | damper release to -60 dB (s) |", "|---|---|"]
    for name, pitch in KEYS.items():
        lines.append(f"| {name} | {release_time(model, pitch):.2f} |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    text = report(load_model(args.ckpt))
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
