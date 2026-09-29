"""Measure what the model does on isolated notes and check it against docs/physical_parameters.md.

    python -m pianonn.diagnostics [--ckpt runs/x/last.pt] [--out report.md]

Prints a per-note table (effective inharmonicity, partial levels, prompt/aftersound
T60, spectral centroid, peak level) and the acceptance checks of section 7 of the
spec, each with the measured value, the target and PASS/FAIL.
"""

import argparse
import copy
import math

import numpy as np
import torch

from .render import load_model

KEYS = {"A0": 21, "C2": 36, "C4": 60, "A4": 69, "C6": 84, "C7": 96, "C8": 108}
# fundamental T60 (prompt, aftersound) ranges exactly as in docs/physical_parameters.md, section 2
T60_TARGETS = {"A0": ((25, 40), None), "C2": ((15, 20), (20, 30)), "C4": ((6, 8), (20, 35)), "A4": ((5, 6), (15, 30)),
               "C6": ((2.5, 3.5), (8, 15)), "C7": ((1.5, 2.0), (3, 6)), "C8": ((0.7, 1.2), None)}


def _variant(model, **flags):
    m = copy.copy(model)
    m.cfg = type(model.cfg)(**{**model.cfg.to_dict(), **flags})
    return m


def _perf(model, n, notes, sustain=0.0):
    F = model.n_frames(n)
    t = torch.tensor(notes, dtype=torch.float32)[None]
    return {"pitch": t[..., 0].long(), "onset": t[..., 1], "offset": t[..., 2], "velocity": t[..., 3],
            "mask": torch.ones(1, len(notes), dtype=torch.bool), "condition": torch.tensor([9]),
            "sustain": torch.full((1, F), float(sustain)), "soft": torch.zeros(1, F), "sostenuto": torch.zeros(1, F)}


@torch.no_grad()
def _render(model, seconds, notes, sustain=0.0):
    n = int(seconds * model.cfg.sample_rate)
    return {k: v[0].double().numpy() for k, v in
            model(_perf(model, n, notes, sustain), n, block_seconds=2.0, generator=torch.Generator().manual_seed(0)).items()}


def _t60(t, db):
    slope = np.polyfit(t, db, 1)[0]
    return -60 / slope if slope < 0 else np.inf


def _edc_t60(x, sr, f, lo=-3.0, hi=-13.0):
    """Prompt T60 of the partial at ``f`` from its energy decay curve (Schroeder), fitted from ``lo`` to ``hi`` dB.
    Backward integration averages out unison beating, which ruins envelope slope fits."""
    t = np.arange(len(x)) / sr
    k = max(1, int(0.02 * sr))
    p = np.abs(np.convolve(x * np.exp(-2j * np.pi * f * t), np.ones(k) / k, mode="same")) ** 2
    edc = 10 * np.log10(np.cumsum(p[::-1])[::-1] + 1e-30)
    edc -= edc[int(0.05 * sr)]
    sel = (edc <= lo) & (edc >= hi) & (t > 0.05)
    return _t60(t[sel], edc[sel]) if sel.sum() > 10 else float("nan")


def _partials(x, sr, freqs, t0=0.05, t1=2.0):
    seg = x[int(t0 * sr): int(t1 * sr)]
    n_fft = 8 * len(seg)
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), n_fft))
    hz = np.fft.rfftfreq(n_fft, 1 / sr)
    meas, lev = [], []
    for fk in freqs:
        band = (hz > fk * 0.985) & (hz < fk * 1.015)
        if fk > 0.45 * sr or not band.any():
            break
        i = np.argmax(spec[band])
        meas.append(hz[band][i])
        lev.append(20 * np.log10(spec[band][i] + 1e-15))
    return np.array(meas), np.array(lev)


def _b_eff(meas):
    n = np.arange(1, len(meas) + 1)
    y = (meas / (n * meas[0])) ** 2 - 1
    return max(0.0, np.polyfit(n**2 - 1, y, 1)[0]) if len(meas) >= 5 else float("nan")


def _centroid(x, sr, t0=0.02, t1=0.12):
    frame = x[int(t0 * sr): int(t1 * sr)]
    s = np.abs(np.fft.rfft(frame * np.hanning(len(frame))))
    return float((s * np.fft.rfftfreq(len(frame), 1 / sr)).sum() / (s.sum() + 1e-15))


def _model_freqs(model, pitch, velocity):
    ki = torch.tensor([[pitch - 21]])
    m = model.physics.modes(ki, torch.tensor([[velocity / 127]]), torch.zeros(1, 1), torch.tensor([9]))
    return m["freq"][0, 0, :12, 0].numpy()


def _release_time(x, sr, t_release):
    k = int(0.02 * sr)
    rms = np.sqrt(np.convolve(x**2, np.ones(k) / k, mode="same")) + 1e-15
    i0 = int(t_release * sr)
    below = np.nonzero(rms[i0:] < rms[i0 - k] * 1e-3)[0]
    return below[0] / sr if len(below) else np.inf


def _rms_db(x):
    return 10 * np.log10(np.mean(x**2) + 1e-30)


@torch.no_grad()
def run(model):
    sr = model.cfg.sample_rate
    dry = _variant(model, use_noise=False, use_sympathetic=False, use_room=False)
    radiated = _variant(model, use_noise=False, use_sympathetic=False)
    rows, checks = [], []

    def check(name, value, target, ok, fmt="{:.3g}"):
        checks.append((name, value if isinstance(value, str) else fmt.format(value), target, "PASS" if ok else "FAIL"))

    # --- per-note table + decay / inharmonicity checks on the dry string signal ---
    notes = {}
    for name, pitch in KEYS.items():
        for v in (40, 80, 120):
            x = _render(dry, 12.0, [(pitch, 0.0, 12.0, v)])["audio"]
            meas, lev = _partials(x, sr, _model_freqs(model, pitch, v))
            f1 = meas[0]
            after_alpha = model.physics.modes(torch.tensor([[pitch - 21]]), torch.tensor([[v / 127]]), torch.zeros(1, 1),
                                             torch.tensor([9]))["alpha"][0, 0, 0, 1].item()
            notes[(name, v)] = dict(f1=f1, B=_b_eff(meas), lev=lev - lev[0], t60p=_edc_t60(x, sr, f1),
                                    t60a=6.91 / after_alpha,
                                    centroid=_centroid(x, sr), peak=20 * np.log10(np.abs(x).max() + 1e-15))
            r = notes[(name, v)]
            rows.append(f"| {name} | {v} | {f1:.2f} | {r['B']:.2e} | {' '.join(f'{d:+.0f}' for d in r['lev'][1:6])} | "
                        f"{r['t60p']:.1f} | {r['t60a']:.1f} | {r['centroid']:.0f} | {r['peak']:.1f} |")

    for name, (prompt, after) in T60_TARGETS.items():
        r = notes[(name, 80)]
        check(f"{name} prompt T60 (s)", r["t60p"], f"{prompt[0]}-{prompt[1]}", prompt[0] <= r["t60p"] <= prompt[1])
        if after:  # analytic from the mode parameters: beating between aftersound modes defeats any fit
            check(f"{name} aftersound T60, model (s)", r["t60a"], f"{after[0]}-{after[1]}", after[0] <= r["t60a"] <= after[1])
    for name, target in (("C4", 3.8e-4), ("C6", 3.5e-3)):
        b = notes[(name, 80)]["B"]
        check(f"{name} effective B", b, f"{target:.1e} (x1.5)", target / 1.5 <= b <= target * 1.5)
    lev = notes[("C4", 80)]["lev"]
    check("C4 mf partials 2-6 re p1 (dB)", " ".join(f"{d:+.0f}" for d in lev[1:6]), "-30..+3",
          bool(np.all((lev[1:6] >= -30) & (lev[1:6] <= 3))))
    check("C4 mf partial 10 re p1 (dB)", lev[9], "<= -30", lev[9] <= -30, "{:+.0f}")
    ratio = notes[("C4", 120)]["centroid"] / notes[("C4", 40)]["centroid"]
    check("C4 centroid vel120 / vel40", ratio, "1.1-2", 1.1 <= ratio <= 2.0, "{:.2f}")
    for name in ("C2", "C4", "C6", "C7"):
        cs = [_centroid(_render(dry, 0.5, [(KEYS[name], 0.0, 0.5, v)])["audio"], sr) for v in (30, 60, 90, 120)]
        check(f"{name} brightness rises with velocity (centroid Hz, vel 30/60/90/120)", " ".join(f"{c:.0f}" for c in cs),
              "increasing", bool(np.all(np.diff(cs) > 0)))

    # --- radiated bass fundamentals (body high-pass) ---
    for name in ("A0", "C2"):
        pitch = KEYS[name]
        x = _render(radiated, 3.0, [(pitch, 0.0, 3.0, 80)])["audio"]
        _, lv = _partials(x, sr, _model_freqs(model, pitch, 80), 0.05, 2.5)
        check(f"{name} radiated fundamental below strongest partial (dB)", lv.max() - lv[0], ">= 10",
              lv.max() - lv[0] >= 10, "{:.0f}")

    # --- dampers ---
    for name, lo, hi in (("A0", 0.5, np.inf), ("A4", 0.2, 0.4)):
        x = _render(dry, 4.0, [(KEYS[name], 0.0, 1.0, 80)])["audio"]
        t = _release_time(x, sr, 1.0)
        check(f"{name} release to -60 dB (s)", t, f">= {lo}" if hi == np.inf else f"{lo}-{hi}", lo <= t <= hi, "{:.2f}")
    released = _render(dry, 1.0, [(KEYS["C8"], 0.0, 0.3, 80)])["audio"]
    held = _render(dry, 1.0, [(KEYS["C8"], 0.0, 1.0, 80)])["audio"]
    w = slice(int(0.35 * sr), int(0.6 * sr))
    d = _rms_db(released[w]) - _rms_db(held[w])
    check("C8 released vs held (dB, undamped)", d, "0 +-1", abs(d) <= 1, "{:+.1f}")

    # --- sustain-pedal halo and noise levels, full model ---
    chord = [(48, 0.0, 0.5, 90), (55, 0.0, 0.5, 90), (64, 0.0, 0.5, 90)]
    full = _variant(model, use_room=False)
    ped, noped = _render(full, 3.0, chord, sustain=1.0), _render(full, 3.0, chord, sustain=0.0)
    halo = _rms_db(ped["symp"][: 2 * sr]) - _rms_db(ped["strings"][: 2 * sr])
    check("pedal halo, symp re strings (dB)", halo, "-40..-25", -40 <= halo <= -25, "{:.0f}")
    diff = _rms_db(ped["symp"][: 2 * sr]) - _rms_db(noped["symp"][: 2 * sr])
    check("halo with vs without pedal (dB)", diff, ">= 10", diff >= 10, "{:.0f}")
    no_symp = _variant(model, use_room=False, use_sympathetic=False)
    for v, target in ((120, -25), (25, -12)):
        out = _render(no_symp, 1.0, [(60, 0.1, 0.8, v)])
        w = slice(int(0.1 * sr), int(0.16 * sr))
        lvl = _rms_db(out["noise"][w]) - _rms_db(out["strings"][w])
        check(f"knock re tone, first 60 ms, vel {v} (dB)", lvl, f"{target} +-3", abs(lvl - target) <= 3, "{:.0f}")
    out = _render(no_symp, 1.5, [(60, 0.0, 0.8, 64)])
    rel = 0.8 + model.cfg.damper_delay
    lvl = _rms_db(out["noise"][int(rel * sr): int((rel + 0.04) * sr)]) - _rms_db(out["strings"][int((rel - 0.1) * sr): int(rel * sr)])
    check("damper noise re released note (dB)", lvl, "-45..-35", -45 <= lvl <= -35, "{:.0f}")
    perf = _perf(model, 2 * sr, [(60, 0.0, 1.9, 64)])
    t = torch.arange(perf["sustain"].shape[-1]) * model.cfg.hop / sr
    perf["sustain"] = ((t > 0.5) & (t < 1.2)).float()[None]
    out = {k: v[0].double().numpy() for k, v in no_symp(perf, 2 * sr, generator=torch.Generator().manual_seed(0)).items()}
    w = slice(int(0.5 * sr), int(0.6 * sr))
    lvl = _rms_db(out["noise"][w]) - _rms_db(out["strings"][w])
    check("pedal-press noise re mf note (dB)", lvl, "-35 +-5", abs(lvl + 35) <= 5, "{:.0f}")
    x = _render(no_symp, 1.0, [(60, 0.2, 0.8, 100)])["noise"]
    pre = float((x[: int(0.2 * sr)] ** 2).sum() / ((x**2).sum() + 1e-30))
    check("noise energy before the hammer strikes (fraction)", pre, "< 0.01", pre < 0.01, "{:.3f}")

    table = ["| key | vel | f1 Hz | B_eff | partials 2..6 dB re f1 | T60 prompt s (EDC) | T60 after s (model) | centroid Hz | peak dBFS |",
             "|---|---|---|---|---|---|---|---|---|"] + rows
    acc = ["| check | measured | target | result |", "|---|---|---|---|"] + [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in checks]
    n_pass = sum(c[3] == "PASS" for c in checks)
    return "\n".join(["## Isolated notes (dry strings)", "", *table, "",
                      f"## Acceptance checks: {n_pass}/{len(checks)} pass", "", *acc])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    text = run(load_model(args.ckpt))
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
