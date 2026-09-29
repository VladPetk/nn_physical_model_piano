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
# Measured on the Iowa Steinway B with pianonn.calibration (docs/calibration_iowa.md), both channels in power,
# medians over mf + ff and +-3 semitones. Per landmark: decay profile (dB re peak of the power-summed decay partials
# at 0.5/1/2/4/8/16 s; nan = at the noise floor), early spectral slope at mf (dB/oct), tuning (cents re A4).
# A0 (2 notes) and C8 (3 notes) are not used as targets: too few, partly mislabelled recordings.
IOWA = {
    "C1": ((-3, -6, -10, -18, -27, -33), 0.7, -15.9),
    "C2": ((-5, -10, -12, -22, -24, -35), -3.5, -4.3),
    "C3": ((-5, -10, -22, -24, -32, -45), -5.3, 0.5),
    "C4": ((-12, -21, -25, -31, -45, -55), -15.4, -1.4),
    "A4": ((-15, -26, -29, -38, -53, -64), -19.2, 0.0),
    "C5": ((-20, -21, -29, -37, -53, -70), -20.3, -0.4),
    "C6": ((-15, -22, -36, -47, -69, None), -28.5, 5.3),
    "C7": ((-19, -34, -53, -66, None, None), -37.2, 14.0),
}
IOWA_PITCH = {"C1": 24, "C2": 36, "C3": 48, "C4": 60, "A4": 69, "C5": 72, "C6": 84, "C7": 96}


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
            x = _render(dry, 3.0, [(pitch, 0.0, 3.0, v)])["audio"]
            meas, lev = _partials(x, sr, _model_freqs(model, pitch, v))
            f1 = meas[0]
            notes[(name, v)] = dict(f1=f1, B=_b_eff(meas), lev=lev - lev[0],
                                    centroid=_centroid(x, sr), peak=20 * np.log10(np.abs(x).max() + 1e-15))
            r = notes[(name, v)]
            rows.append(f"| {name} | {v} | {f1:.2f} | {r['B']:.2e} | {' '.join(f'{d:+.0f}' for d in r['lev'][1:6])} | "
                        f"{r['centroid']:.0f} | {r['peak']:.1f} |")

    # --- decays, brightness and stretch vs the Iowa recordings, analysed by the same code ---
    from .calibration import PROFILE_TIMES, analyze_note, render_note

    measured = {}
    for name, p in IOWA_PITCH.items():
        r = analyze_note(render_note(model, p, 64, 20.0), sr, p)  # dry strings, 60 dB noise floor
        spec = analyze_note(render_note(model, p, 64, 1.5, body=True), sr, p)  # through the soundboard body
        r["slope_db_oct"] = spec.get("slope_db_oct", float("nan"))
        # decay profile aggregated like the recordings' targets (median over neighbouring keys and mf + ff),
        # so single-note beat nulls are averaged out on both sides
        profiles = [analyze_note(render_note(model, q, vel, 20.0), sr, q).get("profile_db")
                    for q in range(max(21, p - 3), min(108, p + 3) + 1, 2) for vel in (64, 110)]
        profiles = np.array([pr for pr in profiles if pr], dtype=float)
        r["profile_db"] = list(np.nanmedian(profiles, 0)) if len(profiles) else r.get("profile_db")
        measured[name] = r
    a4 = measured["A4"]["cents"]
    for name, (profile, slope, cents) in IOWA.items():
        r = measured[name]
        got = r.get("profile_db", [float("nan")] * len(PROFILE_TIMES))
        pairs = [(t, g, w) for t, g, w in zip(PROFILE_TIMES, got, profile) if w is not None and np.isfinite(g)]
        dev = max(abs(g - w) for _, g, w in pairs) if pairs else float("inf")
        shown = " ".join(f"{g:.0f}/{w}" for _, g, w in pairs)
        check(f"{name} decay profile (median, +-3 keys, mf+ff), model/recording dB at {'/'.join(f'{t:g}' for t, _, _ in pairs)} s", shown,
              "each +-6 dB", dev <= 6)
        v = r["slope_db_oct"]
        check(f"{name} early spectral slope, mf, radiated (dB/oct)", v, f"{slope} +-5", abs(v - slope) <= 5, "{:.1f}")
        v = r["cents"] - a4
        check(f"{name} tuning re A4 (cents) [regression: the prior copies these values]", v, f"{cents:+.1f} +-4",
              abs(v - cents) <= 4, "{:+.1f}")

    for name, target in (("C4", 3.3e-4), ("C6", 2.85e-3)):
        b = notes[(name, 80)]["B"]
        check(f"{name} effective B vs Rigaud et al. [regression: the prior is this curve]", b, f"{target:.2e} (x1.5)",
              target / 1.5 <= b <= target * 1.5)
    for name, target in (("C2", 1.17e-4), ("C4", 3.26e-4)):  # measured on the Iowa Steinway B (independent of the prior)
        b = measured[name]["B"]
        check(f"{name} effective B vs Iowa (x2: bass is piano-specific)", b, f"{target:.2e} (x2)", target / 2 <= b <= target * 2)
    lev = notes[("C4", 80)]["lev"]  # v1 memory-based sanity ranges, kept (tag M)
    check("C4 vel 80 partials 2-6 re p1 (dB) [M]", " ".join(f"{d:+.0f}" for d in lev[1:6]), "-30..+3",
          bool(np.all((lev[1:6] >= -30) & (lev[1:6] <= 3))))
    check("C4 vel 80 partial 10 re p1 (dB) [M]", lev[9], "<= -30", lev[9] <= -30, "{:+.0f}")
    # Hall, Five Lectures, Fig. 15: C4 spectral slope over partials 1-10 at pp / mf / ff = -18 / -15 / -11 dB/oct.
    # The velocity law (HAMMER_ORDER_VEL) is still Hall's, so its steps are a regression check. The absolute
    # slope now comes from the Iowa attack spectra (fit_spectra.py), which make C4 ~3-4 dB/oct flatter than
    # Hall's piano: checked against Hall with the tolerance of the Iowa slope checks.
    ki = torch.tensor([[39]])
    slopes = {}
    for vel in (30, 64, 110):
        a = model.physics.modes(ki, torch.tensor([[vel / 127]]), torch.zeros(1, 1), torch.tensor([9]))["amp"][0, 0, :10, 0].abs().numpy()
        slopes[vel] = np.polyfit(np.log2(np.arange(1, 11)), 20 * np.log10(a / a[0] + 1e-12), 1)[0]
    check("C4 spectral slope, partials 1-10, vel 64 (dB/oct) vs Hall (another piano)", slopes[64], "-15 +-5",
          abs(slopes[64] + 15) <= 5, "{:.1f}")
    for (v0, v1), target in (((30, 64), 3.0), ((64, 110), 4.0)):
        d = slopes[v1] - slopes[v0]
        check(f"C4 slope step vel {v0} -> {v1} (dB/oct) [regression: velocity law from Hall]", d, f"+{target:.0f} +-1.5",
              abs(d - target) <= 1.5, "{:+.1f}")
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
    # the halo is checked even when the model has the bank switched off, so the prior stays valid
    chord = [(48, 0.0, 0.5, 90), (55, 0.0, 0.5, 90), (64, 0.0, 0.5, 90)]
    full = _variant(model, use_room=False, use_sympathetic=True)
    ped, noped = _render(full, 3.0, chord, sustain=1.0), _render(full, 3.0, chord, sustain=0.0)
    off = "" if model.cfg.use_sympathetic else " [bank disabled in this model]"
    halo = _rms_db(ped["symp"][: 2 * sr]) - _rms_db(ped["strings"][: 2 * sr])
    check("pedal halo, symp re strings (dB)" + off, halo, "-40..-25", -40 <= halo <= -25, "{:.0f}")
    diff = _rms_db(ped["symp"][: 2 * sr]) - _rms_db(noped["symp"][: 2 * sr])
    check("halo with vs without pedal (dB)" + off, diff, ">= 10", diff >= 10, "{:.0f}")
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
    # at f/ff the key bottoms out a few ms *before* the hammer reaches the string (Askenfelt & Jansson,
    # Figs. 4-5), so sound in the last 6 ms before the strike is correct; nothing earlier should sound
    x = _render(no_symp, 1.0, [(60, 0.2, 0.8, 100)])["noise"]
    pre = float((x[: int(0.194 * sr)] ** 2).sum() / ((x**2).sum() + 1e-30))
    check("noise energy > 6 ms before the hammer strikes (fraction)", pre, "< 0.01", pre < 0.01, "{:.3f}")
    ff = _render(no_symp, 1.0, [(60, 0.2, 0.8, 127)])["noise"]
    early = float((ff[int(0.194 * sr): int(0.2 * sr)] ** 2).sum() / ((ff**2).sum() + 1e-30))
    check("ff: key-bottom thump before the hammer (fraction of noise energy)", early, "> 0.005", early > 0.005, "{:.3f}")

    table = ["| key | vel | f1 Hz | B_eff | partials 2..6 dB re f1 | centroid Hz | peak dBFS |",
             "|---|---|---|---|---|---|---|"] + rows
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
