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
# Measured on the Iowa Steinway B with pianonn.calibration (docs/calibration_iowa.md): robust medians over
# partials 1-4 (mf + ff), smoothed +-3 semitones. Keys: prompt T60 s, aftersound T60 s, aftersound knee of the
# fundamental dB re peak (partials 2-4 saturate in this fit, in recordings and model alike, so only partial 1 is
# used; C1's -13 dB is an outlier between neighbours at -29/-28), early spectral slope at mf dB/oct, tuning
# cents re A4.
IOWA = {"A0": (26.4, 113, -29, -1, -16), "C1": (19.8, 110, None, 1.5, -15), "C2": (26.7, 93, -28, -3.5, -4),
        "C3": (15.7, 51, -27, -4.7, 0), "C4": (10.3, 26, -25, -13.7, -1), "A4": (5.8, 18, -24, -17.3, 0),
        "C5": (5.7, 14, -24, -25.8, 0), "C6": (5.0, 9.1, -16, -26.8, 6), "C7": (None, 4.6, -14, -32.6, 14)}
IOWA_PITCH = {"A0": 21, "C1": 24, "C2": 36, "C3": 48, "C4": 60, "A4": 69, "C5": 72, "C6": 84, "C7": 96}


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
            x = _render(dry, 3.0, [(pitch, 0.0, 3.0, v)])["audio"]
            meas, lev = _partials(x, sr, _model_freqs(model, pitch, v))
            f1 = meas[0]
            notes[(name, v)] = dict(f1=f1, B=_b_eff(meas), lev=lev - lev[0],
                                    centroid=_centroid(x, sr), peak=20 * np.log10(np.abs(x).max() + 1e-15))
            r = notes[(name, v)]
            rows.append(f"| {name} | {v} | {f1:.2f} | {r['B']:.2e} | {' '.join(f'{d:+.0f}' for d in r['lev'][1:6])} | "
                        f"{r['centroid']:.0f} | {r['peak']:.1f} |")

    # --- decays, knee, brightness and stretch vs the Iowa recordings, analysed by the same code ---
    from .calibration import analyze_note

    pre = int(0.3 * sr)
    body = _variant(model, use_noise=False, use_sympathetic=False)
    with torch.no_grad():
        saved = body.room.log_gain.clone()
        body.room.log_gain.fill_(-30.0)  # near-field recording: soundboard, no hall
    measured = {}
    for name, p in IOWA_PITCH.items():
        n = int(45 * sr) + pre
        x = _render(dry, n / sr, [(p, pre / sr, n / sr, 64)])["audio"]
        x = x + 1e-7 * np.random.default_rng(p).standard_normal(len(x))  # a noise floor, as in a recording
        r = analyze_note(x, sr, p)
        y = _render(body, 1.5 + pre / sr, [(p, pre / sr, 1.5 + pre / sr, 64)])["audio"]
        y = y + 1e-7 * np.random.default_rng(p).standard_normal(len(y))
        r["slope_db_oct"] = analyze_note(y, sr, p).get("slope_db_oct", float("nan"))
        measured[name] = r
    with torch.no_grad():
        body.room.log_gain.copy_(saved)
    a4 = measured["A4"]["cents"]
    for name, (prompt, after, knee, slope, cents) in IOWA.items():
        r = measured[name]
        med = lambda key, lo=0, hi=4: float(np.nanmedian(r.get(key, [np.nan])[lo:hi]))
        if prompt:
            v = med("t60_prompt")
            check(f"{name} prompt T60, partials 1-4 (s)", v, f"{prompt} (x/1.35)", prompt / 1.35 <= v <= prompt * 1.35, "{:.1f}")
        v = med("t60_after")
        check(f"{name} aftersound T60, partials 1-4 (s)", v, f"{after} (x/1.35)", after / 1.35 <= v <= after * 1.35, "{:.1f}")
        if knee is not None:
            v = r.get("after_level_db", [np.nan])[0]
            check(f"{name} aftersound knee of the fundamental (dB re peak)", v, f"{knee} +-4", abs(v - knee) <= 4, "{:.0f}")
        v = r["slope_db_oct"]
        check(f"{name} early spectral slope, mf, radiated (dB/oct)", v, f"{slope} +-5", abs(v - slope) <= 5, "{:.1f}")
        v = r["cents"] - a4
        check(f"{name} tuning re A4 (cents)", v, f"{cents:+d} +-4", abs(v - cents) <= 4, "{:+.1f}")

    for name, target in (("C4", 3.3e-4), ("C6", 2.85e-3)):  # Rigaud et al. (DAFx 2011)
        b = notes[(name, 80)]["B"]
        check(f"{name} effective B (Rigaud et al.)", b, f"{target:.2e} (x1.5)", target / 1.5 <= b <= target * 1.5)
    for name, target in (("C2", 1.19e-4), ("C4", 3.04e-4)):  # measured on the Iowa Steinway B
        b = measured[name]["B"]
        check(f"{name} effective B (Iowa, x2: bass is piano-specific)", b, f"{target:.2e} (x2)", target / 2 <= b <= target * 2)
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
