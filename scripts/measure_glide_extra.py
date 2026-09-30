"""N3 glide and N12 extra peaks (docs/tone_measures.md, section 11) on the note bench: recordings against a model.

    python scripts/measure_glide_extra.py data/maestro24k --model runs/round2/b_control/last.pt --out runs/round2/glide_extra

- N3 on bass-to-middle notes (R2-R4) that ring to the end of the window (held, or the pedal not up), by register,
  velocity and pedal. The model has fixed-frequency partials, so its reading is the measure's own bias (unison
  double decay, context): recording - model is the glide.
- N12 on pedal-up notes (R3-R7) held at least 0.45 s (any length for the undamped keys), by register, with the
  model as rendered and with its sympathetic bank switched on (the undamped strings ring with the pedal up).
- Checks through the renderer first: a known glide warped into the model's renders; known tones added to them.
Both groups of the bench are used (the model has none of these effects, so there is nothing to fit).
"""

import argparse
import copy
import json
import math
import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.6,max_split_size_mb:256")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.interpolate import CubicSpline  # noqa: E402

from pianonn import measures as M  # noqa: E402
from pianonn.config import year_to_condition  # noqa: E402
from pianonn.render import load_model  # noqa: E402

UNDAMPED = 89  # F6: the first key without a damper; its f0 (1397 Hz) is the lowest an undamped string can answer at
GLIDE_END, EXTRA_END = 0.62, 0.65
LABELS = ("phantom", "near", "between")
SOURCES = ("recording", "model", "model+symp")


def held(n):
    return n["offset"] - n["onset"]


def onset_of(x, sr, f0):
    t = M.onset(x, sr, M.PRE, f0)
    return t if np.isfinite(t) else M.PRE


def warp(x, sr, t_on, cents, tau):
    """``x`` with a pitch glide: ``cents`` sharp at ``t_on``, decaying with ``tau`` s (everything, room included)."""
    t = np.arange(len(x)) / sr
    u = np.clip(t - t_on, 0, None)
    src = t + math.log(2) / 1200 * cents * tau * (1 - np.exp(-u / tau))
    return CubicSpline(t, x, axis=0)(np.clip(src, 0, t[-1]))


def expected_glide(f0, cents, tau, t_end=GLIDE_END, early=0.01, span=0.1):
    win = max(0.04, 4.0 / f0)
    ce = early + win / 2 + np.linspace(0, span, 50)
    cl = t_end - win / 2 - np.linspace(0, span, 50)
    return cents * (np.exp(-ce / tau).mean() - np.exp(-cl / tau).mean())


def add_tones(x, sr, t_on, partials, t1, level_db=-30.0, t0=0.05):
    """``x`` plus steady tones from ``t_on`` above partial 2 and below partial 4 (or above partial 1), each
    ``level_db`` re its partial's peak power in N12's window, per channel, and max(2 %, 3 exclusion zones) away
    from it (within 0.4 f0). Returns the clip and the tones' frequencies."""
    seg = M._segment(x, sr, t_on + t0, t_on + t1)
    n = len(seg)
    res = 2.0 / (t1 - t0)
    n_fft = 1 << int(math.ceil(math.log2(n * 4)))
    hz = np.fft.rfftfreq(n_fft, 1 / sr)
    own = M.own_partials(M._power(seg, n_fft), hz, partials, 8000.0, track=True)
    S = np.hanning(n).sum()
    X = np.abs(np.fft.rfft(seg * np.hanning(n)[:, None], n_fft, axis=0)) ** 2  # [bins, ch]
    t = np.arange(len(x)) / sr
    on = np.clip((t - t_on) / 0.005, 0, 1)
    y, freqs = x.copy(), []
    for k, sign in ((1, 1), (3, -1), (0, 1)):
        if len(freqs) == 2 or k >= len(own):
            continue
        off = max(0.02 * own[k], 3 * (2 * res + 0.0015 * own[k]))
        f = own[k] + sign * off
        if off > 0.4 * own[0] or f >= 8000:
            continue
        i = int(round(own[k] * n_fft / sr))
        pk = X[max(0, i - 2): i + 3].max(0)  # the partial's peak power per channel
        a = 2 * np.sqrt(10 ** (level_db / 10) * pk) / S
        y += on[:, None] * a[None, :] * np.sin(2 * np.pi * f * t + 0.7)[:, None]
        freqs.append(f)
    return y, freqs


def render(root, notes, models, cfg, dev, chunk=48):
    """``{"recording": [...], label: [...]}``; a model with its sympathetic bank on renders 2 clips at a time
    (the bank's recurrence over all 88 keys does not fit 16)."""
    out = {}
    for s in range(0, len(notes), chunk):
        for j, (label, model, residual) in enumerate(models):
            batch = 2 if model.cfg.use_sympathetic else 16
            clips = M.render_clips(root, notes[s: s + chunk], [(label, model, residual)], cfg, dev, batch=batch)
            for k, v in clips.items():
                if k != "recording" or j == 0:
                    out.setdefault(k, []).extend(v)
        print(f"rendered {min(s + chunk, len(notes))}/{len(notes)}", flush=True)
    return out


def boot_median(v, n=1000, seed=0):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    b = np.median(v[rng.integers(0, len(v), (n, len(v)))], 1)
    return float(np.median(v)), float(np.percentile(b, 5)), float(np.percentile(b, 95))


def fmt(m):
    return "–" if not np.isfinite(m[0]) else f"{m[0]:+.2f} [{m[1]:+.2f}, {m[2]:+.2f}]"


def med(v):
    return f"{np.median(v):+.1f}" if len(v) else "–"


def recurring(rows, src, tol=1.5, min_notes=4, min_keys=3):
    """Frequencies (Hz) at which ``src`` has a peak in at least ``min_notes`` notes of ``min_keys`` keys: a fixed
    resonance (a body mode, a string), not something that follows the note. ``[(Hz, notes, keys)]``, most first."""
    pk = sorted((p[0], i, r["pitch"]) for i, r in enumerate(rows) for p in r[src].get("N12 peaks", []))
    out, j = [], 0
    for a in range(len(pk)):
        while pk[a][0] - pk[j][0] > 2 * tol:
            j += 1
        grp = pk[j: a + 1]
        notes, keys = len({g[1] for g in grp}), len({g[2] for g in grp})
        if notes >= min_notes and keys >= min_keys:
            f = float(np.median([g[0] for g in grp]))
            if not out or abs(out[-1][0] - f) > 2 * tol:
                out.append((f, notes, keys))
            elif notes > out[-1][1]:
                out[-1] = (f, notes, keys)
    return sorted(out, key=lambda c: -c[1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data")
    ap.add_argument("--bench", default="runs/measurements/bench_2018.json")
    ap.add_argument("--model", required=True, help="a physics-only checkpoint")
    ap.add_argument("--max-notes", type=int, default=0, help="cap each note set (smoke runs)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.6)
    os.makedirs(args.out, exist_ok=True)
    with open(args.bench) as f:
        bench = json.load(f)
    year = bench["years"][0]
    model = load_model(args.model, device=dev)
    cfg = model.cfg
    sr = cfg.sample_rate
    table = M.partial_table(model, year_to_condition(year), dev)
    symp = copy.deepcopy(model)
    symp.cfg.use_sympathetic = True

    g_notes = [n for n in bench["notes"] if n["register"] in ("R2", "R3", "R4")
               and (held(n) >= GLIDE_END + 0.02 or n["pedal_bin"] != "up")]
    x_notes = [n for n in bench["notes"] if n["register"] in ("R3", "R4", "R5", "R6", "R7") and n["pedal_bin"] == "up"
               and (held(n) >= 0.45 or n["pitch"] >= UNDAMPED)]
    if args.max_notes:
        g_notes, x_notes = g_notes[: args.max_notes], x_notes[: args.max_notes]
    L = [f"# N3 glide and N12 extra peaks: `{args.model}` vs the recordings ({year}, both groups)", ""]

    # ------------------------------------------------------------ N3
    clips = render(args.data, g_notes, [("model", model, False)], cfg, dev)
    g_rows, val_g = [], []
    for i, n in enumerate(g_notes):
        ps = table[n["pitch"] - 21]
        r = {k: n[k] for k in ("piece", "pitch", "velocity", "register", "vel_bin", "pedal_bin")}
        for src in ("recording", "model"):
            x = clips[src][i]
            t_on = onset_of(x, sr, float(ps[0]))
            r[src] = M.glide(x, sr, t_on, ps)
            if src == "model" and i % 4 == 0 and "N3 glide" in r[src]:
                w = M.glide(warp(x, sr, t_on, 3.0, 0.3), sr, t_on, ps)
                if "N3 glide" in w:
                    val_g.append((w["N3 glide"] - r[src]["N3 glide"], expected_glide(float(ps[0]), 3.0, 0.3)))
        g_rows.append(r)
    del clips

    # ------------------------------------------------------------ N12
    clips = render(args.data, x_notes, [("model", model, False), ("model+symp", symp, False)], cfg, dev)
    x_rows, val_x = [], []
    for i, n in enumerate(x_notes):
        ps = table[n["pitch"] - 21]
        t1 = EXTRA_END if n["pitch"] >= UNDAMPED else min(EXTRA_END, held(n) - 0.03)
        r = {k: n[k] for k in ("piece", "pitch", "velocity", "register", "vel_bin")}
        for src in SOURCES:
            x = clips[src][i]
            t_on = onset_of(x, sr, float(ps[0]))
            r[src] = M.extra_peaks(x, sr, t_on, ps, t1=t1)
            if src == "model" and i % 2 == 0 and "N12 n" in r[src]:
                res = 2.0 / (t1 - 0.05)
                for db in (-30.0, -45.0):
                    y, fs = add_tones(x, sr, t_on, ps, t1, db)
                    got = M.extra_peaks(y, sr, t_on, ps, t1=t1).get("N12 peaks", [])
                    for f in fs:
                        hit = [p for p in got if abs(p[0] - f) <= res + 1.0]
                        val_x.append((db, n["register"], hit[0][3] if hit else None))
        x_rows.append(r)
    del clips

    # ------------------------------------------------------------ report: checks
    d = np.array(val_g)
    L += ["## Checks through the renderer", ""]
    if len(d):
        L.append(f"- **N3**: a glide of 3 cents at the onset, decaying with 0.3 s, warped into {len(d)} of the model's "
                 f"renders (after the room: this checks the measure in real clips, not how the room smears a string's "
                 f"glide). Reads {np.median(d[:, 0]):+.2f} cents (IQR {np.percentile(d[:, 0], 25):+.2f} … "
                 f"{np.percentile(d[:, 0], 75):+.2f}); the definition gives {np.median(d[:, 1]):+.2f}.")
    for db in (-30.0, -45.0):
        v = [(reg, lv) for lvl, reg, lv in val_x if lvl == db]
        if not v:
            continue
        per = ", ".join(f"{reg} {100 * np.mean([lv is not None for rg, lv in v if rg == reg]):.0f} %"
                        for reg in ("R3", "R4", "R5", "R6", "R7") if any(rg == reg for rg, _ in v))
        lv = [x for _, x in v if x is not None]
        L.append(f"- **N12**: steady tones {db:+.0f} dB re a partial (above partial 2, below partial 4; max(2 %, 3 "
                 f"exclusion zones) away) added to the model's renders: {len(v)} tones, {100 * len(lv) / len(v):.0f} % "
                 f"found ({per}); found ones read {med(lv)} dB re their partial.")
    L.append("")

    # ------------------------------------------------------------ report: N3
    L += ["## N3 glide (cents; early re late; positive: starts sharp)", "",
          "Median [90 % bootstrap interval]. The model has no glide: its reading is the measure's bias.", "",
          "| register | velocity | n | recording | model | recording − model |", "|---|---|---|---|---|---|"]
    ok = lambda r: "N3 glide" in r["recording"] and "N3 glide" in r["model"]
    for reg in ("R2", "R3", "R4"):
        for vb, _, _ in M.VELOCITY:
            rr = [r for r in g_rows if r["register"] == reg and r["vel_bin"] == vb and ok(r)]
            if len(rr) < 3:
                continue
            a = [r["recording"]["N3 glide"] for r in rr]
            b = [r["model"]["N3 glide"] for r in rr]
            L.append(f"| {reg} | {vb} | {len(rr)} | {fmt(boot_median(a))} | {fmt(boot_median(b))} | "
                     f"{fmt(boot_median(np.subtract(a, b)))} |")
    L += ["", "By pedal (R2-R4, all velocities):", "", "| pedal | n | recording − model |", "|---|---|---|"]
    for pb, _, _ in M.PEDAL:
        rr = [r for r in g_rows if r["pedal_bin"] == pb and ok(r)]
        if len(rr) >= 3:
            L.append(f"| {pb} | {len(rr)} | {fmt(boot_median([r['recording']['N3 glide'] - r['model']['N3 glide'] for r in rr]))} |")
    L.append("")

    # ------------------------------------------------------------ report: N12
    L += ["## N12 extra peaks (pedal up)", "",
          "Per note: narrow peaks that are not the note's partials, new with the note, ≥ 10 dB over the local floor "
          "and the partials' sidelobes. \"phantom\": at an f_j + f_k; \"near\": within 25 cents of a partial "
          "(unison splitting beyond the main lobe, undamped strings tuned to the note, duplex segments tuned near "
          "it); \"between\": the rest.", "",
          "| register | n | source | per note: phantom / near / between | notes with near or between | level re partials: phantom / near / between (median dB) |",
          "|---|---|---|---|---|---|"]
    for reg in ("R3", "R4", "R5", "R6", "R7"):
        rr = [r for r in x_rows if r["register"] == reg and all("N12 n" in r[s] for s in SOURCES)]
        if not rr:
            continue
        for src in SOURCES:
            v = [r[src] for r in rr]
            per = " / ".join(f"{np.mean([m[f'N12 n {k}'] for m in v]):.2f}" for k in LABELS)
            share = np.mean([m["N12 n near"] + m["N12 n between"] > 0 for m in v])
            lv = " / ".join(med([m[f"N12 level {k}"] for m in v if f"N12 level {k}" in m]) for k in LABELS)
            L.append(f"| {reg} | {len(v)} | {src} | {per} | {100 * share:.0f} % | {lv} |")
    L += ["", "Phantom peaks per note by velocity (phantoms grow with the square of the amplitude):", "",
          "| register | velocity | n | recording | model |", "|---|---|---|---|---|"]
    for reg in ("R3", "R4", "R5", "R6", "R7"):
        for vb, _, _ in M.VELOCITY:
            rr = [r for r in x_rows if r["register"] == reg and r["vel_bin"] == vb and "N12 n" in r["recording"] and "N12 n" in r["model"]]
            if len(rr) >= 3:
                L.append(f"| {reg} | {vb} | {len(rr)} | {np.mean([r['recording']['N12 n phantom'] for r in rr]):.2f} | "
                         f"{np.mean([r['model']['N12 n phantom'] for r in rr]):.2f} |")
    L += ["", "Near peaks below 1397 Hz on damped keys (R4-R6): no undamped string answers there, "
          "so these are unison splitting, duplex segments or something else.", "",
          "| register | n | source | near peaks per note | notes with one | median dB re partial |", "|---|---|---|---|---|---|"]
    for reg in ("R4", "R5", "R6"):
        rr = [r for r in x_rows if r["register"] == reg and r["pitch"] < UNDAMPED and all("N12 n" in r[s] for s in SOURCES)]
        if not rr:
            continue
        for src in SOURCES:
            v = [[p for p in r[src]["N12 peaks"] if p[4] == "near" and p[0] < 1397] for r in rr]
            L.append(f"| {reg} | {len(rr)} | {src} | {np.mean([len(x) for x in v]):.2f} | "
                     f"{100 * np.mean([len(x) > 0 for x in v]):.0f} % | {med([p[3] for x in v for p in x])} |")
    L += ["", "Where the recordings' near and between peaks sit (cents from the nearest partial):", "",
          "| cents | near + between peaks | median dB re that partial |", "|---|---|---|"]
    pk = [p for r in x_rows for p in r["recording"].get("N12 peaks", []) if p[4] != "phantom"]
    edges = (-1200, -100, -25, 0, 25, 100, 1200)
    for a, b in zip(edges[:-1], edges[1:]):
        s = [p for p in pk if a <= p[1] < b]
        L.append(f"| {a} … {b} | {len(s)} | {med([p[3] for p in s])} |")
    L += ["", "Nearest partial number of those peaks: " + ", ".join(
        f"{k}: {sum(1 for p in pk if p[2] == k)}" for k in range(1, 11)) + f", above: {sum(1 for p in pk if p[2] > 10)}", ""]
    L += ["Recurring frequencies: a peak at the same frequency (± 1.5 Hz) in ≥ 4 notes of ≥ 3 keys is a fixed "
          "resonance, not something that follows the note.", "",
          "| source | peaks | in recurring frequencies | the most frequent (Hz: notes, keys) |", "|---|---|---|---|"]
    for src in SOURCES:
        rec = recurring(x_rows, src)
        allp = [p[0] for r in x_rows for p in r[src].get("N12 peaks", [])]
        inrec = sum(1 for f in allp if any(abs(f - c[0]) <= 1.5 for c in rec))
        L.append(f"| {src} | {len(allp)} | {inrec} | " + ", ".join(f"{c[0]:.0f}: {c[1]}, {c[2]}" for c in rec[:8]) + " |")
    L.append("")

    text = "\n".join(L)
    with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8") as f:
        f.write(text)
    with open(os.path.join(args.out, "rows.json"), "w") as f:
        json.dump({"glide": g_rows, "extra": x_rows}, f)
    print(text)


if __name__ == "__main__":
    main()
