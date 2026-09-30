"""The note bench (docs/tone_measures.md, section 3): isolated notes of the recordings against models' renders.

    python scripts/note_bench.py data/maestro24k --years 2018 --out runs/round2/bench --validate --music 24 \\
        --model control=runs/round2/b_control/last.pt:physics --model trial=runs/round1_trial/main/best.pt:residual

Builds (or reads) ``runs/measurements/bench_<years>.json``, renders the evaluation group's notes and releases with
every model in their context, and writes ``report.md`` and ``rows.json`` to ``--out``:
- N0-N8 per register, with flags (|model - recording| > max(JND, half the recording's IQR)) and spread ratios;
- N1, N2, N6 knock, N4 percussive and N9 by velocity, N6 by pedal (E1), N9 phantoms, N10 releases;
- P4 (room) on free decays and on the notes (the image), F4 on re-strike runs under the pedal;
- with ``--music K``: E4 (attacks in music) and P3 (texture statistics) on K test excerpts of 6 s;
- with ``--validate``: known parameter changes pushed through the first model, and what each measure reports.
The partial frequencies of the first model (measured and frozen in round 2) are the comb for every clip.
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

from pianonn import measures as M  # noqa: E402
from pianonn.config import year_to_condition  # noqa: E402
from pianonn.data import MaestroSegments  # noqa: E402
from pianonn.render import load_model  # noqa: E402
from pianonn.train import fixed_batches  # noqa: E402

GRID = ["N1 level", "N2 slope", "N2 centroid", "N2 partial 2", "N2 partial 4", "N2 partial 8", "N4 rise",
        "N4 percussive 250", "N4 percussive 1000", "N4 percussive 4000", "N4 percussive 8000",
        "N5 attack 500", "N5 attack 1000", "N5 attack 2000", "N5 attack 4000", "N5 attack 8000",
        "N5 early 1000", "N5 early 4000", "N5 sustain 1000", "N5 sustain 4000", "N5 sustain 8000",
        "N6 knock 125", "N6 knock 250", "N6 knock 500", "N6 knock 1000", "N6 knock 2000", "N6 knock 4000", "N6 knock 8000",
        "N6 sustain all", "N7 pre-onset low", "N7 pre-onset high", "N8 drop 1-4", "N8 drop 5-8", "N9 phantom", "N9 control",
        "P4 iacc early 500", "P4 iacc early 2000", "P4 iacc late 500", "P4 iacc late 2000", "P4 lr 500", "P4 lr 2000",
        "onset_ms"]
SPREAD = ["N1 level", "N2 slope", "N5 attack 2000", "N6 knock 2000", "N8 drop 1-4", "onset_ms"]
ROOM_BANDS = (125,) + M.OCTAVES
UNITS = {"N2 slope": "dB/oct", "N2 centroid": "cents re f0", "N4 rise": "ms", "onset_ms": "ms re MIDI"}


def measure_notes(root, notes, models, cfg, dev, table, chunk=48):
    rows = []
    sr = cfg.sample_rate
    for s in range(0, len(notes), chunk):
        part = notes[s: s + chunk]
        clips = M.render_clips(root, part, models, cfg, dev)
        for i, n in enumerate(part):
            ps = table[n["pitch"] - 21]
            r = {k: n[k] for k in ("piece", "pitch", "velocity", "register", "vel_bin", "pedal_bin", "pedal")}
            for src, xs in clips.items():
                r[src] = M.note_measures(xs[i], sr, M.PRE, ps, n["clear_before"], n["clear_after"])
                if np.isfinite(r[src]["onset_ms"]):
                    r[src].update(M.channel_measures(xs[i], sr, M.PRE + r[src]["onset_ms"] / 1000))
            rows.append(r)
        print(f"notes {min(s + chunk, len(notes))}/{len(notes)}", flush=True)
    return rows


def floor_bands(model, cond, cfg, dev):
    """The model's stationary floor per octave band (``band_envelope`` power): P4 fits stop 10 dB above it."""
    with torch.no_grad():
        fl = model.room.floor_noise(torch.tensor([cond], device=dev), int(5 * cfg.sample_rate),
                                    torch.Generator(device=dev).manual_seed(0))[0]
    return M.band_floor(fl.T.cpu().numpy().astype(np.float64), cfg.sample_rate, ROOM_BANDS)


def measure_decays(root, decays, models, cfg, dev, floors, chunk=48):
    """P4 on free decays; the recording is fitted above the first model's floor, each model above its own."""
    rows = []
    for s in range(0, len(decays), chunk):
        part = decays[s: s + chunk]
        clips = M.render_clips(root, part, models, cfg, dev, pre=M.DECAY_PRE, post=M.DECAY_POST, at="stop")
        for i, d in enumerate(part):
            t_end = M.DECAY_PRE + min(d["next"] - d["stop"], M.DECAY_POST)
            r = {"piece": d["piece"]}
            for src, xs in clips.items():
                r[src] = M.room_measures(xs[i], cfg.sample_rate, M.DECAY_PRE, t_end, floors.get(src, floors["recording"]), ROOM_BANDS)
            rows.append(r)
        print(f"decays {min(s + chunk, len(decays))}/{len(decays)}", flush=True)
    return rows


def measure_repeats(root, runs, models, cfg, dev, table, chunk=32):
    """F4: the level of the key's partials after each strike of a re-strike run."""
    rows = []
    for s in range(0, len(runs), chunk):
        part = runs[s: s + chunk]
        clips = M.render_clips(root, part, models, cfg, dev, pre=M.PRE, post=M.REPEAT_POST)
        for i, run in enumerate(part):
            r = {k: run[k] for k in ("piece", "pitch", "register", "times", "velocities")}
            for src, xs in clips.items():
                r[src] = M.restrike_levels(xs[i], cfg.sample_rate, M.PRE, run["times"], table[run["pitch"] - 21])
            rows.append(r)
        print(f"repeats {min(s + chunk, len(runs))}/{len(runs)}", flush=True)
    return rows


def restrike_build_up(rows, label, base="recording"):
    """F4 per run and strike k >= 2: (label - base at strike k) - (label - base at the first strike), dB."""
    out = []
    for r in rows:
        a, b = np.array(r[base]), np.array(r[label])
        for k in range(1, len(a)):
            out.append((r["register"], k + 1, float((b[k] - a[k]) - (b[0] - a[0]))))
    return out


def measure_releases(root, offs, models, cfg, dev, table, chunk=48):
    """N10 on note-offs in any texture, from the released note's partials clear of the other ringing notes'."""
    rows = []
    for s in range(0, len(offs), chunk):
        part = offs[s: s + chunk]
        clips = M.render_clips(root, part, models, cfg, dev, pre=M.REL_PRE, post=M.REL_POST, at="offset")
        for i, n in enumerate(part):
            r = {k: n[k] for k in ("piece", "pitch", "velocity", "register")}
            others = [table[p - 21] for p in n["others"] if 21 <= p <= 108]
            for src, xs in clips.items():
                r[src] = M.release_tracks(xs[i], cfg.sample_rate, M.REL_PRE, table[n["pitch"] - 21], others)
            rows.append(r)
        print(f"releases {min(s + chunk, len(offs))}/{len(offs)}", flush=True)
    return rows


def measure_music(root, years, models, cfg, dev, k):
    ds = MaestroSegments(root, "test", cfg, 6.0, 1.0, 12.0, length=k, deterministic=True, years=years, seed=13)
    rows, flux = [], {"recording": []} | {label: [] for label, _, _ in models}
    sr = cfg.sample_rate
    rng = np.random.default_rng(0)
    for b in fixed_batches(ds, k, 4, dev):
        n, s0 = b["audio"].shape[-1], int(b["loss_start"][0])
        sig = {"recording": b["audio"][..., s0:].cpu().numpy().astype(np.float64)}
        for label, model, residual in models:
            with torch.no_grad():
                y = model(b, n, residual=residual, generator=torch.Generator(device=dev).manual_seed(0))["audio"]
            sig[label] = y[..., s0:].float().cpu().numpy().astype(np.float64)
        for j in range(b["audio"].shape[0]):
            on = (b["onset"][j] - s0 / sr).cpu().numpy()
            m = (b["mask"][j].cpu().numpy()) & (on >= 0.01) & (on < 5.9)
            regs = [M.register_of(int(p)) for p in b["pitch"][j].cpu().numpy()[m]]
            # the flux of noise-like content (room, reverb) is not an attack: each excerpt's baseline is the flux at
            # up to 20 times at least 80 ms from any onset, and E4 is the contrast (onset - baseline)
            grid = np.arange(0.05, 5.9, 0.01)
            far = grid[np.min(np.abs(grid[:, None] - on[m][None, :]), 1) > 0.08] if m.any() else grid
            away = rng.choice(far, min(len(far), 20), replace=False) if len(far) else np.array([])
            r = {}
            for src, X in sig.items():
                x = X[j].T
                r[src] = M.texture_stats(x, sr)
                if len(away) < 5 or not m.any():
                    continue
                base = np.median(M.onset_flux(x, sr, away), 0)
                f = M.onset_flux(x, sr, on[m]) - base
                flux[src] += [(reg, row) for reg, row in zip(regs, f)]
            rows.append(r)
    return rows, flux


def validate(root, bench, notes, rels, model, residual, cfg, dev, table, parts=("notes", "releases", "decays", "repeats")):
    """Known changes to the first model: what each measure reports (median over events of changed - unchanged)."""
    cond = year_to_condition(notes[0]["year"])
    sel = [n for n in notes if n["register"] in ("R3", "R4", "R5") and n["pedal_bin"] == "up"][:24] or notes[:24]
    rel_sel = [r for r in rels if r["register"] in ("R3", "R4", "R5")][:48]
    dec_sel = [d for d in bench.get("decays", []) if d["group"] == "eval"][:24]
    rep_sel = [r for r in bench.get("repeats", []) if r["group"] == "eval" and len(r["times"]) >= 3][:24]
    floor = floor_bands(model, cond, cfg, dev)

    def with_change(fn):
        m = copy.deepcopy(model)
        with torch.no_grad():
            fn(m)
        return m

    def shift_body(m, samples=120):
        b = m.room.body.data
        m.room.body.data = torch.cat([torch.zeros_like(b[..., :samples]), b[..., :-samples]], -1)

    def delay_dampers(m):
        orig = m.physics.damper_delay
        m.physics.damper_delay = lambda c: orig(c) + 0.02

    changes = [
        ("latency +5 ms (body FIR shifted)", shift_body, "onset_ms", "+5.0 ms", lambda d: abs(d - 5) < 0.7, "notes"),
        ("mic gain +6 dB", lambda m: m.room.mic_gain_db.data[cond].add_(6.0), "N1 level", "+6 dB (floor unchanged)", lambda d: abs(d - 6) < 0.7, "notes"),
        ("knock noise +20 dB", lambda m: m.noise.knock.add_(math.log(10.0)), "N6 knock 4000", "> +10", lambda d: d > 10.0, "notes"),
        ("knock noise +20 dB", lambda m: m.noise.knock.add_(math.log(10.0)), "N4 percussive 8000", "> +10", lambda d: d > 10.0, "notes"),
        ("knock noise +20 dB", lambda m: m.noise.knock.add_(math.log(10.0)), "N5 attack 4000", "up", lambda d: d > 1.0, "notes"),
        ("knock impulse +20 dB", lambda m: m.physics.prior_impulse_db.add_(20.0), "N6 knock 500", "> +10", lambda d: d > 10.0, "notes"),
        ("contact time x0.7 (brighter)", lambda m: m.physics.prior_log_tc.add_(math.log(0.7)), "N2 slope", "up", lambda d: d > 0.5, "notes"),
        ("contact time x0.7 (brighter)", lambda m: m.physics.prior_log_tc.add_(math.log(0.7)), "N2 centroid", "up", lambda d: d > 20, "notes"),
        ("contact time x0.7 (brighter)", lambda m: m.physics.prior_log_tc.add_(math.log(0.7)), "N6 knock 1000", "within ±3 (not a knock)", lambda d: abs(d) < 3.0, "notes"),
        ("prompt-decay ratio x1.5", lambda m: m.physics.prior_prompt_ratio.mul_(1.5), "N8 drop 1-4", "up", lambda d: d > 1.0, "notes"),
        ("prompt-decay ratio x1.5", lambda m: m.physics.prior_prompt_ratio.mul_(1.5), "N6 knock 1000", "within ±2 (not a knock)", lambda d: abs(d) < 2.0, "notes"),
        ("phantoms +20 dB", lambda m: m.physics.prior_phantom_db.add_(20.0), "N9 phantom", "up", lambda d: d > 5.0, "notes"),
        ("channel 0 gain +3 dB", lambda m: m.room.mic_gain_db.data[cond, 0].add_(3.0), "P4 lr 1000", "+3 dB", lambda d: abs(d - 3) < 0.3, "notes"),
        ("one body for both channels", lambda m: m.room.body.data[cond, 1].copy_(m.room.body.data[cond, 0]), "P4 iacc early 1000",
         "up (the model's two bodies decorrelate even the direct sound)", lambda d: d > 0.05, "notes"),
        ("hall T60 x1.3", lambda m: m.room.prior_log_t60.add_(math.log(1.3)), "P4 T60 1000", "up ~30 %", lambda d: d > 0.15, "decays"),
        ("hall T60 x1.3", lambda m: m.room.prior_log_t60.add_(math.log(1.3)), "P4 T60 4000", "up ~30 %", lambda d: d > 0.1, "decays"),
        ("hall -6 dB", lambda m: m.room.log_gain.data[cond].sub_(math.log(2.0)), "P4 early 1000", "up (~+6 if the direct sound dominates)", lambda d: d > 3.0, "decays"),
        ("no re-strike damping", lambda m: m.physics.prior_restrike.mul_(0.0), "F4 build-up", "up", lambda d: d > 0.1, "repeats"),
        ("re-strike damping x3", lambda m: m.physics.prior_restrike.mul_(3.0), "F4 build-up", "down", lambda d: d < -0.1, "repeats"),
        ("damper delay +20 ms", delay_dampers, "N10 delay", "+20 ms", lambda d: abs(d - 20) < 6, "releases"),
        ("damper delay +20 ms", delay_dampers, "N10 step", "~0 (a delay, not a change)", lambda d: abs(d) < 1.0, "releases"),
    ]

    changes = [c for c in changes if c[-1] in parts]

    def run(label, m):
        src = [(label, m, residual)]
        return {"notes": measure_notes(root, sel, src, cfg, dev, table) if "notes" in parts else [],
                "releases": measure_releases(root, rel_sel, src, cfg, dev, table) if rel_sel and "releases" in parts else [],
                "decays": measure_decays(root, dec_sel, src, cfg, dev, {"recording": floor}) if dec_sel and "decays" in parts else [],
                "repeats": measure_repeats(root, rep_sel, src, cfg, dev, table) if rep_sel and "repeats" in parts else []}

    base = run("base", model)
    lines = ["## Validation: known changes through the renderer", "",
             f"First model: {len(sel)} notes (R3-R5, pedal up), {len(rel_sel)} releases, {len(dec_sel)} free decays and "
             f"{len(rep_sel)} re-strike runs. Median over events of (changed − unchanged); F4: the build-up of "
             "(changed − unchanged) from the first strike to the later ones.", "",
             "| change | measure | expected | measured | pass |", "|---|---|---|---|---|"]
    cache = {}
    for name, fn, key, expected, ok, kind in changes:
        if name not in cache:
            cache[name] = run("changed", with_change(fn))
        rows_c, rows_b = cache[name][kind], base[kind]
        if kind == "repeats":
            both = [{**b, "changed": c["changed"]} for b, c in zip(rows_b, rows_c)]
            d = np.array([v for _, _, v in restrike_build_up(both, "changed", "base")])
        else:
            d = [c["changed"].get(key, np.nan) - b["base"].get(key, np.nan) for b, c in zip(rows_b, rows_c)]
            d = np.array([v for v in d if np.isfinite(v)])
        med = float(np.median(d)) if len(d) else float("nan")
        lines.append(f"| {name} | {key} | {expected} | {med:+.2f} (n={len(d)}) | {'yes' if len(d) and ok(med) else '**no**'} |")
    return lines


def fmt_cell(s):
    return f"{s['diff']:+.1f}{'*' if s['flag'] else ''}"


def report(rows, rel_rows, dec_rows, rep_rows, labels, music, validation, bench, args):
    regs = [r for r, _, _ in M.REGISTERS]
    L = [f"# Note bench: {', '.join(labels)} vs the recordings ({', '.join(map(str, args.years))}, group {args.group})", "",
         f"{len(rows)} isolated notes and {len(rel_rows)} note-offs from `{args.bench}` "
         f"(no other onset 0.3 s before to 0.65 s after; at most {bench['cap']} per register × velocity × pedal). "
         "Definitions: `pianonn/measures.py`; design: docs/tone_measures.md.", "",
         "Cells: model − recording, median of the paired differences; * = beyond max(JND, half the recording's IQR). "
         "Units dB unless noted: " + ", ".join(f"{k} in {u}" for k, u in UNITS.items()) + ".", ""]
    L += ["## N0: onsets", "", "| register | notes | recording: onset re MIDI, median [IQR] ms | "
          + " | ".join(f"{l}: failed, model − recording [IQR] ms" for l in labels) + " |", "|---|---|---|" + "---|" * len(labels)]
    for reg in regs:
        rs = [r for r in rows if r["register"] == reg]
        if not rs:
            continue
        rec = np.array([r["recording"]["onset_ms"] for r in rs])
        ok = np.isfinite(rec)
        cells = []
        for l in labels:
            mo = np.array([r[l]["onset_ms"] for r in rs])
            both = ok & np.isfinite(mo)
            d = mo[both] - rec[both]
            cells.append(f"{np.mean(~np.isfinite(mo)) * 100:.0f} %, {np.median(d):+.1f} [{np.percentile(d, 25):+.1f}, {np.percentile(d, 75):+.1f}]" if both.any() else "-")
        L.append(f"| {reg} | {len(rs)} (rec failed {np.mean(~ok) * 100:.0f} %) | "
                 + (f"{np.median(rec[ok]):+.1f} [{np.percentile(rec[ok], 25):+.1f}, {np.percentile(rec[ok], 75):+.1f}]" if ok.any() else "-")
                 + " | " + " | ".join(cells) + " |")
    for l in labels:
        L += ["", f"## {l}: model − recording by register", "", "| measure | " + " | ".join(regs) + " |", "|---|" + "---|" * len(regs)]
        L.append("| n (notes) | " + " | ".join(str(sum(r["register"] == g for r in rows)) for g in regs) + " |")
        for key in GRID:
            s = M.paired_summary(rows, l, key, lambda r: r["register"])
            L.append(f"| {key} | " + " | ".join(fmt_cell(s[g]) if g in s else "" for g in regs) + " |")
        L += ["", f"Recording's median for reference: " + "; ".join(
            f"{key} " + "/".join(f"{M.paired_summary(rows, l, key, lambda r: r['register'])[g]['rec']:+.1f}"
                                 if g in M.paired_summary(rows, l, key, lambda r: r["register"]) else "-" for g in regs)
            for key in ("N2 slope", "N5 attack 2000", "N6 attack all", "N8 drop 1-4")) + " (R1..R7)."]
        L += ["", f"Spread ratio (model / recording IQR of residuals after a linear fit on pitch and velocity; 1 = as varied as the piano):", "",
              "| measure | " + " | ".join(regs) + " |", "|---|" + "---|" * len(regs)]
        for key in SPREAD:
            s = M.paired_summary(rows, l, key, lambda r: r["register"])
            L.append(f"| {key} | " + " | ".join(f"{s[g]['spread']:.2f}" if g in s else "" for g in regs) + " |")
    L += ["", "## By velocity (registers R2-R6 pooled)", "", "| measure | model | " + " | ".join(v for v, _, _ in M.VELOCITY) + " |",
          "|---|---|" + "---|" * len(M.VELOCITY)]
    mid = [r for r in rows if r["register"] in ("R2", "R3", "R4", "R5", "R6")]
    for key in ("N1 level", "N2 slope", "N2 centroid", "N5 attack 2000", "N6 knock 1000", "N4 percussive 4000", "N9 phantom"):
        for l in labels:
            s = M.paired_summary(mid, l, key, lambda r: r["vel_bin"])
            L.append(f"| {key} | {l} | " + " | ".join(fmt_cell(s[v]) if v in s else "" for v, _, _ in M.VELOCITY) + " |")
    L += ["", "## E1: pedal halo (energy between the partials, 100-400 ms, 0.2-8 kHz, dB re the band)", "",
          "| register | source | " + " | ".join(p for p, _, _ in M.PEDAL) + " | down − up |", "|---|---|" + "---|" * (len(M.PEDAL) + 1)]
    for reg in regs:
        for src in ["recording"] + labels:
            vals = {}
            for p, _, _ in M.PEDAL:
                v = [r[src].get("N6 sustain all", np.nan) for r in rows if r["register"] == reg and r["pedal_bin"] == p]
                v = [x for x in v if np.isfinite(x)]
                vals[p] = (np.median(v), len(v)) if len(v) >= 5 else None
            if not any(vals.values()):
                continue
            du = f"{vals['down'][0] - vals['up'][0]:+.1f}" if vals["down"] and vals["up"] else ""
            L.append(f"| {reg} | {src} | " + " | ".join(f"{v[0]:+.1f} (n={v[1]})" if v else "" for v in vals.values()) + f" | {du} |")
    L += ["", "## N10: releases (note-offs with the pedal up; only the released note's partials clear of other ringing notes)", "",
          "Each partial's level track is fitted by two lines with a step between them. Delay: the breakpoint re the MIDI "
          "note-off (where the direct sound has fallen to the room's level: the damper's contact plus part of its fall); "
          "step: the drop at the breakpoint; slopes before and after (dB/s). Recording: median; models: median of the "
          "paired differences. Damped: share of partials with a clear break.", "",
          "| register | n | source | partials | damped | delay ms | step dB | slope before | slope after |", "|---|---|---|---|---|---|---|---|---|"]
    for reg in regs:
        rs = [r for r in rel_rows if r["register"] == reg]
        if len(rs) < 5:
            continue
        med = lambda src, k: np.nanmedian([r[src].get(k, np.nan) for r in rs]) if any(k in r[src] for r in rs) else np.nan
        L.append(f"| {reg} | {len(rs)} | recording | {med('recording', 'N10 n'):.0f} | {med('recording', 'N10 damped'):.2f} | "
                 f"{med('recording', 'N10 delay'):+.0f} | {med('recording', 'N10 step'):+.1f} | {med('recording', 'N10 slope before'):+.0f} | "
                 f"{med('recording', 'N10 slope after'):+.0f} |")
        for src in labels:
            pd = lambda k: np.nanmedian([r[src][k] - r["recording"][k] for r in rs if k in r[src] and k in r["recording"]] or [np.nan])
            L.append(f"| {reg} | {len(rs)} | {src} − rec | {med(src, 'N10 n') - med('recording', 'N10 n'):+.0f} | "
                     f"{med(src, 'N10 damped') - med('recording', 'N10 damped'):+.2f} | {pd('N10 delay'):+.0f} | {pd('N10 step'):+.1f} | "
                     f"{pd('N10 slope before'):+.0f} | {pd('N10 slope after'):+.0f} |")
    L += ["", "## N9: phantom partials (level at 2f_j re partial 2j, 30-430 ms; median over resolvable j)", "",
          "Control: the same detector half-way down to the nearest transverse partial, where nothing is expected. A phantom "
          "is there only where the phantom position reads above the control.", "",
          "| register | velocity | source | n | phantom dB | control dB |", "|---|---|---|---|---|---|"]
    for reg in regs:
        for vb, _, _ in M.VELOCITY:
            rs = [r for r in rows if r["register"] == reg and r["vel_bin"] == vb]
            for src in ["recording"] + labels:
                v = [(r[src]["N9 phantom"], r[src]["N9 control"]) for r in rs if "N9 phantom" in r[src]]
                if len(v) >= 5:
                    v = np.array(v)
                    L.append(f"| {reg} | {vb} | {src} | {len(v)} | {np.median(v[:, 0]):+.1f} | {np.median(v[:, 1]):+.1f} |")
    if dec_rows:
        L += ["", f"## P4: room, from {len(dec_rows)} free decays (every key up, pedal up, ≥ 0.7 s to the next onset)", "",
              "T60 (s) from the level 0.4 s (below 500 Hz), 0.25 s (to 1 kHz) or 0.15 s (above) after the MIDI's last damper on, while 10 dB above the "
              "floor; early (dB): the 50 ms before the stop over that line, i.e. the direct sound and early reflections of "
              "what was sounding. Recording: median [IQR]; models: median of the paired differences.", "",
              "| band | n | recording T60 | " + " | ".join(f"{l} − rec T60" for l in labels) + " | recording early | "
              + " | ".join(f"{l} − rec early" for l in labels) + " |", "|---|---|---|" + "---|" * (2 * len(labels) + 1)]
        for c in ROOM_BANDS:
            k1, k2 = f"P4 T60 {c}", f"P4 early {c}"
            rec = [r["recording"][k1] for r in dec_rows if k1 in r["recording"]]
            if len(rec) < 5:
                continue
            q = lambda v: f"{np.median(v):.2f} [{np.percentile(v, 25):.2f}, {np.percentile(v, 75):.2f}]"
            pd = lambda k, l: np.median([r[l][k] - r["recording"][k] for r in dec_rows if k in r[l] and k in r["recording"]] or [np.nan])
            rec2 = [r["recording"][k2] for r in dec_rows if k2 in r["recording"]]
            L.append(f"| {c} | {len(rec)} | {q(rec)} | " + " | ".join(f"{pd(k1, l):+.2f}" for l in labels) + f" | {q(rec2)} | "
                     + " | ".join(f"{pd(k2, l):+.1f}" for l in labels) + " |")
        knee = [r["recording"]["P4 knee ms"] for r in dec_rows if "P4 knee ms" in r["recording"]]
        if knee:
            L += ["", f"Knee re the MIDI's last damper, recording: median {np.median(knee):+.0f} ms (the 10 ms smoothing and "
                  "the walk back put it ~13 ms early on an instant stop); " + "; ".join(
                      f"{l} − rec {np.median([r[l]['P4 knee ms'] - r['recording']['P4 knee ms'] for r in dec_rows if 'P4 knee ms' in r[l] and 'P4 knee ms' in r['recording']] or [np.nan]):+.0f} ms"
                      for l in labels) + "."]
    if rep_rows:
        L += ["", f"## F4: re-strike ({len(rep_rows)} runs of one key struck again 0.08-0.6 s apart under the pedal)", "",
              "Build-up: (model − recording at strike k) − (model − recording at the first strike), level of the key's "
              "partials 20-100 ms after each strike. Positive: the model keeps more of the ringing string than the piano.", "",
              "| register | strike | " + " | ".join(labels) + " |", "|---|---|" + "---|" * len(labels)]
        for reg in regs:
            for k, name in ((2, "2"), (3, "3"), (4, "4+")):
                cells, n = [], 0
                for l in labels:
                    v = [d for g, kk, d in restrike_build_up(rep_rows, l) if g == reg and (kk == k or (k == 4 and kk >= 4))]
                    n = max(n, len(v))
                    cells.append(f"{np.median(v):+.1f} (n={len(v)})" if len(v) >= 5 else "")
                if n >= 5:
                    L.append(f"| {reg} | {name} | " + " | ".join(cells) + " |")
    if music:
        mrows, flux = music
        L += ["", f"## E4: attacks in music ({len(mrows)} test excerpts of 6 s): positive spectral flux around onsets, dB, median", "",
              "Contrast: the flux in -5..+40 ms around each onset minus the excerpt's flux away from onsets (the recordings' "
              "room and reverb fluctuate more everywhere, so raw flux is not an attack measure).", "",
              "| register | onsets | source | " + " | ".join(f"{c} Hz" for c in M.OCTAVES) + " |", "|---|---|---|" + "---|" * len(M.OCTAVES)]
        for reg in regs:
            for src in ["recording"] + labels:
                f = np.array([row for g, row in flux[src] if g == reg])
                if len(f) < 10:
                    continue
                L.append(f"| {reg} | {len(f)} | {src} | " + " | ".join(f"{np.median(f[:, k]):.1f}" for k in range(len(M.OCTAVES))) + " |")
        keys = sorted(k for k in mrows[0]["recording"] if not k.startswith("P3 mean"))
        L += ["", "## P3: texture statistics of music, model − recording (median over excerpts)", "",
              "CV: envelope fluctuation (std/mean of amplitude^0.3); mod: share of modulation power in the band (dB); "
              "corr: adjacent bands' envelope correlation.", "", "| statistic | " + " | ".join(labels) + " | recording |", "|---|" + "---|" * (len(labels) + 1)]
        for k in keys:
            rec = np.array([r["recording"][k] for r in mrows])
            cells = [f"{np.nanmedian(np.array([r[l][k] for r in mrows]) - rec):+.2f}" for l in labels]
            L.append(f"| {k} | " + " | ".join(cells) + f" | {np.nanmedian(rec):.2f} |")
    if validation:
        L += [""] + validation
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data")
    ap.add_argument("--years", type=int, nargs="*", default=[2018])
    ap.add_argument("--bench", help="bench JSON (default runs/measurements/bench_<years>.json; built if missing)")
    ap.add_argument("--group", default="eval", choices=("eval", "calib"))
    ap.add_argument("--model", action="append", required=True, help="label=checkpoint:physics|residual")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--music", type=int, default=0, help="E4 and P3 on this many 6 s test excerpts")
    ap.add_argument("--max-notes", type=int, default=0, help="cap the notes (smoke runs)")
    ap.add_argument("--parts", default="notes,releases,decays,repeats", help="which event sets to measure (comma list)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.6)
    os.makedirs(args.out, exist_ok=True)
    args.bench = args.bench or f"runs/measurements/bench_{'_'.join(map(str, args.years))}.json"
    if os.path.exists(args.bench):
        with open(args.bench) as f:
            bench = json.load(f)
        if any(k not in bench for k in ("decays", "repeats", "offs")):  # a bench from before P4, F4, N10: add them, same draw
            new = M.build_bench(args.data, args.years)
            assert new["notes"] == bench["notes"] and new["releases"] == bench["releases"], "the rebuilt bench differs"
            bench = new
            with open(args.bench, "w") as f:
                json.dump(bench, f)
            print(f"{args.bench}: {len(bench['decays'])} free decays, {len(bench['repeats'])} re-strike runs, "
                  f"{len(bench['offs'])} note-offs", flush=True)
    else:
        bench = M.build_bench(args.data, args.years)
        os.makedirs(os.path.dirname(args.bench), exist_ok=True)
        with open(args.bench, "w") as f:
            json.dump(bench, f)
        print(f"built {args.bench}: {len(bench['notes'])} notes, {len(bench['releases'])} releases", flush=True)
    notes = [n for n in bench["notes"] if n["group"] == args.group]
    rels = [o for o in bench["offs"] if o["group"] == args.group]
    decays = [d for d in bench["decays"] if d["group"] == args.group]
    repeats = [r for r in bench["repeats"] if r["group"] == args.group]
    if args.max_notes:
        notes, rels, decays, repeats = (v[: args.max_notes] for v in (notes, rels, decays, repeats))
    models = []
    for spec in args.model:
        label, rest = spec.split("=", 1)
        ckpt, mode = rest.rsplit(":", 1)
        models.append((label, load_model(ckpt, device=dev), mode == "residual"))
    cfg = models[0][1].cfg
    table = M.partial_table(models[0][1], year_to_condition(args.years[0]), dev)
    parts = set(args.parts.split(","))
    rows = measure_notes(args.data, notes, models, cfg, dev, table) if "notes" in parts else []
    rel_rows = measure_releases(args.data, rels, models, cfg, dev, table) if "releases" in parts else []
    cond = year_to_condition(args.years[0])
    floors = {"recording": floor_bands(models[0][1], cond, cfg, dev)} | {l: floor_bands(m, cond, cfg, dev) for l, m, _ in models}
    dec_rows = measure_decays(args.data, decays, models, cfg, dev, floors) if "decays" in parts else []
    rep_rows = measure_repeats(args.data, repeats, models, cfg, dev, table) if "repeats" in parts else []
    music = measure_music(args.data, args.years, models, cfg, dev, args.music) if args.music else None
    val = validate(args.data, bench, notes, rels, models[0][1], models[0][2], cfg, dev, table, parts) if args.validate else None
    labels = [l for l, _, _ in models]
    text = report(rows, rel_rows, dec_rows, rep_rows, labels, music, val, bench, args)
    with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8") as f:
        f.write(text)
    with open(os.path.join(args.out, "rows.json"), "w") as f:
        json.dump({"notes": rows, "releases": rel_rows, "decays": dec_rows, "repeats": rep_rows, "music": music[0] if music else None}, f)
    print(text)


if __name__ == "__main__":
    main()
