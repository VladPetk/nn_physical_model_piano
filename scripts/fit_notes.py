"""Fit parameters on isolated notes of the bench (phase 3, step 2; docs/tone_measures.md 12.1).

    python scripts/fit_notes.py data/maestro24k --model s1=runs/phase3/step1/hall/model.pt:physics \\
        --fit velmap --out runs/phase3/step2/velmap

Fits the named parameter set on the bench's calibration notes with the aligned note term (``pianonn.notefit``:
band levels in windows placed at each clip's own N0, L1 in dB, counted where the recording's note stands over its
background, or the model's over its own) plus the model's smoothness regulariser, and checks on the evaluation notes. Writes ``model.pt`` (the
fitted model, loadable with ``pianonn.render.load_variant``), ``report.md`` (term before and after on both groups,
the bias gate, residuals by register and velocity, the fitted velocity map and per-key changes) and ``log.txt``.

Parameter sets (``--fit``):
- ``velmap``: the velocity map (``cond_vel_map``), the level (per-key gain and velocity slope, the condition's
  velocity slope and curve, the microphone gain with both channels tied) and the per-key brightness at mf (contact
  time and roll-off order, the condition's contact time). The velocity exponents of the contact time and the order
  stay as they are: the map carries the velocity dependence, one curve for every key.
- ``nomap``: the same without the map (the control: what per-key level and brightness do alone).
- ``knock``: the attack's existing parts (the knock noise's per-key spectrum, velocity slope and decay; the key-bottom
  thump's gain and decay; the knock impulse's per-key level and velocity slope), on an attack window (-3 to 33 ms,
  flat-topped as N6's, every band, below f0 too): its level re the early window (the attack's excess over the tone)
  and the energy between the partials (the knock alone, as N6), with the early window as a guard. Step 2b: what is
  left after it is what new attack physics has to explain.

The per-key tables move by a piecewise-linear correction with knots every ``--key-step`` keys (0: every key free);
with 1-20 notes per key a free table follows single notes. A free level per piece, averaging zero, is fitted
alongside (``--no-piece-levels`` to drop it): the recordings' level at equal key and velocity differs between pieces
by ~2 dB (sd), which no model parameter should absorb. Held out, the term is reported as a level (median over pieces
of each piece's median) and a shape (mean |difference - its piece's median|).
"""

import argparse
import math
import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.6,max_split_size_mb:256")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from pianonn import measures as M  # noqa: E402
from pianonn import notefit as NF  # noqa: E402
from pianonn.config import year_to_condition  # noqa: E402
from pianonn.dsp import bounded  # noqa: E402
from pianonn.render import load_variant  # noqa: E402
from pianonn.train import lr_scale  # noqa: E402

LEVEL = ("physics.gain_db", "physics.raw_vel_slope", "physics.cond_vel_curve", "physics.cond_vel_slope", "room.mic_gain_db")
BRIGHT = ("physics.raw_log_tc", "physics.raw_order", "physics.cond_log_tc")
KNOCK = ("noise.knock", "noise.knock_vel", "noise.raw_knock_tau", "noise.thump_log_gain", "noise.raw_thump_tau",
         "physics.raw_impulse_db", "physics.raw_impulse_vel")
FITS = {"velmap": ("physics.cond_vel_map",) + LEVEL + BRIGHT, "nomap": LEVEL + BRIGHT, "knock": KNOCK}
# the attack's excess over the tone (attack re early, every band), the energy between the partials (the knock without
# the partials' onset), and the early window as a guard: a band-level term alone let the knock noise stand in for high
# partials the tone lacks (step 2b, first run)
ATTACK = {"attack re early": (-0.003, 0.033, {"flat": 0.7, "rel": "early", "all_bands": True}),
          "attack gaps": (-0.003, 0.033, {"flat": 0.7, "gaps": True}), "early": NF.WINDOWS["early"]}
WINDOWS = {"velmap": NF.WINDOWS, "nomap": NF.WINDOWS, "knock": ATTACK}
PER_KEY = ("physics.gain_db", "physics.raw_vel_slope", "physics.raw_log_tc", "physics.raw_order", "noise.knock",
           "noise.knock_vel", "noise.raw_knock_tau", "physics.raw_impulse_db")
MAP_VEL = (20, 32, 48, 64, 80, 96, 112)


def fmt_table(rows, head):
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(r) + " |" for r in rows]


def save_eval(path, ev):
    np.savez_compressed(path, ok=ev["ok"], **{f"{k}_{w}": v for k in ("rec", "mod", "cells") for w, v in ev[k].items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data")
    ap.add_argument("--years", type=int, nargs="*", default=[2018])
    ap.add_argument("--bench")
    ap.add_argument("--model", required=True, help="label=checkpoint:physics[:options] (pianonn.render.load_variant)")
    ap.add_argument("--fit", default="velmap", choices=sorted(FITS))
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-3, help="base learning rate (x20 for dB parameters, train.lr_scale)")
    ap.add_argument("--reg", type=float, default=1.0, help="weight of the smoothness regulariser")
    ap.add_argument("--key-step", type=int, default=8, help="knot spacing (keys) of the per-key corrections; 0: every key free")
    ap.add_argument("--no-piece-levels", action="store_true", help="no free level per piece")
    ap.add_argument("--knock-max-hz", type=float, default=0.0,
                    help="keep the knock noise's spectrum above this frequency as it is (0: fit every band): between the "
                         "partials, an isolated note's attack rarely stands over its background at 4-8 kHz")
    ap.add_argument("--max-notes", type=int, default=0, help="cap the notes of each group (smoke runs)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(args.out, exist_ok=True)
    logf = open(os.path.join(args.out, "log.txt"), "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.6)
    with open(args.bench or f"runs/measurements/bench_{'_'.join(map(str, args.years))}.json") as f:
        bench = json.load(f)
    label, model, _ = load_variant(args.model, device=dev, log=log)
    cfg, cond = model.cfg, year_to_condition(args.years[0])
    table = M.partial_table(model, cond, dev)
    groups = {}
    for g in ("calib", "eval"):
        notes = [n for n in bench["notes"] if n["group"] == g]
        notes = notes[: args.max_notes] if args.max_notes else notes
        groups[g] = NF.NoteSet(args.data, notes, cfg, table, log=lambda m, g=g: log(f"{g}: {m}"))
    fit = groups["calib"]
    term = NF.NoteTerm(cfg.sample_rate, dev, WINDOWS[args.fit])

    for p in model.parameters():
        p.requires_grad_(False)
    names = [n for n, _ in model.named_parameters() if n in FITS[args.fit]]
    params = dict(model.named_parameters())
    snap = {n: p.detach().clone() for n, p in params.items()}
    before = {g: NF.evaluate(model, s, term, dev) for g, s in groups.items()}
    speed0 = model.physics.hammer_speed(torch.tensor([MAP_VEL], device=dev) / 127.0, torch.tensor([cond], device=dev))[0].tolist()
    for g in groups:
        log(f"{g} before: " + ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}"
                                        for k, v in NF.level_and_shape(before[g], groups[g], term.windows).items()))
    knotted = [n for n in names if n in PER_KEY] if args.key_step > 0 else []
    knots = {}
    for mod in ("physics", "noise"):
        short = [n.split(".", 1)[1] for n in knotted if n.startswith(mod + ".")]
        knots.update({f"{mod}.{n}": k for n, k in NF.smooth_per_key(getattr(model, mod), short, args.key_step).items()})
    train = [(n, params[n]) for n in names if n not in knotted] + list(knots.items())
    for n, p in train:
        p.requires_grad_(True)
    if "room.mic_gain_db" in names:  # one level for both channels: the term sums their power and cannot see the balance
        params["room.mic_gain_db"].register_hook(lambda g: g.mean(-1, keepdim=True).expand_as(g))
    frozen_hz = ""
    if args.knock_max_hz > 0 and "noise.knock" in names:
        keep = (model.noise.centers <= args.knock_max_hz).float()
        knots.get("noise.knock", params["noise.knock"]).register_hook(lambda g, keep=keep: g * keep)
        frozen_hz = f"; the knock noise's spectrum above {args.knock_max_hz:.0f} Hz kept as it was"
    pieces = None if args.no_piece_levels else NF.PieceLevels(len(fit.pieces)).to(dev)
    groups_opt = [{"params": [p], "lr": args.lr * lr_scale(n), "name": n} for n, p in train]
    if pieces is not None:
        groups_opt.append({"params": [pieces.db], "lr": args.lr * 20.0, "name": "piece levels"})
    opt = torch.optim.Adam(groups_opt)
    for g in opt.param_groups:
        g["base_lr"] = g["lr"]
    log(f"fitting {', '.join(names)} on {len(fit)} notes of {len(fit.pieces)} pieces ({args.steps} steps of {args.batch}); "
        f"per-key knots every {args.key_step} keys on {', '.join(knotted) or 'none'}; piece levels {'off' if pieces is None else 'on'}{frozen_hz}")

    rng = np.random.default_rng(args.seed)
    order, pos = rng.permutation(len(fit)), 0
    run, t0 = [], time.time()
    cvec = torch.tensor([cond], device=dev)
    for step in range(1, args.steps + 1):
        if pos + args.batch > len(order):
            order, pos = rng.permutation(len(fit)), 0
        idx = order[pos: pos + args.batch].tolist()
        pos += args.batch
        b, audio = fit.batch(idx, dev)
        y = model(b, fit.n, residual=False, generator=torch.Generator(device=dev).manual_seed(step))["audio"]
        t_mod = NF.model_onsets(y, cfg.sample_rate, fit.t_midi, fit.f0[idx])
        ok = np.isfinite(t_mod)
        rec = term.levels(audio, np.array([fit.t_rec[i] for i in idx]), fit.partials[idx])
        mod = term.levels(y, np.where(ok, t_mod, fit.t_midi), fit.partials[idx])
        note = term.loss(mod, rec, term.cells(rec, fit.f0[idx], mod), torch.as_tensor(ok, dtype=torch.float32, device=dev),
                         None if pieces is None else pieces(fit.piece[idx]))
        reg = model.physics.regularizer(cvec)
        loss = note + args.reg * reg
        opt.zero_grad(set_to_none=True)
        loss.backward()
        frac = step / args.steps
        for g in opt.param_groups:  # cosine decay to a tenth
            g["lr"] = g["base_lr"] * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
        opt.step()
        run.append((float(note), float(reg)))
        if step % 25 == 0 or step == args.steps:
            r = np.mean(run[-25:], 0)
            log(f"step {step}: note {r[0]:.3f} dB, reg {r[1]:.4f}, {(time.time() - t0) / step:.2f} s/step")

    for n in knots:
        mod, short = n.split(".", 1)
        NF.parametrize.remove_parametrizations(getattr(model, mod), short)
    params = dict(model.named_parameters())
    model.eval()
    after = {g: NF.evaluate(model, s, term, dev) for g, s in groups.items()}
    for g in groups:
        save_eval(os.path.join(args.out, f"eval_{g}_before.npz"), before[g])
        save_eval(os.path.join(args.out, f"eval_{g}_after.npz"), after[g])
    speed1 = model.physics.hammer_speed(torch.tensor([MAP_VEL], device=dev) / 127.0, cvec)[0].tolist()
    torch.save({"cfg": cfg.to_dict(), "model": model.state_dict(), "source": args.model, "fit": args.fit, "step": 0},
               os.path.join(args.out, "model.pt"))

    # ---- report
    L = [f"# Note fit `{args.fit}` ({args.years[0]})", "",
         f"Model: `{args.model}`. Fitted: {', '.join(f'`{n}`' for n in names)}. {args.steps} steps of {args.batch} notes, "
         f"lr {args.lr}, regulariser weight {args.reg}{frozen_hz}. Calibration notes: {len(fit)}; evaluation notes: "
         f"{len(groups['eval'])} (N0 found on the recording). Term: band levels (1/3 octave, 100 Hz - 8 kHz) in "
         + ", ".join(f"{w} {win[0] * 1000:.0f} to {win[1] * 1000:.0f} ms" for w, win in term.windows.items())
         + f" re each clip's own N0, counted where the recording or the model stands {NF.MIN_OVER_BG:.0f} dB over its own background; "
           "cells: model − recording, dB.", "", "## The term", ""]
    rows = []
    for g in groups:
        a, b_ = NF.level_and_shape(before[g], groups[g], term.windows), NF.level_and_shape(after[g], groups[g], term.windows)
        rows.append([g, f"{a['abs']:.3f} → {b_['abs']:.3f}", f"{a['level']:+.2f} → {b_['level']:+.2f}",
                     f"{a['piece_sd']:.2f} → {b_['piece_sd']:.2f} ({a['pieces']})", f"{a['shape']:.3f} → {b_['shape']:.3f}", str(a["n"]),
                     f"{before[g]['ok'].mean() * 100:.0f} / {after[g]['ok'].mean() * 100:.0f} %"])
    L += fmt_table(rows, ["group", "mean abs (dB)", "level: median of piece medians (dB)", "sd of piece medians (pieces)",
                          "shape: mean abs re piece median (dB)", "cells", "N0 found on the model"])
    if pieces is not None:
        v = (pieces.db - pieces.db.mean()).detach().cpu().numpy()
        L += ["", f"Fitted piece levels (recording re the model, dB): sd {v.std():.2f}, range {v.min():+.1f} … {v.max():+.1f} "
                  f"({len(v)} pieces)."]
    for g in groups:
        L += ["", f"## Bias gate, {g} group", "", "Within each register × velocity stratum: the term's optimum level (median of model − recording over the counted "
              "cells) minus the energy match over the same cells; bias = the median over the strata (optimum and energy "
              "columns: medians over the strata too). Pass: the median over the bands within ±0.5 dB, every band within ±1 dB. "
              "Negative: the term's optimum leaves the model's energy under the recordings'.", ""]
        for tag, ev in (("before", before[g]), ("after", after[g])):
            gate, ok = NF.bias_gate(ev, groups[g], term.windows)
            L += [f"**{tag}** ({'pass' if ok else 'fail'}):", ""]
            L += fmt_table([[r["window"], str(r["band"]), f"{r['strata']} / {r['n']}", f"{r['median']:+.2f}", f"{r['energy']:+.2f}",
                             f"{r['bias']:+.2f}"] for r in gate], ["window", "octave", "strata / cells", "optimum", "energy", "bias"])
            L.append("")
    for key, vals in (("register", ("R1", "R2", "R3", "R4", "R5", "R6", "R7")), ("vel_bin", ("p", "mp-mf", "mf-f", "ff"))):
        for g in groups:
            tb = NF.residual_table(before[g], groups[g], term.windows, key=key)
            ta = NF.residual_table(after[g], groups[g], term.windows, key=key)
            L += [f"## Residual by {key}, {g} group (before → after, median dB)", ""]
            head = [key, "window"] + [str(o) for o in NF.OCTAVES]
            rows = []
            for v in vals:
                for w in term.windows:
                    cells = [f"{tb[(v, w, o)][0]:+.1f} → {ta[(v, w, o)][0]:+.1f}" if (v, w, o) in tb and (v, w, o) in ta else ""
                             for o in NF.OCTAVES]
                    if any(cells):
                        rows.append([v, w] + cells)
            L += fmt_table(rows, head) + [""]
    L += ["## Fitted parameters", "", "Velocity map (hammer speed, m/s; the prior is 2.8 exp(0.023 (v − 64))):", ""]
    L += fmt_table([["before"] + [f"{s:.2f}" for s in speed0], ["after"] + [f"{s:.2f}" for s in speed1]], ["MIDI velocity"] + [str(v) for v in MAP_VEL])
    keys = list(range(3, 88, 6))
    rows = []
    now = {n: p.detach() for n, p in model.named_parameters()}
    centers = model.noise.centers.cpu().numpy()
    near = lambda f: int(np.argmin(np.abs(np.log2(centers / f))))
    with torch.no_grad():
        per_key = (("contact time x", lambda d: torch.exp(bounded(d["physics.raw_log_tc"], 1.0))),
                   ("order x", lambda d: torch.exp(bounded(d["physics.raw_order"], 0.9))),
                   ("gain dB", lambda d: d["physics.gain_db"]),
                   ("velocity slope dB/u", lambda d: 40.0 * torch.exp(bounded(d["physics.raw_vel_slope"], 0.7))))
        if args.fit == "knock":  # the noise tables are log amplitudes: x 20 / ln 10 for dB
            per_key = tuple((f"knock noise {f} Hz dB", lambda d, f=f: d["noise.knock"][:, near(f)] * 20 / math.log(10))
                            for f in (250, 1000, 4000, 8000)) + (
                ("knock noise velocity slope dB/u", lambda d: d["noise.knock_vel"] * 20 / math.log(10)),
                ("knock noise decay x", lambda d: torch.exp(bounded(d["noise.raw_knock_tau"], 1.0))),
                ("knock impulse dB", lambda d: bounded(d["physics.raw_impulse_db"], 20.0)))
        for name, fn in per_key:
            a, b_ = fn(snap), fn(now)
            d = (b_ / a) if name.endswith("x") else (b_ - a)
            rows.append([name + (" (after / before)" if name.endswith("x") else " (after − before)")]
                        + [f"{float(d[k]):.3f}" if name.endswith("x") else f"{float(d[k]):+.2f}" for k in keys])
    L += ["", "Per key (every 6th key from MIDI 24):", ""]
    L += fmt_table(rows, ["MIDI"] + [str(k + 21) for k in keys])
    scal = []
    show = lambda t: ", ".join(f"{float(x):+.3f}" for x in t.reshape(-1))
    for n in ("physics.cond_log_tc", "physics.cond_vel_slope", "physics.cond_vel_curve", "room.mic_gain_db"):
        if n in names:
            scal.append(f"`{n}`: {show(snap[n][cond])} → {show(params[n].detach()[cond])}")
    for n in ("noise.thump_log_gain", "noise.raw_thump_tau", "physics.raw_impulse_vel"):
        if n in names:
            scal.append(f"`{n}` (raw): {show(snap[n])} → {show(params[n].detach())}")
    L += [""] + [f"- {s}" for s in scal]
    text = "\n".join(L) + "\n"
    with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8") as f:
        f.write(text)
    log(text)


if __name__ == "__main__":
    main()
