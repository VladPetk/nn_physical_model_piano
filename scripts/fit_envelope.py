"""Fit the strings' decay on the whole note's envelope (N13; docs/tone_measures.md 17).

    python scripts/fit_envelope.py data/maestro24k --model p5=runs/phase5/train_run/train/last.pt:physics \\
        --out runs/phase6/env_fit

Events: N13's (``scripts/note_envelope.py``'s ``mine``: notes of MIDI 36-88 in any texture, read while they sound
freely, at most 3 s), drawn with ``--seed``; the training pieces' are fitted, the validation and test pieces' held out.
The term (``pianonn.envfit``): the L1 distance of each partial's fade (0.2-2.5 s re 0.1 s) between the model's render
and the recording, over N13's cells, plus the model's smoothness regulariser. Fitted: the strings' decay (per-key
prompt loss, b1 and b3, the loss exponent, the bridge conductance over frequency) and the aftersound's level per key
and mode; per-key tables move by piecewise-linear corrections with knots every ``--key-step`` keys. Writes
``model.pt``, ``report.md`` (the term and its medians by partial group and time, before and after, both groups) and
``log.txt``.
"""

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import note_envelope as NE  # noqa: E402
from pianonn import envfit as EF  # noqa: E402
from pianonn import measures as M  # noqa: E402
from pianonn import notefit as NF  # noqa: E402
from pianonn.config import year_to_condition  # noqa: E402
from pianonn.render import load_variant  # noqa: E402
from pianonn.train import lr_scale  # noqa: E402

FIT = ("physics.raw_prompt", "physics.raw_log_b1", "physics.raw_log_b3", "physics.raw_bridge_g", "physics.raw_decay_p",
       "physics.raw_after")
PER_KEY = ("physics.raw_prompt", "physics.raw_log_b1", "physics.raw_log_b3", "physics.raw_after")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data")
    ap.add_argument("--model", required=True, help="label=checkpoint:physics[:options] (pianonn.render.load_variant)")
    ap.add_argument("--year", type=int, default=2018)
    ap.add_argument("--cap", type=int, default=15, help="as scripts/note_envelope.py --cap")
    ap.add_argument("--seed", type=int, default=1, help="event draw (the measure's own run used 0)")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--reg", type=float, default=1.0)
    ap.add_argument("--key-step", type=int, default=8)
    ap.add_argument("--max-notes", type=int, default=0, help="cap each group (smoke runs)")
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
    label, model, _ = load_variant(args.model, device=dev, log=log)
    cfg, cond = model.cfg, year_to_condition(args.year)
    sr = cfg.sample_rate
    table = M.partial_table(model, cond, dev)
    tabs = {p: table[p - 21][(table[p - 21] > 0) & (table[p - 21] < 6500)] for p in range(21, 109)}
    events, _ = NE.mine(args.data, args.year, args.cap, args.seed)
    groups = {}
    for g, splits in (("calib", ("train",)), ("eval", ("validation", "test"))):
        evs = [e for e in events if e["split"] in splits]
        evs = evs[:: max(1, len(evs) // args.max_notes)][: args.max_notes] if args.max_notes else evs
        groups[g] = EF.EnvelopeSet(args.data, evs, cfg, table, tabs, log=lambda m, g=g: log(f"{g}: {m}"))
    fit = groups["calib"]
    term = EF.EnvelopeTerm(sr)

    for p in model.parameters():
        p.requires_grad_(False)
    params = dict(model.named_parameters())
    names = [n for n in FIT if n in params]
    before = {g: EF.evaluate(model, s, term, dev) for g, s in groups.items()}
    knotted = [n for n in names if n in PER_KEY] if args.key_step > 0 else []
    knots = {f"physics.{n}": k for n, k in
             NF.smooth_per_key(model.physics, [n.split(".", 1)[1] for n in knotted], args.key_step).items()}
    train = [(n, params[n]) for n in names if n not in knotted] + list(knots.items())
    for n, p in train:
        p.requires_grad_(True)
    opt = torch.optim.Adam([{"params": [p], "lr": args.lr * lr_scale(n), "name": n} for n, p in train])
    for g in opt.param_groups:
        g["base_lr"] = g["lr"]
    log(f"fitting {', '.join(names)} on {len(fit)} notes ({args.steps} steps of {args.batch}); knots every "
        f"{args.key_step} keys on {', '.join(knotted) or 'none'}")
    rng = np.random.default_rng(args.seed)
    order, pos = rng.permutation(len(fit)), 0
    run, t0 = [], time.time()
    cvec = torch.tensor([cond], device=dev)
    for step in range(1, args.steps + 1):
        if pos + args.batch > len(order):
            order, pos = rng.permutation(len(fit)), 0
        idx = order[pos: pos + args.batch].tolist()
        pos += args.batch
        y = model(fit.batch(idx, dev), fit.n, residual=False,
                  generator=torch.Generator(device=dev).manual_seed(step))["audio"].float()
        tot, cnt = 0.0, 0
        for j, i in enumerate(idx):
            ev = fit.notes[i]
            s, c, _, _ = term.note(y[j], fit.rec[i], table[ev["pitch"] - 21], fit.t_on, ev["t_end"])
            tot, cnt = tot + s, cnt + c
        env = tot / max(cnt, 1)
        reg = model.physics.regularizer(cvec)
        loss = env + args.reg * reg
        opt.zero_grad(set_to_none=True)
        loss.backward()
        frac = step / args.steps
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
        opt.step()
        run.append((float(env), float(reg), cnt))
        if step % 10 == 0 or step == args.steps:
            r = np.mean(run[-10:], 0)
            log(f"step {step}: envelope {r[0]:.3f} dB over {r[2]:.0f} cells, reg {r[1]:.4f}, {(time.time() - t0) / step:.2f} s/step")

    for n in knots:
        NF.parametrize.remove_parametrizations(model.physics, n.split(".", 1)[1])
    model.eval()
    after = {g: EF.evaluate(model, s, term, dev) for g, s in groups.items()}
    torch.save({"cfg": cfg.to_dict(), "model": model.state_dict(), "source": args.model, "fit": "envelope", "step": 0},
               os.path.join(args.out, "model.pt"))
    np.savez_compressed(os.path.join(args.out, "eval.npz"),
                        **{f"{g}|{t}|{k}": v[k] for g in groups for t, v in (("before", before[g]), ("after", after[g])) for k in v})

    L = [f"# Envelope fit ({args.year})", "",
         f"Model: `{args.model}`. Fitted: {', '.join(f'`{n}`' for n in names)}; knots every {args.key_step} keys on the "
         f"per-key tables. {args.steps} steps of {args.batch} notes, lr {args.lr}, regulariser weight {args.reg}. "
         f"Calibration notes (training pieces): {len(fit)}; evaluation notes (validation and test pieces): "
         f"{len(groups['eval'])}. Term: N13 fades (each partial's level at 0.2-2.5 s re 0.1 s, held at its floor), "
         "model − recording, dB, over N13's cells.", "", "## The term (mean |model − recording| over the cells)", "",
         "| group | before | after | cells |", "|---|---|---|---|"]
    for g in groups:
        b_, a_ = before[g]["d"], after[g]["d"]
        L.append(f"| {g} | {np.nanmean(np.abs(b_)):.3f} | {np.nanmean(np.abs(a_)):.3f} | {int(np.isfinite(b_).sum())} |")
    times = [t for t in M.ENV_TIMES[1:]]
    for g in groups:
        sb = EF.summary(before[g], groups[g], NE.GROUPS, regs=[r for r, _, _ in NE.REGS])
        sa = EF.summary(after[g], groups[g], NE.GROUPS, regs=[r for r, _, _ in NE.REGS])
        for r in [None, "R2", "R3", "R4"]:
            L += ["", f"## {g}, {'all registers' if r is None else r}: median model − recording fade, before → after (dB)", "",
                  "| partials | " + " | ".join(f"{t:g} s" for t in times) + " |", "|---|" + "---|" * len(times)]
            for gname, _, _ in NE.GROUPS:
                cells = []
                for t in times:
                    (mb, nb), (ma, na) = sb[(r, gname, t)], sa[(r, gname, t)]
                    cells.append("" if nb < 5 else f"{mb:+.1f} → {ma:+.1f} ({nb})")
                L.append(f"| {gname} | " + " | ".join(cells) + " |")
    with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    log("\n".join(L))


if __name__ == "__main__":
    main()
