"""Which interaction tables the training loss consistently pushes (docs/physics_revamp.md 7, 10 step 5), before any run.

At the identity (every table zero) the training step's gradient is taken on N batches of the training set, as train.py
computes it (the composite as an energy score: a second draw without gradient, the piece gains of the checkpoint, the
per-strike variation on, the physics alone). Per table and per reference parameter:

- agreement: the mean cosine between the gradients of two different batches (0 for pure noise, at most 1); its
  standard error from the per-batch means;
- cells: the share of the table's cells whose gradient has the same sign batch after batch, |mean / (sd / sqrt N)| > 3
  (0.3 % for pure noise); cells never reached (gradient 0 in every batch) are left out;
- the gradient's size: the norm of the mean gradient against the typical batch's (1 / sqrt N for pure noise).

The reference parameters are ones the model already trains (aftersound decay, the strings' pitch offsets, the bridge
loss, the per-key level), to read the tables' numbers against.

    python scripts/coherence_probe.py --ckpt runs/physics_revamp/start/model.pt --out runs/physics_revamp/probe
"""

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch  # noqa: E402

from pianonn.composite import CompositeLoss  # noqa: E402
from pianonn.data import MaestroSegments  # noqa: E402
from pianonn.interactions import TABLES  # noqa: E402
from pianonn.render import load_model  # noqa: E402
from pianonn.train import fixed_batches  # noqa: E402

REFERENCE = ["physics.raw_log_b1", "physics.raw_prompt", "physics.gain_db", "physics.coupled.raw_cents",
             "physics.coupled.raw_sh"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="data/maestro24k")
    ap.add_argument("--years", type=int, nargs="*", default=[2018])
    ap.add_argument("--batches", type=int, default=24)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = torch.device(args.device)
    torch.cuda.set_per_process_memory_fraction(0.8)
    os.makedirs(args.out, exist_ok=True)
    log_f = open(os.path.join(args.out, "probe.log"), "w", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        log_f.write(msg + "\n")

    model = load_model(args.ckpt, device=dev, ema=False, interactions=True)
    model.train()
    model.strike_on = True
    cfg = model.cfg
    sr = cfg.sample_rate
    state = torch.load(args.ckpt, map_location="cpu")
    saved = dict(zip(state["piece_gain"]["ids"], state["piece_gain"]["db"].tolist())) if "piece_gain" in state else {}
    ds = MaestroSegments(args.data, "train", cfg, 2.0, 1.0, 12.0, length=args.batches * args.batch, deterministic=True,
                         years=args.years, seed=args.seed)
    piece_db = torch.tensor([saved.get(p["id"], 0.0) for p in ds.pieces], device=dev)
    comp = CompositeLoss(sr).to(dev)
    names = {f"physics.inter.raw_{t}": t for t in TABLES}
    params = dict(model.named_parameters())
    watch = list(names) + [k for k in REFERENCE if k in params]
    for p in model.parameters():
        p.requires_grad_(False)
    for k in watch:
        params[k].requires_grad_(True)
    grads = {k: [] for k in watch}
    scores = []
    log(f"{args.ckpt}: {args.batches} batches of {args.batch} training excerpts of {args.years}, physics only")
    for i, b in enumerate(fixed_batches(ds, args.batches * args.batch, args.batch, dev)):
        n, s = b["audio"].shape[-1], int(b["loss_start"][0])
        g = (10 ** (piece_db[b["piece"]] / 20))[:, None, None]
        model.zero_grad(set_to_none=True)
        with model.physics.shared_core(b["condition"]):
            with torch.no_grad():
                second = model(b, n, residual=False)["audio"].float() * g
            out = model(b, n, residual=False)
        score, _ = comp((out["audio"] * g).float(), b["audio"].float(), b, s, out["partials"], second)
        score.backward()
        for k in watch:
            gk = params[k].grad
            grads[k].append(torch.zeros_like(params[k]).flatten() if gk is None else gk.detach().flatten().clone())
        scores.append(float(score))
        out = second = score = None
        if (i + 1) % 8 == 0:
            log(f"  batch {i + 1}/{args.batches}: score {sum(scores[-8:]) / 8:.4f}")
    torch.save({k: torch.stack(v).cpu() for k, v in grads.items()}, os.path.join(args.out, "grads.pt"))
    rows = {}
    for k in watch:
        G = torch.stack(grads[k]).double()  # [N, d]
        N = G.shape[0]
        reached = G.abs().sum(0) > 0
        G = G[:, reached]
        if G.shape[1] == 0:
            rows[k] = None
            continue
        U = G / G.norm(dim=1, keepdim=True).clamp(min=1e-300)
        C = U @ U.T
        off = ~torch.eye(N, dtype=torch.bool, device=G.device)
        per = (C * off).sum(1) / (N - 1)  # each batch's mean agreement with the others
        mean, sd = G.mean(0), G.std(0).clamp(min=1e-300)
        t = mean / (sd / math.sqrt(N))
        rows[k] = {"agreement": float(per.mean()), "agreement_se": float(per.std() / math.sqrt(N)),
                   "cells": int(G.shape[1]), "cells_t3": float((t.abs() > 3).double().mean()),
                   "mean_vs_batch_norm": float(mean.norm() / G.norm(dim=1).mean()), "null_mean_vs_batch": 1 / math.sqrt(N)}
    log("\n{:<34s} {:>14s} {:>7s} {:>10s} {:>12s}".format("parameter", "agreement", "cells", "|t|>3", "|mean|/|g|"))
    for k in watch:
        r = rows[k]
        label = ("table " + names[k]) if k in names else k
        if r is None:
            log(f"{label:<34s} never reached")
            continue
        log(f"{label:<34s} {r['agreement']:+.3f} ± {r['agreement_se']:.3f} {r['cells']:7d} {100 * r['cells_t3']:9.1f} % "
            f"{r['mean_vs_batch_norm']:12.3f}")
    log(f"(pure noise: agreement 0, |t|>3 in 0.3 % of cells, |mean|/|g| {1 / math.sqrt(args.batches):.3f})")
    with open(os.path.join(args.out, "probe.json"), "w") as f:
        json.dump({"args": vars(args), "rows": rows, "scores": scores}, f, indent=1)


if __name__ == "__main__":
    main()
