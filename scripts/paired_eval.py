"""Paired composite scores of models on the same excerpts, with their uncertainty (docs/physics_revamp.md 12).

    python scripts/paired_eval.py data/maestro24k --out runs/physics_revamp/paired \\
        --model B_comp=runs/loss_compare/B_comp/train/last.pt:physics \\
        --model coupled=runs/physics_revamp/coupled_run/train/last.pt:physics

Each model (``load_variant`` specs: ``label=checkpoint:physics|residual[:key=value,...]``) renders the same excerpts of
each set (2018 validation, test and training pieces by default; ``--examples`` each, the training's validation draws,
seed 1, first): two draws per excerpt (seeds 0 and 1, the energy score's second draw), each render level matched to its
piece's recording on the piece's other excerpts in the set, as ``pianonn.train.validate_composite``. The read-by-read
terms and the onset term are kept per excerpt (batches of one), the pooled terms' cells per excerpt too, so any subset
of excerpts can be scored again: the composite of a set is the pooled terms over its excerpts plus the means of the
others, each less half its distance to the second draw.

The uncertainty is a bootstrap over pieces (excerpts of one piece share its recording, its music and its level, so they
are not independent): ``--boot`` resamples of each set's pieces with replacement, every model scored on the same
resample. Reported: each model's composite with its standard error, each model's difference from the first with a 95 %
interval, per term; the held-out sets pooled (validation + test); the difference between held-out and training pieces
of each model's difference from the first (the sign of over-fitting: does a model gain on the training pieces what it
does not gain on the others?); per piece, how much leaving it out moves each difference.

Per-excerpt data are saved (``<out>/data/<set>__<label>.pt``) and reused when present, so models can be added later.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from pianonn.composite import POOLED, CompositeLoss, scored_notes  # noqa: E402
from pianonn.data import MaestroSegments  # noqa: E402
from pianonn.losses import highpass  # noqa: E402
from pianonn.metrics import piece_gains_loo  # noqa: E402
from pianonn.partial_view import POOL_MIN, pooled_across  # noqa: E402
from pianonn.render import load_variant  # noqa: E402
from pianonn.train import fixed_batches  # noqa: E402

READ = ("band", "partials", "between", "onset")


@torch.no_grad()
def score_set(model, residual, batches, comp, log):
    """Per-excerpt terms of one model on one set: ``{"read": {term: [n]}, "cells": {term: (P, T, cnt, eps)},
    "piece": [n], "gain_db": [n]}`` with every term also as ``<term>_self`` (against the second draw)."""
    model.eval()
    comp.eval()
    sr = model.cfg.sample_rate
    renders, e_y, e_t, groups = [], [], [], []
    for b in batches:
        n, s = b["audio"].shape[-1], int(b["loss_start"][0])
        draw = lambda seed: model(b, n, residual=residual,
                                  generator=torch.Generator(device=b["audio"].device).manual_seed(seed))
        out = draw(0)
        full, second = out["audio"].float(), draw(1)["audio"].float()
        renders.append((full.cpu(), second.cpu(), out["partials"]))
        energy_of = lambda x: highpass(x[..., s:], sr).pow(2).sum((1, 2)).double().cpu()
        e_y.append(energy_of(full)), e_t.append(energy_of(b["audio"].float())), groups.append(b["piece"].cpu())
    gains, _ = piece_gains_loo(torch.cat(e_y).numpy(), torch.cat(e_t).numpy(), torch.cat(groups).numpy())
    read = {}
    cells = {}
    for i, (b, (full, second, partials)) in enumerate(zip(batches, renders)):
        dev = b["audio"].device
        n, s = b["audio"].shape[-1], int(b["loss_start"][0])
        g = 10 ** (gains[i] / 20)
        full, second = full.to(dev) * g, second.to(dev) * g
        (p, crop), (t, _) = comp.scored(full, s), comp.scored(b["audio"], s)
        notes = scored_notes(partials, b, s, sr)
        cache = {}
        terms, sums = comp.read(p, t, notes, cache=cache, crop=crop)
        terms["onset"] = comp.onset(full, b["audio"].float(), b, s / sr, cache=cache)
        own, own_sums = comp.read(p, comp.scored(second, s)[0], notes, cache=cache, crop=crop)
        own["onset"] = comp.self_onset(full, second, b, s / sr, cache=cache)
        terms.update({k + "_self": v for k, v in own.items()})
        sums.update({k + "_self": v for k, v in own_sums.items()})
        for k, v in terms.items():
            read.setdefault(k, []).append(float(v))
        for k, c in sums.items():
            eps = c[3] if c[3].dim() == 2 else c[3][None]
            cells.setdefault(k, []).append(tuple(x.detach().float().cpu() for x in (c[0], c[1], c[2], eps)))
        if (i + 1) % 64 == 0:
            log(f"    {i + 1}/{len(batches)}")
    return {"read": {k: torch.tensor(v) for k, v in read.items()},
            "cells": {k: tuple(torch.cat([c[j] for c in cs]) for j in range(4)) for k, cs in cells.items()},
            "piece": torch.cat(groups), "gain_db": torch.tensor(gains)}


def composite(d, idx, w, dev):
    """The composite and its terms (each d - d_self / 2) of the excerpts ``idx`` (repeats allowed) of one model's data."""
    terms = {}
    for k in READ:
        terms[k] = float(d["read"][k][idx].mean() - 0.5 * d["read"][k + "_self"][idx].mean())
    for k in POOLED:
        v = []
        for kk in (k, k + "_self"):
            P, T, cnt, eps = (x[idx].to(dev) for x in d["cells"][kk])
            v.append(float(pooled_across(P, T, cnt, eps.amin(0), POOL_MIN.get(k, 1))))
        terms[k] = v[0] - 0.5 * v[1]
    return sum(w[k] * terms[k] for k in terms if w.get(k, 0)), terms


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data")
    ap.add_argument("--model", action="append", required=True, help="label=checkpoint:physics|residual[:options]")
    ap.add_argument("--sets", nargs="*", default=["validation", "test", "train"])
    ap.add_argument("--examples", type=int, default=256)
    ap.add_argument("--years", type=int, nargs="*", default=[2018])
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    dev = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(0.5)
    os.makedirs(os.path.join(args.out, "data"), exist_ok=True)
    log_f = open(os.path.join(args.out, "paired.log"), "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    labels = [s.split("=", 1)[0] for s in args.model]
    data = {}
    sets, comp, piece_names = {}, None, {}
    for spec in args.model:
        label = spec.split("=", 1)[0]
        todo = [st for st in args.sets if not os.path.exists(os.path.join(args.out, "data", f"{st}__{label}.pt"))]
        if todo:
            _, model, residual = load_variant(spec, device=dev, log=lambda m: None)
            if comp is None:
                comp = CompositeLoss(model.cfg.sample_rate).to(dev)
            for st in todo:
                if st not in sets:
                    ds = MaestroSegments(args.data, st, model.cfg, 2.0, 1.0, 12.0, length=args.examples,
                                         deterministic=True, years=args.years, seed=1)
                    sets[st] = fixed_batches(ds, args.examples, 1, dev)
                    piece_names[st] = [p["id"] for p in ds.pieces]
                log(f"{label}: {st} ({args.examples} excerpts)")
                d = score_set(model, residual, sets[st], comp, log)
                d["spec"] = spec
                torch.save(d, os.path.join(args.out, "data", f"{st}__{label}.pt"))
            del model
            torch.cuda.empty_cache()
        for st in args.sets:
            data[(st, label)] = torch.load(os.path.join(args.out, "data", f"{st}__{label}.pt"))
    if comp is None:
        comp = CompositeLoss(24000).to(dev)
    w = comp.w
    terms_order = list(READ) + list(POOLED)

    # the held-out sets pooled: validation + test (piece ids offset so they stay distinct)
    groups = {st: data[(st, labels[0])]["piece"].numpy() for st in args.sets}
    for st in args.sets:
        for lab in labels:
            assert np.array_equal(data[(st, lab)]["piece"].numpy(), groups[st]), (st, lab, "different excerpts")
    held = [st for st in args.sets if st != "train"]
    if len(held) > 1:
        for lab in labels:
            ds_ = [data[(st, lab)] for st in held]
            data[("held-out", lab)] = {
                "read": {k: torch.cat([d["read"][k] for d in ds_]) for k in ds_[0]["read"]},
                "cells": {k: _cat_cells([d["cells"][k] for d in ds_]) for k in ds_[0]["cells"]},
                "piece": torch.cat([d["piece"] + 1000 * j for j, d in enumerate(ds_)])}
        groups["held-out"] = data[("held-out", labels[0])]["piece"].numpy()
    report_sets = args.sets + (["held-out"] if len(held) > 1 else [])

    rng = np.random.default_rng(0)
    point, boots = {}, {}
    for st in report_sets:
        g = groups[st]
        pieces = np.unique(g)
        members = [np.flatnonzero(g == p) for p in pieces]
        resamples = [np.concatenate([members[j] for j in rng.integers(0, len(pieces), len(pieces))])
                     for _ in range(args.boot)]
        for lab in labels:
            d = data[(st, lab)]
            point[(st, lab)] = composite(d, np.arange(len(g)), w, dev)
            boots[(st, lab)] = [composite(d, ix, w, dev) for ix in resamples]
        log(f"bootstrap {st} done ({len(pieces)} pieces)")

    def vec(st, lab, k=None):
        return np.array([b[0] if k is None else b[1][k] for b in boots[(st, lab)]])

    lines = [f"# Paired composite scores ({args.examples} excerpts per set, {args.boot} piece resamples)", "",
             "Models: " + "; ".join(f"`{s}`" for s in args.model), "",
             "Each cell: the score on the whole set, ± its bootstrap standard error over pieces. Differences: model − "
             f"`{labels[0]}` on the same excerpts, 95 % interval in brackets (percentiles of the paired resamples).", ""]
    for st in report_sets:
        n_p = len(np.unique(groups[st]))
        lines += [f"## {st} ({len(groups[st])} excerpts, {n_p} pieces)", "",
                  "| model | composite | " + " | ".join(terms_order) + " |", "|---|" + "---|" * (len(terms_order) + 1)]
        for lab in labels:
            tot, tm = point[(st, lab)]
            cells = [f"{tm[k]:.4f}" for k in terms_order]
            lines.append(f"| {lab} | {tot:.4f} ± {vec(st, lab).std():.4f} | " + " | ".join(cells) + " |")
        for lab in labels[1:]:
            dt = point[(st, lab)][0] - point[(st, labels[0])][0]
            bd = vec(st, lab) - vec(st, labels[0])
            cells = []
            for k in terms_order:
                dk = point[(st, lab)][1][k] - point[(st, labels[0])][1][k]
                bk = vec(st, lab, k) - vec(st, labels[0], k)
                cells.append(f"{dk:+.4f} [{np.percentile(bk, 2.5):+.4f}, {np.percentile(bk, 97.5):+.4f}]")
            lines.append(f"| {lab} − {labels[0]} | {dt:+.4f} [{np.percentile(bd, 2.5):+.4f}, {np.percentile(bd, 97.5):+.4f}] | "
                         + " | ".join(cells) + " |")
        lines.append("")
    if "train" in args.sets and len(labels) > 1:
        lines += ["## Held-out against training pieces", "",
                  f"Each model's difference from `{labels[0]}` on the held-out pieces less the same on the training "
                  "pieces (independent resamples of the two sets): positive where the model gains more on the training "
                  "pieces than on the others.", "", "| model | sets | difference of differences [95 %] |", "|---|---|---|"]
        for lab in labels[1:]:
            for st in [s for s in report_sets if s != "train"]:
                dd = ((point[(st, lab)][0] - point[(st, labels[0])][0])
                      - (point[("train", lab)][0] - point[("train", labels[0])][0]))
                bd = (vec(st, lab) - vec(st, labels[0])) - (vec("train", lab) - vec("train", labels[0]))
                lines.append(f"| {lab} | {st} − train | {dd:+.4f} [{np.percentile(bd, 2.5):+.4f}, "
                             f"{np.percentile(bd, 97.5):+.4f}] |")
        lines.append("")
    if len(labels) > 1:
        lines += ["## Pieces that move the differences most", "",
                  "Leave one piece out: the change in each model's difference from the first (positive: the piece "
                  "makes the model look better than without it).", ""]
        for st in report_sets:
            g = groups[st]
            pieces = np.unique(g)
            for lab in labels[1:]:
                base = point[(st, lab)][0] - point[(st, labels[0])][0]
                infl = []
                for p in pieces:
                    ix = np.flatnonzero(g != p)
                    dlo = composite(data[(st, lab)], ix, w, dev)[0] - composite(data[(st, labels[0])], ix, w, dev)[0]
                    infl.append((dlo - base, int(p), int((g == p).sum())))
                infl.sort(key=lambda x: -abs(x[0]))
                names = piece_names.get(st)
                lines.append(f"- {st}, {lab}: " + "; ".join(
                    f"{(names[p] if names and p < len(names) else p)} ({n} exc.) {v:+.4f}" for v, p, n in infl[:4]))
        lines.append("")
    with open(os.path.join(args.out, "paired.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log("\n".join(lines))


def _cat_cells(cs):
    """Concatenate per-excerpt cells of several sets; the last axes can differ in size only if the views differ."""
    return tuple(torch.cat([c[j] for c in cs]) for j in range(4))


if __name__ == "__main__":
    main()
