"""Paired comparison of checkpoints on the same held-out excerpts, with the render-noise spread for scale.

    python scripts/compare_runs.py data/maestro24k --years 2018 --out runs/round2/compare.md \\
        --model control=runs/round2/b_control/last.pt:physics --model residual=runs/round2/a_residual/last.pt:residual

Each ``--model`` is ``label=checkpoint:physics|residual``. Every model renders the excerpts of
``scripts/evaluate.py`` (test split, seed 11) under several noise seeds. Reported, per distance:
* each model's mean, and how far that mean moves between noise seeds (the spread a difference must beat);
* for every model against the first one, the paired per-excerpt difference, averaged over the seeds, with
  two standard errors over excerpts, and in how many seeds its sign agrees.
One training seed per model, so the spread between training runs is not in these numbers.
"""

import argparse
import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.6,max_split_size_mb:256")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from pianonn.data import MaestroSegments  # noqa: E402
from pianonn.metrics import make_terms, render  # noqa: E402
from pianonn.render import load_model  # noqa: E402
from pianonn.train import fixed_batches  # noqa: E402

KEYS = ("new: total", "new: band", "new: fine", "new: attack", "old: MR-STFT", "log-mel (dB)")


@torch.no_grad()
def per_excerpt(model, batches, residual, seed):
    terms = make_terms(model.cfg.sample_rate, batches[0]["audio"].device)
    acc = {k: [] for k in KEYS}
    for b in batches:
        s = int(b["loss_start"][0])
        d = terms(render(model, b, residual, seed), b["audio"][..., s:], b["onset"] - s / model.cfg.sample_rate, b["mask"])
        for k in KEYS:
            acc[k].append(d[k].cpu())
    return {k: torch.cat(v).numpy() for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data")
    ap.add_argument("--years", type=int, nargs="*", default=[2018])
    ap.add_argument("--split", default="test")
    ap.add_argument("--examples", type=int, default=96)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--model", action="append", required=True, help="label=checkpoint:physics|residual")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.6)
    specs = []
    for m in args.model:
        label, rest = m.split("=", 1)
        ckpt, mode = rest.rsplit(":", 1)
        specs.append((label, ckpt, mode == "residual"))

    res = {}  # label -> {key: [seeds, excerpts]}
    batches = None
    for label, ckpt, residual in specs:
        model = load_model(ckpt, device=dev)
        if batches is None:
            ds = MaestroSegments(args.data, args.split, model.cfg, 2.0, 1.0, 12.0, length=args.examples,
                                 deterministic=True, years=args.years, seed=11)
            batches = fixed_batches(ds, args.examples, 8, dev)
        runs = [per_excerpt(model, batches, residual, s) for s in range(args.seeds)]
        res[label] = {k: np.stack([r[k] for r in runs]) for k in KEYS}
        del model
        torch.cuda.empty_cache()
        print("done", label, flush=True)

    fmt = lambda k, v: f"{v:.2f}" if k == "log-mel (dB)" else f"{v:.4f}"
    lines = [f"# Paired comparison on {args.examples} {args.split} excerpts, {args.seeds} noise seeds", "",
             "Models: " + "; ".join(f"{lab} = `{ck}` ({'with' if r else 'without'} the residual)" for lab, ck, r in specs), "",
             "## Means", "", "Mean over excerpts, averaged over seeds; in brackets, the range of that mean across seeds.", "",
             "| | " + " | ".join(KEYS) + " |", "|---|" + "---|" * len(KEYS)]
    for label in res:
        cells = []
        for k in KEYS:
            m = res[label][k].mean(1)
            cells.append(f"{fmt(k, m.mean())} [{fmt(k, m.max() - m.min())}]")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    base = specs[0][0]
    lines += ["", f"## Paired differences against `{base}`", "",
              "Per excerpt, model − base, averaged over seeds: mean ± 2 standard errors over excerpts; then the number "
              "of seeds whose mean difference has the same sign. Negative means the model is nearer the recording.", "",
              "| | " + " | ".join(KEYS) + " |", "|---|" + "---|" * len(KEYS)]
    for label in list(res)[1:]:
        cells = []
        for k in KEYS:
            d = res[label][k] - res[base][k]  # [seeds, excerpts]
            per_ex = d.mean(0)
            se = per_ex.std(ddof=1) / np.sqrt(len(per_ex))
            agree = int((np.sign(d.mean(1)) == np.sign(per_ex.mean())).sum())
            cells.append(f"{fmt(k, per_ex.mean())} ± {fmt(k, 2 * se)} ({agree}/{args.seeds})")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    text = "\n".join(lines) + "\n"
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
