"""The composite score of training checkpoints on held-out and on training pieces, level matched per piece.

    python scripts/composite_eval.py data/maestro24k --out runs/composite_train/smoke2/eval.md \\
        --model start=runs/phase6/train_run/train/last.pt --model end=runs/composite_train/smoke2/train/last.pt

Each checkpoint (its averaged weights where it has them) is scored as ``pianonn.train.validate_composite`` scores it
in training (the energy score: each term less half its distance to a second draw; per-strike variation as the
checkpoint's config), with and without its residual, on ``--examples`` excerpts of the validation pieces and as many
of the training pieces (the training's own validation draws, seed 1, plus more), every render level matched to its
piece's recording on the piece's other excerpts. Written: a table per set, the terms per model.
"""

import argparse
import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.6,max_split_size_mb:256")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch  # noqa: E402

from pianonn.composite import CompositeLoss  # noqa: E402
from pianonn.data import MaestroSegments  # noqa: E402
from pianonn.losses import PianoLoss  # noqa: E402
from pianonn.render import load_variant  # noqa: E402
from pianonn.train import fixed_batches, validate_composite  # noqa: E402

TERMS = ("band", "partials", "between", "pooled_exposed", "between_pooled", "level", "onset")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data")
    ap.add_argument("--model", action="append", required=True, help="label=checkpoint (scored with and without the residual)")
    ap.add_argument("--examples", type=int, default=64)
    ap.add_argument("--years", type=int, nargs="*", default=[2018])
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--no-level-match", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    dev = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(0.3)
    rows = {}
    sets = None
    for spec in args.model:
        label, ck = spec.split("=", 1)
        _, model, _ = load_variant(f"{label}={ck}:residual", device=dev, log=lambda m: None)
        sr = model.cfg.sample_rate
        if sets is None:
            sets = {split: fixed_batches(MaestroSegments(args.data, split, model.cfg, 2.0, 1.0, 12.0, length=args.examples,
                                                         deterministic=True, years=args.years, seed=1),
                                         args.examples, args.batch, dev)
                    for split in ("validation", "train")}
        comp, old = CompositeLoss(sr).to(dev), PianoLoss(sr, level_weight=0.5).to(dev)
        for split, batches in sets.items():
            for residual in (False, True):
                total, per = validate_composite(model, batches, comp, residual, True, old,
                                                level_match=not args.no_level_match)
                rows[(split, label, residual)] = (total, per)
                print(f"{split:10s} {label:12s} {'residual' if residual else 'physics':8s} {total:.4f} old {per['old']:.4f}",
                      flush=True)
        del model
        torch.cuda.empty_cache()
    lines = [f"# Composite score of checkpoints ({args.examples} excerpts per set, "
             f"{'not ' if args.no_level_match else ''}level matched per piece)", "",
             "The energy score as training's validation computes it (`pianonn.train.validate_composite`): each term "
             "d(render, recording) - d(render, second draw) / 2, the pooled terms pooled over the set, per-strike variation "
             "as in each checkpoint's config. `old`: PianoLoss with its level term at 0.5 on the first draw. Models: "
             + "; ".join(f"{s.split('=', 1)[0]} = `{s.split('=', 1)[1]}`" for s in args.model), ""]
    for split in ("validation", "train"):
        lines += [f"## {'Held-out (validation) pieces' if split == 'validation' else 'Training pieces'}", "",
                  "| model | composite | old | " + " | ".join(TERMS) + " |", "|---|" + "---|" * (len(TERMS) + 2)]
        for spec in args.model:
            label = spec.split("=", 1)[0]
            for residual in (False, True):
                total, per = rows[(split, label, residual)]
                cells = [f"{per[k] - 0.5 * per.get(k + '_self', 0.0):.4f}" for k in TERMS]
                lines.append(f"| {label} {'with' if residual else 'without'} the residual | {total:.4f} | {per['old']:.4f} | "
                             + " | ".join(cells) + " |")
        lines.append("")
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
