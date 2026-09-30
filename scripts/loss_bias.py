"""Level bias of the loss terms on real excerpts: the gate for any loss change (docs/plan_round2.md, A1).

    python scripts/loss_bias.py runs/round1_trial/main/best.pt data/maestro24k --years 2018

For each held-out excerpt and octave band, the model's render is turned up or down in that band only, and
each term's optimum gain is compared with the gain that matches the model's energy in the band to the
recording's. ``bias = optimum - energy match`` (dB, median over excerpts): an unbiased term reads 0 whatever
the model's shape errors. A term passes if the median over all excerpts and bands is within +-0.5 dB and
every band's median within +-1 dB.

Blank cells: the term does not cover that band (the fine term stops at 2 kHz, the attack term starts at 400 Hz).

Also: can each term see a doubled inharmonicity in the bass (MIDI 21-50)? Its distance between the model and
the model with B x 2 is compared with its distance between two noise seeds of the same model.
"""

import argparse
import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.6,max_split_size_mb:256")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch  # noqa: E402

from pianonn.config import PianoConfig  # noqa: E402
from pianonn.data import MaestroSegments  # noqa: E402
from pianonn.metrics import bias_report, inharmonicity_visibility, level_bias  # noqa: E402
from pianonn.render import load_weights  # noqa: E402
from pianonn.synth import NeuralPhysicalPiano  # noqa: E402
from pianonn.train import fixed_batches  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt")
    ap.add_argument("data")
    ap.add_argument("--years", type=int, nargs="*", default=[2018])
    ap.add_argument("--split", default="test")
    ap.add_argument("--examples", type=int, default=24)
    ap.add_argument("--residual", type=int, default=-1, help="1/0: residual on/off (default: on if the checkpoint is in stage 2)")
    ap.add_argument("--out", help="write the table here (markdown)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.6)
    state = torch.load(args.ckpt, map_location=dev)
    cfg = PianoConfig.from_dict(state["cfg"])
    model = load_weights(NeuralPhysicalPiano(cfg), state["model"]).to(dev).eval()
    residual = bool(args.residual) if args.residual >= 0 else state.get("stage", 1) >= 2
    ds = MaestroSegments(args.data, args.split, cfg, 2.0, 1.0, 12.0, length=args.examples, deterministic=True,
                         years=args.years, seed=11)
    batches = fixed_batches(ds, args.examples, 8, dev)
    bias, err = level_bias(model, batches, residual)
    vis = inharmonicity_visibility(model, batches, residual)
    lines = [f"# Level bias of the loss terms: `{args.ckpt}`, {args.examples} {args.split} excerpts, residual {'on' if residual else 'off'}",
             "", "Bias = the term's optimum gain in the band minus the energy-matching gain (dB, median over excerpts).", ""]
    lines += bias_report(bias, err)
    lines += ["", "Visibility of bass B x 2 (MIDI 21-50): distance to the unperturbed model / distance between two noise "
              "seeds (> 1: the term sees it beyond the noise).", "", "| term | ratio |", "|---|---|"]
    lines += [f"| {k} | {v:.2f} |" for k, v in vis.items()]
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    main()
