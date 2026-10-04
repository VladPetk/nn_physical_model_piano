"""The coupled strings' start from a trained mode model (docs/physics_revamp.md 10, step 4; 11): the mode checkpoint's
weights in a ``string_model=coupled`` model (``init_coupled``), the coupled parameters fitted to its partials' energy
envelopes and the longitudinal level calibrated against its phantoms (``fit_init.distill_coupled``). Writes a
checkpoint in train.py's format (``cfg``, ``model``, the mode run's piece gains), so ``train.py --init-from`` and the
scripts load it without distilling again, and ``distill.log`` beside it.

    python scripts/coupled_start.py runs/loss_compare/B_comp/train/best.pt runs/physics_revamp/start/model.pt
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch  # noqa: E402

from pianonn.config import PianoConfig, year_to_condition  # noqa: E402
from pianonn.fit_init import distill_coupled  # noqa: E402
from pianonn.render import load_weights  # noqa: E402
from pianonn.synth import NeuralPhysicalPiano  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode_ckpt")
    ap.add_argument("out")
    ap.add_argument("--year", type=int, default=2018, help="the condition the envelopes are fitted at")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--cfg", nargs="*", default=[], metavar="KEY=VALUE", help="further config overrides")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    from pianonn.render import _parse_value

    dev = torch.device(args.device)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    log_path = os.path.join(os.path.dirname(args.out) or ".", "distill.log")
    log_f = open(log_path, "w", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        log_f.write(msg + "\n")

    state = torch.load(args.mode_ckpt, map_location=dev)
    base = state["cfg"]
    assert base.get("string_model", "modes") != "coupled", "a mode model's checkpoint"
    overrides = {k: _parse_value(v) for k, v in (o.split("=", 1) for o in args.cfg)}
    teacher = NeuralPhysicalPiano(PianoConfig.from_dict(base)).to(dev)
    load_weights(teacher, state["model"], log=lambda *_: None)
    model = NeuralPhysicalPiano(PianoConfig.from_dict({**base, "string_model": "coupled", **overrides})).to(dev)
    load_weights(model, state["model"], log=log)
    log(f"from {args.mode_ckpt} (its training weights), condition of {args.year}")
    distill_coupled(model, teacher, year_to_condition(args.year), steps=args.steps, log=log)
    out = {"cfg": model.cfg.to_dict(), "model": model.state_dict(), "source": args.mode_ckpt}
    if "piece_gain" in state:
        out["piece_gain"] = state["piece_gain"]
    torch.save(out, args.out)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
