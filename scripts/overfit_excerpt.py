"""Go/no-go (review 3, phase 1, step 2): overfit one MAESTRO excerpt with growing parameter sets.

From the same data-initialised prior, fit a single excerpt for the same number of steps with
(A) the recording chain and per-condition scalars only, (B) all the physics as well, and
(C) everything including the learned residual. If (B) cannot get well below (A), the physical
forms are missing something that matters on real audio; (C) - (B) is what the residual adds.

    python scripts/overfit_excerpt.py data/maestro24k --years 2018 --out runs/overfit
"""

import argparse
import copy
import json
import os
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.6,max_split_size_mb:256")

import soundfile as sf  # noqa: E402
import torch  # noqa: E402

from pianonn.config import PianoConfig  # noqa: E402
from pianonn.data import MaestroSegments  # noqa: E402
from pianonn.fit_init import initialise_from_data  # noqa: E402
from pianonn.losses import PianoLoss  # noqa: E402
from pianonn.synth import NeuralPhysicalPiano  # noqa: E402
from pianonn.train import fixed_batches, param_groups, pan_smoothness, set_stage  # noqa: E402

RUNGS = {
    "A_chain_and_scalars": dict(stage=1, only=("room.", "physics.cond_"), residual=False),
    "B_physics": dict(stage=1, only=None, residual=False),
    "C_everything": dict(stage=2, only=None, residual=True),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data")
    ap.add_argument("--years", type=int, nargs="*")
    ap.add_argument("--out", default="runs/overfit")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=7, help="which validation excerpt")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.8)
    torch.manual_seed(0)
    cfg = PianoConfig()
    os.makedirs(args.out, exist_ok=True)
    base = NeuralPhysicalPiano(cfg).to(dev)
    init_set = MaestroSegments(args.data, "train", cfg, 2.0, 1.0, 12.0, length=32, deterministic=True, years=args.years, seed=3)
    initialise_from_data(base, fixed_batches(init_set, 32, 8, dev), silence=init_set.silence_clips())
    ex_set = MaestroSegments(args.data, "validation", cfg, args.seconds, 1.0, 12.0, length=1, deterministic=True,
                             years=args.years, seed=args.seed)
    (batch,) = fixed_batches(ex_set, 1, 1, dev)
    n, s = batch["audio"].shape[-1], int(batch["loss_start"][0])
    target = batch["audio"][..., s:]
    sf.write(os.path.join(args.out, "target.wav"), batch["audio"][0].T.cpu().numpy(), cfg.sample_rate)
    recon = PianoLoss(cfg.sample_rate).to(dev)
    onsets = (batch["onset"] - s / cfg.sample_rate, batch["mask"])
    print(f"excerpt: piece {int(batch['piece'])} at {int(batch['start']) / cfg.sample_rate:.1f}s, "
          f"{int(batch['mask'].sum())} notes incl. lookback", flush=True)

    results = {}
    for name, rung in RUNGS.items():
        model = copy.deepcopy(base)
        opt = torch.optim.Adam(param_groups(model, args.lr))
        set_stage(model, opt, rung["stage"], physics_lr=1.0, only=rung["only"])
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        curve, t0 = [], time.time()
        for step in range(args.steps + 1):
            g = torch.Generator(device=dev).manual_seed(0)
            out = model(batch, n, residual=rung["residual"], generator=g)
            loss, _ = recon(out["audio"][..., s:], target, *onsets)
            if step % 10 == 0 or step == args.steps:
                curve.append((step, float(loss)))
            if step == args.steps:
                break
            total = loss + model.physics.regularizer(batch["condition"]) + pan_smoothness(model, batch["condition"])
            opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)  # spikes only
            opt.step()
        with torch.no_grad():
            g = torch.Generator(device=dev).manual_seed(0)
            y = model(batch, n, residual=rung["residual"], generator=g)["audio"][0]
            sf.write(os.path.join(args.out, f"{name}.wav"), y.T.clamp(-1, 1).cpu().numpy(), cfg.sample_rate)
        results[name] = {"trainable": n_train, "start": curve[0][1], "end": curve[-1][1], "curve": curve,
                         "seconds": time.time() - t0}
        print(f"{name}: {n_train} trainable, loss {curve[0][1]:.4f} -> {curve[-1][1]:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    with open(os.path.join(args.out, "ladder.json"), "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
