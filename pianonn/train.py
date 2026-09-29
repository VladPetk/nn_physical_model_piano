"""Fit the physical model to MAESTRO (or to a perturbed copy of itself with --synthetic)."""

import argparse
import itertools
import json
import os
import time

import torch
from torch.utils.data import DataLoader

from .config import PianoConfig
from .data import MaestroSegments, SyntheticPerformances, collate
from .losses import MultiResolutionDiscriminator, MultiResolutionSTFTLoss, discriminator_loss, generator_adv_loss
from .synth import NeuralPhysicalPiano


@torch.no_grad()
def perturb_physics(model, scale=0.3, seed=0):
    """Randomise the learnable physical offsets: a stand-in 'real piano' for sanity checks."""
    g = torch.Generator().manual_seed(seed)
    for name, p in model.physics.named_parameters():
        if name.startswith("raw_") or name == "partial_gain":
            p.add_(scale * torch.randn(p.shape, generator=g))
    return model


def param_groups(model, lr, ir_lr_scale=0.03):
    """Thousands of FIR taps each taking a full Adam step drown out the physics, so the IR learns slower."""
    rest = [p for name, p in model.named_parameters() if name != "ir"]
    return [{"params": rest, "lr": lr}, {"params": [model.ir], "lr": lr * ir_lr_scale}]


def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", help="directory produced by scripts/prepare_maestro.py")
    ap.add_argument("--synthetic", action="store_true", help="fit a randomly perturbed teacher instead of MAESTRO")
    ap.add_argument("--out", default="runs/default")
    ap.add_argument("--sr", type=int, default=24000)
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--segment", type=float, default=2.0)
    ap.add_argument("--warmup", type=float, default=1.0)
    ap.add_argument("--lookback", type=float, default=4.0)
    ap.add_argument("--reg", type=float, default=1.0, help="weight of the key-smoothness regulariser")
    ap.add_argument("--adv-start", type=int, default=-1, help="step to switch on the GAN loss (-1: never)")
    ap.add_argument("--adv-weight", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=2000)
    args = ap.parse_args(argv)

    cfg = PianoConfig(sample_rate=args.sr)
    device = torch.device(args.device)
    model = NeuralPhysicalPiano(cfg).to(device)
    teacher = None
    if args.synthetic:
        seconds = args.warmup + args.segment
        dataset = SyntheticPerformances(cfg, seconds=seconds, length=args.steps * args.batch)
        teacher = perturb_physics(NeuralPhysicalPiano(cfg), seed=1).to(device).eval()
    else:
        dataset = MaestroSegments(args.data, "train", cfg, args.segment, args.warmup, args.lookback,
                                  length=args.steps * args.batch)
    loader = DataLoader(dataset, batch_size=args.batch, collate_fn=collate, num_workers=args.workers,
                        persistent_workers=args.workers > 0)

    recon = MultiResolutionSTFTLoss()
    opt = torch.optim.Adam(param_groups(model, args.lr))
    disc = disc_opt = None
    if args.adv_start >= 0:
        disc = MultiResolutionDiscriminator().to(device)
        disc_opt = torch.optim.Adam(disc.parameters(), lr=2e-4, betas=(0.5, 0.9))

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump({"model": cfg.to_dict(), "args": vars(args)}, f, indent=2)

    t_start = time.time()
    for step, batch in enumerate(itertools.islice(loader, args.steps), 1):
        batch = to_device(batch, device)
        if teacher is not None:
            n = int(round((args.warmup + args.segment) * cfg.sample_rate))
            with torch.no_grad():
                target = teacher(batch, n)["audio"]
            s = int(args.warmup * cfg.sample_rate)
        else:
            target, n, s = batch["audio"], batch["audio"].shape[-1], int(batch["loss_start"][0])
        pred = model(batch, n)["audio"]
        pred, target = pred[:, s:], target[:, s:]

        logs = {"recon": recon(pred, target), "reg": model.physics.regularizer()}
        loss = logs["recon"] + args.reg * logs["reg"]
        if disc is not None and step >= args.adv_start:
            disc_opt.zero_grad()
            logs["disc"] = discriminator_loss(disc, target, pred)
            logs["disc"].backward()
            disc_opt.step()
            logs["adv"], logs["fm"] = generator_adv_loss(disc, target, pred)
            loss = loss + args.adv_weight * (logs["adv"] + 2.0 * logs["fm"])

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % args.log_every == 0:
            msg = " ".join(f"{k}={float(v.detach()):.4f}" for k, v in logs.items())
            print(f"step {step} {msg} ({(time.time() - t_start) / step:.2f}s/step)", flush=True)
        if step % args.save_every == 0 or step == args.steps:
            torch.save({"cfg": cfg.to_dict(), "model": model.state_dict(), "step": step},
                       os.path.join(args.out, "last.pt"))


if __name__ == "__main__":
    main()
