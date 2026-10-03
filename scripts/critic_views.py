"""What tells the model's renders from the recordings: a fresh critic trained on one restricted view of both sides at a
time, scored on held-out excerpts (AUC; 0.5 = cannot tell). A view in which it still separates them holds a cue; a view
in which it cannot, holds none it can find in this many steps. Several cues can each be enough on their own, so the
table says where cues are, not which one a full critic uses.

Why: the wide critic (``runs/gan_check/wide1``) told ``env_fit2``'s renders from recordings at AUC 1.000, but its push
did not follow the band error, and the GAN trained on it degraded the model (``runs/loss_compare/C_gan``).

    python scripts/critic_views.py --model runs/loss_compare/B_stage1/train/last.pt --out runs/gan_check/views1
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pianonn.data import MaestroSegments, collate  # noqa: E402
from pianonn.losses import discriminator_loss, highpass, make_discriminator  # noqa: E402
from pianonn.render import load_model  # noqa: E402
from pianonn.train import to_device  # noqa: E402


def band(lo, hi, sr):
    """A zero-phase band of ``x[..., T]`` between ``lo`` and ``hi`` Hz (raised-cosine edges a third of an octave wide)."""
    def f(x):
        fr = torch.fft.rfftfreq(x.shape[-1], 1 / sr).to(x.device)
        lf = torch.log2(fr.clamp(min=1.0))
        g = torch.ones_like(fr)
        w = 1 / 3
        if lo:
            g = g * (0.5 + 0.5 * torch.sin(math.pi * ((lf - math.log2(lo)) / w).clamp(-0.5, 0.5)))
        if hi:
            g = g * (0.5 - 0.5 * torch.sin(math.pi * ((lf - math.log2(hi)) / w).clamp(-0.5, 0.5)))
        return torch.fft.irfft(torch.fft.rfft(x) * g, x.shape[-1])
    return f


def loudnorm(x):
    """Each excerpt (both channels together) scaled to the same RMS: no absolute level to go by."""
    return x / x.pow(2).mean((-2, -1), keepdim=True).sqrt().clamp(min=1e-8) * 0.05


VIEWS = {
    "full": lambda sr: (lambda x: x),
    "loudness equalised": lambda sr: loudnorm,
    "< 500 Hz": lambda sr: band(0, 500, sr),
    "500 Hz - 2 kHz": lambda sr: band(500, 2000, sr),
    "2 - 6 kHz": lambda sr: band(2000, 6000, sr),
    "> 6 kHz": lambda sr: band(6000, 0, sr),
    "mid (L+R)/2": lambda sr: (lambda x: x.mean(-2, keepdim=True)),
    "side (L-R)/2": lambda sr: (lambda x: 0.5 * (x[..., :1, :] - x[..., 1:, :])),
    "side, loudness equalised": lambda sr: (lambda x: loudnorm(0.5 * (x[..., :1, :] - x[..., 1:, :]))),
}


def pools(args, model, cfg, device, log):
    sr = cfg.sample_rate

    def draw(split, n, seed, render):
        ds = MaestroSegments(args.data, split, cfg, 2.0, 1.0, 12.0, length=n, deterministic=True, years=args.years,
                             seed=seed)
        rec, ren = [], []
        with torch.no_grad():
            for b in DataLoader(ds, batch_size=8, collate_fn=collate, num_workers=0):
                b = to_device(b, device)
                s = int(b["loss_start"][0])
                rec.append(highpass(b["audio"][..., s:].float(), sr))
                if render:
                    ren.append(highpass(model(b, b["audio"].shape[-1], residual=args.residual)["audio"][..., s:].float(), sr))
        return torch.cat(rec), (torch.cat(ren) if render else None)

    _, fake = draw("train", args.pool, 5, True)
    real, _ = draw("train", args.pool, 7, False)  # other excerpts: the critic never sees a render's own recording
    val_real, val_fake = draw("validation", args.val, 1, True)
    log(f"pools: {len(fake)} renders and {len(real)} recordings of the training pieces; {len(val_real)} held-out pairs")
    return real, fake, val_real, val_fake


def score(disc, x):
    B = x.shape[0]
    return torch.stack([o.reshape(B, -1).mean(1) for o, _ in disc(x)]).mean(0)


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/maestro24k")
    ap.add_argument("--years", type=int, nargs="*", default=[2018])
    ap.add_argument("--model", default="runs/loss_compare/B_stage1/train/last.pt")
    ap.add_argument("--critic", default="wide")
    ap.add_argument("--residual", action="store_true", help="render with the residual (default: the physics alone)")
    ap.add_argument("--views", nargs="*", default=list(VIEWS))
    ap.add_argument("--pool", type=int, default=384)
    ap.add_argument("--val", type=int, default=64)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(0.8)
    device = torch.device("cuda")
    os.makedirs(args.out, exist_ok=True)
    logf = open(os.path.join(args.out, "log.txt"), "a", encoding="utf-8")

    def log(m):
        print(m, flush=True)
        logf.write(m + "\n")
        logf.flush()

    log(f"=== {time.ctime()}: {vars(args)}")
    model = load_model(args.model, device)
    cfg = model.cfg
    real, fake, val_real, val_fake = pools(args, model, cfg, device, log)
    del model
    torch.cuda.empty_cache()
    res = {}
    for name in args.views:
        view = VIEWS[name](cfg.sample_rate)
        torch.manual_seed(args.seed)
        g = torch.Generator().manual_seed(args.seed)
        disc = make_discriminator(args.critic).to(device)
        dopt = torch.optim.Adam(disc.parameters(), lr=2e-4, betas=(0.5, 0.9))
        t0 = time.time()
        losses = []
        for step in range(args.steps):
            r = view(real[torch.randint(len(real), (8,), generator=g)])
            f = view(fake[torch.randint(len(fake), (8,), generator=g)])
            dopt.zero_grad()
            dl = discriminator_loss(disc, r, f)
            dl.backward()
            dopt.step()
            losses.append(float(dl))
        disc.requires_grad_(False)
        sr_, sf_ = [], []
        with torch.no_grad():
            for i in range(0, len(val_real), 8):
                sr_ += score(disc, view(val_real[i:i + 8])).tolist()
                sf_ += score(disc, view(val_fake[i:i + 8])).tolist()
        n = len(disc.discs)
        res[name] = {"auc": auc(sr_, sf_), "loss_end": float(np.mean(losses[-50:])),
                     "seconds": time.time() - t0}
        log(f"{name:26s} held-out AUC {res[name]['auc']:.3f}   critic loss, last 50 steps {res[name]['loss_end']:.3f} "
            f"(chance {0.5 * n:.1f})   {res[name]['seconds']:.0f} s")
        del disc, dopt
        torch.cuda.empty_cache()
    with open(os.path.join(args.out, "result.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
