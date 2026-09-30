"""Fit the physical model to MAESTRO (or to a perturbed copy of itself with --synthetic).

Staged, so the physics keeps first claim on the data (review 3, sections 4.3 and 9.2):

* stage 1: the physical forms, the recording chain (body, hall, mic gain, noise floor) and the
  per-condition scalars. The tables that can absorb form errors (``partial_gain``, the per-key
  colouration) and the learned residual are frozen, and the residual is switched off.
* stage 2 (from ``--stage2-at``): those tables and the residual (context net R1/R2/R3) are
  unfrozen, the physics learning rate drops (x``--stage2-physics-lr``), and a residual budget
  penalises what the residual explains, so a physics explanation wins at equal loss.

Validation reports the held-out loss with the residual on and off: the gap is what the physics
still leaves unexplained. Audio of fixed validation excerpts is written at every dump.

    python -m pianonn.train --data data/maestro24k --years 2018 --out runs/trial --minutes 120
"""

import argparse
import itertools
import json
import math
import os
import time

# must be set before the first CUDA allocation (see the memory fraction in main)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.6,max_split_size_mb:256")

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from .config import PianoConfig  # noqa: E402
from .data import MaestroSegments, SyntheticPerformances, collate
from .dsp import bounded
from .losses import (LogMelLoss, MultiResolutionDiscriminator, MultiResolutionSTFTLoss, band_energies,
                     discriminator_loss, generator_adv_loss)
from .synth import ContextNet, NeuralPhysicalPiano

STAGE2_ONLY = ("context.", "noise.att", "physics.partial_gain", "physics.color")
DB_PARAMS = ("physics.gain_db", "physics.cond_gain_db", "physics.cond_vel_slope", "physics.soft_gain_db",
             "physics.raw_phantom_db", "physics.raw_impulse_db", "physics.raw_impulse_vel", "room.mic_gain_db",
             "room.floor_db", "room.raw_pan")
CENTS_PARAMS = ("physics.raw_cents", "physics.cond_cents")


@torch.no_grad()
def perturb_physics(model, scale=0.3, seed=0):
    """Randomise the learnable physical offsets: a stand-in 'real piano' for sanity checks."""
    g = torch.Generator().manual_seed(seed)
    for name, p in model.physics.named_parameters():
        if name.startswith("raw_") or name == "partial_gain":
            p.add_(scale * torch.randn(p.shape, generator=g))
    return model


def lr_scale(name):
    """Per-parameter learning-rate multiplier by unit: Adam moves every parameter by ~lr per step, so dB and
    cents need larger steps than nats, and thousands of FIR taps must move slower than the physics."""
    if name == "room.body":
        return 0.03
    if name.startswith(DB_PARAMS):
        return 20.0
    if name.startswith(CENTS_PARAMS):
        return 5.0
    if name.startswith("context."):
        return 0.5
    return 1.0


def param_groups(model, lr, fir_lr_scale=None):
    """One group per parameter, so stages can rescale learning rates by name."""
    groups = []
    for name, p in model.named_parameters():
        s = fir_lr_scale if (fir_lr_scale is not None and name == "room.body") else lr_scale(name)
        groups.append({"params": [p], "lr": lr * s, "name": name, "base_lr": lr * s})
    return groups


def set_stage(model, opt, stage, physics_lr=1.0, only=None, frozen=()):
    """Freeze/unfreeze by stage and rescale the physical groups' learning rate.

    ``only``: optional tuple of name prefixes that are the *only* trainable parameters (overfit ladders);
    ``frozen``: name prefixes that stay frozen in every stage (ablations)."""
    for g in opt.param_groups:
        name = g["name"]
        residual = name.startswith(STAGE2_ONLY)
        train = ((stage >= 2 or not residual) and (only is None or name.startswith(only))
                 and not (frozen and name.startswith(tuple(frozen))))
        for p in g["params"]:
            p.requires_grad_(train)
            if not train:
                p.grad = None
        g["lr"] = g["base_lr"] * (1.0 if residual or stage < 2 else physics_lr)


def to_device(batch, device, non_blocking=False):
    return {k: v.to(device, non_blocking=non_blocking) for k, v in batch.items()}


def residual_budget(model, out, mask):
    """What the residual explains, as penalties: per-note corrections and band gains (normalised by their bounds),
    and the residual noise's share of the band energy of the dry signal."""
    terms = {}
    ctx, frame = out.get("ctx") or {}, out.get("frame_ctx")
    if ctx:
        w = mask.float()
        tot = 0.0
        for name, (n, s) in ContextNet.NOTE.items():
            v = (ctx[name] / s) ** 2
            v = v.mean(-1) if v.dim() == 3 else v
            tot = tot + (v * w).sum() / w.sum().clamp(min=1)
        terms["note"] = tot / len(ContextNet.NOTE)
    if frame is not None:
        terms["frame"] = ((frame["band_gain"] / ContextNet.FRAME["band_gain"][1]) ** 2).mean()
    if "noise_res" in out:
        M = model.noise.band_masks(512)
        e_res = band_energies(out["noise_res"], M)
        e_dry = band_energies(out.get("dry_phys", out["dry"]).mean(1), M).detach()
        terms["additive"] = (e_res / (e_res + e_dry + 1e-12)).mean()
    return terms


def pan_smoothness(model, cond):
    x = model.room.raw_pan[torch.unique(cond)]
    return ((x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]) ** 2).mean()


@torch.no_grad()
def validate(model, batches, recon, residual):
    model.eval()
    tot, per = 0.0, {}
    for b in batches:
        n, s = b["audio"].shape[-1], int(b["loss_start"][0])
        g = torch.Generator(device=b["audio"].device).manual_seed(0)
        pred = model(b, n, residual=residual, generator=g)["audio"][..., s:]
        loss, terms = recon(pred, b["audio"][..., s:], per_resolution=True)
        tot += float(loss)
        for k, v in terms.items():
            per[k] = per.get(k, 0.0) + float(v)
    model.train()
    return tot / len(batches), {k: v / len(batches) for k, v in per.items()}


@torch.no_grad()
def dump_audio(model, examples, out_dir, tag, residual_too):
    import soundfile as sf

    model.eval()
    os.makedirs(out_dir, exist_ok=True)
    sr = model.cfg.sample_rate
    for i, b in enumerate(examples):
        n = b["audio"].shape[-1]
        g = torch.Generator(device=b["audio"].device).manual_seed(0)
        if tag == "prior" or not os.path.exists(os.path.join(out_dir, f"ex{i}_target.wav")):
            sf.write(os.path.join(out_dir, f"ex{i}_target.wav"), b["audio"][0].T.cpu().numpy(), sr)
        pred = model(b, n, residual=False, generator=g)["audio"][0]
        sf.write(os.path.join(out_dir, f"ex{i}_{tag}_physics.wav"), pred.T.clamp(-1, 1).cpu().numpy(), sr)
        if residual_too:
            pred = model(b, n, residual=True, generator=g)["audio"][0]
            sf.write(os.path.join(out_dir, f"ex{i}_{tag}_residual.wav"), pred.T.clamp(-1, 1).cpu().numpy(), sr)
    model.train()


def fixed_batches(dataset, n, batch, device):
    items = [dataset[i] for i in range(n)]
    return [to_device(collate(items[i: i + batch]), device) for i in range(0, n, batch)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="directory produced by scripts/prepare_maestro.py")
    ap.add_argument("--years", type=int, nargs="*", help="only these MAESTRO years")
    ap.add_argument("--synthetic", action="store_true", help="fit a randomly perturbed teacher instead of MAESTRO")
    ap.add_argument("--out", default="runs/default")
    ap.add_argument("--resume", help="checkpoint to continue from (model, optimiser, step, stage)")
    ap.add_argument("--sr", type=int, default=24000)
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--minutes", type=float, help="stop after this much wall-clock time in the training loop "
                                                   "(validation and dumps included; the initialisation is not)")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--segment", type=float, default=2.0)
    ap.add_argument("--warmup", type=float, default=1.0)
    ap.add_argument("--lookback", type=float, default=12.0)
    ap.add_argument("--reg", type=float, default=1.0, help="weight of the smoothness regulariser")
    ap.add_argument("--mel-weight", type=float, default=0.0,
                    help="weight of a log-mel band-energy term (insensitive to partial misalignment); 0 = off")
    ap.add_argument("--freeze", nargs="*", default=[], help="parameter-name prefixes kept frozen in every stage")
    ap.add_argument("--stage2-at", type=float, default=0.6,
                    help="switch to stage 2 at this step (>= 1) or this fraction of --minutes / --steps (< 1); -1: never")
    ap.add_argument("--stage2-physics-lr", type=float, default=0.3)
    ap.add_argument("--budget", type=float, nargs=3, default=(0.05, 0.05, 0.1), metavar=("NOTE", "FRAME", "ADD"),
                    help="residual budget weights: per-note corrections, band gains, residual noise share")
    ap.add_argument("--no-init", action="store_true", help="skip the data-driven initialisation of the recording chain")
    ap.add_argument("--mined", help="scripts/mine_notes.py output: start B and stretch from isolated-note measurements")
    ap.add_argument("--init-examples", type=int, default=32)
    ap.add_argument("--val-examples", type=int, default=24)
    ap.add_argument("--val-every", type=int, default=250)
    ap.add_argument("--dump-every", type=int, default=1000)
    ap.add_argument("--dump-seconds", type=float, default=8.0)
    ap.add_argument("--adv-start", type=int, default=-1, help="step to switch on the GAN loss (-1: never)")
    ap.add_argument("--adv-weight", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--gpu-mem-fraction", type=float, default=0.8)
    ap.add_argument("--amp", action="store_true",
                    help="autocast the forward pass to bfloat16 (GPU only). Off by default: the physics is "
                         "exp/log/sqrt/sin of wide-range quantities and sub-Hz beating, and matmuls are a small "
                         "share of the compute, so the speedup is modest and the precision risk is real.")
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--save-every", type=int, default=500)
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    cfg = PianoConfig(sample_rate=args.sr)
    device = torch.device(args.device)
    if device.type == "cuda":
        # Every chunk renders a different number of notes and partials, so tensor sizes vary and the caching
        # allocator keeps reserving new blocks. On Windows (WDDM) the driver then silently spills past the
        # card's memory into system RAM and steps get 10x slower. Capping the fraction makes the allocator
        # free and reuse its cache instead.
        torch.cuda.set_per_process_memory_fraction(args.gpu_mem_fraction)
    use_amp = args.amp and device.type == "cuda"
    model = NeuralPhysicalPiano(cfg).to(device)
    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "log.jsonl")

    def log(msg, **rec):
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(json.dumps({"time": time.time(), "msg": msg, **rec}) + "\n")

    teacher = None
    n_steps = args.steps
    if args.synthetic:
        seconds = args.warmup + args.segment
        dataset = SyntheticPerformances(cfg, seconds=seconds, length=args.steps * args.batch)
        teacher = perturb_physics(NeuralPhysicalPiano(cfg), seed=1).to(device).eval()
        val_set = None
    else:
        dataset = MaestroSegments(args.data, "train", cfg, args.segment, args.warmup, args.lookback,
                                  length=args.steps * args.batch, years=args.years)
        val_set = MaestroSegments(args.data, "validation", cfg, args.segment, args.warmup, args.lookback,
                                  length=args.val_examples, deterministic=True, years=args.years, seed=1)
        dump_set = MaestroSegments(args.data, "validation", cfg, args.dump_seconds - args.warmup, args.warmup,
                                   args.lookback, length=3, deterministic=True, years=args.years, seed=2)
    loader = DataLoader(dataset, batch_size=args.batch, collate_fn=collate, num_workers=args.workers,
                        persistent_workers=args.workers > 0, pin_memory=device.type == "cuda",
                        prefetch_factor=4 if args.workers > 0 else None)

    recon = MultiResolutionSTFTLoss()
    mel_loss = LogMelLoss(cfg.sample_rate).to(device) if args.mel_weight > 0 else None
    opt = torch.optim.Adam(param_groups(model, args.lr))
    disc = disc_opt = None
    if args.adv_start >= 0:
        disc = MultiResolutionDiscriminator().to(device)
        disc_opt = torch.optim.Adam(disc.parameters(), lr=2e-4, betas=(0.5, 0.9))

    step, stage, best, elapsed0 = 0, 1, math.inf, 0.0
    if args.resume:
        from .render import load_weights

        state = torch.load(args.resume, map_location=device)
        load_weights(model, state["model"], log=log)
        try:
            opt.load_state_dict(state["opt"])
        except ValueError as e:  # parameter set changed: restart the optimiser's moments
            log(f"optimiser state not restored ({e})")
        for g in opt.param_groups:  # the learning-rate policy is this run's (--lr, lr_scale), not the checkpoint's
            g["base_lr"] = args.lr * lr_scale(g["name"])
        step, stage = state["step"], state["stage"]
        best, elapsed0 = state.get("best", math.inf), state.get("elapsed", 0.0)
        log(f"resumed from {args.resume} at step {step}, stage {stage}, {elapsed0 / 60:.1f} min in, best val {best:.4f}")
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump({"model": cfg.to_dict(), "args": vars(args)}, f, indent=2)

    val_batches = fixed_batches(val_set, args.val_examples, args.batch, device) if val_set else None
    dump_examples = fixed_batches(dump_set, 3, 1, device) if val_set else None
    if val_batches and not args.resume:
        v_raw, _ = validate(model, val_batches, recon, residual=False)
        log(f"val (untrained prior, before init): {v_raw:.4f}", kind="val", step=0, val_physics=v_raw, tag="raw_prior")
        if args.mined:
            from .fit_init import apply_mined_priors

            with open(args.mined) as f:
                est = apply_mined_priors(model, json.load(f), log=log)
            log("mined priors", kind="init", **est)
        if not args.no_init:
            from .fit_init import initialise_from_data

            for year in (args.years or sorted({p["year"] for p in dataset.pieces})):  # one recording condition at a time
                init_set = MaestroSegments(args.data, "train", cfg, args.segment, args.warmup, args.lookback,
                                           length=args.init_examples, deterministic=True, years=[year], seed=3)
                est = initialise_from_data(model, fixed_batches(init_set, args.init_examples, args.batch, device),
                                           log=log, tuning=not args.mined)
                log(f"init estimates {year}", kind="init", year=year, **{k: v for k, v in est.items()})
            v0, per0 = validate(model, val_batches, recon, residual=False)
            log(f"val (prior after init): {v0:.4f}", kind="val", step=0, val_physics=v0, per_res=per0, tag="init_prior")
        dump_audio(model, dump_examples, os.path.join(args.out, "audio"), "prior", residual_too=False)

    t_train, steps_run = 0.0, 0
    budget_s = args.minutes * 60 if args.minutes else None
    t_loop = time.time()
    elapsed = lambda: elapsed0 + time.time() - t_loop  # wall clock of the training loop, validation included

    def stage2_due():
        if args.stage2_at < 0:
            return False
        if args.stage2_at >= 1:
            return step >= args.stage2_at
        return (elapsed() >= args.stage2_at * budget_s) if budget_s else step >= args.stage2_at * n_steps

    set_stage(model, opt, stage, args.stage2_physics_lr, frozen=args.freeze)
    amp_ctx = lambda: torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp)

    def save(tag):
        torch.save({"cfg": cfg.to_dict(), "model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                    "stage": stage, "best": best, "elapsed": elapsed(), "args": vars(args)}, os.path.join(args.out, f"{tag}.pt"))

    def run_validation():
        nonlocal best
        if not val_batches:
            return
        v_phys, per = validate(model, val_batches, recon, residual=False)
        rec = {"kind": "val", "step": step, "stage": stage, "val_physics": v_phys, "per_res": per}
        msg = f"val step {step}: physics {v_phys:.4f}"
        v = v_phys
        if stage >= 2:
            v_res, _ = validate(model, val_batches, recon, residual=True)
            rec["val_residual"] = v_res
            msg += f"  with residual {v_res:.4f}  (unexplained by physics: {v_phys - v_res:+.4f})"
            v = v_res
        log(msg, **rec)
        if v < best:
            best = v
            save("best")

    model.train()
    t_last = time.time()
    it = iter(loader)
    while step < n_steps:
        if stage == 1 and stage2_due():
            stage = 2
            set_stage(model, opt, stage, args.stage2_physics_lr, frozen=args.freeze)
            log(f"step {step}: stage 2 (partial_gain, colouration and the residual unfrozen; physics lr x{args.stage2_physics_lr})")
            run_validation()
        batch = to_device(next(it), device, non_blocking=device.type == "cuda")
        t0 = time.time()
        step += 1
        if teacher is not None:
            n = int(round((args.warmup + args.segment) * cfg.sample_rate))
            with torch.no_grad(), amp_ctx():
                target = teacher(batch, n, residual=False)["audio"]
            s = int(args.warmup * cfg.sample_rate)
        else:
            target, n, s = batch["audio"], batch["audio"].shape[-1], int(batch["loss_start"][0])
        residual = stage >= 2
        with amp_ctx():
            out = model(batch, n, residual=residual)
        pred, tgt = out["audio"][..., s:].float(), target[..., s:].float()

        logs = {"recon": recon(pred, tgt), "reg": model.physics.regularizer(batch["condition"]) + pan_smoothness(model, batch["condition"])}
        loss = logs["recon"] + args.reg * logs["reg"]
        if mel_loss is not None:
            logs["mel"] = mel_loss(pred, tgt)
            loss = loss + args.mel_weight * logs["mel"]
        if residual:
            weights = dict(zip(("note", "frame", "additive"), args.budget))
            for k, v in residual_budget(model, out, batch["mask"]).items():
                logs["budget_" + k] = v
                loss = loss + weights[k] * v
        if disc is not None and step >= args.adv_start:
            disc_opt.zero_grad()
            logs["disc"] = discriminator_loss(disc, tgt, pred)
            logs["disc"].backward()
            disc_opt.step()
            logs["adv"], logs["fm"] = generator_adv_loss(disc, tgt, pred)
            loss = loss + args.adv_weight * (logs["adv"] + 2.0 * logs["fm"])

        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = {}
        for mod in ("physics", "room", "noise", "context"):
            gs = [p.grad.norm() for name, p in model.named_parameters() if name.startswith(mod + ".") and p.grad is not None]
            gnorm[mod] = float(torch.stack(gs).norm()) if gs else 0.0
        total = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(total):
            log(f"step {step}: non-finite gradient norm, step skipped", kind="warn", step=step)
            opt.zero_grad(set_to_none=True)
            continue
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_train += time.time() - t0
        steps_run += 1

        if step % args.log_every == 0:
            vals = {k: float(v.detach()) for k, v in logs.items()}
            dt = time.time() - t_last
            t_last = time.time()
            mem = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
            log(f"step {step} stage {stage} " + " ".join(f"{k}={v:.4f}" for k, v in vals.items())
                + f" | grad {' '.join(f'{k}={v:.2g}' for k, v in gnorm.items())} | {dt / args.log_every:.2f}s/step "
                  f"(compute {t_train / steps_run:.2f}) mem {mem:.1f}G notes {batch['pitch'].shape[1]}",
                kind="train", step=step, stage=stage, grad=gnorm, **vals)
        if step % args.val_every == 0:
            run_validation()
        if dump_examples and step % args.dump_every == 0:
            dump_audio(model, dump_examples, os.path.join(args.out, "audio"), f"step{step}", residual_too=stage >= 2)
        if step % args.save_every == 0:
            save("last")
        if budget_s and elapsed() >= budget_s:
            log(f"time limit reached after {step} steps ({elapsed() / 60:.1f} min, {t_train / 60:.1f} of them in steps)")
            break

    run_validation()
    if dump_examples:
        dump_audio(model, dump_examples, os.path.join(args.out, "audio"), f"step{step}", residual_too=stage >= 2)
    save("last")
    log(f"done: {step} steps, best val {best:.4f}")


if __name__ == "__main__":
    main()
