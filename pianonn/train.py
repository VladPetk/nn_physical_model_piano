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

The loss is :class:`pianonn.losses.PianoLoss` (round 2: log band energies, a fine term below 2 kHz and an
onset-window term; the trial's spectral-convergence term set the level ~1.8 dB low; ``--level-weight`` adds its
whole-excerpt level term). ``--piece-gain`` fits a free gain per training piece (a nuisance, averaging zero, not used
in validation or evaluation): at equal key and velocity the pieces of one year differ by ~2 dB, and without it the
level parameters follow each batch's pieces (docs/tone_measures.md 12.5, 12.9). ``--no-residual`` runs
stage 2 without the residual (the control for its gain), and ``--adv-with-stage2`` adds the GAN in stage 2:
the discriminator judges the audio, but its gradients reach only the noise bank and the residual
(``GAN_PARAMS``), through a view of the output in which everything else is detached.

``--env-weight`` adds the whole note's envelope (N13, ``pianonn.envfit``; docs/tone_measures.md 17): each step also
renders ``--env-batch`` of N13's notes of the training pieces in their contexts (with the residual in stage 2) and adds
the L1 distance of each partial's fade (0.2-2.5 s re 0.1 s) to the recording's, in log10 power (dB / 10, PianoLoss's
unit). The log reports it, the cosine between its gradient and the rest of the loss's on the strings' decay
(``ENV_PARAMS``: below 0 they pull against each other) and its gradient's size re theirs; each validation reports it on
N13's notes of the validation and test pieces, with the median fade error of partials 9-40 at 1 and 1.5 s.

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

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from .config import PianoConfig  # noqa: E402
from .data import MaestroSegments, SyntheticPerformances, collate
from .dsp import bounded
from .notefit import PieceLevels
from .losses import (LogMelLoss, MultiResolutionDiscriminator, MultiResolutionSTFTLoss, OnsetLoss, PianoLoss,
                     band_energies, discriminator_loss, generator_adv_loss, highpass)
from .synth import ContextNet, NeuralPhysicalPiano

STAGE2_ONLY = ("context.", "noise.att", "physics.partial_gain", "physics.color")
DB_PARAMS = ("physics.gain_db", "physics.cond_gain_db", "physics.cond_vel_slope", "physics.cond_vel_curve", "physics.soft_gain_db",
             "physics.raw_phantom_db", "physics.raw_impulse_db", "physics.raw_impulse_vel", "room.mic_gain_db",
             "room.raw_pan")  # the floor and hum (bounded +-3 dB around a measurement) learn at the base rate
GAN_PARAMS = ("noise.", "context.")  # what the critic may change (enforced by the texture view, see synth.forward)
MODULES = ("physics", "room", "noise", "context")
CENTS_PARAMS = ("physics.raw_cents", "physics.cond_cents")
ENV_PARAMS = ("physics.raw_prompt", "physics.raw_log_b1", "physics.raw_log_b3", "physics.raw_bridge_g",
              "physics.raw_decay_p", "physics.raw_after")  # the strings' decay and the aftersound's level


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
    if name == "piece_gain":
        return 50.0  # each of 2018's 70 training pieces is in ~1 batch in 9 (batch 8): it must reach ~2 dB within a 30-min run
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
        g["stage_scale"] = 1.0 if residual or stage < 2 else physics_lr
        g["lr"] = g["base_lr"] * g["stage_scale"]


def to_device(batch, device, non_blocking=False):
    return {k: v.to(device, non_blocking=non_blocking) for k, v in batch.items()}


def residual_budget(model, out, mask):
    """What the residual explains, as penalties: per-note corrections and band gains (normalised by their bounds),
    and the residual noise's share of the output's band energy at the microphones (floor included). Measured
    against the physics alone, as in the trial, the share was 1 wherever the physics is silent, whatever the
    residual added, and those cells made up three quarters of the penalty (review 4, section 5)."""
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
    if "noise_res_out" in out:
        M = model.noise.band_masks(512)
        e_res = band_energies(out["noise_res_out"], M)
        e_out = band_energies(out["audio"], M).detach()
        terms["additive"] = (e_res / (e_res.detach() + e_out + 1e-12)).clamp(max=1.0).mean()
    return terms


def pan_smoothness(model, cond):
    x = model.room.raw_pan[torch.unique(cond)]
    return ((x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]) ** 2).mean()


def onsets_of(batch, s, sr):
    """Onsets re the loss window's first sample, and their mask (for the attack term)."""
    return batch["onset"] - s / sr, batch["mask"]


@torch.no_grad()
def validate(model, batches, loss_fn, residual, old=None, onset=None):
    """Mean loss and terms over fixed batches; with ``old``, also the trial's MR-STFT loss (``terms["mrstft"]``); with
    ``onset`` = (``OnsetLoss``, weight), its term too (in the total with its weight; pooled over each batch, the
    running pool left as training left it)."""
    model.eval()
    if onset is not None:
        onset[0].eval()
    tot, per = 0.0, {}
    for b in batches:
        n, s = b["audio"].shape[-1], int(b["loss_start"][0])
        g = torch.Generator(device=b["audio"].device).manual_seed(0)
        full = model(b, n, residual=residual, generator=g)["audio"]
        pred, tgt = full[..., s:], b["audio"][..., s:]
        loss, terms = loss_fn(pred, tgt, *onsets_of(b, s, model.cfg.sample_rate))
        terms = {k: float(v) for k, v in terms.items()}
        if onset is not None:
            v = onset[0](full.float(), b["audio"].float(), b, s / model.cfg.sample_rate)
            terms["onset"] = float(v)
            loss = loss + onset[1] * v
        if old is not None:
            terms["mrstft"] = float(old(pred, tgt))
        tot += float(loss)
        for k, v in terms.items():
            per[k] = per.get(k, 0.0) + v
    model.train()
    if onset is not None:
        onset[0].train()
    return tot / len(batches), {k: v / len(batches) for k, v in per.items()}


@torch.no_grad()
def dump_audio(model, examples, out_dir, tag, residual_too, strike=None):
    """Render the fixed excerpts; ``strike`` switches the per-strike variation on or off for them (None: as it is)."""
    import soundfile as sf

    model.eval()
    keep = model.strike_on
    if strike is not None:
        model.strike_on = strike
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
    model.strike_on = keep
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
    ap.add_argument("--init-from", help="start from this checkpoint's weights and model config (a training checkpoint or "
                                        "a fitted model.pt): fresh optimiser, step 0, stage 1, no data initialisation")
    ap.add_argument("--strike-train", action="store_true",
                    help="draw the per-strike variation (config strike_*) in training and validation too. By default it "
                         "is off there, since median-seeking terms would pull the fitted parameters (docs/tone_measures.md "
                         "12.7), and on in the audio dumps; the checkpoints keep it in their config for rendering")
    ap.add_argument("--sr", type=int, default=24000)
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--minutes", type=float, help="stop after this much wall-clock time in the training loop "
                                                   "(validation and dumps included; the initialisation is not)")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-decay-at", type=float, default=-1.0,
                    help="from this step (>= 1) or fraction of --minutes / --steps (< 1) the learning rate falls on a "
                         "half cosine to --lr-final times its value at the end; -1: constant")
    ap.add_argument("--lr-final", type=float, default=0.1)
    ap.add_argument("--lr-warmup", type=int, default=0,
                    help="steps over which each parameter's learning rate rises linearly from 0, counted from when it "
                         "first trains (the start, or stage 2 for the tables and the residual): a fresh optimiser's "
                         "first steps are full-sized whatever the gradient noise, which pushes a fitted model off")
    ap.add_argument("--segment", type=float, default=2.0)
    ap.add_argument("--warmup", type=float, default=1.0)
    ap.add_argument("--lookback", type=float, default=12.0)
    ap.add_argument("--reg", type=float, default=1.0, help="weight of the smoothness regulariser")
    ap.add_argument("--mel-weight", type=float, default=0.0,
                    help="weight of a log-mel band-energy term (insensitive to partial misalignment); 0 = off")
    ap.add_argument("--onset-weight", type=float, default=0.0,
                    help="weight of the onset-aligned attack term (losses.OnsetLoss: windows at each note's expected "
                         "sound onset, pooled over the batch; docs/tone_measures.md 12.6); 0 = off")
    ap.add_argument("--onset-relative", action="store_true",
                    help="the onset term compares the attack re the same onset's early window (30-100 ms; 12.8)")
    ap.add_argument("--onset-pool", type=float, default=0.0,
                    help="the onset term's pool runs across steps, decaying by this factor per step (e.g. 0.99; 12.8)")
    ap.add_argument("--level-weight", type=float, default=0.0,
                    help="weight of PianoLoss's whole-excerpt level term (log band energies summed over the excerpt; "
                         "its optimum is the energy match whatever the time structure); 0 = off")
    ap.add_argument("--env-weight", type=float, default=0.0,
                    help="weight of the whole-note envelope term (N13 fades, dB / 10; see above); 0 = off")
    ap.add_argument("--env-batch", type=int, default=2, help="N13 notes rendered per step for the envelope term")
    ap.add_argument("--env-cap", type=int, default=40, help="N13 events per cell (as scripts/note_envelope.py --cap)")
    ap.add_argument("--env-seed", type=int, default=1, help="N13 event draw and the order of its notes")
    ap.add_argument("--piece-gain", action="store_true",
                    help="fit a free gain (dB, averaging zero) per training piece, applied to the rendered audio before "
                         "every loss term; validation and evaluation render without it")
    ap.add_argument("--freeze", nargs="*", default=[], help="parameter-name prefixes kept frozen in every stage")
    ap.add_argument("--cfg", nargs="*", default=[], metavar="KEY=VALUE",
                    help="model config overrides, e.g. bridge_end_comb=1 body_q_max=50 (pianonn.config.PianoConfig)")
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
    ap.add_argument("--adv-with-stage2", action="store_true", help="switch the GAN on together with stage 2")
    ap.add_argument("--adv-weight", type=float, default=0.1)
    ap.add_argument("--no-residual", action="store_true",
                    help="stage 2 without the residual: the control run for what the residual adds")
    ap.add_argument("--loss-weights", type=float, nargs=3, default=(1.0, 0.25, 0.5), metavar=("BAND", "FINE", "ATTACK"))
    ap.add_argument("--clip", type=float, default=4.0,
                    help="clip each module's gradient norm at this multiple of its running typical norm")
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
    from .render import _parse_value
    overrides = {k: _parse_value(v) for k, v in (o.split("=", 1) for o in args.cfg)}
    unknown = set(overrides) - set(PianoConfig.__dataclass_fields__)
    assert not unknown, f"unknown config fields {sorted(unknown)}"
    source = args.resume or args.init_from  # its model config is the base; --cfg overrides it
    base_cfg = torch.load(source, map_location="cpu")["cfg"] if source else {}
    cfg = PianoConfig.from_dict({**base_cfg, "sample_rate": args.sr, **overrides})
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

    if args.init_from and not args.resume:
        from .render import load_weights

        load_weights(model, torch.load(args.init_from, map_location=device)["model"], log=log)
        log(f"weights from {args.init_from}")
    has_strike = model.strike_on
    model.strike_on = has_strike and args.strike_train
    if has_strike:
        log(f"per-strike variation {'on' if args.strike_train else 'off'} in training and validation, on in the dumps")

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

    recon = PianoLoss(cfg.sample_rate, weights=args.loss_weights, level_weight=args.level_weight).to(device)
    old_loss = MultiResolutionSTFTLoss()  # the trial's loss, reported for continuity
    mel_loss = LogMelLoss(cfg.sample_rate).to(device) if args.mel_weight > 0 else None
    onset_loss = (OnsetLoss(cfg.sample_rate, relative=args.onset_relative, pool_decay=args.onset_pool).to(device)
                  if args.onset_weight > 0 else None)
    onset_val = (onset_loss, args.onset_weight) if onset_loss is not None else None
    envs = env_val = env_term = None
    if args.env_weight > 0 and teacher is None:
        from . import envfit as EF
        from . import measures as M
        from .config import year_to_condition

        assert args.years and len(args.years) == 1, "the envelope term reads one year's recordings: --years Y"
        table = M.partial_table(model, year_to_condition(args.years[0]), device)
        tabs = {p: table[p - 21][(table[p - 21] > 0) & (table[p - 21] < 6500)] for p in range(21, 109)}
        events, _ = EF.mine(args.data, args.years[0], args.env_cap, args.env_seed)
        envs = EF.EnvelopeSet(args.data, [e for e in events if e["split"] == "train"], cfg, table, tabs,
                              log=lambda m: log(f"envelope term, training pieces: {m}"))
        env_val = EF.EnvelopeSet(args.data, [e for e in events if e["split"] != "train"], cfg, table, tabs,
                                 log=lambda m: log(f"envelope term, validation and test pieces: {m}"))
        env_term = EF.EnvelopeTerm(cfg.sample_rate)
        env_rng = np.random.default_rng(args.env_seed)
    env_hist = []

    def env_report(residual):
        """N13 on the held-out notes: mean |model - recording fade| (dB) and the median for partials 9-40 at 1, 1.5 s."""
        model.eval()
        r = EF.evaluate(model, env_val, env_term, device, residual=residual)
        model.train()
        d = r["d"]
        rec = {"env_val": float(np.nanmean(np.abs(d))), "env_cells": int(np.isfinite(d).sum())}
        for t in (1.0, 1.5):
            v = d[:, M.ENV_TIMES.index(t), 8:]
            v = v[np.isfinite(v)]
            rec[f"env_9+_{t:g}s"] = float(np.median(v)) if len(v) else float("nan")
        msg = (f"envelope {rec['env_val']:.3f} dB over {rec['env_cells']} cells, partials 9-40 at 1 s "
               f"{rec['env_9+_1s']:+.1f}, at 1.5 s {rec['env_9+_1.5s']:+.1f} dB")
        return rec, msg
    pieces = PieceLevels(len(dataset.pieces)).to(device) if args.piece_gain and teacher is None else None
    groups = param_groups(model, args.lr)
    if pieces is not None:
        groups.append({"params": [pieces.db], "lr": args.lr * lr_scale("piece_gain"), "name": "piece_gain",
                       "base_lr": args.lr * lr_scale("piece_gain")})
    opt = torch.optim.Adam(groups)
    disc = disc_opt = None
    if args.adv_start >= 0 or args.adv_with_stage2:
        disc = MultiResolutionDiscriminator().to(device)
        disc_opt = torch.optim.Adam(disc.parameters(), lr=2e-4, betas=(0.5, 0.9))

    step, stage, best, elapsed0 = 0, 1, math.inf, 0.0
    if args.resume:
        from .render import load_weights

        state = torch.load(args.resume, map_location=device)
        load_weights(model, state["model"], log=log)
        if pieces is not None and "piece_gain" in state:
            saved = dict(zip(state["piece_gain"]["ids"], state["piece_gain"]["db"].tolist()))
            with torch.no_grad():
                pieces.db.copy_(torch.tensor([saved.get(p["id"], 0.0) for p in dataset.pieces], device=device))
        try:
            opt.load_state_dict(state["opt"])
        except ValueError as e:  # parameter set changed: restart the optimiser's moments
            log(f"optimiser state not restored ({e})")
        for g in opt.param_groups:  # the learning-rate policy is this run's (--lr, lr_scale), not the checkpoint's
            g["base_lr"] = args.lr * lr_scale(g["name"])
        step, stage = state["step"], state["stage"]
        best, elapsed0 = state.get("best", math.inf), state.get("elapsed", 0.0)
        log(f"resumed from {args.resume} at step {step}, stage {stage}, {elapsed0 / 60:.1f} min in, best val {best:.4f}")
    for g in opt.param_groups:  # where each parameter's warm-up starts (kept in the optimiser's state across resumes)
        g.setdefault("warm_from", step)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump({"model": cfg.to_dict(), "args": vars(args)}, f, indent=2)

    val_batches = fixed_batches(val_set, args.val_examples, args.batch, device) if val_set else None
    dump_examples = fixed_batches(dump_set, 3, 1, device) if val_set else None
    if val_batches and not args.resume:
        v_raw, _ = validate(model, val_batches, recon, residual=False, old=old_loss, onset=onset_val)
        log(f"val ({'start model' if args.init_from else 'untrained prior, before init'}): {v_raw:.4f}", kind="val",
            step=0, val_physics=v_raw, tag="raw_prior")
        if env_val is not None:
            erec, emsg = env_report(False)
            log(f"val step 0: {emsg}", kind="env_val", step=0, **erec)
        if args.mined:
            from .fit_init import apply_mined_priors

            with open(args.mined) as f:
                est = apply_mined_priors(model, json.load(f), log=log)
            log("mined priors", kind="init", **est)
        if not args.no_init and not args.init_from:  # a checkpoint's recording chain is already fitted
            from .fit_init import initialise_from_data

            for year in (args.years or sorted({p["year"] for p in dataset.pieces})):  # one recording condition at a time
                init_set = MaestroSegments(args.data, "train", cfg, args.segment, args.warmup, args.lookback,
                                           length=args.init_examples, deterministic=True, years=[year], seed=3)
                est = initialise_from_data(model, fixed_batches(init_set, args.init_examples, args.batch, device),
                                           log=log, tuning=not args.mined, silence=dataset.silence_clips(year))
                log(f"init estimates {year}", kind="init", year=year, **{k: v for k, v in est.items()})
            v0, per0 = validate(model, val_batches, recon, residual=False, old=old_loss, onset=onset_val)
            log(f"val (prior after init): {v0:.4f}", kind="val", step=0, val_physics=v0, per_res=per0, tag="init_prior")
        dump_audio(model, dump_examples, os.path.join(args.out, "audio"), "prior", residual_too=False, strike=has_strike)

    t_train, steps_run = 0.0, 0
    budget_s = args.minutes * 60 if args.minutes else None
    t_loop = time.time()
    elapsed = lambda: elapsed0 + time.time() - t_loop  # wall clock of the training loop, validation included

    def lr_factor():
        """The learning-rate decay's multiplier: 1 until --lr-decay-at, then a half cosine to --lr-final at the end."""
        if args.lr_decay_at < 0:
            return 1.0
        if args.lr_decay_at >= 1:
            start, now, end = args.lr_decay_at, step, n_steps
        elif budget_s:
            start, now, end = args.lr_decay_at * budget_s, elapsed(), budget_s
        else:
            start, now, end = args.lr_decay_at * n_steps, step, n_steps
        x = min(max((now - start) / max(end - start, 1e-9), 0.0), 1.0)
        return args.lr_final + (1.0 - args.lr_final) * 0.5 * (1.0 + math.cos(math.pi * x))

    def stage2_due():
        if args.stage2_at < 0:
            return False
        if args.stage2_at >= 1:
            return step >= args.stage2_at
        return (elapsed() >= args.stage2_at * budget_s) if budget_s else step >= args.stage2_at * n_steps

    frozen = tuple(args.freeze) + (("context.", "noise.att") if args.no_residual else ())
    set_stage(model, opt, stage, args.stage2_physics_lr, frozen=frozen)
    amp_ctx = lambda: torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp)

    def save(tag):
        extra = {} if pieces is None else {"piece_gain": {"ids": [p["id"] for p in dataset.pieces],
                                                          "db": (pieces.db - pieces.db.mean()).detach().cpu()}}
        torch.save({"cfg": cfg.to_dict(), "model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                    "stage": stage, "best": best, "elapsed": elapsed(), "args": vars(args), **extra},
                   os.path.join(args.out, f"{tag}.pt"))

    use_residual = lambda: stage >= 2 and not args.no_residual

    def run_validation():
        nonlocal best
        if not val_batches:
            return
        v_phys, per = validate(model, val_batches, recon, residual=False, old=old_loss, onset=onset_val)
        rec = {"kind": "val", "step": step, "stage": stage, "val_physics": v_phys, "per_res": per}
        msg = f"val step {step}: physics {v_phys:.4f} (" + " ".join(f"{k} {v:.4f}" for k, v in per.items()) + ")"
        v = v_phys
        if use_residual():
            v_res, per_res = validate(model, val_batches, recon, residual=True, old=old_loss, onset=onset_val)
            rec["val_residual"], rec["per_res_residual"] = v_res, per_res
            msg += f"  with residual {v_res:.4f}  (unexplained by physics: {v_phys - v_res:+.4f})"
            v = v_res
        if env_val is not None:
            erec, emsg = env_report(use_residual())
            rec.update(erec)
            msg += f"; {emsg}"
        log(msg, **rec)
        if v < best:
            best = v
            save("best")

    model.train()
    t_last = time.time()
    it = iter(loader)
    clip_state = {}  # per module: running typical gradient norm (the trial's global clip at 1.0 scaled every step)
    adv_from = args.adv_start if args.adv_start >= 0 else math.inf
    if args.adv_with_stage2 and stage >= 2:
        adv_from = step
    while step < n_steps:
        if stage == 1 and stage2_due():
            stage = 2
            set_stage(model, opt, stage, args.stage2_physics_lr, frozen=frozen)
            for g in opt.param_groups:
                if g["name"].startswith(STAGE2_ONLY):
                    g["warm_from"] = step
            log(f"step {step}: stage 2 (partial_gain, colouration{'' if args.no_residual else ' and the residual'} unfrozen; "
                f"physics lr x{args.stage2_physics_lr})")
            if args.adv_with_stage2 and disc is not None:
                adv_from = step
                log(f"step {step}: GAN on (critic gradients reach only {', '.join(GAN_PARAMS)})")
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
        oom = False
        try:
            residual = use_residual()
            gan = disc is not None and step >= adv_from
            extras = (("residual_out",) if residual else ()) + (("texture_view",) if gan else ())
            with amp_ctx():
                out = model(batch, n, residual=residual, extras=extras)
            if pieces is not None:  # the recording's level for this piece: a nuisance, discarded at evaluation
                out["audio"] = out["audio"] * (10 ** (pieces(batch["piece"]) / 20))[:, None, None]
            pred, tgt = out["audio"][..., s:].float(), target[..., s:].float()

            logs = {}
            logs["recon"], parts = recon(pred, tgt, *onsets_of(batch, s, cfg.sample_rate))
            logs.update(parts)
            logs["reg"] = model.physics.regularizer(batch["condition"]) + pan_smoothness(model, batch["condition"])
            loss = logs["recon"] + args.reg * logs["reg"]
            if mel_loss is not None:
                logs["mel"] = mel_loss(pred, tgt)
                loss = loss + args.mel_weight * logs["mel"]
            if onset_loss is not None:
                logs["onset"] = onset_loss(out["audio"].float(), target.float(), batch, s / cfg.sample_rate)
                loss = loss + args.onset_weight * logs["onset"]
            if residual:
                weights = dict(zip(("note", "frame", "additive"), args.budget))
                for k, v in residual_budget(model, out, batch["mask"]).items():
                    logs["budget_" + k] = v
                    loss = loss + weights[k] * v
            if gan:
                real = highpass(tgt, cfg.sample_rate)  # no infrasound giveaway: the loss ignores it too
                fake = highpass(out["audio_texture"][..., s:].float(), cfg.sample_rate)
                disc.requires_grad_(True)
                disc_opt.zero_grad()
                logs["disc"] = discriminator_loss(disc, real, fake)
                logs["disc"].backward()
                disc_opt.step()
                disc.requires_grad_(False)
                logs["adv"], logs["fm"] = generator_adv_loss(disc, real, fake)
                loss = loss + args.adv_weight * (logs["adv"] + 2.0 * logs["fm"])  # reaches GAN_PARAMS only

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if envs is not None:
                dec = [p for name, p in model.named_parameters() if name.startswith(ENV_PARAMS) and p.grad is not None]
                g_rest = [p.grad.detach().clone() for p in dec]
                idx = env_rng.choice(len(envs), args.env_batch, replace=False).tolist()
                with amp_ctx():
                    y = model(envs.batch(idx, device), envs.n, residual=residual)["audio"].float()
                tot, cnt = 0.0, 0
                for j, i in enumerate(idx):
                    ev = envs.notes[i]
                    sm, c, _, _ = env_term.note(y[j], envs.rec[i], envs.table[ev["pitch"] - 21], envs.t_on, ev["t_end"])
                    tot, cnt = tot + sm, cnt + c
                if cnt:
                    env = tot / cnt / 10  # dB -> log10 power
                    (args.env_weight * env).backward()
                    if dec:
                        ga = torch.cat([g.flatten() for g in g_rest])
                        ge = torch.cat([(p.grad - g).flatten() for p, g in zip(dec, g_rest)])
                        cos = float(ga @ ge / (ga.norm() * ge.norm() + 1e-30))
                        ratio = float(ge.norm() / (ga.norm() + 1e-30))
                    else:
                        cos = ratio = float("nan")
                    env_hist.append((float(env.detach()) * 10, cos, ratio, cnt))
                del y
        except torch.OutOfMemoryError:  # a rare batch with many notes (stage 2): skip it rather than crash
            oom = True
            out = pred = tgt = loss = logs = y = None
        if oom:
            opt.zero_grad(set_to_none=True)
            if disc_opt is not None:
                disc_opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            log(f"step {step}: out of GPU memory ({batch['pitch'].shape[1]} notes), batch skipped", kind="warn", step=step)
            continue
        gnorm, finite = {}, True
        for mod in MODULES:
            ps = [p for name, p in model.named_parameters() if name.startswith(mod + ".") and p.grad is not None]
            if not ps:
                gnorm[mod] = 0.0
                continue
            norm = float(torch.stack([p.grad.norm() for p in ps]).norm())
            gnorm[mod] = norm
            if not math.isfinite(norm):
                finite = False
                continue
            typical = clip_state.get(mod)
            limit = args.clip * typical if typical else math.inf
            if norm > limit:  # a spike in this module: scale it back, leave the others alone
                for p in ps:
                    p.grad.mul_(limit / norm)
            clip_state[mod] = norm if typical is None else 0.98 * typical + 0.02 * min(norm, limit)
        if not finite:
            log(f"step {step}: non-finite gradient norm, step skipped", kind="warn", step=step)
            opt.zero_grad(set_to_none=True)
            continue
        decay = lr_factor()
        for g in opt.param_groups:
            warm = min(1.0, (step - g["warm_from"]) / args.lr_warmup) if args.lr_warmup > 0 else 1.0
            g["lr"] = g["base_lr"] * g["stage_scale"] * decay * warm
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_train += time.time() - t0
        steps_run += 1

        if step % args.log_every == 0:
            vals = {k: float(v.detach()) for k, v in logs.items()}
            if pieces is not None:
                vals["piece_sd"] = float(pieces.db.std())
            if env_hist:
                h = np.array(env_hist[-args.log_every:])
                vals.update(env_db=float(h[:, 0].mean()), env_cos=float(np.nanmean(h[:, 1])),
                            env_gratio=float(np.nanmedian(h[:, 2])), env_cells=float(h[:, 3].mean()))
            dt = time.time() - t_last
            t_last = time.time()
            mem = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
            log(f"step {step} stage {stage} " + " ".join(f"{k}={v:.4f}" for k, v in vals.items())
                + f" | grad {' '.join(f'{k}={v:.2g}' for k, v in gnorm.items())} | {dt / args.log_every:.2f}s/step "
                  f"(compute {t_train / steps_run:.2f}) mem {mem:.1f}G notes {batch['pitch'].shape[1]} lr x{decay:.3g}",
                kind="train", step=step, stage=stage, grad=gnorm, lr_factor=decay, **vals)
        if step % args.val_every == 0:
            run_validation()
        if dump_examples and step % args.dump_every == 0:
            dump_audio(model, dump_examples, os.path.join(args.out, "audio"), f"step{step}", residual_too=use_residual(),
                       strike=has_strike)
        if step % args.save_every == 0:
            save("last")
        if budget_s and elapsed() >= budget_s:
            log(f"time limit reached after {step} steps ({elapsed() / 60:.1f} min, {t_train / 60:.1f} of them in steps)")
            break

    run_validation()
    if dump_examples:
        dump_audio(model, dump_examples, os.path.join(args.out, "audio"), f"step{step}", residual_too=use_residual(),
                   strike=has_strike)
    save("last")
    log(f"done: {step} steps, best val {best:.4f}")


if __name__ == "__main__":
    main()
