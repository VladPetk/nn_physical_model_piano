"""Sanity checks of the GAN (a critic of ``pianonn.losses``, ``--critic patch`` or ``wide``, with train.py's settings)
before it trains anything. Round 2's patch critic, never checked before, failed all three (``runs/gan_check/check1``).

1. ``tilt``: a known answer without the renderer. Recordings against other recordings with a shelf of ``theta`` dB
   above ~1 kHz, ``theta`` the only generator parameter, trained on the adversarial term alone (the feature matching
   is paired and would solve it trivially). Started at +3 and -3 dB: does ``theta`` return to 0?
2. ``model``: the critic trained against the model's renders (generator frozen, the renders drawn once), unpaired
   recordings of the training pieces. On held-out excerpts of the validation pieces: can it tell them apart (AUC),
   and which way does its gradient push each octave band, against the measured band error of the same renders?
3. ``share``: with that critic, the gradient on the residual (``context.``) of train.py's GAN term
   (``adv_weight * (adv + fm_weight fm)``) against the composite energy score's, on training batches, the residual
   fresh; through the texture view or the output itself (``--reach``, as train.py's ``--critic-reach``).
4. ``timing``: a stage-2 composite step (energy score, residual on) without the GAN and with it at either reach, on
   the same batches; synchronised timers per part and the memory peak.

    python scripts/gan_check.py --out runs/gan_check/check1 --critic patch --reach texture --parts tilt model share
    python scripts/gan_check.py --out runs/gan_check/wide1 --critic wide --reach residual
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
from pianonn.composite import CompositeLoss  # noqa: E402
from pianonn.config import PianoConfig  # noqa: E402
from pianonn.data import MaestroSegments, collate  # noqa: E402
from pianonn.losses import (critic_step, discriminator_loss, generator_adv_loss, highpass,  # noqa: E402
                            make_discriminator)
from pianonn.render import _parse_value, load_weights  # noqa: E402
from pianonn.synth import NeuralPhysicalPiano  # noqa: E402
from pianonn.train import to_device  # noqa: E402

OCT = 31.25 * 2.0 ** np.arange(9)  # octave bands 31 Hz-8 kHz


def shelf(x, sr, db):
    """``x[..., T]`` with ``db`` added above ~1 kHz (a tanh step over about an octave in log frequency)."""
    f = torch.fft.rfftfreq(x.shape[-1], 1 / sr).to(x.device)
    s = 0.5 * (1 + torch.tanh(torch.log2(f.clamp(min=1.0) / 1000.0) / 0.5))
    return torch.fft.irfft(torch.fft.rfft(x) * 10 ** (db * s / 20), x.shape[-1])


def octave_weights(n, sr, device):
    """Raised-cosine octave bands (sum to 1 between the outer centres) over the rfft bins of ``n`` samples."""
    f = torch.fft.rfftfreq(n, 1 / sr).to(device)
    lf = torch.log2(f.clamp(min=1.0))
    c = torch.log2(torch.tensor(OCT, device=device, dtype=torch.float32))
    w = torch.cos(0.5 * math.pi * (lf[None] - c[:, None]).clamp(-1, 1)) ** 2
    w[0] = torch.where(lf <= c[0], 1.0, w[0])
    w[-1] = torch.where(lf >= c[-1], 1.0, w[-1])
    return w  # [bands, bins]


def band_gains(x, sr, g, W):
    return torch.fft.irfft(torch.fft.rfft(x) * 10 ** ((g @ W) / 20), x.shape[-1])


def adv_only(disc, fake):
    return sum(((lf - 1) ** 2).mean() for lf, _ in disc(fake))


def critic_score(disc, x):
    """Per excerpt: the critic's mean output over patches, resolutions and channels (higher = more real)."""
    B = x.shape[0]
    outs = disc(x)  # each [B*ch, 1, F, T]
    return torch.stack([o.reshape(B, -1).mean(1) for o, _ in outs]).mean(0)


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean()))


def check_tilt(args, cfg, device, log):
    ds = MaestroSegments(args.data, "train", cfg, 2.0, 1.0, 12.0, length=args.tilt_steps * 16, years=args.years)
    res = {}
    for start in (3.0, -3.0):
        torch.manual_seed(0)
        loader = DataLoader(ds, batch_size=16, collate_fn=collate, num_workers=args.workers)
        disc = make_discriminator(args.critic, args.mono).to(device)
        dopt = torch.optim.Adam(disc.parameters(), lr=2e-4, betas=(0.5, 0.9))
        theta = torch.tensor(start, device=device, requires_grad=True)
        gopt = torch.optim.Adam([theta], lr=args.tilt_lr)
        trace = []
        t0 = time.time()
        for step, b in enumerate(loader):
            a = b["audio"][..., int(b["loss_start"][0]):].to(device)
            real, src = highpass(a[:8], cfg.sample_rate), highpass(a[8:], cfg.sample_rate)
            fake = shelf(src, cfg.sample_rate, theta)
            disc.requires_grad_(True)
            dopt.zero_grad()
            dl = discriminator_loss(disc, real, fake)
            dl.backward()
            dopt.step()
            disc.requires_grad_(False)
            gopt.zero_grad()
            adv_only(disc, fake).backward()
            gopt.step()
            trace.append(float(theta))
            if step % 50 == 0:
                log(f"tilt from {start:+.0f} dB: step {step} theta {float(theta):+.2f} dB, critic loss {float(dl):.3f}")
        tail = trace[-100:]
        res[f"{start:+.0f}"] = {"trace": trace, "end_mean": float(np.mean(tail)), "end_sd": float(np.std(tail)),
                                 "seconds": time.time() - t0}
        log(f"tilt from {start:+.0f} dB: last 100 steps {np.mean(tail):+.2f} +- {np.std(tail):.2f} dB")
    return res


def check_octaves(args, cfg, device, log):
    """Recordings against other recordings through per-octave gains (9, from a random +-3 dB), the gains the only
    generator parameters, on the adversarial term alone, with the critic as in training (R1, mono). Judged by the mean
    of each gain over the second half (a GAN circles its answer): the RMS of those means should be near 0."""
    ds = MaestroSegments(args.data, "train", cfg, 2.0, 1.0, 12.0, length=args.oct_steps * 16, years=args.years)
    torch.manual_seed(0)
    sr = cfg.sample_rate
    loader = DataLoader(ds, batch_size=16, collate_fn=collate, num_workers=args.workers)
    disc = make_discriminator(args.critic, args.mono).to(device)
    dopt = torch.optim.Adam(disc.parameters(), lr=2e-4, betas=(0.5, 0.9))
    g0 = (torch.rand(len(OCT), generator=torch.Generator().manual_seed(1)) * 6 - 3).to(device)
    gains = g0.clone().requires_grad_(True)
    gopt = torch.optim.Adam([gains], lr=args.oct_lr)
    W, trace, t0 = None, [], time.time()
    for step, b in enumerate(loader):
        a = b["audio"][..., int(b["loss_start"][0]):].to(device)
        real, src = highpass(a[:8], sr), highpass(a[8:], sr)
        if W is None:
            W = octave_weights(src.shape[-1], sr, device)
        fake = band_gains(src, sr, gains, W)
        spec = disc.spectra(real) if hasattr(disc, "spectra") else None
        dl, _ = critic_step(disc, dopt, real, fake, spec, args.r1, args.r1_every, step)
        gopt.zero_grad()
        adv_only(disc, fake).backward()
        gopt.step()
        trace.append(gains.detach().cpu().numpy().copy())
        if step % 100 == 0:
            log(f"octaves: step {step} RMS {np.sqrt(np.mean(trace[-1] ** 2)):.2f} dB, critic loss {float(dl):.3f}")
    tr = np.array(trace)
    half = tr[len(tr) // 2:]
    mean = half.mean(0)
    res = {"start": g0.cpu().tolist(), "trace": tr.tolist(), "second_half_mean": mean.tolist(),
           "rms_start": float(np.sqrt(np.mean(g0.cpu().numpy() ** 2))), "rms_second_half_mean": float(np.sqrt(np.mean(mean ** 2))),
           "rms_second_half": float(np.sqrt(np.mean(half ** 2))), "seconds": time.time() - t0}
    log("octaves: band   " + " ".join(f"{f:>6.0f}" for f in OCT))
    log("octaves: start  " + " ".join(f"{v:+6.2f}" for v in g0.cpu().numpy()))
    log("octaves: 2nd-half mean " + " ".join(f"{v:+6.2f}" for v in mean))
    log(f"octaves: RMS {res['rms_start']:.2f} dB at the start; of the second half's means {res['rms_second_half_mean']:.2f} "
        f"dB; over the second half {res['rms_second_half']:.2f} dB")
    return res


def build_model(args, device, log):
    state = torch.load(args.model, map_location="cpu")
    over = {k: _parse_value(v) for k, v in (o.split("=", 1) for o in args.cfg)}
    cfg = PianoConfig.from_dict({**state["cfg"], "sample_rate": 24000, **over})
    model = NeuralPhysicalPiano(cfg).to(device)
    sd = state["ema"]["weights"] if "ema" in state else state["model"]
    load_weights(model, {k: v for k, v in sd.items() if not k.startswith("context.")}, log=log)  # the residual fresh
    return cfg, model


def check_model(args, cfg, model, device, log):
    sr = cfg.sample_rate
    model.eval()
    # the renders (frozen generator), with their recordings for the band error
    pool_set = MaestroSegments(args.data, "train", cfg, 2.0, 1.0, 12.0, length=args.pool * 8, deterministic=True,
                               years=args.years, seed=5)
    fakes = []
    with torch.no_grad():
        for b in DataLoader(pool_set, batch_size=8, collate_fn=collate, num_workers=args.workers):
            b = to_device(b, device)
            s = int(b["loss_start"][0])
            fakes.append(highpass(model(b, b["audio"].shape[-1], residual=False)["audio"][..., s:].float(), sr))
    fakes = torch.cat(fakes)
    log(f"model: {len(fakes)} renders drawn")
    real_set = MaestroSegments(args.data, "train", cfg, 2.0, 1.0, 12.0, length=args.critic_steps * 8,
                               years=args.years, seed=7)
    disc = make_discriminator(args.critic, args.mono).to(device)
    dopt = torch.optim.Adam(disc.parameters(), lr=2e-4, betas=(0.5, 0.9))
    g = torch.Generator().manual_seed(0)
    for step, b in enumerate(DataLoader(real_set, batch_size=8, collate_fn=collate, num_workers=args.workers)):
        real = highpass(b["audio"][..., int(b["loss_start"][0]):].to(device), sr)
        fake = fakes[torch.randint(len(fakes), (8,), generator=g)]
        spec = disc.spectra(real) if hasattr(disc, "spectra") else None
        dl, _ = critic_step(disc, dopt, real, fake, spec, args.r1, args.r1_every, step)
        if step % 100 == 0:
            log(f"model: critic step {step} loss {float(dl):.3f}")
    # held out: the validation pieces, render and recording of the same excerpts
    val = MaestroSegments(args.data, "validation", cfg, 2.0, 1.0, 12.0, length=args.val, deterministic=True,
                          years=args.years, seed=1)
    sr_, sf_, Ef, Er = [], [], 0.0, 0.0
    push_adv, push_fm = 0.0, 0.0
    W = None
    for b in DataLoader(val, batch_size=8, collate_fn=collate, num_workers=args.workers):
        b = to_device(b, device)
        s = int(b["loss_start"][0])
        with torch.no_grad():
            fake = highpass(model(b, b["audio"].shape[-1], residual=False)["audio"][..., s:].float(), sr)
        real = highpass(b["audio"][..., s:].float(), sr)
        with torch.no_grad():
            sr_ += critic_score(disc, real).tolist()
            sf_ += critic_score(disc, fake).tolist()
        if W is None:
            W = octave_weights(fake.shape[-1], sr, device)
        Ef = Ef + (W @ (torch.fft.rfft(fake).abs() ** 2).sum((0, 1))).double()
        Er = Er + (W @ (torch.fft.rfft(real).abs() ** 2).sum((0, 1))).double()
        gb = torch.zeros(len(OCT), device=device, requires_grad=True)
        adv, fm = generator_adv_loss(disc, real, band_gains(fake, sr, gb, W))
        ga, = torch.autograd.grad(adv, gb, retain_graph=True)
        gf, = torch.autograd.grad(fm, gb)
        push_adv, push_fm = push_adv - ga.double(), push_fm - gf.double()
    err = (10 * torch.log10(Ef / Er)).cpu().numpy()  # render - recording, dB per octave band
    pa, pf = push_adv.cpu().numpy(), push_fm.cpu().numpy()
    want = -(err - err.mean())  # the tilt that would fix it (the level aside)
    agree = lambda p: float(np.mean(np.sign(p - p.mean()) == np.sign(want)))
    corr = lambda p: float(np.corrcoef(p, want)[0, 1])
    res = {"auc": auc(sr_, sf_), "n_val": len(sr_), "bands_hz": OCT.tolist(), "err_db": err.tolist(),
           "push_adv": pa.tolist(), "push_fm": pf.tolist(), "sign_agree_adv": agree(pa), "sign_agree_fm": agree(pf),
           "corr_adv": corr(pa), "corr_fm": corr(pf)}
    log(f"model: held-out AUC {res['auc']:.3f} over {len(sr_)} + {len(sf_)} excerpts")
    log("model: band  " + " ".join(f"{f:>7.0f}" for f in OCT))
    log("model: err dB" + " ".join(f"{v:+7.2f}" for v in err))
    log("model: adv  " + " ".join(f"{v:+7.3f}" for v in pa / np.abs(pa).max()))
    log("model: fm   " + " ".join(f"{v:+7.3f}" for v in pf / np.abs(pf).max()))
    log(f"model: push vs the tilt that would fix the error: adv sign agreement {res['sign_agree_adv']:.2f}, "
        f"corr {res['corr_adv']:+.2f}; fm {res['sign_agree_fm']:.2f}, corr {res['corr_fm']:+.2f}")
    return res, disc


def check_share(args, cfg, model, disc, device, log):
    sr = cfg.sample_rate
    model.train()
    comp = CompositeLoss(sr).to(device)
    ds = MaestroSegments(args.data, "train", cfg, 2.0, 1.0, 12.0, length=args.share_batches * 8, deterministic=True,
                         years=args.years, seed=9)
    params = [p for n, p in model.named_parameters() if n.startswith("context.")]
    for p in model.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    flat = lambda gs: torch.cat([(x if x is not None else torch.zeros_like(p)).flatten() for x, p in zip(gs, params)])
    rows = []
    for b in DataLoader(ds, batch_size=8, collate_fn=collate, num_workers=args.workers):
        b = to_device(b, device)
        n, s = b["audio"].shape[-1], int(b["loss_start"][0])
        with torch.no_grad():
            second = model(b, n, residual=True)["audio"].float()
        out = model(b, n, residual=True, extras=("texture_view",) if args.reach == "texture" else ())
        score, _ = comp(out["audio"].float(), b["audio"].float(), b, s, out["partials"], second)
        gc = flat(torch.autograd.grad(score, params, retain_graph=True, allow_unused=True))
        real = highpass(b["audio"][..., s:].float(), sr)
        fake = highpass(out["audio_texture" if args.reach == "texture" else "audio"][..., s:].float(), sr)
        adv, fm = generator_adv_loss(disc, real, fake)
        ga = flat(torch.autograd.grad(args.adv_weight * adv, params, retain_graph=True, allow_unused=True))
        gf = flat(torch.autograd.grad(args.adv_weight * args.fm_weight * fm, params, allow_unused=True))
        cos = lambda a, c: float((a @ c) / (a.norm() * c.norm()).clamp(min=1e-30))
        rows.append({"composite": float(gc.norm()), "adv": float(ga.norm()), "fm": float(gf.norm()),
                     "cos_adv_comp": cos(ga, gc), "cos_fm_comp": cos(gf, gc)})
        log("share: " + " ".join(f"{k} {v:.3g}" for k, v in rows[-1].items()))
    med = {k: float(np.median([r[k] for r in rows])) for k in rows[0]}
    log("share, medians: " + " ".join(f"{k} {v:.3g}" for k, v in med.items()) +
        f"; GAN / composite norm {(med['adv'] + med['fm']) / med['composite']:.3f} (adv weight {args.adv_weight}, "
        f"fm weight {args.fm_weight}, reach {args.reach})")
    return {"rows": rows, "median": med}


def check_timing(args, cfg, model, device, log):
    """Seconds per part of a stage-2 composite step: without the GAN, with it through the texture view, and with it
    through the output (a second backward into GAN_PARAMS alone), on the same batches (the first two discarded)."""
    sr = cfg.sample_rate
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)
    comp = CompositeLoss(sr).to(device)
    disc = make_discriminator(args.critic, args.mono).to(device)
    dopt = torch.optim.Adam(disc.parameters(), lr=2e-4, betas=(0.5, 0.9))
    gan_params = [p for n, p in model.named_parameters() if n.startswith(("noise.", "context."))]
    ds = MaestroSegments(args.data, "train", cfg, 2.0, 1.0, 12.0, length=8 * args.timing_batches, deterministic=True,
                         years=args.years, seed=11)
    sync = lambda: (torch.cuda.synchronize(), time.time())[1]
    T = {}
    for i, b in enumerate(DataLoader(ds, batch_size=8, collate_fn=collate, num_workers=args.workers)):
        b = to_device(b, device)
        n, s = b["audio"].shape[-1], int(b["loss_start"][0])
        for mode in ("plain", "texture", "residual"):
            torch.cuda.reset_peak_memory_stats()
            t = {"start": sync()}
            with torch.no_grad():
                second = model(b, n, residual=True)["audio"].float()
            t["second"] = sync()
            out = model(b, n, residual=True, extras=("texture_view",) if mode == "texture" else ())
            t["render"] = sync()
            loss, _ = comp(out["audio"].float(), b["audio"].float(), b, s, out["partials"], second)
            t["score"] = sync()
            gan = None
            if mode != "plain":
                real = highpass(b["audio"][..., s:].float(), sr)
                fake = highpass(out["audio_texture" if mode == "texture" else "audio"][..., s:].float(), sr)
                spec = disc.spectra(real) if hasattr(disc, "spectra") else None
                disc.requires_grad_(True)
                dopt.zero_grad()
                discriminator_loss(disc, real, fake, spec).backward()
                dopt.step()
                disc.requires_grad_(False)
                t["critic step"] = sync()
                adv, fm = generator_adv_loss(disc, real, fake, spec)
                gan = args.adv_weight * (adv + args.fm_weight * fm)
                t["critic fwd"] = sync()
                if mode == "texture":
                    loss, gan = loss + gan, None
            if gan is not None:
                torch.autograd.backward(gan, inputs=gan_params, retain_graph=True)
                t["gan backward"] = sync()
            loss.backward()
            t["backward"] = sync()
            model.zero_grad(set_to_none=True)
            out = loss = gan = second = None
            if i < 2:
                continue
            keys = list(t)
            for a, c in zip(keys, keys[1:]):
                T.setdefault(mode, {}).setdefault(c, []).append(t[c] - t[a])
            T[mode].setdefault("step", []).append(t[keys[-1]] - t["start"])
            T[mode].setdefault("peak GB", []).append(torch.cuda.max_memory_allocated() / 2**30)
    res = {m: {k: float(np.mean(v)) for k, v in d.items()} for m, d in T.items()}
    for m, d in res.items():
        log(f"timing {m:8s} " + "  ".join(f"{k} {v:.3f}" for k, v in d.items()))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/maestro24k")
    ap.add_argument("--years", type=int, nargs="*", default=[2018])
    ap.add_argument("--model", default="runs/phase6/env_fit2/model.pt")
    ap.add_argument("--cfg", nargs="*", default=["residual_kind=aware", "res_curve_partials=32", "res_noise_bands=32",
                                                 "res_note_noise=16", "res_latent=8"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--parts", nargs="*", default=["octaves", "model", "share"])
    ap.add_argument("--mono", action="store_true", help="the critic judges the channels' mean (train.py --critic-mono)")
    ap.add_argument("--r1", type=float, default=0.0)
    ap.add_argument("--r1-every", type=int, default=4)
    ap.add_argument("--oct-steps", type=int, default=800)
    ap.add_argument("--oct-lr", type=float, default=0.01)
    ap.add_argument("--critic", choices=("patch", "wide"), default="wide")
    ap.add_argument("--reach", choices=("texture", "residual"), default="residual")
    ap.add_argument("--fm-weight", type=float, default=2.0)
    ap.add_argument("--timing-batches", type=int, default=6)
    ap.add_argument("--tilt-steps", type=int, default=600)
    ap.add_argument("--tilt-lr", type=float, default=0.02)
    ap.add_argument("--pool", type=int, default=48, help="batches of 8 renders the critic trains against")
    ap.add_argument("--critic-steps", type=int, default=600)
    ap.add_argument("--val", type=int, default=64)
    ap.add_argument("--share-batches", type=int, default=6)
    ap.add_argument("--adv-weight", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(0.8)
    device = torch.device("cuda")
    os.makedirs(args.out, exist_ok=True)
    logf = open(os.path.join(args.out, "log.txt"), "a")

    def log(m):
        print(m, flush=True)
        logf.write(m + "\n")
        logf.flush()

    log(f"=== {time.ctime()}: {vars(args)}")
    cfg, model = build_model(args, device, log)
    out = {}
    if "octaves" in args.parts:
        out["octaves"] = check_octaves(args, cfg, device, log)
    if "tilt" in args.parts:
        out["tilt"] = check_tilt(args, cfg, device, log)
    disc = None
    if "model" in args.parts or "share" in args.parts:
        out["model"], disc = check_model(args, cfg, model, device, log)
    if "share" in args.parts:
        out["share"] = check_share(args, cfg, model, disc, device, log)
    if "timing" in args.parts:
        out["timing"] = check_timing(args, cfg, model, device, log)
    with open(os.path.join(args.out, "result.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
