"""The ceiling of the residual's output language: how far could perfect corrections go?

    python scripts/residual_ceiling.py data/maestro24k --years 2018 --out runs/residual/ceiling \\
        --model runs/phase3/step4/train/last.pt

The model is frozen. Its context net is replaced by free outputs per excerpt, in the net's own language and bounds
(``ContextNet.NOTE`` per note, ``ContextNet.FRAME`` per frame), fitted by Adam to each test excerpt's recording on the
training loss (``PianoLoss``), a fresh noise draw at every step as in training. No budget, no network: what any
residual with these outputs could at best do on these excerpts, the noise of the render aside. Variants:
``note`` (per-note outputs only), ``frame`` (per-frame band gains and noise path only), ``both``.

Every variant and the baselines (``physics``: the residual off; ``net``: the trained context net; ``zero``: the free
outputs at zero, the residual's paths on but neutral) are then scored on the excerpts of ``scripts/compare_runs.py``
(test split, seed 11) under render-noise seeds the fit never saw. The per-strike variation is off (as in training).
Written: ``report.md``, ``scores.npz`` (variant -> [seeds, excerpts] per distance), ``outputs.npz`` (the fitted
outputs with each note's pitch, velocity, onset and offset, for a check of how much of them the context predicts),
``curves.json`` (the fit's loss per step).

Variant ``aware`` instead trains the physics-aware residual itself (``pianonn.residual``, from ``--aware-from``) on each
batch of excerpts, as long as the free outputs fit: a memorisation test of that architecture against the ceiling.
"""

import argparse
import copy
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from pianonn.data import MaestroSegments  # noqa: E402
from pianonn.losses import PianoLoss  # noqa: E402
from pianonn.metrics import make_terms, render  # noqa: E402
from pianonn.render import load_variant  # noqa: E402
from pianonn.synth import ContextNet  # noqa: E402
from pianonn.train import fixed_batches  # noqa: E402

KEYS = ("new: total", "new: band", "new: fine", "new: attack", "log-mel (dB)")
N_NOTE = sum(n for n, _ in ContextNet.NOTE.values())
N_FRAME = sum(n for n, _ in ContextNet.FRAME.values())


class FreeContext(nn.Module):
    """Stands in for the context net: free outputs per example, bounded as the net's (``s * tanh``, ``scale`` x its
    bounds). Zero-initialised, so it starts where the residual starts."""

    def __init__(self, B, N, F, parts, scale=1.0):
        super().__init__()
        self.note = nn.Parameter(torch.zeros(B, N, N_NOTE)) if "note" in parts else None
        self.frame = nn.Parameter(torch.zeros(B, F, N_FRAME)) if "frame" in parts else None
        self.scale = scale

    def _split(self, z, spec):
        out, i = {}, 0
        for name, (n, s) in spec.items():
            v = self.scale * s * torch.tanh(z[..., i: i + n])
            out[name] = v[..., 0] if n == 1 else v
            i += n
        return out

    def forward(self, onset_roll, key_down, pedals, ki, u, onset_frame, cond, hist_frames):
        B, N = ki.shape
        F = onset_roll.shape[-1] - hist_frames
        zn = self.note if self.note is not None else onset_roll.new_zeros(B, N, N_NOTE)
        zf = self.frame if self.frame is not None else onset_roll.new_zeros(B, F, N_FRAME)
        frame = self._split(zf, ContextNet.FRAME)
        return self._split(zn, ContextNet.NOTE), {k: v.transpose(1, 2) for k, v in frame.items()}


@torch.no_grad()
def score(model, b, residual, seeds, terms):
    s = int(b["loss_start"][0])
    rows = []
    for seed in seeds:
        d = terms(render(model, b, residual, seed), b["audio"][..., s:], b["onset"] - s / model.cfg.sample_rate, b["mask"])
        rows.append(np.stack([d[k].cpu().numpy() for k in KEYS]))  # [keys, B]
    return np.stack(rows)  # [seeds, keys, B]


def fit(model, b, parts, steps, lr, scale, recon, log, aware=None):
    n, s = b["audio"].shape[-1], int(b["loss_start"][0])
    B, N = b["pitch"].shape
    if aware is not None:  # the aware residual's own weights, a fresh copy per batch
        free = copy.deepcopy(aware).train()
        free.requires_grad_(True)
    else:
        free = FreeContext(B, N, model.n_frames(n), parts, scale).to(b["audio"].device)
    model.context = free
    opt = torch.optim.Adam(free.parameters(), lr=lr)
    tgt = b["audio"][..., s:].float()
    on, om = b["onset"] - s / model.cfg.sample_rate, b["mask"]
    curve, t0 = [], time.time()
    for k in range(steps):
        pred = model(b, n, residual=True)["audio"][..., s:].float()
        loss, _ = recon(pred, tgt, on, om)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        curve.append(float(loss))
        if k % 25 == 0 or k == steps - 1:
            log(f"    {'+'.join(parts)} step {k}: {curve[-1]:.4f} ({time.time() - t0:.0f} s)")
    return free, curve


def outputs_of(free, model, b):
    """The fitted outputs in their own units (dB, log, ...), per note and per frame."""
    with torch.no_grad():
        n = b["audio"].shape[-1]
        H = int(b["hist_frames"].reshape(-1)[0])
        F = H + model.n_frames(n)
        dummy = b["audio"].new_zeros(b["pitch"].shape[0], 1, F)
        note, frame = free(dummy, None, None, b["pitch"], None, None, None, H)
    return {**{"note/" + k: v.cpu().numpy() for k, v in note.items()},
            **{"frame/" + k: v.cpu().numpy() for k, v in frame.items()}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data")
    ap.add_argument("--years", type=int, nargs="*", default=[2018])
    ap.add_argument("--model", default="runs/phase6/env_fit2/model.pt")
    ap.add_argument("--examples", type=int, default=24)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=0.05, help="Adam on the outputs before the tanh (unitless)")
    ap.add_argument("--scale", type=float, default=1.0, help="the outputs' bounds as a multiple of the net's")
    ap.add_argument("--variants", nargs="*", default=["note", "frame", "both"], help="note, frame, both, aware")
    ap.add_argument("--aware-from", default="runs/loss_compare/B_comp/train/last.pt",
                    help="checkpoint of the aware residual for variant aware (its weights are the start)")
    ap.add_argument("--aware-lr", type=float, default=1e-3)
    ap.add_argument("--aware-steps", type=int, help="steps for variant aware (default --steps)")
    ap.add_argument("--max-batches", type=int, help="only the first batches of the excerpts")
    ap.add_argument("--seeds", type=int, nargs="*", default=[101, 102], help="render-noise seeds for the scores")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(args.out, exist_ok=True)
    logf = open(os.path.join(args.out, "log.txt"), "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.8)
    torch.manual_seed(0)
    _, model, _ = load_variant(f"m={args.model}:residual", device=dev, log=log)
    model.strike_on = False
    for p in model.parameters():
        p.requires_grad_(False)
    net = model.context
    ds = MaestroSegments(args.data, "test", model.cfg, 2.0, 1.0, 12.0, length=args.examples, deterministic=True,
                         years=args.years, seed=11)
    batches = fixed_batches(ds, args.examples, args.batch, dev)
    if args.max_batches:
        batches = batches[:args.max_batches]
    aware = None
    if "aware" in args.variants:
        _, am, _ = load_variant(f"a={args.aware_from}:residual", device=dev, log=log)
        assert am.cfg.residual_kind == "aware", args.aware_from
        aware = am.context
        for k in ("res_groups", "res_control", "res_dim", "res_heads", "res_layers", "res_curve_db"):
            setattr(model.cfg, k, getattr(am.cfg, k))
        del am
    terms = make_terms(model.cfg.sample_rate, dev)
    recon = PianoLoss(model.cfg.sample_rate).to(dev)
    log(f"{args.model}: {args.examples} test excerpts, variants {args.variants}, {args.steps} steps, lr {args.lr}, "
        f"bounds x{args.scale}, scored on seeds {args.seeds}")

    scores = {v: [] for v in ["physics", "net", "zero"] + args.variants}
    if aware is not None:  # the aware residual as trained, before fitting it to these excerpts
        scores["aware_start"] = []
    outputs, curves = {v: [] for v in args.variants}, {v: [] for v in args.variants}
    feats = []
    t_start = time.time()
    for bi, b in enumerate(batches):
        log(f"batch {bi + 1}/{len(batches)} ({time.time() - t_start:.0f} s)")
        feats.append({k: b[k].cpu().numpy() for k in ("pitch", "velocity", "onset", "offset", "mask")})
        model.context = net
        scores["physics"].append(score(model, b, False, args.seeds, terms))
        scores["net"].append(score(model, b, True, args.seeds, terms))
        model.context = FreeContext(*b["pitch"].shape, model.n_frames(b["audio"].shape[-1]), ()).to(dev)
        scores["zero"].append(score(model, b, True, args.seeds, terms))
        if aware is not None:
            model.context, model.cfg.residual_kind = aware.eval(), "aware"
            scores["aware_start"].append(score(model, b, True, args.seeds, terms))
            model.cfg.residual_kind = "gru"
        for v in args.variants:
            parts = ("note", "frame") if v == "both" else (v,)
            if v == "aware":
                model.cfg.residual_kind = "aware"
                free, curve = fit(model, b, parts, args.aware_steps or args.steps, args.aware_lr, args.scale, recon, log,
                                  aware=aware)
                free.eval()
                scores[v].append(score(model, b, True, args.seeds, terms))
                model.cfg.residual_kind = "gru"
            else:
                free, curve = fit(model, b, parts, args.steps, args.lr, args.scale, recon, log)
                scores[v].append(score(model, b, True, args.seeds, terms))
                outputs[v].append(outputs_of(free, model, b))
            curves[v].append(curve)
            log(f"  {v}: physics {scores['physics'][-1][:, 0].mean():.4f}  net {scores['net'][-1][:, 0].mean():.4f}  "
                f"fitted {scores[v][-1][:, 0].mean():.4f} (total, unseen seeds)")
            del free
            torch.cuda.empty_cache()
    model.context = net

    res = {v: np.concatenate(x, -1) for v, x in scores.items()}  # [seeds, keys, excerpts]
    np.savez(os.path.join(args.out, "scores.npz"), keys=np.array(KEYS), **res)
    flat = {}
    for v, outs in outputs.items():
        for bi, o in enumerate(outs):
            for k, a in o.items():
                flat[f"{v}/b{bi}/{k}"] = a
    for bi, f in enumerate(feats):
        for k, a in f.items():
            flat[f"feat/b{bi}/{k}"] = a
    np.savez(os.path.join(args.out, "outputs.npz"), **flat)
    with open(os.path.join(args.out, "curves.json"), "w") as f:
        json.dump(curves, f)

    fmt = lambda k, x: f"{x:.2f}" if k == "log-mel (dB)" else f"{x:.4f}"
    n_ex = sum(b["pitch"].shape[0] for b in batches)
    lines = [f"# The residual's ceiling on {n_ex} test excerpts", "",
             f"Model `{args.model}`, frozen, per-strike variation off. Free outputs fitted per excerpt ({args.steps} Adam "
             f"steps, lr {args.lr}, bounds x{args.scale} the net's), scored on render-noise seeds {args.seeds} (not used "
             f"in the fit). `physics`: the residual off; `net`: the trained context net; `zero`: the residual's paths "
             f"on at zero output. Wall clock {(time.time() - t_start) / 60:.0f} min.", "",
             "## Means", "", "| | " + " | ".join(KEYS) + " |", "|---|" + "---|" * len(KEYS)]
    for v, a in res.items():
        lines.append(f"| {v} | " + " | ".join(fmt(k, a[:, i].mean()) for i, k in enumerate(KEYS)) + " |")
    lines += ["", "## Paired differences against `physics`", "",
              "Per excerpt, variant − physics, averaged over seeds: mean ± 2 standard errors over excerpts.", "",
              "| | " + " | ".join(KEYS) + " |", "|---|" + "---|" * len(KEYS)]
    for v, a in res.items():
        if v == "physics":
            continue
        cells = []
        for i, k in enumerate(KEYS):
            d = (a[:, i] - res["physics"][:, i]).mean(0)
            cells.append(f"{fmt(k, d.mean())} ± {fmt(k, 2 * d.std(ddof=1) / np.sqrt(len(d)))}")
        lines.append(f"| {v} | " + " | ".join(cells) + " |")
    lines += ["", "## The fit", "", "Training loss (`PianoLoss`, a fresh noise draw per step), mean over excerpts: first "
              "step, last 10 steps.", ""]
    for v, cs in curves.items():
        c = np.array(cs)
        lines.append(f"- {v}: {c[:, 0].mean():.4f} → {c[:, -10:].mean():.4f}")
    text = "\n".join(lines) + "\n"
    with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8") as f:
        f.write(text)
    log(text)


if __name__ == "__main__":
    main()
