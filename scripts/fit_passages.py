"""Review 6, step 2: what the residual could at best do on a passage, by ear.

    python scripts/fit_passages.py data/maestro24k --passages runs/fitted_passages/passages/passages.json \\
        --model runs/phase6/train_run/train/last.pt --out runs/fitted_passages/run1 --samples samples/fitted

For each passage (``scripts/pick_passages.py``: ~10 s of held-out music), the phase-6 model is frozen and the
physics-aware residual is replaced by free outputs fitted to that passage's recording on the training loss
(``PianoLoss``, level term 0.5; a fresh noise draw per step; the per-strike variation off). That is deliberately
unfair: a fit to one take is the best any residual with those outputs could do, not what a trained one will do.
Versions, every one rendered with the same noise seed and its level matched to the recording over the passage:

* ``recording``; ``physics``: phase 6 without the residual;
* ``current``: the residual's present outputs fitted: per note a gain curve over time for each octave group of its
  partials, the onset-time corrections (level, brightness, decay, knock, attack noise), and the residual noise path
  (16 bands per 20 ms). Not the frame-wide band gains: free per 20 ms they paint the recording's envelope onto
  anything, like a vocoder (review 6, section 8.2 discussion);
* ``extended``: one gain curve per partial (partials 1-31, the rest share one), the noise path in the noise bank's
  32 bands, and twice the bounds on everything;
* ``extended_no_noise``, ``extended_no_curves``: the extended fit with its noise path, or its gain curves, set back
  to zero: which part of the fit the ear needs;
* ``shuffled``: every note struck in the passage gets the fitted extended outputs of a note of similar pitch and
  velocity from *another* passage, aligned at the onset: the right amount of variety, not this take's;
* ``typical``: every such note gets the median of those notes' outputs: no variety, roughly the best a residual that
  always predicts the same correction could reach. Both use the passage's noise path held at its median per band.
  (Some fitted values are extreme because they cover model errors; shuffling moves those to other notes.)

Written: ``<out>/report.md`` (the fit's loss per version on an unseen noise seed; for the record, not the decision),
``<out>/fits.pt`` (the fitted outputs), ``<out>/log.txt``; and per passage ``<k>_<version>.wav`` and ``<k>_ab.wav``
(versions in the listening order, half a second apart) in ``--samples``, with a README.
"""

import argparse
import json
import math
import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.6,max_split_size_mb:256")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

from pianonn.data import MaestroSegments  # noqa: E402
from pianonn.losses import PianoLoss, highpass  # noqa: E402
from pianonn.render import load_variant  # noqa: E402
from pianonn.residual import group_centers  # noqa: E402
from pianonn.synth import ContextNet  # noqa: E402
from pianonn.train import collate, to_device  # noqa: E402

N_NOTE = sum(n for n, _ in ContextNet.NOTE.values())
ORDER = ["recording", "physics", "current", "extended", "extended_no_noise", "extended_no_curves", "typical", "shuffled"]


class FreeOutputs(nn.Module):
    """Stands in for the aware residual (same call, same outputs): free values per note and per control frame,
    bounded as the net's (``s * tanh(z)``) times ``scale``. ``groups`` gain curves per note: octave groups (the
    net's) or, with the model's ``curve_partials``, one per partial. All start at zero: the physics."""

    def __init__(self, cfg, N, Cn, groups, noise_bands, scale):
        super().__init__()
        self.cfg, self.scale, self.noise_bands = cfg, scale, noise_bands
        self.register_buffer("centers", torch.tensor(group_centers(cfg.res_groups)), persistent=False)
        self.curves = nn.Parameter(torch.zeros(1, N, groups, Cn))
        self.note = nn.Parameter(torch.zeros(1, N, N_NOTE))
        self.noise = nn.Parameter(torch.zeros(1, noise_bands, Cn))

    def curve_bound(self):
        return self.scale * self.cfg.res_curve_db / (20 / math.log(10))  # dB -> log amplitude

    def values(self):
        """The outputs in their own units: curves (log amplitude), note outputs (raw vector), noise (log amplitude)."""
        out, i = {}, 0
        for name, (n, s) in ContextNet.NOTE.items():
            v = self.scale * s * torch.tanh(self.note[..., i: i + n])
            out[name] = v[..., 0] if n == 1 else v
            i += n
        return (self.curve_bound() * torch.tanh(self.curves), out,
                self.scale * ContextNet.FRAME["noise_level"][1] * torch.tanh(self.noise))

    def forward(self, onset_roll, key_down, pedals, ki, u, onset, offset, mask, cond, H, n_samples, *rest):
        curves, note, noise = self.values()
        m = mask.float()
        note = {k: v * (m if v.dim() == 2 else m[..., None]) for k, v in note.items()}
        K = self.cfg.res_control
        Fw = onset_roll.shape[-1] - H
        x = F.interpolate(noise, size=(noise.shape[-1] - 1) * K, mode="linear", align_corners=False)[..., :Fw]
        if x.shape[-1] < Fw:
            x = torch.cat([x, x[..., -1:].expand(-1, -1, Fw - x.shape[-1])], -1)
        frame = {"noise_level": x, "band_gain": x.new_zeros(1, ContextNet.FRAME["band_gain"][0], Fw)}
        return note, frame, curves * m[..., None, None]

    def set_values(self, curves=None, note=None, noise=None):
        """Set the raw parameters so the outputs equal the given values (clipped inside the bounds)."""
        inv = lambda v, b: torch.atanh((v / b).clamp(-0.999, 0.999))
        with torch.no_grad():
            if curves is not None:
                self.curves.copy_(inv(curves, self.curve_bound()))
            if note is not None:
                i = 0
                for name, (n, s) in ContextNet.NOTE.items():
                    self.note[..., i: i + n].copy_(inv(note[..., i: i + n], self.scale * s))
                    i += n
            if noise is not None:
                self.noise.copy_(inv(noise, self.scale * ContextNet.FRAME["noise_level"][1]))

    def note_vector(self):
        """The note outputs as one vector per note ``[N, N_NOTE]`` in their units."""
        _, note, _ = self.values()
        return torch.cat([v[..., None] if v.dim() == 2 else v for v in note.values()], -1)[0]


def make_free(model, b, kind):
    n = b["audio"].shape[-1]
    N = b["pitch"].shape[1]
    Cn = model.context.n_control(n) if hasattr(model.context, "n_control") else n // (model.cfg.res_control * model.cfg.hop) + 2
    if kind == "current":
        model.curve_partials = 0
        return FreeOutputs(model.cfg, N, Cn, model.cfg.res_groups, ContextNet.FRAME["noise_level"][0], 1.0)
    model.curve_partials = 32
    return FreeOutputs(model.cfg, N, Cn, 32, model.cfg.noise_bands, 2.0)


def render(model, b, seed, residual=True):
    n, s = b["audio"].shape[-1], int(b["loss_start"][0])
    return model(b, n, residual=residual, generator=torch.Generator(device=b["audio"].device).manual_seed(seed))["audio"][..., s:]


def fit(model, b, free, steps, lr, loss_fn, log, tag):
    model.context = free
    opt = torch.optim.Adam(free.parameters(), lr=lr)
    s = int(b["loss_start"][0])
    tgt = b["audio"][..., s:].float()
    on, om = b["onset"] - s / model.cfg.sample_rate, b["mask"]
    curve, t0 = [], time.time()
    for k in range(steps):
        pred = model(b, b["audio"].shape[-1], residual=True)["audio"][..., s:].float()
        loss, _ = loss_fn(pred, tgt, on, om)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        curve.append(float(loss))
        if k % 25 == 0 or k == steps - 1:
            log(f"    {tag} step {k}: {curve[-1]:.4f} ({time.time() - t0:.0f} s)")
    return curve


@torch.no_grad()
def scored(model, b, loss_fn, seed=7, residual=True):
    s = int(b["loss_start"][0])
    y = render(model, b, seed, residual).float()
    tot, parts = loss_fn(y, b["audio"][..., s:].float(), b["onset"] - s / model.cfg.sample_rate, b["mask"])
    return float(tot), {k: float(v) for k, v in parts.items()}


def level_matched(y, t, sr):
    e_y, e_t = highpass(y[None], sr).pow(2).sum(), highpass(t[None], sr).pow(2).sum()
    g = float((e_t / e_y.clamp(min=1e-20)).sqrt())
    return y * g, 20 * math.log10(g)


def note_table(b, free, model):
    """Per note struck inside the scored passage: pitch, velocity, onset (s re the window), its fitted curves as a
    function of age (``[groups, A]`` on the control grid from the onset, the last value held) and its note vector."""
    sr, K, hop = model.cfg.sample_rate, model.cfg.res_control, model.cfg.hop
    dt = K * hop / sr
    curves, _, noise = free.values()
    vec = free.note_vector()
    s = int(b["loss_start"][0]) / sr
    rows = []
    for i in range(b["pitch"].shape[1]):
        o = float(b["onset"][0, i])
        if not bool(b["mask"][0, i]) or o < s:
            continue
        c = curves[0, i]  # [G, Cn] on the window's grid
        t = torch.arange(c.shape[-1], device=c.device) * dt
        ages = torch.arange(0, c.shape[-1], device=c.device) * dt
        pos = ((o + ages) / dt).clamp(max=c.shape[-1] - 1)  # fractional grid index at each age
        i0 = pos.floor().long().clamp(max=c.shape[-1] - 2)
        w = pos - i0
        aged = c[:, i0] * (1 - w) + c[:, i0 + 1] * w  # [G, A]
        rows.append({"i": i, "pitch": int(b["pitch"][0, i]), "vel": float(b["velocity"][0, i]), "onset": o,
                     "aged": aged.detach().cpu(), "vec": vec[i].detach().cpu()})
    return rows, noise.detach()


def to_window(aged, onset, Cn, dt):
    """Age-aligned curves ``[G, A]`` back onto a window's control grid for a note struck at ``onset``."""
    t = torch.arange(Cn) * dt
    age = (t - onset) / dt
    i0 = age.floor().long().clamp(0, aged.shape[-1] - 2)
    w = (age - i0).clamp(0, 1)
    v = aged[:, i0] * (1 - w) + aged[:, i0 + 1] * w
    return torch.where(age[None] >= 0, v, torch.zeros_like(v))


def candidates(row, pool, k_self):
    for dp, dv in ((3, 15), (6, 25), (12, 40)):
        c = [r for r in pool if r["k"] != k_self and abs(r["pitch"] - row["pitch"]) <= dp and abs(r["vel"] - row["vel"]) <= dv]
        if c:
            return c
    return sorted((r for r in pool if r["k"] != k_self), key=lambda r: abs(r["pitch"] - row["pitch"]))[:5]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data")
    ap.add_argument("--passages", required=True)
    ap.add_argument("--model", default="runs/phase6/train_run/train/last.pt")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.05, help="Adam on the outputs before the tanh (unitless)")
    ap.add_argument("--only", type=int, nargs="*", help="only these passages (indices)")
    ap.add_argument("--seed", type=int, default=0, help="render-noise seed of the listening files")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", required=True)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.samples, exist_ok=True)
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
    assert model.cfg.residual_kind == "aware", "the free outputs stand in for the aware residual"
    model.strike_on = False
    for p in model.parameters():
        p.requires_grad_(False)
    sr = model.cfg.sample_rate
    dt = model.cfg.res_control * model.cfg.hop / sr
    with open(args.passages, encoding="utf-8") as f:
        spec = json.load(f)
    passages = spec["passages"]
    ds = {}
    loss_fn = PianoLoss(sr, level_weight=0.5).to(dev)
    log(f"{args.model}: {len(passages)} passages of {spec['seconds']} s, {args.steps} steps, lr {args.lr}")

    batches, fits, results, pool = {}, {}, {}, []
    idx = args.only if args.only is not None else range(len(passages))
    for k in idx:
        p = passages[k]
        split = p.get("split", "test")
        if split not in ds:
            ds[split] = MaestroSegments(args.data, split, model.cfg, spec["seconds"], spec["warmup"], 12.0, length=1,
                                        deterministic=True, years=[2018])
        b = to_device(collate([ds[split].item_at(p["piece"], p["start_s"])]), dev)
        batches[k] = b
        log(f"passage {k} ({p['kind']}): {p['piece']} at {p['start_s']} s, {int(b['mask'].sum())} notes in the window")
        clips = {"recording": b["audio"][0, :, int(b["loss_start"][0]):]}
        res = {}
        with torch.no_grad():
            model.curve_partials = 0
            clips["physics"] = render(model, b, args.seed, residual=False)[0]
        res["physics"] = scored(model, b, loss_fn, residual=False)
        for kind in ("current", "extended"):
            free = make_free(model, b, kind).to(dev)
            curve = fit(model, b, free, args.steps, args.lr, loss_fn, log, kind)
            fits[(k, kind)] = {n: v.detach().cpu() for n, v in free.state_dict().items()}
            res[kind] = scored(model, b, loss_fn)
            res[kind][1]["fit_first"], res[kind][1]["fit_last10"] = curve[0], float(np.mean(curve[-10:]))
            with torch.no_grad():
                clips[kind] = render(model, b, args.seed)[0]
            if kind == "extended":
                saved = {n: v.detach().clone() for n, v in free.state_dict().items()}
                with torch.no_grad():
                    free.noise.zero_()
                    clips["extended_no_noise"] = render(model, b, args.seed)[0]
                    res["extended_no_noise"] = scored(model, b, loss_fn)
                    free.load_state_dict(saved)
                    free.curves.zero_()
                    clips["extended_no_curves"] = render(model, b, args.seed)[0]
                    res["extended_no_curves"] = scored(model, b, loss_fn)
                    free.load_state_dict(saved)
                rows, noise = note_table(b, free, model)
                for r in rows:
                    pool.append({**r, "k": k})
                results.setdefault(k, {})["noise_median"] = noise.median(-1, keepdim=True).values.expand_as(noise).cpu()
            del free
            torch.cuda.empty_cache()
        results[k].update({"clips": {n: c.cpu() for n, c in clips.items()}, "scores": res})
        log("  " + "  ".join(f"{v} {s[0]:.4f}" for v, s in res.items()))

    # shuffled and typical: each passage's struck notes take outputs from notes of the other passages
    rng = np.random.default_rng(1)
    for k in idx:
        b = batches[k]
        free = make_free(model, b, "extended").to(dev)
        Cn = free.curves.shape[-1]
        G = free.curves.shape[2]
        for variant in ("shuffled", "typical"):
            curves = torch.zeros(1, b["pitch"].shape[1], G, Cn)
            vecs = torch.zeros(1, b["pitch"].shape[1], N_NOTE)
            for r in (r for r in pool if r["k"] == k):
                c = candidates(r, pool, k)
                if variant == "shuffled":
                    src = c[int(rng.integers(len(c)))]
                    aged, vec = src["aged"], src["vec"]
                else:
                    A = min(x["aged"].shape[-1] for x in c)
                    aged = torch.stack([x["aged"][:, :A] for x in c]).median(0).values
                    vec = torch.stack([x["vec"] for x in c]).median(0).values
                curves[0, r["i"]] = to_window(aged, r["onset"], Cn, dt)
                vecs[0, r["i"]] = vec
            free.set_values(curves.to(dev), vecs.to(dev), results[k]["noise_median"].to(dev))
            model.context = free
            with torch.no_grad():
                results[k]["clips"][variant] = render(model, b, args.seed)[0].cpu()
                results[k]["scores"][variant] = scored(model, b, loss_fn)
        del free
        torch.cuda.empty_cache()

    # files
    gap = np.zeros((sr // 2, model.cfg.channels), dtype=np.float32)
    lines = [f"# Fitted passages ({time.strftime('%Y-%m-%d')})", "",
             f"Model `{args.model}`, frozen, per-strike variation off; free outputs fitted per passage, {args.steps} Adam "
             f"steps, lr {args.lr}. Distances: the training loss (`PianoLoss`, level 0.5) on noise seed 7, which the fits "
             "never saw; for the record, the ear decides. Gain: what level matching applied to the listening file.", ""]
    readme = ["# Fitted passages: what the residual could at best do", "",
              "Review 6, step 2 (`docs/reviews/review_6_training.md` 8.2; `scripts/fit_passages.py`). For each passage of "
              "held-out music the phase-6 model is frozen and the residual's outputs are fitted to that recording. Each "
              "`<k>_ab.wav` plays, half a second apart, in this order:", ""]
    desc = {"recording": "the recording", "physics": "phase 6 without the residual",
            "current": "the residual's present outputs, fitted to this take (octave groups, attack corrections, "
                       "16-band noise)",
            "extended": "richer outputs, fitted: a gain curve per partial, 32-band noise, wider ranges",
            "extended_no_noise": "the extended fit without its noise path",
            "extended_no_curves": "the extended fit without its gain curves",
            "typical": "each note the median fitted outputs of similar notes in the other passages: no variety",
            "shuffled": "each note the fitted outputs of one similar note from another passage: variety, not this take's"}
    readme += [f"{j + 1}. `{v}`: {desc[v]}" for j, v in enumerate(ORDER)]
    readme += ["", "Every version uses the same noise seed and is level matched to the recording over the passage "
               "(`report.md` in the run folder has the gains). Suggested order: recording, physics, extended, typical; "
               "the rest if those differ.", "", "## Passages", ""]
    for k in idx:
        p, r = passages[k], results[k]
        rec = r["clips"]["recording"].float()
        files = []
        lines += [f"## {k}: {p['kind']} (`{p['piece']}` at {p['start_s']} s)", "", p.get("why", ""), "",
                  "| version | total | band | fine | attack | level | gain (dB) |", "|---|---|---|---|---|---|---|"]
        for v in ORDER:
            if v not in r["clips"]:
                continue
            y = r["clips"][v].float()
            g_db = 0.0
            if v != "recording":
                y, g_db = level_matched(y, rec, sr)
            y = y.clamp(-1, 1)
            sf.write(os.path.join(args.samples, f"{k}_{v}.wav"), y.T.numpy(), sr)
            files.append(y.T.numpy())
            if v != "recording":
                t, parts = r["scores"][v]
                lines.append(f"| {v} | {t:.4f} | {parts.get('band', 0):.4f} | {parts.get('fine', 0):.4f} | "
                             f"{parts.get('attack', 0):.4f} | {parts.get('level', 0):.4f} | {g_db:+.1f} |")
        sf.write(os.path.join(args.samples, f"{k}_ab.wav"), np.concatenate([x for c in files for x in (c, gap)]), sr)
        for v in ("current", "extended"):
            if v in r["scores"]:
                lines.append(f"\nFit `{v}`: {r['scores'][v][1]['fit_first']:.4f} → {r['scores'][v][1]['fit_last10']:.4f} "
                             "(training loss, first step → mean of the last 10)")
        lines.append("")
        readme.append(f"{k}. **{p['kind']}**: `{p['piece']}`, {p['start_s'] + 1:.0f}-{p['start_s'] + 1 + spec['seconds']:.0f} "
                      f"s of the piece. {p.get('why', '')}. {p.get('onsets_per_s')} onsets/s, mean velocity "
                      f"{p.get('velocity')}, pedal {100 * p.get('pedal', 0):.0f} %.")
    torch.save(fits, os.path.join(args.out, "fits.pt"))
    with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(args.samples, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(readme) + "\n")
    log("\n".join(lines))


if __name__ == "__main__":
    main()
