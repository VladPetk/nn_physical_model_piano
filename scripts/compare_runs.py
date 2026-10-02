"""Paired comparison of checkpoints on the same held-out excerpts, with a scale for every number (review 6, 2.1, 7).

    python scripts/compare_runs.py data/maestro24k --years 2018 --out runs/round2/compare.md \\
        --model control=runs/round2/b_control/last.pt:physics --model residual=runs/round2/a_residual/last.pt:residual

Each ``--model`` is ``label=checkpoint:physics|residual``, optionally with config overrides after the mode
(``:key=value,...``, as in ``pianonn.render.load_variant``). Every model renders the excerpts of
``scripts/evaluate.py`` (test split, seed 11) under several noise seeds, with the per-strike variation off, as in
training. Reported, per distance:

* **Level matched per piece.** Each piece was recorded at its own level, which no MIDI predicts (the validation
  pieces of 2018 sit -3.8 ... +3.1 dB from the phase-6 model; review 6, 2.1). Each excerpt's render is scaled to its
  piece's level, measured on the piece's *other* excerpts (leave one out). Excerpts alone in their piece keep 0 dB.
  The tables without this are at the end.
* **Errors grouped by piece.** Excerpts of one piece are not independent: every ± is two standard errors with the
  excerpts grouped by piece, so it counts pieces, not excerpts.
* **The like-for-like floor.** The model's render (variation off) against a draw of the same model with its
  per-strike variation on and another noise seed: what this distance reads if the piano *were* the model. The gap
  over it is what a better model could still take off. The floor is the model's stand-in for the piano's take-to-take
  difference, which cannot be measured (one take per piece), and it scales with the variation that was set by hand
  (which the bench finds smaller than the piano's). A model without per-strike variation gets a floor of noise
  seeds only, labelled so.
* For every model against the first one, the paired per-excerpt difference, averaged over the seeds, and in how
  many seeds its sign agrees.

One training seed per model, so the spread between training runs is not in these numbers.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from pianonn.data import MaestroSegments  # noqa: E402
from pianonn.losses import highpass  # noqa: E402
from pianonn.metrics import clustered_mean, make_terms, piece_gains_loo, render  # noqa: E402
from pianonn.render import load_variant  # noqa: E402
from pianonn.train import fixed_batches  # noqa: E402

BASE_KEYS = ("new: total", "new: band", "new: fine", "new: attack", "old: MR-STFT", "log-mel (dB)")


def _onsets(b, sr):
    s = int(b["loss_start"][0])
    return b["onset"] - s / sr, b["mask"]


@torch.no_grad()
def score(model, batches, residual, seed, terms, keys, gains=None, cache=None):
    """Per-excerpt distances ``{key: [excerpts]}`` of the model (variation off) against the recordings, the render
    scaled by ``gains`` (dB per excerpt) if given. ``cache``: renders to reuse (list per batch, on the CPU)."""
    sr = model.cfg.sample_rate
    model.strike_on = False
    acc, k = {key: [] for key in keys}, 0
    for i, b in enumerate(batches):
        s = int(b["loss_start"][0])
        y = cache[i].to(b["audio"].device) if cache is not None else render(model, b, residual, seed)
        if gains is not None:
            y = y * torch.as_tensor(10 ** (gains[k: k + len(y)] / 20), device=y.device, dtype=y.dtype)[:, None, None]
        d = terms(y, b["audio"][..., s:], *_onsets(b, sr))
        for key in keys:
            acc[key].append(d[key].cpu())
        k += len(y)
    return {key: torch.cat(v).numpy() for key, v in acc.items()}


@torch.no_grad()
def renders_and_energies(model, batches, residual, seed):
    """Seed ``seed``'s renders (variation off, kept on the CPU) and each excerpt's energy, model and recording, after
    the loss's 20 Hz high-pass."""
    sr = model.cfg.sample_rate
    model.strike_on = False
    cache, e_y, e_t = [], [], []
    for b in batches:
        s = int(b["loss_start"][0])
        y = render(model, b, residual, seed)
        cache.append(y.cpu())
        e_y.append(highpass(y, sr).pow(2).sum((1, 2)).cpu())
        e_t.append(highpass(b["audio"][..., s:], sr).pow(2).sum((1, 2)).cpu())
    return cache, torch.cat(e_y).numpy(), torch.cat(e_t).numpy()


@torch.no_grad()
def floor(model, batches, residual, seed, terms, keys, has_strike):
    """The model's render (variation off, ``seed``) against a varied draw of itself (variation on, another seed)."""
    sr = model.cfg.sample_rate
    acc = {key: [] for key in keys}
    for b in batches:
        model.strike_on = False
        y0 = render(model, b, residual, seed)
        model.strike_on = has_strike
        y1 = render(model, b, residual, seed + 1000)
        d = terms(y0, y1, *_onsets(b, sr))
        for key in keys:
            acc[key].append(d[key].cpu())
    model.strike_on = False
    return {key: torch.cat(v).numpy() for key, v in acc.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data")
    ap.add_argument("--years", type=int, nargs="*", default=[2018])
    ap.add_argument("--split", default="test")
    ap.add_argument("--examples", type=int, default=96)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--floor-seeds", type=int, default=2)
    ap.add_argument("--level-weight", type=float, default=0.5,
                    help="weight of PianoLoss's whole-excerpt level term in 'train total' (the runs since phase 4 "
                         "train with 0.5); 'new: total' stays without it, as in earlier tables")
    ap.add_argument("--model", action="append", required=True, help="label=checkpoint:physics|residual[:key=value,...] (render.load_variant)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.6)
    keys = (("train total", "new: level") if args.level_weight else ()) + BASE_KEYS
    main_key = keys[0]
    res, res_raw, floors, notes = {}, {}, {}, {}  # label -> {key: [seeds, excerpts]} / {key: [excerpts]} / str
    batches, pieces, specs = None, None, []
    for spec in args.model:
        label, model, residual = load_variant(spec, device=dev)
        specs.append((label, spec.split("=", 1)[1], residual))
        if batches is None:
            ds = MaestroSegments(args.data, args.split, model.cfg, 2.0, 1.0, 12.0, length=args.examples,
                                 deterministic=True, years=args.years, seed=11)
            batches = fixed_batches(ds, args.examples, 8, dev)
            pieces = torch.cat([b["piece"].cpu() for b in batches]).numpy()
        terms = make_terms(model.cfg.sample_rate, dev, level_weight=args.level_weight)
        has_strike = model.strike_on
        cache, e_y, e_t = renders_and_energies(model, batches, residual, 0)
        gains, alone = piece_gains_loo(e_y, e_t, pieces)
        runs = [score(model, batches, residual, s, terms, keys, gains, cache if s == 0 else None)
                for s in range(args.seeds)]
        res[label] = {k: np.stack([r[k] for r in runs]) for k in keys}
        raw = [score(model, batches, residual, 0, terms, keys, None, cache)]
        res_raw[label] = {k: np.stack([r[k] for r in raw]) for k in keys}
        fl = [floor(model, batches, residual, s, terms, keys, has_strike) for s in range(args.floor_seeds)]
        floors[label] = {k: np.mean([f[k] for f in fl], 0) for k in keys}
        per_piece = {p: gains[pieces == p].mean() for p in np.unique(pieces)}
        notes[label] = (f"piece gains {min(per_piece.values()):+.1f} … {max(per_piece.values()):+.1f} dB "
                        f"(sd {np.std(list(per_piece.values())):.1f} over {len(per_piece)} pieces; {alone} excerpts "
                        f"alone in their piece kept 0 dB); floor: "
                        + ("variation on vs off" if has_strike else "noise seeds only (no per-strike variation in this model)"))
        del model, cache
        torch.cuda.empty_cache()
        print("done", label, flush=True)

    fmt = lambda k, v: f"{v:.2f}" if k == "log-mel (dB)" else f"{v:.4f}"
    n_pieces = len(np.unique(pieces))
    lines = [f"# Paired comparison on {args.examples} {args.split} excerpts ({n_pieces} pieces), {args.seeds} noise seeds",
             "", "Models: " + "; ".join(f"{lab} = `{ck}` ({'with' if r else 'without'} the residual)" for lab, ck, r in specs),
             "", "Every render has the per-strike variation off, as in training. ± = 2 standard errors with the excerpts "
             "grouped by piece.", ""]
    for label in res:
        lines.append(f"- {label}: {notes[label]}")

    def means_table(R, with_floor):
        head = "| | " + " | ".join(keys) + (f" | floor ({main_key}) | gap over the floor |" if with_floor else " |")
        out = [head, "|---|" + "---|" * (len(keys) + (2 if with_floor else 0))]
        for label in R:
            cells = []
            for k in keys:
                m = R[label][k].mean(1)
                cells.append(f"{fmt(k, m.mean())}" + (f" [{fmt(k, m.max() - m.min())}]" if len(m) > 1 else ""))
            if with_floor:
                f = floors[label][main_key]
                g, se, _ = clustered_mean(R[label][main_key].mean(0) - f, pieces)
                cells += [fmt(main_key, f.mean()), f"{fmt(main_key, g)} ± {fmt(main_key, 2 * se)}"]
            out.append(f"| {label} | " + " | ".join(cells) + " |")
        return out

    def paired_table(R):
        base = list(R)[0]
        out = ["| | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
        for label in list(R)[1:]:
            cells = []
            for k in keys:
                d = R[label][k] - R[base][k]  # [seeds, excerpts]
                m, se, _ = clustered_mean(d.mean(0), pieces)
                agree = int((np.sign(d.mean(1)) == np.sign(m)).sum())
                cells.append(f"{fmt(k, m)} ± {fmt(k, 2 * se)} ({agree}/{d.shape[0]})")
            out.append(f"| {label} | " + " | ".join(cells) + " |")
        return out

    base = specs[0][0]
    lines += ["", "## Means, level matched per piece", "",
              "Mean over excerpts, averaged over seeds; in brackets, the range of that mean across seeds. The floor and "
              f"the gap are in `{main_key}`.", ""] + means_table(res, True)
    if len(res) > 1:
        lines += ["", f"## Paired differences against `{base}`, level matched per piece", "",
                  "Per excerpt, model − base, averaged over seeds: mean ± 2 se (grouped by piece); then the number of "
                  "seeds whose mean difference has the same sign. Negative means the model is nearer the recording.",
                  ""] + paired_table(res)
    lines += ["", "## As rendered (no level match, seed 0)", ""] + means_table(res_raw, False)
    if len(res) > 1:
        lines += [""] + paired_table(res_raw)
    text = "\n".join(lines) + "\n"
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
