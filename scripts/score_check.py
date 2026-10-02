"""Does a score see what it should? Known changes pushed through the renderer (review 6, step 3).

    python scripts/score_check.py data/maestro24k --model runs/phase6/train_run/train/last.pt --out runs/score_check/run1

On held-out excerpts the model (variation off, noise seed 0) is the reference. Its "take" is a draw of the same model
with the per-strike variation on and another noise seed: a piano that *is* the model, played again. For every term:

* **floor**: the reference against the take (what the term reads for a perfect model);
* **gap**: the reference against the recording, minus the floor (what the term sees of the real model's errors);
* **known changes**: the reference with one change against the take, minus the reference against the take; in units
  of the floor, and as a multiple of its standard error (grouped by piece). A term should register an audible change
  clearly; a change it cannot see is reported as such. The changes:
  upper partials (9+) fading twice as fast; partials 1-8 fading 1.3 times as fast; partials 5-8 +3 dB; the notes
  from C5 up +3 dB (their strings); the aftersound (the unison's other strings, the beating's depth) +6 dB; the
  mechanical noise +6 dB (energy between the partials); the knock +5 dB (the attack parts' power and the knock
  impulse).
* **sweeps**: the reference with one setting moved both ways against the take: each term's best setting (0 is right;
  elsewhere, the term is biased by the take's variation) and how much it rises at the sweep's ends (what it sees of
  the setting, either way). The knock at -8 ... +4 dB (the take's onsets carry the per-strike timing scatter: a term
  that prefers the knock lower punishes the scatter); the decay rates of partials 1-8 and of partials 9+ times
  2^-0.5 ... 2^0.5; the aftersound -6 ... +6 dB.

Two tables. **Per excerpt**: PianoLoss's band, fine, attack and level; the onset term (OnsetLoss, pooled per batch);
PartialView's partials, pooled, pooled_exposed, between and between_pooled (each pooled within its excerpt); and
``old total``, phase 6's training loss without the envelope term (band + 0.25 fine + 0.5 attack + 0.5 level + 0.5
onset). Errors grouped by piece. **The composite** (``pianonn.composite``): its pooled terms pooled across all the
excerpts (each weighted by 1 / the reference's mean power), its read-by-read terms as means, and the weighted sum;
errors by leaving one piece out at a time (jackknife). The per-excerpt terms are saved (``terms.npz``).

``--energy``: every term as an energy score, ``d(ref, take) - d(ref, ref') / 2`` with ``ref'`` a second draw of the
reference (variation on, seed 2, the same change): a proper score, whose optimum is the take's distribution itself,
spreads included (a plain distance between two random draws shrinks when one draw gets less random). The floor column
stays the plain distance (the unit of the changes).
"""

import argparse
import collections
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from pianonn.composite import POOLED, WEIGHTS, level_sums  # noqa: E402
from pianonn.data import MaestroSegments  # noqa: E402
from pianonn.losses import OnsetLoss, PianoLoss  # noqa: E402
from pianonn.metrics import clustered_mean  # noqa: E402
from pianonn.partial_view import POOL_MIN, PartialView, example_weights, pooled_across  # noqa: E402
from pianonn.render import load_model  # noqa: E402
from pianonn.train import fixed_batches  # noqa: E402

CHANGES = ("upper partials fade x2", "partials 1-8 fade x1.3", "partials 5-8 +3 dB", "C5 up +3 dB", "aftersound +6 dB",
           "noise +6 dB", "knock +5 dB")
OLD_TOTAL = {"band": 1.0, "fine": 0.25, "attack": 0.5, "level": 0.5, "onset": 0.5}
SWEEPS = {  # name: (unit, settings); evenly spaced: parabolic() needs it
    "knock": ("dB", (-8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0)),
    "decay 1-8": ("log2 rate", (-0.5, -0.25, 0.0, 0.25, 0.5)),
    "decay 9+": ("log2 rate", (-0.5, -0.25, 0.0, 0.25, 0.5)),
    "aftersound": ("dB", (-6.0, -3.0, 0.0, 3.0, 6.0)),
}


class Change:
    """Patches the model's instance for one known change (``name``) or one point ``x`` of a sweep; undone on exit."""

    def __init__(self, model, name=None, sweep=None, x=0.0):
        self.model, self.name, self.sweep, self.x = model, name, sweep, x
        self.knock = x if sweep == "knock" else (5.0 if name == "knock +5 dB" else 0.0)

    def __enter__(self):
        m, name, knock, sweep, x = self.model, self.name, self.knock, self.sweep, self.x
        modes0, noise0, attack0 = m.physics.modes, m.render_noise, m.noise.attack_power

        def modes(*a, **k):
            out = modes0(*a, **k)
            if not k.get("phantoms", True):  # the sympathetic bank's strings: unchanged
                return out
            out = dict(out)
            if name == "upper partials fade x2":
                al = out["alpha"].clone()
                al[..., 8:, :] = al[..., 8:, :] * 2
                out["alpha"] = al
            elif name == "partials 1-8 fade x1.3":
                al = out["alpha"].clone()
                al[..., :8, :] = al[..., :8, :] * 1.3
                out["alpha"] = al
            elif name == "C5 up +3 dB":
                amp = out["amp"].clone()
                amp[a[0] >= 72 - 21] = amp[a[0] >= 72 - 21] * 10 ** (3 / 20)  # the first argument: key index
                out["amp"] = amp
            elif name == "partials 5-8 +3 dB":
                amp = out["amp"].clone()
                amp[..., 4:8, :] = amp[..., 4:8, :] * 10 ** (3 / 20)
                out["amp"] = amp
            elif name == "aftersound +6 dB":
                amp = out["amp"].clone()
                amp[..., 1:] = amp[..., 1:] * 2
                out["amp"] = amp
            if sweep in ("decay 1-8", "decay 9+") and x:
                al = out["alpha"].clone()
                sl = slice(0, 8) if sweep == "decay 1-8" else slice(8, None)
                al[..., sl, :] = al[..., sl, :] * 2 ** x
                out["alpha"] = al
            if sweep == "aftersound" and x:
                amp = out["amp"].clone()
                amp[..., 1:] = amp[..., 1:] * 10 ** (x / 20)
                out["amp"] = amp
            if knock:
                out["impulse"] = out["impulse"] * 10 ** (knock / 20)
            return out

        def render_noise(*a, **k):
            n, r, w, wr = noise0(*a, **k)
            return (n * 2 if name == "noise +6 dB" else n), r, w, wr

        def attack_power(*a, **k):
            return attack0(*a, **k) * 10 ** (knock / 10)

        m.physics.modes, m.render_noise, m.noise.attack_power = modes, render_noise, attack_power
        self.saved = (modes0, noise0, attack0)

    def __exit__(self, *exc):
        for obj, attr in ((self.model.physics, "modes"), (self.model, "render_noise"), (self.model.noise, "attack_power")):
            obj.__dict__.pop(attr, None)


def quad_fit(xs, v):
    """Least-squares parabola through the sweep: ``(vertex, slope at 0)``; the vertex is nan where it opens downwards."""
    c2, c1, _ = np.polyfit(np.asarray(xs), np.asarray(v), 2)
    return (-c1 / (2 * c2) if c2 > 0 else float("nan")), c1


def sweep_cells(xs, get, groups, unit_scale):
    """A sweep's summary: the fitted best setting +- its jackknife error, the slope at 0 as a multiple of its error (the
    pull a training step would feel at the right setting), and the rise at the ends as a share of ``unit_scale`` (its
    multiple of the error). ``get(x, index)``: the term at setting ``x`` over the examples ``index``."""
    every = np.arange(len(groups))
    best = lambda i: quad_fit(xs, [get(x, i) for x in xs])
    b0, s0 = best(every)
    ids = np.unique(groups)
    th = np.array([best(np.nonzero(groups != g)[0]) for g in ids])
    G = len(ids)
    se = np.sqrt((G - 1) / G * np.nansum((th - np.nanmean(th, 0)) ** 2, 0))
    rise, rse = jackknife(lambda i: 0.5 * (get(xs[0], i) + get(xs[-1], i)) - get(0.0, i), groups)
    return (f"{b0:+.2f} ± {se[0]:.2f} | {s0 / se[1]:+.1f} | {rise / unit_scale:+.3f} ({rise / rse:+.1f})"
            if rse > 0 else f"{b0:+.2f} | | ")


def jackknife(stat, groups):
    """``stat(index)`` over all examples and its standard error by leaving one group out at a time."""
    full = stat(np.arange(len(groups)))
    ids = np.unique(groups)
    th = np.array([stat(np.nonzero(groups != g)[0]) for g in ids])
    G = len(ids)
    return full, float(np.sqrt((G - 1) / G * ((th - th.mean()) ** 2).sum()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data")
    ap.add_argument("--model", default="runs/phase6/train_run/train/last.pt")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--examples", type=int, default=48)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--sweeps", nargs="*", default=list(SWEEPS), help=f"of {list(SWEEPS)}; none: no sweeps")
    ap.add_argument("--no-sweep", action="store_true")
    ap.add_argument("--exposure-pow", type=float, default=2.0, help="PartialView's exposure_pow (pooled_exposed)")
    ap.add_argument("--energy", action="store_true", help="energy scores (implies --ref-strike); see above")
    ap.add_argument("--take-seed", type=int, default=1, help="the take's seed (the reference's is 0, its second draw's 2)")
    ap.add_argument("--ref-strike", action="store_true",
                    help="the reference with the per-strike variation on too (seed 0; the take seed 1): the terms as a "
                         "training step that renders with the variation would see them")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    args.ref_strike = args.ref_strike or args.energy
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device(args.device)
    torch.cuda.set_per_process_memory_fraction(0.6)
    model = load_model(args.model, device=dev)
    has_strike = model.strike_on
    sr = model.cfg.sample_rate
    ds = MaestroSegments(args.data, args.split, model.cfg, 2.0, 1.0, 12.0, length=args.examples, deterministic=True,
                         years=[2018], seed=1)
    batches = fixed_batches(ds, args.examples, args.batch, dev)
    band = PianoLoss(sr, level_weight=0.5).to(dev)
    onset = OnsetLoss(sr).to(dev).eval()
    view = PartialView(sr, exposure_pow=args.exposure_pow).to(dev)
    print(f"{args.model}: {args.examples} {args.split} excerpts; per-strike variation {'on' if has_strike else 'OFF'} in the take")

    @torch.no_grad()
    def render(b, seed, strike):
        model.strike_on = strike and has_strike
        out = model(b, b["audio"].shape[-1], residual=False, generator=torch.Generator(device=dev).manual_seed(seed))
        model.strike_on = False
        return out

    @torch.no_grad()
    def terms(full, ref_full, b, notes):
        """Per-excerpt terms ``{name: [B]}`` and the pooled terms' cells ``{name: (P, T, count, eps)}``."""
        s = int(b["loss_start"][0])
        p, t = full[..., s:].float(), ref_full[..., s:].float()
        d = band.terms(p, t, b["onset"] - s / sr, b["mask"])
        v = view(p, t, notes)
        sums = dict(v.pop("sums"), level=level_sums(band, p, t))
        d.update(v)
        d["onset"] = onset(full.float(), ref_full.float(), b, s / sr).expand(p.shape[0])  # pooled per batch
        d["old total"] = sum(w * d[k] for k, w in OLD_TOTAL.items())
        sums = {k: tuple(x.cpu().numpy() for x in sums[k]) for k in POOLED}
        return {k: x.cpu().numpy() for k, x in d.items()}, sums

    sweeps = [] if args.no_sweep else args.sweeps
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    acc_sums = collections.defaultdict(lambda: collections.defaultdict(list))

    def add(name, res):
        for k, v in res[0].items():
            acc[name][k].append(v)
        for k, v in res[1].items():
            acc_sums[name][k].append(v)

    pieces = []
    for bi, b in enumerate(batches):
        s = int(b["loss_start"][0])
        base = render(b, 0, args.ref_strike)
        notes = {k: v for k, v in base["partials"].items()}
        notes["onset"], notes["t_ref"] = notes["onset"] - s / sr, -s / sr
        take = render(b, args.take_seed, True)["audio"]
        ref, rec = base["audio"], b["audio"]
        add("floor", terms(ref, take, b, notes))
        add("vs recording", terms(ref, rec, b, notes))
        if args.energy:
            add("floor self", terms(ref, render(b, 2, True)["audio"], b, notes))
        for c in CHANGES:
            with Change(model, c):
                y = render(b, 0, args.ref_strike)["audio"]
                y2 = render(b, 2, True)["audio"] if args.energy else None
            add(c, terms(y, take, b, notes))
            if args.energy:
                add(c + " self", terms(y, y2, b, notes))
        for sw in sweeps:
            for x in SWEEPS[sw][1]:
                with Change(model, sweep=sw, x=x):
                    y = render(b, 0, args.ref_strike)["audio"]
                    y2 = render(b, 2, True)["audio"] if args.energy else None
                add(f"{sw} {x:+g}", terms(y, take, b, notes))
                if args.energy:
                    add(f"{sw} {x:+g} self", terms(y, y2, b, notes))
        pieces.append(b["piece"].cpu().numpy())
        print(f"batch {bi + 1}/{len(batches)}", flush=True)
    pieces = np.concatenate(pieces)
    R = R0 = {name: {k: np.concatenate(v) for k, v in d.items()} for name, d in acc.items()}
    S = {name: {k: [np.concatenate(x) if x[0].ndim > 1 else x[0] for x in zip(*v)] for k, v in d.items()}
         for name, d in acc_sums.items()}
    np.savez(os.path.join(args.out, "terms.npz"), pieces=pieces,
             **{f"{name}|{k}": v for name, d in R.items() for k, v in d.items()})
    self_of = lambda name: "floor self" if name == "vs recording" else name + " self"
    scale = {k: R["floor"][k].mean() for k in R["floor"]}  # the plain floor: the unit of the changes
    if args.energy:  # per excerpt: d(y, take) - d(y, y') / 2
        R = {name: {k: v - 0.5 * R[self_of(name)][k] for k, v in d.items()} for name, d in R.items()
             if not name.endswith(" self")}

    # ---- per excerpt
    keys = ("band", "fine", "attack", "level", "onset", "partials", "pooled", "pooled_exposed", "between",
            "between_pooled", "old total")
    lines = [f"# Score check: {args.examples} {args.split} excerpts, `{args.model}`", "",
             f"Reference: the model, variation {'on' if args.ref_strike else 'off'}, noise seed 0. Take: the model with its "
             f"per-strike variation, seed {args.take_seed} (a piano that is the model)."
             + (" Energy scores: each term less half its distance to a second draw of the reference (seed 2); the floor "
                "column is the plain distance." if args.energy else "")
             + " Units: log10 power (1 = 10 dB).", "",
             "## Per excerpt", "",
             "Mean per excerpt. Changes: the change minus the reference, both against the take, as a share of the floor; "
             "in brackets the multiple of its standard error (excerpts grouped by piece; the onset term is pooled per "
             "batch, so its error is rough).",
             "", "| term | floor | gap (recording) | gap / floor | " + " | ".join(CHANGES) + " |",
             "|---|---|---|---|" + "---|" * len(CHANGES)]
    for k in keys:
        fl, sc = R["floor"][k], scale[k]
        g, gse, _ = clustered_mean(R["vs recording"][k] - fl, pieces)
        cells = []
        for c in CHANGES:
            m, se, _ = clustered_mean(R[c][k] - fl, pieces)
            cells.append(f"{m / sc:+.3f} ({m / se:+.1f})" if se > 0 else f"{m / sc:+.3f}")
        lines.append(f"| {k} | {sc:.4f} | {g:+.4f} ± {2 * gse:.4f} | {g / sc:.2f} | " + " | ".join(cells) + " |")
    for sw in sweeps:
        unit, xs = SWEEPS[sw]
        lines += ["", f"### Sweep: {sw} ({unit})", "",
                  "Each term against the take with the reference's setting moved: the mean; the best setting (the "
                  "vertex of a least-squares parabola, 0 is right) +- its standard error; the slope at 0 as a multiple of "
                  "its error; the rise at the ends (their mean minus the middle's) as a share of the floor, in brackets "
                  "its multiple of the error (errors by leaving one piece out at a time).", "",
                  "| term | " + " | ".join(f"{x:+g}" for x in xs) + " | best | slope at 0 (z) | rise at the ends |",
                  "|---|" + "---|" * (len(xs) + 3)]
        for k in keys:
            v = np.array([R[f"{sw} {x:+g}"][k].mean() for x in xs])
            get = lambda x, i, k=k, sw=sw: float(R[f"{sw} {x:+g}"][k][i].mean())
            lines.append(f"| {k} | " + " | ".join(f"{x:.4f}" for x in v) + " | "
                         + sweep_cells(xs, get, pieces, scale[k]) + " |")

    # ---- the composite: pooled terms across the excerpts
    def stat(name, k, idx, plain=False):
        if k == "composite":
            return sum(w * stat(name, kk, idx, plain) for kk, w in WEIGHTS.items())
        if args.energy and not plain:
            return stat(name, k, idx, True) - 0.5 * stat(self_of(name), k, idx, True)
        if k in POOLED:
            P, T, cnt, eps = S[name][k]
            P, T, cnt = (torch.from_numpy(x[idx]).double() for x in (P, T, cnt))
            eps = torch.from_numpy(eps).double()
            eps = eps.amin(0) if eps.dim() == 2 else eps
            P0, _, c0, _ = S["floor"][k]  # every setting weighted as the unchanged reference (training's weights are
            w = example_weights(torch.from_numpy(P0[idx]).double(), torch.from_numpy(c0[idx]).double())  # detached)
            return float(pooled_across(P, T, cnt, eps, POOL_MIN.get(k, 1), w))
        return float((R0[name][k] if args.energy else R[name][k])[idx].mean())

    ckeys = tuple(WEIGHTS) + ("composite",)
    every = np.arange(len(pieces))
    lines += ["", "## The composite", "",
              "Pooled terms pooled across all the excerpts (each weighted by 1 / the reference's mean power: the energy "
              "match over the set), read-by-read terms and the onset term as means, and their weighted sum "
              f"(weights {WEIGHTS}). Changes as above; in brackets the multiple of the standard error by leaving one "
              "piece out at a time. The gap of a pooled term includes each piece's recording level.", "",
              "| term | floor | gap (recording) | gap / floor | " + " | ".join(CHANGES) + " |",
              "|---|---|---|---|" + "---|" * len(CHANGES)]
    for k in ckeys:
        fl = stat("floor", k, every, True)  # the plain distance: the unit
        g, gse = jackknife(lambda i: stat("vs recording", k, i) - stat("floor", k, i), pieces)
        cells = []
        for c in CHANGES:
            m, se = jackknife(lambda i: stat(c, k, i) - stat("floor", k, i), pieces)
            cells.append(f"{m / fl:+.3f} ({m / se:+.1f})" if se > 0 else f"{m / fl:+.3f}")
        lines.append(f"| {k} | {fl:.4f} | {g:+.4f} ± {2 * gse:.4f} | {g / fl:.2f} | " + " | ".join(cells) + " |")
    if sweeps:
        lines += ["", "### Sweeps", "", "Per sweep and term: the best setting +- its error (0 is right), the slope at "
                  "0 as a multiple of its error, and the rise at the ends as a share of the floor (its multiple of the "
                  "error); errors by leaving one piece out at a time.", "",
                  "| term | " + " | ".join(f"{sw}: best | slope z | rise" for sw in sweeps) + " |",
                  "|---|" + "---|---|---|" * len(sweeps)]
        for k in ckeys:
            cells = []
            for sw in sweeps:
                get = lambda x, i, k=k, sw=sw: stat(f"{sw} {x:+g}", k, i)
                cells.append(sweep_cells(SWEEPS[sw][1], get, pieces, stat("floor", k, every, True)))
            lines.append(f"| {k} | " + " | ".join(cells) + " |")
    text = "\n".join(lines) + "\n"
    with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
