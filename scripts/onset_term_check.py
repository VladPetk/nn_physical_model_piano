"""Checks of the onset-aligned attack term, ``pianonn.losses.OnsetLoss`` (phase 3, docs/tone_measures.md 12.6).

    python scripts/onset_term_check.py data/maestro24k --model k25=runs/phase3/step2/knock_2k5/model.pt:physics \\
        --out runs/phase3/onset_term/pooled

The training loss sees the attack only through 10.7 ms windows with no band below ~400 Hz, and the bass only through
windows of 341 ms (band term) or, bin by bin and not tied to onsets, 43-171 ms (fine term). ``OnsetLoss`` places a
window from 3 ms before to 33 ms after each note's expected sound onset (MIDI + the recordings' delay law), counts a
band where either side stands 6 dB over its own background, and sums the counted windows' power over the batch before
the log. This script renders validation segments (2 s after 1 s of warm-up, as in training) with the model, stores
every window's band powers (``windows.npz``) and computes from them, for batches of k segments:
1. tolerance: each term between the model and itself with another noise draw, and shifted by 1-10 ms (the per-note
   scatter of N0 around the delay law is about +-4 ms, IQR);
2. scans (the first ``--scan-segments``): the knock (all bands; below 400 Hz only), the knock impulse and the contact
   time moved over a range, each term against the recordings, the noise draw fixed: where each term's minimum lies and
   how sharp it is, and where the training loss's minimum moves with the term added at a weight;
3. the bias gate: per band, the optimum of a level offset (the median over batches of the pooled log ratio) against
   the energy match over all segments, for k = 1, 2, 4, 8, 16 and for the running pool across steps (simulated on
   random orders of the batches); and the spread of the attack levels, model and recordings.
Each for both forms of the term: ``abs`` (the attack's level) and ``rel`` (the attack re the same onset's early window,
30-100 ms, 12.8). ``--from`` recomputes the report from a saved ``windows.npz``.
"""

import argparse
import math
import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.6,max_split_size_mb:256")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from pianonn import measures as M  # noqa: E402

SHIFTS_MS = (1, 2, 5, 10)
POOLS = (1, 2, 4, 8, 16)
RUNNING = (0.875, 0.9375, 0.96875)  # pool decays per batch of 2: about 8, 16 and 32 batches
FORMS = ("abs", "rel")
SCAN_K = 2  # segments per batch in the scans: the step-4 run renders 2 (the sympathetic bank's memory)
WEIGHTS = (0.5, 1.0)
CURRENT = {"band": 1.0, "fine": 0.25, "attack": 0.5}
SCANS = {"knock (dB)": (-6, -3, 3, 6), "knock below 400 Hz (dB)": (-6, -3, 3, 6), "knock impulse (dB)": (-6, -3, 3, 6),
         "contact time (%)": (-20, -10, 10, 20)}


def render_all(args, log):
    from pianonn.data import MaestroSegments
    from pianonn.losses import OnsetLoss, PianoLoss
    from pianonn.render import load_variant
    from pianonn.train import collate, to_device

    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.6)
    _, model, _ = load_variant(args.model, device=dev, log=print)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    sr = model.cfg.sample_rate
    ds = MaestroSegments(args.data, "validation", model.cfg, segment=2.0, warmup=1.0, length=args.segments,
                         deterministic=True, years=args.years)
    recon = PianoLoss(sr).to(dev)
    ot = OnsetLoss(sr, relative=True).to(dev)  # the early window's powers too

    def render(b, seed=0):
        with torch.no_grad():
            return model(b, b["audio"].shape[-1], residual=False,
                         generator=torch.Generator(device=dev).manual_seed(seed))["audio"].float()

    low = (model.noise.centers < 400).float()
    db = math.log(10) / 20
    moves = {"knock (dB)": lambda x: ("knock", x * db), "knock below 400 Hz (dB)": lambda x: ("knock", x * db * low),
             "knock impulse (dB)": lambda x: ("impulse", float(x)), "contact time (%)": lambda x: ("contact", math.log(1 + x / 100))}

    def apply(kind, v):
        if kind == "knock":
            model.noise.knock.data += v
        elif kind == "impulse":
            model.physics.prior_impulse_db += v
        else:
            model.physics.prior_log_tc += v

    store = {}

    def put(key, v):
        store.setdefault(key, []).append(np.asarray(v))

    for j in range(0, args.segments, args.batch):
        b = to_device(collate([ds[i] for i in range(j, min(j + args.batch, args.segments))]), dev)
        s, n = int(b["loss_start"][0]), b["audio"].shape[-1]
        on_loss, mask = b["onset"] - s / sr, b["mask"]
        rows, times, vels = ot.anchors(b, s / sr, n / sr - ot.window[1])
        rec, ref = b["audio"].float(), render(b)
        outs = {"ref": ref, "seed 1": render(b, 1)}
        for d_ms in SHIFTS_MS:
            outs[f"shift {d_ms} ms"] = torch.nn.functional.pad(ref, (int(round(d_ms * 1e-3 * sr)), 0))[..., :n]
        if j < args.scan_segments:
            for name, offs in SCANS.items():
                for o in offs:
                    kind, v = moves[name](o)
                    apply(kind, v)
                    outs[f"{name} {o:+d}"] = render(b)
                    apply(kind, -v)
        put("rows", rows + j)
        put("vel", vels)
        for k, x in [("rec", rec)] + list(outs.items()):
            if len(rows):
                pw = ot.powers(x, rows, times)
                put(f"att|{k}", pw["attack"].cpu().numpy())
                put(f"bg|{k}", pw["bg"].cpu().numpy())
                put(f"early|{k}", pw["early"].cpu().numpy())
            if k == "rec":
                continue
            for tag, other in (("vs rec", rec), ("vs ref", ref)):
                if tag == "vs ref" and (k == "ref" or not (k == "seed 1" or k.startswith("shift"))):
                    continue
                for name, v in recon.terms(x[..., s:], other[..., s:], on_loss, mask).items():
                    put(f"{name}|{k}|{tag}", 10 * v.cpu().numpy())
        print(f"segments {j + b['audio'].shape[0]}/{args.segments}: {len(rows)} onset groups", flush=True)
    out = {k: np.concatenate(v) for k, v in store.items()}
    out["centers"] = ot.centers.cpu().numpy()
    out["over"] = np.array(ot.over)
    out["scan_segments"] = np.array(min(args.scan_segments, args.segments))
    np.savez_compressed(os.path.join(args.out, "windows.npz"), **out)
    return out


def analyse(z, log, model_label):
    rows, vel, fc, over = z["rows"], z["vel"], z["centers"], float(z["over"])
    n_seg = int(max(len(z["band|ref|vs rec"]), rows.max() + 1))
    n_scan = int(z["scan_segments"])

    def cells(a, b, form="abs"):
        """Either side over its background (``rel``: its early window too); a scan's windows cover only the first
        segments, so both are cut to the shorter (the windows are stored in segment order)."""
        n = min(len(z[f"att|{a}"]), len(z[f"att|{b}"]))

        def stands(k):
            c = z[f"att|{k}"][:n] >= over * z[f"bg|{k}"][:n]
            return c & (z[f"early|{k}"][:n] >= over * z[f"bg|{k}"][:n]) if form == "rel" else c
        return stands(a) | stands(b)

    def level(key, c, w, form):
        """10 log10 of the counted attack power summed over the windows ``w`` (``rel``: re the early window's)."""
        att = (z[f"att|{key}"][:len(c)][w] * c[w]).sum(0)
        if form == "abs":
            return 10 * np.log10(np.maximum(att, 1e-30))
        return 10 * np.log10(np.maximum(att, 1e-30) / np.maximum((z[f"early|{key}"][:len(c)][w] * c[w]).sum(0), 1e-30))

    def pooled(a, b, k, segs, c=None, form="abs"):
        """Mean over batches of k consecutive segments (among ``segs``) of the pooled loss (dB), and per band the
        batches' log ratios ``[batches, bands]`` (NaN where no window counts)."""
        c = cells(a, b, form) if c is None else c
        vals, ratios = [], []
        for g0 in range(0, segs, k):
            w = (rows[:len(c)] >= g0) & (rows[:len(c)] < min(g0 + k, segs))
            has = c[w].any(0)
            r = np.where(has, level(a, c, w, form) - level(b, c, w, form), np.nan)
            ratios.append(r)
            if has.any():
                vals.append(np.nanmean(np.abs(r)))
        return (float(np.mean(vals)) if vals else float("nan")), np.array(ratios)

    def running(a, b, decay, segs, c, form, epochs=12, seed=0):
        """The running pool's log ratios ``[steps, bands]`` over random orders of the batches of 2 segments (the
        first epoch dropped as the pool's warm-up)."""
        sums = []
        for g0 in range(0, segs, 2):
            w = (rows >= g0) & (rows < g0 + 2)
            e = [(z[f"att|{k}"][w] * c[w]).sum(0) for k in (a, b)]
            e += [(z[f"early|{k}"][w] * c[w]).sum(0) if form == "rel" else np.ones(c.shape[1]) for k in (a, b)]
            sums.append(np.stack(e))
        sums = np.array(sums)
        rng = np.random.default_rng(seed)
        run, out = np.zeros_like(sums[0]), []
        for ep in range(epochs):
            for i in rng.permutation(len(sums)):
                run = decay * run + sums[i]
                if ep:
                    lg = lambda x: 10 * np.log10(np.maximum(x, 1e-30))
                    out.append(np.where(run[0] > 0, lg(run[0]) - lg(run[2]) - lg(run[1]) + lg(run[3]), np.nan))
        return np.array(out)

    def per_window(a, b, segs, form="abs"):
        c = cells(a, b, form) & (rows < segs)[:, None]
        d = 10 * np.log10(z[f"att|{a}"][c] / z[f"att|{b}"][c])
        if form == "rel":
            d = d - 10 * np.log10(z[f"early|{a}"][c] / z[f"early|{b}"][c])
        return float(np.abs(d).mean())

    def cur(name, key, tag, segs):
        return float(z[f"{name}|{key}|{tag}"][:segs].mean())

    log(f"# Onset-aligned attack term (`pianonn.losses.OnsetLoss`): checks\n\nModel `{model_label}`; {n_seg} validation "
        f"segments of 2018 (2 s after 1 s of warm-up), {len(rows)} onset groups (scans on the first {n_scan}). "
        "Band, fine and attack: the training loss's terms in dB (10 x its log10 units); pooled k: the onset term over "
        "batches of k segments; per window: the same windows and cells, L1 per window (not pooled).\n")

    log("## 1. Tolerance: each term between the model and itself (dB)\n")
    log("| model vs | band | fine | attack | per window | pooled 1 | pooled 2 | pooled 4 | rel: per window | rel: pooled 2 |")
    log("|---|---|---|---|---|---|---|---|---|---|")
    for k in ["seed 1"] + [f"shift {d} ms" for d in SHIFTS_MS]:
        log(f"| {k} | " + " | ".join(f"{cur(t, k, 'vs ref', n_seg):.3f}" for t in CURRENT) + f" | {per_window(k, 'ref', n_seg):.3f} | "
            + " | ".join(f"{pooled(k, 'ref', kk, n_seg)[0]:.3f}" for kk in (1, 2, 4))
            + f" | {per_window(k, 'ref', n_seg, 'rel'):.3f} | {pooled(k, 'ref', 2, n_seg, form='rel')[0]:.3f} |")

    log(f"\n## 2. Scans: each term against the recordings as one parameter moves (dB re the reference; {n_scan} segments)\n")
    log(f"The noise draw is fixed; the onset term's cells stay the reference's (training's gradient holds each step's "
        f"cells fixed, so the minimum with them held is where training settles). `onset`: pooled over "
        f"batches of {SCAN_K} segments; `onset, pool`: pooled over every scanned segment (what a running pool approaches); "
        "`rel`: the attack re the early window, the same two ways; `current`: band + 0.25 fine + 0.5 attack; "
        "`+ w rel, pool`: the current loss with the relative, pooled term at weight w. Vertex: the minimum of a parabola through the five points, and its 10-90 % range over "
        "200 bootstrap draws of the batches (`-`: the parabola opens downwards or its vertex lies beyond twice the "
        "range). Curvature: the parabola's rise one step (3 dB or 10 %) from its vertex, in dB.\n")
    cols = ("band", "fine", "attack", "onset", "onset, pool", "rel", "rel, pool", "current") + tuple(f"+ {w:g} rel, pool" for w in WEIGHTS)
    groups = [np.arange(g0, min(g0 + SCAN_K, n_scan)) for g0 in range(0, n_scan, SCAN_K)]
    rng = np.random.default_rng(0)
    boots = [rng.integers(0, len(groups), len(groups)) for _ in range(200)]

    def point(key, gsel):
        v = {t: float(np.mean([z[f"{t}|{key}|vs rec"][g].mean() for g in (groups[i] for i in gsel)])) for t in CURRENT}
        for form, col in (("abs", "onset"), ("rel", "rel")):
            c = c_ref[form]  # the reference's cells: training's gradient holds each step's cells fixed
            ons = []
            for i in gsel:
                w = np.isin(rows[:len(c)], groups[i])
                has = c[w].any(0)
                if has.any():
                    ons.append(np.mean(np.abs(level(key, c, w, form) - level("rec", c, w, form))[has]))
            v[col] = float(np.mean(ons))
            w = np.isin(rows[:len(c)], np.concatenate([groups[i] for i in gsel]))
            has = c[w].any(0)
            v[f"{col}, pool"] = float(np.mean(np.abs(level(key, c, w, form) - level("rec", c, w, form))[has]))
        v["current"] = sum(CURRENT[t] * v[t] for t in CURRENT)
        for w_ in WEIGHTS:
            v[f"+ {w_:g} rel, pool"] = v["current"] + w_ * v["rel, pool"]
        return v

    def vertex(xs, y):
        a, b_, _ = np.polyfit(xs, y, 2)
        if a <= 0:
            return float("nan"), a
        v = -b_ / (2 * a)
        return (v if abs(v) <= 2 * np.max(np.abs(xs)) else float("nan")), a

    allg = np.arange(len(groups))
    c_ref = {f: cells("ref", "rec", f)[: int(np.sum(rows < n_scan))] for f in FORMS}
    summary = {}
    for name, offs in SCANS.items():
        xs = np.array(sorted(offs + (0,)), float)
        keys = [f"{name} {int(x):+d}" if x else "ref" for x in xs]
        pts = [point(k, allg) for k in keys]
        i0 = list(xs).index(0)
        log(f"\n### {name}\n")
        log("| offset | " + " | ".join(cols) + " |")
        log("|---|" + "---|" * len(cols))
        for i, x in enumerate(xs):
            log(f"| {x:+g} | " + " | ".join(f"{pts[i][c] - pts[i0][c]:+.3f}" for c in cols) + " |")
        bpts = [[point(k, idx) for k in keys] for idx in boots]
        vc, cc = [], []
        for c in cols:
            v0, a = vertex(xs, np.array([p[c] for p in pts]))
            vb = np.array([vertex(xs, np.array([p[c] for p in bp]))[0] for bp in bpts])
            ok = np.isfinite(vb)
            rs = (f" [{np.percentile(vb[ok], 10):+.1f}, {np.percentile(vb[ok], 90):+.1f}]" if ok.mean() >= 0.8
                  else f" ({ok.mean():.0%} of draws)")
            vc.append(("-" if not np.isfinite(v0) else f"{v0:+.1f}") + rs)
            cc.append(f"{a * xs[i0 + 1] ** 2:.3f}")
            summary[(name, c)] = (v0, a * xs[i0 + 1] ** 2)
        log("| vertex | " + " | ".join(vc) + " |")
        log("| curvature | " + " | ".join(cc) + " |")

    log("\n## 3. Bias gate\n")
    log("Per band: the optimum of a level offset for the pooled term over batches of k segments (minus the median over "
        "the batches of their pooled log ratio, model re recording) minus the energy match over all segments (minus the "
        "log ratio of the summed powers); + = the term would leave the model louder than the energy match. Running: the "
        "pool across steps (batches of 2 in random orders, 12 epochs, the first dropped), decaying by the factor given "
        "per step; its optimum is the median over steps of the running log ratio. `rel`: the attack re the early window "
        "(the offset moves the attack only). Cells: either side over its background. Pass: every band within +-1 dB, the "
        "median over bands within +-0.5 dB. Last column: 10-90 % bootstrap range of the band's bias over the batches, "
        "k = 2.\n")
    lg = lambda x: 10 * np.log10(np.maximum(x, 1e-30))
    for form in FORMS:
        c = cells("ref", "rec", form)
        allw = np.ones(len(c), bool)
        energy = np.where(c.any(0), level("ref", c, allw, form) - level("rec", c, allw, form), np.nan)
        bias = {}
        for k in POOLS:
            _, r = pooled("ref", "rec", k, n_seg, c, form)
            bias[k] = energy - np.nanmedian(r, 0)
        for dcy in RUNNING:
            bias[f"run {dcy:g}"] = energy - np.nanmedian(running("ref", "rec", dcy, n_seg, c, form), 0)
        _, r2 = pooled("ref", "rec", 2, n_seg, c, form)
        rng = np.random.default_rng(1)
        bb = []
        for _ in range(200):
            idx = rng.integers(0, len(r2), len(r2))
            # the energy match over the same draw of batches (windows repeated as drawn)
            wi = np.concatenate([np.flatnonzero(rows == s_) for i in idx for s_ in (2 * i, 2 * i + 1)])
            tot = lambda key, k_: (z[f"{key}|{k_}"][wi] * c[wi]).sum(0)
            e = lg(tot("att", "ref")) - lg(tot("att", "rec"))
            if form == "rel":
                e = e - lg(tot("early", "ref")) + lg(tot("early", "rec"))
            bb.append(e - np.nanmedian(r2[idx], 0))
        bb = np.array(bb)
        per_win = []
        for j in range(len(fc)):
            d = 10 * np.log10(z["att|ref"][c[:, j], j] / z["att|rec"][c[:, j], j])
            if form == "rel":
                d = d - 10 * np.log10(z["early|ref"][c[:, j], j] / z["early|rec"][c[:, j], j])
            per_win.append(energy[j] - np.median(d) if len(d) >= 20 else np.nan)
        cols_ = [f"k = {k}" for k in POOLS] + [f"running {d:g}" for d in RUNNING]
        keys_ = list(POOLS) + [f"run {d:g}" for d in RUNNING]
        log(f"\n### {form}\n")
        log("| band (Hz) | windows counted | per window | " + " | ".join(cols_) + " | k = 2, 10-90 % |")
        log("|---|---|---|" + "---|" * len(cols_) + "---|")
        for j, f in enumerate(fc):
            log(f"| {f:.0f} | {c[:, j].mean():.2f} | {per_win[j]:+.2f} | " + " | ".join(f"{bias[k][j]:+.2f}" for k in keys_)
                + f" | {np.nanpercentile(bb[:, j], 10):+.2f} .. {np.nanpercentile(bb[:, j], 90):+.2f} |")
        log("")
        for k, name in zip(keys_, cols_):
            v = bias[k][np.isfinite(bias[k])]
            ok = np.max(np.abs(v)) <= 1.0 and abs(np.median(v)) <= 0.5
            log(f"{form}, {name}: median over bands {np.median(v):+.2f} dB, largest |band| {np.max(np.abs(v)):.2f} dB: {'pass' if ok else 'fail'}")
        summary[("gate", form)] = {name: bias[k] for k, name in zip(keys_, cols_)}
    c = cells("ref", "rec")

    log("\n**Spread of the attack levels** (sd of 10 log10 power over the counted windows, dB; within velocity strata of "
        "the group's loudest note, median over the strata). Where the recordings' spread exceeds the model's, a median "
        "and an energy mean part ways by about 0.115 (sd_rec^2 - sd_mod^2) dB.\n")
    log("| band (Hz) | recordings | model | 0.115 (sd_rec^2 - sd_mod^2) | rel: recordings | rel: model | rel: 0.115 (...) |")
    log("|---|---|---|---|---|---|---|")
    c_rel = cells("ref", "rec", "rel")
    for j, f in enumerate(fc):
        sd = {"abs": ([], []), "rel": ([], [])}
        for _, lo, hi in M.VELOCITY:
            for form, cc in (("abs", c), ("rel", c_rel)):
                m = cc[:, j] & (vel >= lo) & (vel < hi)
                if m.sum() >= 20:
                    for i, key in enumerate(("rec", "ref")):
                        d = 10 * np.log10(z[f"att|{key}"][m, j])
                        if form == "rel":
                            d = d - 10 * np.log10(z[f"early|{key}"][m, j])
                        sd[form][i].append(np.std(d))
        cells_ = []
        for form in FORMS:
            if sd[form][0]:
                a, b_ = np.median(sd[form][0]), np.median(sd[form][1])
                cells_.append(f"{a:.1f} | {b_:.1f} | {0.115 * (a * a - b_ * b_):+.2f}")
            else:
                cells_.append(" | | ")
        log(f"| {f:.0f} | " + " | ".join(cells_) + " |")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", nargs="?")
    ap.add_argument("--model", help="label=checkpoint:physics[:options] (pianonn.render.load_variant)")
    ap.add_argument("--years", type=int, nargs="*", default=[2018])
    ap.add_argument("--segments", type=int, default=128)
    ap.add_argument("--scan-segments", type=int, default=64)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--from", dest="from_npz", help="recompute the report from this windows.npz")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(args.out, exist_ok=True)
    lines = []

    def log(msg=""):
        print(msg, flush=True)
        lines.append(msg)

    if args.from_npz:
        z = dict(np.load(args.from_npz))
    else:
        z = render_all(args, log)
    analyse(z, log, args.model or args.from_npz)
    with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
