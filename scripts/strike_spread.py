"""Per-strike variation (docs/tone_measures.md 12.7): how much the piano's notes vary at equal key and velocity
beyond what a model's do, and the random spread per dimension that closes the gap.

    python scripts/strike_spread.py runs/phase3/step3/calib_k25/rows.json --label k25 --out runs/phase3/step3/spread

Reads ``rows.json`` files written by ``scripts/note_bench.py`` (notes). Per register and measure, each source's
values are reduced to residuals: a linear fit on pitch and velocity, then each piece's median removed (at equal key
and velocity the pieces differ by ~2 dB in level, 12.5: a session's offset, not a strike's). The residuals are split
into a per-key part (voicing: the same every time the key is struck) and a per-strike part, from pairs of notes of
the same key: ``sd_strike = median |r_i - r_j| / (0.6745 sqrt 2)``, ``sd_key^2 = sd_total^2 - sd_strike^2``, with
robust (IQR) totals. Errors: 10-90 % over 200 bootstrap draws of whole pieces.

With ``--probe label=dim:sd`` (models rendered with one dimension of ``PianoConfig.strike_*`` switched on), the
sensitivity of each measure's per-strike spread to each dimension is read from the probes' added variance, and the
spreads that match the recordings' per-strike excess are solved for (non-negative least squares on variances).
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.optimize import nnls

REGS = ("R2", "R3", "R4", "R5", "R6")
MEASURES = ["N1 level", "N2 slope", "N2 centroid", "N5 attack 500", "N5 attack 1000", "N5 attack 2000", "N5 attack 4000",
            "N6 knock 1000", "N6 knock 2000", "N8 drop 1-4", "onset_ms"]
# the measure each dimension is matched on (the solve uses these rows)
TARGETS = {"N1 level": "level", "N2 slope": "fc", "N5 attack 1000": "knock", "N5 attack 2000": "knock",
           "N5 attack 4000": "knock", "N6 knock 2000": "knock", "N8 drop 1-4": "decay", "onset_ms": "onset"}
Z = 0.6745 * np.sqrt(2)


def robust_sd(z):
    return (np.percentile(z, 75) - np.percentile(z, 25)) / 1.349 if len(z) >= 4 else np.nan


def residuals(vals, pitch, vel, piece):
    X = np.stack([np.ones(len(vals)), pitch, vel], 1).astype(float)
    r = vals - X @ np.linalg.lstsq(X, vals, rcond=None)[0]
    for p in set(piece):
        m = piece == p
        if m.sum() >= 3:
            r[m] -= np.median(r[m])
    return r


def components(r, key):
    """(total, per-strike, per-key) robust sds of residuals ``r`` with key labels ``key``."""
    d = []
    for k in set(key):
        v = r[key == k]
        if len(v) >= 2:
            d.append(np.abs(v[:, None] - v[None, :])[np.triu_indices(len(v), 1)])
    tot = robust_sd(r)
    if not d:
        return tot, np.nan, np.nan
    s = np.median(np.concatenate(d)) / Z
    return tot, s, np.sqrt(max(tot**2 - s**2, 0.0))


def table(rows, sources, measure, reg):
    """{source: residuals}, keys, pieces for the notes of ``reg`` where every source has the measure."""
    sel = [r for r in rows if r["register"] == reg and all(
        measure in r[s] and np.isfinite(r[s][measure]) for s in sources)]
    if len(sel) < 12:
        return None
    pitch = np.array([r["pitch"] for r in sel], float)
    vel = np.array([r["velocity"] for r in sel], float)
    piece = np.array([r["piece"] for r in sel])
    out = {s: residuals(np.array([r[s][measure] for r in sel]), pitch, vel, piece) for s in sources}
    return out, pitch.astype(int), piece


def spread(rows, sources, measure, reg, n_boot=200, seed=0):
    """Per source: (total, strike, key) sds and their bootstrap 10-90 % ranges; n notes."""
    t = table(rows, sources, measure, reg)
    if t is None:
        return None
    res, key, piece = t
    est = {s: components(res[s], key) for s in sources}
    rng = np.random.default_rng(seed)
    pieces = np.unique(piece)
    boot = {s: [] for s in sources}
    for _ in range(n_boot):
        idx = np.concatenate([np.flatnonzero(piece == p) for p in rng.choice(pieces, len(pieces))])
        for s in sources:
            boot[s].append(components(res[s][idx], key[idx]))
    rng_ = {s: np.nanpercentile(np.array(boot[s]), [10, 90], axis=0) for s in sources}
    return {"n": len(key), "est": est, "range": rng_, "boot": {s: np.array(b) for s, b in boot.items()}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rows", nargs="+", help="rows.json files; their note rows are merged by (piece, pitch, velocity)")
    ap.add_argument("--label", required=True, help="the base model's label")
    ap.add_argument("--probe", action="append", default=[], help="label=dim:sd of a probe model")
    ap.add_argument("--regs", default=",".join(REGS))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(args.out, exist_ok=True)
    rows = None
    for path in args.rows:  # runs of one bench group list the same notes in the same order
        with open(path) as f:
            part = json.load(f)["notes"]
        if rows is None:
            rows = part
            continue
        assert len(part) == len(rows) and all(
            (a["piece"], a["pitch"], a["velocity"]) == (b["piece"], b["pitch"], b["velocity"]) for a, b in zip(rows, part)), path
        for a, b in zip(rows, part):
            a.update({k: v for k, v in b.items() if k not in a})
    regs = args.regs.split(",")
    probes = [(p.split("=")[0], p.split("=")[1].split(":")[0], float(p.split(":")[1])) for p in args.probe]
    sources = ["recording", args.label] + [p[0] for p in probes]
    rows = [r for r in rows if all(s in r for s in sources)]

    L = [f"# Per-strike spread: recordings vs `{args.label}`", "",
         f"{len(rows)} notes ({', '.join(args.rows)}). Residuals after a linear fit on pitch and velocity per register "
         "and each piece's median removed; robust sds (IQR / 1.349). **strike**: from pairs of notes of the same key; "
         "**key**: the rest (voicing). Excess = sqrt(rec² − model²), negative when the model varies more. Brackets: "
         "10–90 % over 200 bootstrap draws of pieces.", ""]
    result = {}
    for part, j in (("per strike", 1), ("per key", 2), ("total", 0)):
        L += [f"## {part}: recording / {args.label} / excess", "", "| measure | " + " | ".join(regs) + " |",
              "|---|" + "---|" * len(regs)]
        for m in MEASURES:
            cells = []
            for reg in regs:
                s = result.get((m, reg)) or spread(rows, sources, m, reg)
                result[(m, reg)] = s
                if s is None:
                    cells.append("")
                    continue
                a, b = s["est"]["recording"][j], s["est"][args.label][j]
                bx = s["boot"]["recording"][:, j] ** 2 - s["boot"][args.label][:, j] ** 2
                ex = np.sign(a**2 - b**2) * np.sqrt(abs(a**2 - b**2))
                lo, hi = np.nanpercentile(np.sign(bx) * np.sqrt(np.abs(bx)), [10, 90])
                cells.append(f"{a:.2f} / {b:.2f} / {ex:+.2f} [{lo:+.1f}, {hi:+.1f}] ({s['n']})")
            L.append(f"| {m} | " + " | ".join(cells) + " |")
        L.append("")

    if probes:
        L += ["## Probes: added per-strike variance per unit of each dimension's variance", "",
              "Sensitivity S = sqrt(probe² − base²) / probe sd, per-strike sds, pooled over registers (median).", "",
              "| measure | " + " | ".join(f"{lab} ({dim} {sd:g})" for lab, dim, sd in probes) + " |",
              "|---|" + "---|" * len(probes)]
        S = {}
        for m in MEASURES:
            cells = []
            for lab, dim, sd in probes:
                vals = []
                for reg in regs:
                    s = result.get((m, reg))
                    if s is None:
                        continue
                    b, p = s["est"][args.label][1], s["est"][lab][1]
                    vals.append((p**2 - b**2) / sd**2)
                v = float(np.median(vals)) if vals else np.nan
                S[(m, dim)] = v
                cells.append(f"{np.sign(v) * np.sqrt(abs(v)):.2f}" if np.isfinite(v) else "")
            L.append(f"| {m} | " + " | ".join(cells) + " |")
        dims = [p[1] for p in probes]
        L += ["", "## Solved spreads per register (NNLS on variances over the target measures)", "",
              "| register | " + " | ".join(dims) + " | target excess² → fitted |", "|---|" + "---|" * (len(dims) + 1)]
        solved = {}

        def system(reg_list):
            """Rows: the target measures per register; y the recordings' per-strike excess variance (negative where
            the model varies more), sd its bootstrap sd."""
            A, y, sd, names = [], [], [], []
            for reg in reg_list:
                for m in TARGETS:
                    s = result.get((m, reg))
                    if s is None:
                        continue
                    a, b = s["est"]["recording"][1], s["est"][args.label][1]
                    boot = s["boot"]["recording"][:, 1] ** 2 - s["boot"][args.label][:, 1] ** 2
                    A.append([max(S[(m, d)], 0.0) for d in dims])
                    y.append(a**2 - b**2)
                    sd.append(max(np.nanstd(boot), 0.1))
                    names.append(m if len(reg_list) == 1 else f"{reg} {m}")
            return np.array(A), np.array(y), np.array(sd), names

        def solve(A, y, sd):  # inverse-variance weighted
            return nnls(A / sd[:, None], y / sd)[0]

        for reg in regs:
            A, y, sd, names = system([reg])
            if not len(A):
                continue
            x = solve(A, y, sd)
            solved[reg] = dict(zip(dims, np.sqrt(x).tolist()))
            L.append(f"| {reg} | " + " | ".join(f"{np.sqrt(v):.3g}" for v in x) + " | "
                     + ", ".join(f"{n} {t:.1f}→{f:.1f}" for n, t, f in zip(names, y, A @ x)) + " |")
        A, y, sd, names = system(list(solved))
        x = solve(A, y, sd)
        solved["pooled"] = dict(zip(dims, np.sqrt(x).tolist()))
        L.append(f"| pooled | " + " | ".join(f"{np.sqrt(v):.3g}" for v in x) + " | (all registers' targets) |")
        L += ["", "Pooled over the registers (the per-register solutions scatter with the targets' noise), except the "
              "level, whose excess is confined to R2–R3; per target, the recordings' excess variance → the pooled "
              "solution's:", "", "| measure | " + " | ".join(solved) + " |", "|---|" + "---|" * len(solved)]
        fitted = A @ x
        for m in TARGETS:
            cells = []
            for reg in solved:
                k = f"{reg} {m}"
                i = names.index(k) if k in names else None
                cells.append(f"{y[i]:+.1f} ± {sd[i]:.1f} → {fitted[i]:.1f}" if i is not None else "")
            L.append(f"| {m} | " + " | ".join(cells) + " |")
        cfg = {d: round(solved["pooled"][d], 3) for d in dims}
        if "level_db" in dims:
            cfg["level_db"] = "/".join(f"{solved.get(r, {}).get('level_db', 0.0):.2f}" for r in REGS)
        solved["config"] = {f"strike_{d}": v for d, v in cfg.items()}
        L += ["", "Config: `" + ",".join(f"{k}={v}" for k, v in solved["config"].items()) + "`"]
        with open(os.path.join(args.out, "solved.json"), "w") as f:
            json.dump(solved, f, indent=1)
    with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
