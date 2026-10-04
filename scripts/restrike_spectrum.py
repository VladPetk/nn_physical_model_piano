"""Re-strike runs of the bench (docs/physics_revamp.md 13): per strike, each partial group's level after (20 ms .. min(150 ms, next strike - 10 ms))
and its change over just before (-60 .. -5 ms), recording vs model; the first strike of a run is a fresh strike."""
import json, sys
import numpy as np, torch
sys.path.insert(0, ".")
from pianonn import measures as M
from pianonn.config import year_to_condition
from pianonn.render import load_variant
PRE, POST = 0.5, 1.6
lo_p, hi_p = int(sys.argv[1]), int(sys.argv[2])
specs = sys.argv[3:]
bench = json.load(open("runs/measurements/bench_2018.json"))
runs = [r for r in bench["repeats"] if lo_p <= r["pitch"] <= hi_p]
dev = torch.device("cuda"); torch.cuda.set_per_process_memory_fraction(0.6)
models = [load_variant(s, device=dev) for s in specs]
cfg = models[0][1].cfg; sr = cfg.sample_rate
events = [{"piece": r["piece"], "onset": r["times"][0], "pitch": r["pitch"]} for r in runs]
clips = M.render_clips("data/maestro24k", events, models, cfg, dev, pre=PRE, post=POST, batch=8)
table = M.partial_table(models[0][1], year_to_condition(2018), dev)
GROUPS = {"1": (0, 1), "2-6": (1, 6), "7-24": (6, 24)}
def levels(x, a, b, freqs):
    seg = M._segment(x, sr, a, b)
    n = 1 << int(np.ceil(np.log2(len(seg) * 4)))
    P = M._power(seg, n) / len(seg); f = np.fft.rfftfreq(n, 1 / sr)
    out = []
    for fk in freqs:
        band = (f > fk * 0.985) & (f < fk * 1.015)
        out.append(P[band].max() if band.any() else np.nan)
    return np.array(out)
rows = []
for i, r in enumerate(runs):
    tab = table[r["pitch"] - 21]; tab = tab[(tab > 0) & (tab < 7000)][:24]
    for s in clips:
        x = clips[s][i]
        on = M.onset(x, sr, PRE + 0.004, float(tab[0]))
        t0 = on if np.isfinite(on) else PRE + 0.004
        for k, tk in enumerate(r["times"]):
            t = t0 + tk - r["times"][0]
            nxt = r["times"][k + 1] - tk if k + 1 < len(r["times"]) else 0.6
            b = min(0.15, nxt - 0.01)
            if b < 0.05:
                continue
            A, B = levels(x, t + 0.02, t + b, tab), levels(x, t - 0.06, t - 0.005, tab)
            row = {"run": i, "k": k, "src": s, "pitch": r["pitch"], "vel": r["velocities"][k],
                   "vprev": r["velocities"][k - 1] if k else 0, "gap": tk - r["times"][k - 1] if k else 0}
            for g, (a0, a1) in GROUPS.items():
                la = 10 * np.log10(np.nansum(A[a0:a1]) + 1e-30); lb = 10 * np.log10(np.nansum(B[a0:a1]) + 1e-30)
                row[f"after_{g}"], row[f"change_{g}"] = la, la - lb
            rows.append(row)
json.dump(rows, open(f"runs/physics_revamp/attack/restrike_{lo_p}_{hi_p}.json", "w"))
srcs = list(clips)
print(f"{len(runs)} runs, MIDI {lo_p}-{hi_p}")
def tab_(sel, name):
    print(f"\n{name}")
    for g in GROUPS:
        for key in ("change", "tilt"):
            if key == "tilt" and g == "1":
                continue
            vals = {}
            for s in srcs:
                R = [x for x in rows if x["src"] == s and sel(x)]
                v = np.array([x[f"after_{g}"] - x["after_1"] if key == "tilt" else x[f"change_{g}"] for x in R])
                vals[s] = v
            n = min(len(v) for v in vals.values())
            if n < 5: continue
            rec = vals["recording"][:n]
            lab = f"partials {g}: " + ("change at the strike" if key == "change" else "after, re the fundamental")
            print(f"  {lab:48s} rec {np.median(rec):+6.1f} | " + " | ".join(
                f"{s} − rec {np.median(vals[s][:n] - rec):+5.1f}" for s in srcs[1:]) + f"  (n {n})")
tab_(lambda x: x["k"] == 0, "fresh strikes (first of a run)")
tab_(lambda x: x["k"] >= 1, "re-strikes (key ringing, 0.08-0.6 s after its last strike)")
tab_(lambda x: x["k"] >= 1 and x["vel"] > x["vprev"] + 5, "re-strikes harder than the previous strike (+6 velocity or more)")
tab_(lambda x: x["k"] >= 1 and x["vel"] <= x["vprev"] + 5, "re-strikes not harder")
