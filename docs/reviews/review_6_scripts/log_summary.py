import json, sys, collections
import numpy as np

def load(path):
    out = []
    with open(path) as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out

for path in sys.argv[1:]:
    recs = load(path)
    print("=" * 100)
    print(path, len(recs), "records")
    tr = [r for r in recs if r.get("kind") == "train"]
    va = [r for r in recs if r.get("kind") == "val"]
    other = [r for r in recs if r.get("kind") not in ("train", "val")]
    for r in other[:40]:
        print("  msg:", r["msg"][:230])
    if tr:
        keys = [k for k in tr[0] if k not in ("time", "msg", "kind", "step", "stage", "grad", "lr_factor")]
        allkeys = sorted({k for r in tr for k in r if k not in ("time", "msg", "kind", "step", "stage", "grad", "lr_factor")})
        steps = np.array([r["step"] for r in tr])
        nb = 10
        edges = np.linspace(steps.min(), steps.max() + 1, nb + 1)
        print("  train, means per decile of steps: ", allkeys)
        for i in range(nb):
            sel = [r for r in tr if edges[i] <= r["step"] < edges[i + 1]]
            if not sel:
                continue
            row = f"   {int(edges[i]):6d}-{int(edges[i+1]):6d} st{sel[-1]['stage']} lr{sel[-1].get('lr_factor', 1):.2f} "
            for k in allkeys:
                v = [r[k] for r in sel if k in r and r[k] == r[k]]
                row += f"{k}={np.mean(v):.4f} " if v else f"{k}=- "
            g = collections.defaultdict(list)
            for r in sel:
                for k, v in r.get("grad", {}).items():
                    g[k].append(v)
            row += "| grad " + " ".join(f"{k}={np.median(v):.2g}" for k, v in g.items())
            print(row)
    print("  validation:")
    for r in va:
        pr = r.get("per_res", {})
        s = f"   step {r.get('step')} st{r.get('stage','-')} phys {r.get('val_physics', float('nan')):.4f} "
        if "val_residual" in r:
            s += f"res {r['val_residual']:.4f} (d {r['val_residual'] - r['val_physics']:+.4f}) "
        s += " ".join(f"{k}={v:.4f}" for k, v in pr.items())
        if "per_res_residual" in r:
            s += " || R: " + " ".join(f"{k}={v:.4f}" for k, v in r["per_res_residual"].items())
        for k in ("env_val", "env_9+_1s", "env_9+_1.5s"):
            if k in r:
                s += f" {k}={r[k]:.3f}"
        print(s)
