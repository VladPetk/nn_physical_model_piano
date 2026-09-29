"""Plot a training log (``log.jsonl`` from pianonn.train) as an SVG: training loss (smoothed) and the
validation loss with the residual off and on, against the step, with the stage switch marked.

    python scripts/plot_log.py runs/trial/log.jsonl docs/trial_curves.svg
"""

import json
import sys


def main():
    src, dst = sys.argv[1], sys.argv[2]
    train, val_p, val_r, stage2, prior = [], [], [], None, {}
    with open(src) as f:
        for line in f:
            r = json.loads(line)
            if r.get("kind") == "train":
                train.append((r["step"], r["recon"]))
            elif r.get("kind") == "val":
                if r.get("tag") in ("raw_prior", "init_prior"):
                    prior[r["tag"]] = r["val_physics"]
                    continue
                val_p.append((r["step"], r["val_physics"]))
                if "val_residual" in r:
                    val_r.append((r["step"], r["val_residual"]))
            elif "stage 2" in r.get("msg", "") and stage2 is None:
                stage2 = int(r["msg"].split()[1].rstrip(":"))
    k = 8  # moving average over log points
    smooth = [(train[i][0], sum(v for _, v in train[max(0, i - k + 1): i + 1]) / len(train[max(0, i - k + 1): i + 1]))
              for i in range(len(train))]
    W, H, L, R, T, B = 760, 380, 60, 20, 20, 50
    xs = [s for s, _ in smooth + val_p] or [0, 1]
    ys = [v for _, v in smooth + val_p + val_r] + list(prior.values())
    x0, x1 = 0, max(xs)
    y0, y1 = min(ys) * 0.97, max(ys) * 1.02
    X = lambda s: L + (W - L - R) * (s - x0) / max(1, x1 - x0)
    Y = lambda v: T + (H - T - B) * (1 - (v - y0) / (y1 - y0))
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif" font-size="12">',
           f'<rect width="{W}" height="{H}" fill="white"/>']
    for i in range(6):
        v = y0 + (y1 - y0) * i / 5
        out.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="#e5e5e5"/>'
                   f'<text x="{L - 6}" y="{Y(v) + 4:.1f}" text-anchor="end" fill="#555">{v:.2f}</text>')
    for i in range(6):
        s = x0 + (x1 - x0) * i / 5
        out.append(f'<text x="{X(s):.1f}" y="{H - B + 18}" text-anchor="middle" fill="#555">{s:.0f}</text>')
    out.append(f'<text x="{(L + W - R) / 2}" y="{H - 10}" text-anchor="middle" fill="#333">step</text>')
    if stage2 is not None:
        out.append(f'<line x1="{X(stage2):.1f}" x2="{X(stage2):.1f}" y1="{T}" y2="{H - B}" stroke="#999" stroke-dasharray="4 4"/>'
                   f'<text x="{X(stage2) + 4:.1f}" y="{T + 12}" fill="#666">stage 2: residual on</text>')
    for name, v, col in (("untrained prior", prior.get("raw_prior"), "#b0b0b0"), ("prior after init", prior.get("init_prior"), "#7a7a7a")):
        if v is not None:
            out.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="{col}" stroke-dasharray="2 3"/>'
                       f'<text x="{W - R - 4}" y="{Y(v) - 4:.1f}" text-anchor="end" fill="{col}">{name} (val) {v:.3f}</text>')

    def poly(pts, col, width, dash=""):
        if pts:
            d = " ".join(f"{X(s):.1f},{Y(v):.1f}" for s, v in pts)
            out.append(f'<polyline points="{d}" fill="none" stroke="{col}" stroke-width="{width}" {dash}/>')

    poly(smooth, "#9bbbe0", 1.5)
    poly(val_p, "#1f5fa8", 2.5)
    poly(val_r, "#d9731a", 2.5)
    for pts, col in ((val_p, "#1f5fa8"), (val_r, "#d9731a")):
        for s, v in pts:
            out.append(f'<circle cx="{X(s):.1f}" cy="{Y(v):.1f}" r="3" fill="{col}"/>')
    lx, ly = L + 12, H - B - 58
    for i, (name, col) in enumerate((("train (smoothed)", "#9bbbe0"), ("validation, physics only", "#1f5fa8"),
                                     ("validation, physics + residual", "#d9731a"))):
        out.append(f'<line x1="{lx}" x2="{lx + 20}" y1="{ly + 16 * i}" y2="{ly + 16 * i}" stroke="{col}" stroke-width="3"/>'
                   f'<text x="{lx + 26}" y="{ly + 16 * i + 4}" fill="#333">{name}</text>')
    out.append(f'<text x="{L}" y="{T - 6}" fill="#333">multi-resolution STFT loss</text></svg>')
    with open(dst, "w") as f:
        f.write("\n".join(out))
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
