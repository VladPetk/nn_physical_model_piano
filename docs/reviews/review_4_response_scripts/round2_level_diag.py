"""Round 2, B3 follow-up: where the level errors sit.

1. The broadband level error by velocity tercile (the regression in scripts/evaluate.py, without the linear model).
2. Per octave band, the level error in attack frames (−5…+40 ms around onsets) and in the rest.

    python docs/reviews/review_4_response_scripts/round2_level_diag.py data/maestro24k \\
        trial=runs/round1_trial/main/best.pt:residual control=runs/round2/b_control/last.pt:physics ...

Same excerpts as scripts/evaluate.py (test split, seed 11, 96 excerpts, 2018).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from pianonn.data import MaestroSegments  # noqa: E402
from pianonn.losses import highpass  # noqa: E402
from pianonn.metrics import CENTERS, band_split, render  # noqa: E402
from pianonn.render import load_model  # noqa: E402
from pianonn.train import fixed_batches  # noqa: E402

FRAME = 120  # 5 ms at 24 kHz


@torch.no_grad()
def measure(model, batches, residual):
    sr = model.cfg.sample_rate
    vel, broad, att, sus = [], [], [], []
    for b in batches:
        s = int(b["loss_start"][0])
        y, t = highpass(render(model, b, residual), sr), highpass(b["audio"][..., s:], sr)
        n = (t.shape[-1] // FRAME) * FRAME
        ey = band_split(y[..., :n], sr).pow(2).sum(1).reshape(y.shape[0], len(CENTERS), -1, FRAME).sum(-1)  # [B, bands, frames]
        et = band_split(t[..., :n], sr).pow(2).sum(1).reshape(y.shape[0], len(CENTERS), -1, FRAME).sum(-1)
        fc = (torch.arange(ey.shape[-1], device=y.device) + 0.5) * FRAME / sr
        for i in range(y.shape[0]):
            on = b["onset"][i] - s / sr
            m = b["mask"][i] & (on > -1.0) & (on < n / sr)
            vel.append(float(b["velocity"][i][m].float().mean()) if m.any() else float("nan"))
            broad.append(10 * np.log10(float(y[i].pow(2).mean()) / max(1e-20, float(t[i].pow(2).mean()))))
            d = fc[None, :] - on[m][:, None]
            a = ((d > -0.005) & (d < 0.040)).any(0) if m.any() else torch.zeros_like(fc, dtype=torch.bool)
            r = lambda e_y, e_t: (10 * torch.log10(e_y.sum(-1) / e_t.sum(-1).clamp(min=1e-20))).cpu().numpy()
            att.append(r(ey[i][:, a], et[i][:, a]) if a.sum() >= 4 else np.full(len(CENTERS), np.nan))
            sus.append(r(ey[i][:, ~a], et[i][:, ~a]) if (~a).sum() >= 4 else np.full(len(CENTERS), np.nan))
    return np.array(vel), np.array(broad), np.array(att), np.array(sus)


def main():
    data, specs = sys.argv[1], sys.argv[2:]
    sys.stdout.reconfigure(encoding="utf-8")
    dev = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(0.6)
    batches, out = None, {}
    for spec in specs:
        label, rest = spec.split("=", 1)
        ckpt, mode = rest.rsplit(":", 1)
        model = load_model(ckpt, device=dev)
        if batches is None:
            ds = MaestroSegments(data, "test", model.cfg, 2.0, 1.0, 12.0, length=96, deterministic=True, years=[2018], seed=11)
            batches = fixed_batches(ds, 96, 8, dev)
        out[label] = measure(model, batches, mode == "residual")
        del model
        torch.cuda.empty_cache()
    vel = next(iter(out.values()))[0]
    ok = np.isfinite(vel)
    edges = np.percentile(vel[ok], [100 / 3, 200 / 3])
    terc = np.digitize(vel, edges)
    names = [f"soft (vel < {edges[0]:.0f})", f"middle", f"loud (vel ≥ {edges[1]:.0f})"]
    print("## Broadband level error by velocity tercile (median dB, model − recording)\n")
    print("| | " + " | ".join(names) + " | loud − soft |\n|---|---|---|---|---|")
    for label, (_, broad, _, _) in out.items():
        med = [np.median(broad[ok & (terc == k)]) for k in range(3)]
        print(f"| {label} | " + " | ".join(f"{v:+.2f}" for v in med) + f" | {med[2] - med[0]:+.2f} |")
    print("\n## Level error per octave band, attack frames (−5…+40 ms around onsets) vs the rest (median dB)\n")
    print("| | " + " | ".join(f"{c:.0f} Hz" for c in CENTERS) + " |\n|---|" + "---|" * len(CENTERS))
    for label, (_, _, att, sus) in out.items():
        print(f"| {label}, attack | " + " | ".join(f"{np.nanmedian(att[:, k]):+.1f}" for k in range(len(CENTERS))) + " |")
        print(f"| {label}, rest | " + " | ".join(f"{np.nanmedian(sus[:, k]):+.1f}" for k in range(len(CENTERS))) + " |")


if __name__ == "__main__":
    main()
