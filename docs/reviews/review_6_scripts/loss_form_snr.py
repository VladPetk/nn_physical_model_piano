"""Per-cell gradient signal-to-noise of three loss forms, by simulation (CPU, seconds; docs/mean_matching_loss.md).

    python docs/reviews/review_6_scripts/loss_form_snr.py

Cell powers: P_X = lam_x G1, P_X' = lam_x G2 (two draws of the model), P_Y = lam_y G3 (the take), G ~ Gamma(k, 1/k)
(mean 1): k = 1 is the exponential scatter of a dense cell, a large k a cell that one partial dominates. delta =
ln(lam_x / lam_y); the gradient is with respect to ln(lam_x), so dP_X = P_X and dlog(P_X) = 1.

* L1 energy: the composite's form per cell, sign(log P_X - log P_Y) - sign(log P_X - log P_X').
* squared: 2 (P_X' - P_Y) P_X / nu^2 with nu the take's mean.
* IS exact: Itakura-Saito on an exact lam_x, 1 - P_Y / lam_x.

Also: what clipping does to the mean at the match, and what counting cells by the take's own level does.
"""
import numpy as np

rng = np.random.default_rng(0)
N = 4_000_000


def estimators(k, delta):
    g = rng.gamma(k, 1.0 / k, size=(3, N))
    px, px2, py = np.exp(delta) * g[0], np.exp(delta) * g[1], g[2]
    return {
        "L1 energy": np.sign(np.log(px) - np.log(py)) - np.sign(np.log(px) - np.log(px2)),
        "squared": 2 * (px2 - py) * px,
        "IS exact": 1 - py / np.exp(delta),
    }


delta = 0.1
print(f"signal-to-noise per cell, per unit of delta (delta = {delta}, {N:.0e} draws)")
for k in (1, 4, 10):
    moved, matched = estimators(k, delta), estimators(k, 0.0)
    for name, v in moved.items():
        signal = v.mean() - matched[name].mean()
        print(f"  k = {k:2d}  {name:10s} mean {v.mean():+.4f}  sd {v.std():.3f}  SNR/delta {signal / v.std() / delta:.2f}")

g = rng.gamma(1.0, 1.0, size=(3, N))
print("\nclipping at the match, k = 1 (an unbiased form keeps the mean)")
for c in (2.0, 3.0, 5.0):
    clipped = np.minimum(g[2], c).mean()
    factor = np.clip(g[1] - g[2], -c, c).mean()
    print(f"  at {c:.0f}: clipping P_Y/nu reads the take {10 * np.log10(clipped):+.2f} dB; "
          f"clipping (P_X' - P_Y)/nu leaves a mean of {factor:+.4f}")

print("\ncells counted only where the take stands over t x its mean, at the match, k = 1")
for t in (0.1, 0.5):
    keep = g[2] > t
    print(f"  t = {t}: mean of (P_X' - P_Y)/nu over the counted cells {np.mean((g[1] - g[2])[keep]):+.3f}")
