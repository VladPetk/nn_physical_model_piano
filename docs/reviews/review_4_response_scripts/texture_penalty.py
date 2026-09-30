"""Under L1 of log power, how does a constant at the median score against a random signal with the right statistics,
when the target is random (power ~ exponential per bin, Gamma(k) for a band of k independent bins)?"""
import numpy as np

r = np.random.default_rng(0)
for k in (1, 4, 16, 64):
    X, Y = r.gamma(k, size=1_000_000), r.gamma(k, size=1_000_000)
    rand = np.abs(np.log(X) - np.log(Y)).mean()
    med = np.abs(np.log(X) - np.log(np.median(X))).mean()
    print(f"{k:3d} bins: random vs random {rand:.3f} nats, median vs random {med:.3f} (ratio {rand / med:.2f}); "
          f"median re mean {10 * np.log10(np.median(X) / k):+.2f} dB")
