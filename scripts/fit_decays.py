"""Fit the decay priors (aftersound loss b1, prompt ratio R, aftersound amplitude) to measured decay profiles.

    python scripts/fit_decays.py data/iowa_analysis.json

For each landmark key, the target is the median decay profile of the recordings (level of the
power-summed decay partials at fixed times, mf + ff, +-3 semitones). The model's profile is
computed analytically from its mode parameters (sum over partials and modes of amp^2 exp(-2 alpha t),
beats averaged out), so the fit is instantaneous. Verify with rendered notes afterwards
(pianonn.diagnostics measures them exactly like the recordings).
"""

import json
import math
import sys

import numpy as np
import torch
from scipy.optimize import minimize

from pianonn.calibration import LANDMARKS, PROFILE_TIMES, _at_landmarks, _key_values, decay_partials, landmark_samples
from pianonn.config import PianoConfig
from pianonn.physics import PianoPhysics


def model_profile(phys, k, log_b1, log_rm1, log_after, u=0.5):
    with torch.no_grad():
        phys.prior_log_b1[k] = log_b1
        phys.prior_prompt_ratio[k] = 1 + math.exp(log_rm1)
        phys.prior_log_after[k] = log_after
        m = phys.modes(torch.tensor([[k]]), torch.tensor([[u]]), torch.zeros(1, 1), torch.tensor([0]))
    idx = [n - 1 for n in decay_partials(k + 21)]
    a2 = m["amp"][0, 0, idx].double().numpy() ** 2
    al = m["alpha"][0, 0, idx].double().numpy()
    t = np.array((0.05,) + PROFILE_TIMES)
    P = (a2[..., None] * np.exp(-2 * al[..., None] * t)).sum((0, 1))
    return 10 * np.log10(P[1:] / P[0])


def main(path):
    rec = json.load(open(path))["recordings"]
    targets = {}
    for i, _ in enumerate(PROFILE_TIMES):
        vals = _at_landmarks(_key_values(rec, lambda r, i=i: (r.get("profile_db") or [np.nan] * 6)[i], ("mf", "ff")))
        for name, v in vals.items():
            targets.setdefault(name, []).append(v)
    counts = landmark_samples(rec, lambda r: (r.get("profile_db") or [np.nan])[0], ("mf", "ff"))
    phys = PianoPhysics(PianoConfig())
    fitted = {}
    for name, pitch in LANDMARKS.items():
        k = pitch - 21
        tgt = np.array(targets[name])
        ok = np.isfinite(tgt)
        if ok.sum() < 3:
            print(f"{name}: too few profile points ({ok.sum()}), skipped")
            continue
        x0 = [phys.prior_log_b1[k].item(), math.log(max(phys.prior_prompt_ratio[k].item() - 1, 0.1)), phys.prior_log_after[k].item()]

        def loss(x):
            prof = model_profile(phys, k, *x)
            return float(np.sum((prof[ok] - tgt[ok]) ** 2)) + 0.1 * float(np.sum((np.array(x) - x0) ** 2))

        res = minimize(loss, x0, method="Nelder-Mead", options={"maxiter": 2000, "xatol": 1e-4, "fatol": 1e-4})
        b1, R, after = math.exp(res.x[0]), 1 + math.exp(res.x[1]), math.exp(res.x[2])
        prof = model_profile(phys, k, *res.x)
        fitted[name] = (k, b1, R, after)
        print(f"{name} (n={len(counts[name])}): b1={b1:.3f} R={R:.2f} after={after:.3f} | target "
              + " ".join(f"{v:5.0f}" if np.isfinite(v) else "    -" for v in tgt)
              + " | model " + " ".join(f"{v:5.0f}" for v in prof))
    print("\nkey_curve points:")
    for label, j in (("b1", 1), ("R", 2), ("after", 3)):
        print(label, [(k, round(v[j], 3)) for _, v in sorted(fitted.items(), key=lambda kv: kv[1][0]) for k in [v[0]]])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/iowa_analysis.json")
