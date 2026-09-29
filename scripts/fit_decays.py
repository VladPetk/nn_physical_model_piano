"""Fit the string decay priors to the per-partial decays of measured isolated notes.

    python scripts/fit_decays.py data/iowa_partials.json

Data: ``pianonn.calibration.partial_tables``: the level of every partial below 10 kHz at 0.5, 1, 2, 3, 5
and 8 s re its own attack peak (mf and ff; the decay rates do not depend on the dynamic).

Model (``pianonn.physics``), per key k and partial n at frequency f_n, beats averaged out:

    aftersound   alpha_a = b1(k) + b3(k) f_ref^2 (f_n / f_ref)^p       f_ref = 1 kHz, p fitted (2 = the usual b3 f^2)
    prompt       alpha_p = alpha_a + (R(k) - 1) b1(k) g(f_n)      g: bridge conductance vs frequency, all keys
    power        P(t) = exp(-2 alpha_p t) + m after(k)^2 exp(-2 alpha_a t)   (m aftersound modes: 1 below key 26, else 2)

The prompt stage is the in-phase motion of the unison draining through the bridge, so its extra loss
follows the bridge conductance G(f) (Weinreich): the rate per partial is f0 Z0 G(f_n) up to a constant, a
per-key factor times one curve over frequency. g is normalised to a geometric mean of 1 over its knots.
The fit is analytic (instant); verify with scripts/compare_partials.py, which renders and measures.
Recorded levels below the noise floor enter as upper bounds (one-sided penalty).
"""

import json
import math
import sys

import numpy as np
import torch
import torch.nn.functional as F

from pianonn.calibration import PARTIAL_TIMES
from pianonn.config import PianoConfig
from pianonn.physics import DECAY_EXPONENT, PianoPhysics

KNOTS = [0, 12, 20, 27, 34, 39, 45, 51, 57, 63, 75, 87]
G_KNOTS = [55 * 2**k for k in range(9)]  # 55 Hz .. 14 kHz


def interp(xk, yk, x):
    xk = torch.as_tensor(xk, dtype=torch.float64)
    i = torch.searchsorted(xk, x).clamp(1, len(xk) - 1)
    w = ((x - xk[i - 1]) / (xk[i] - xk[i - 1])).clamp(0, 1)
    return yk[i - 1] + w * (yk[i] - yk[i - 1])


def load(path):
    notes = []
    t = torch.tensor(PARTIAL_TIMES, dtype=torch.float64)
    for r in json.load(open(path)):
        if r["dynamic"] not in ("mf", "ff"):
            continue
        e = np.array(r["early_db"], dtype=float)
        lv = np.array(r["level_db"], dtype=float)
        if lv.ndim != 2 or lv.shape[1] != len(t):
            continue
        good = np.isfinite(e) & (e - np.nanmax(e) > -40)  # partials clearly above the noise at the attack
        if good.sum() < 2:
            continue
        # upper bound for censored levels: noise (re the loudest partial) + 3 dB, re this partial's attack
        bound = np.array(r["noise_db"])[:, None] + 3.0 - (e - np.nanmax(e))[:, None] + 0 * lv
        f0 = r["f0"]
        # loud partials dominate what is heard (and the old power-summed profile): weight 1 at the loudest,
        # 0.32 at -20 dB, floor 0.25
        w = np.maximum(0.25, 10 ** ((e - np.nanmax(e))[good] / 40))
        notes.append(dict(k=r["pitch"] - 21, f=torch.tensor(np.array(r["f"])[good]), t=t, w=torch.tensor(w)[:, None].expand(-1, len(t)),
                          te=0.5 * float(np.clip(2.5 / f0, 0.02, 0.2)),
                          lv=torch.tensor(lv[good]), ok=torch.tensor(np.isfinite(lv[good])),
                          bound=torch.tensor(bound[good]), pitch=r["pitch"]))
    return notes


def params_at(p, k):
    kk = torch.tensor(float(k), dtype=torch.float64)
    return {name: torch.exp(interp(KNOTS, p[name], kk)) for name in ("b1", "rm1", "after", "b3")}


def log_g(p, f):
    lg = p["g"] - p["g"].mean()
    return interp(np.log(G_KNOTS), lg, torch.log(f))


def predict(note, p):
    q = params_at(p, note["k"])
    f = note["f"][:, None]
    aa = q["b1"] + q["b3"] * 1e6 * (f / 1000.0) ** torch.exp(p["log_p"])
    ap = aa + q["rm1"] * q["b1"] * torch.exp(log_g(p, f))
    m = 1.0 if note["k"] < 26 else 2.0

    def P(t):
        return torch.exp(-2 * ap * t) + m * q["after"] ** 2 * torch.exp(-2 * aa * t)
    return 10 * torch.log10(P(note["t"][None]) / P(torch.tensor(note["te"], dtype=torch.float64)))


def loss_fn(notes, p):
    total = 0.0
    for nt in notes:
        pr = predict(nt, p)
        d = (pr - nt["lv"])[nt["ok"]]
        cm = ~nt["ok"] & torch.isfinite(nt["bound"])
        cens = F.relu(pr[cm] - nt["bound"][cm])
        hub = F.huber_loss(d, torch.zeros_like(d), delta=5.0, reduction="none")
        total = total + ((hub * nt["w"][nt["ok"]]).sum() + 0.5 * (cens**2 * nt["w"][cm]).sum()) / nt["w"].sum()
    return total / len(notes)


def reg_fn(p):
    def curv(v):
        return ((v[2:] - 2 * v[1:-1] + v[:-2]) ** 2).sum()
    return sum(curv(p[k]) for k in ("b1", "rm1", "after", "b3")) + curv(p["g"])


def residuals(notes, p, label):
    print(f"\n{label}: median level at " + "/".join(f"{t:g}" for t in PARTIAL_TIMES) + " s re attack  [recording | model]")
    regs = [("A0-B1", 21, 35), ("C2-B2", 36, 47), ("C3-B3", 48, 59), ("C4-B4", 60, 71), ("C5-B5", 72, 83), ("C6-C8", 84, 108)]
    with torch.no_grad():
        rows = []
        for nt in notes:
            pr = predict(nt, p).numpy()
            lv = nt["lv"].numpy()
            for j in range(len(lv)):
                rows.append((nt["pitch"], nt["f"][j].item(), *lv[j], *pr[j]))
    a = np.array(rows)
    T = len(PARTIAL_TIMES)
    for name, lo, hi in regs:
        r = a[(a[:, 0] >= lo) & (a[:, 0] <= hi)]
        print(f" {name}")
        for b in range(len(G_KNOTS) - 1):
            s = r[(r[:, 1] >= G_KNOTS[b]) & (r[:, 1] < G_KNOTS[b + 1])]
            if len(s) < 5:
                continue
            fr = lambda cols: " ".join(f"{np.nanmedian(s[:, c]):6.1f}" if np.isfinite(s[:, c]).sum() >= 5 else "     -" for c in cols)
            print(f"  {G_KNOTS[b]:5.0f}-{G_KNOTS[b + 1]:5.0f} Hz {fr(range(2, 2 + T))} | {fr(range(2 + T, 2 + 2 * T))}  ({len(s)})")
    ok = np.isfinite(a[:, 2:2 + T])
    err = (a[:, 2 + T:] - a[:, 2:2 + T])[ok]
    print(f" rms {np.sqrt(np.mean(err**2)):.2f} dB, median |err| {np.median(np.abs(err)):.2f} dB")


def main(path):
    phys = PianoPhysics(PianoConfig())
    notes = load(path)
    print(f"{len(notes)} notes")
    ki = torch.tensor(KNOTS)
    p0 = {"b1": phys.prior_log_b1[ki].double(), "rm1": torch.log(phys.prior_prompt_ratio[ki] - 1).double(),
          "after": phys.prior_log_after[ki].double(), "b3": phys.prior_log_b3[ki].double(),
          "g": phys.prior_log_bridge_g.double().clone(), "log_p": torch.tensor(math.log(DECAY_EXPONENT), dtype=torch.float64)}
    residuals(notes, p0, "current prior")
    p = {k: v.clone().requires_grad_(True) for k, v in p0.items()}
    opt = torch.optim.Adam(p.values(), lr=0.03)
    for it in range(800):
        loss = loss_fn(notes, p)
        reg = reg_fn(p)
        opt.zero_grad()
        (loss + 0.3 * reg).backward()
        opt.step()
        if it % 100 == 0 or it == 799:
            print(f"step {it}: data {loss.item():.3f} reg {reg.item():.3f}", flush=True)
    p = {k: v.detach() for k, v in p.items()}
    residuals(notes, p, "fitted")
    fmt = lambda v, d=3: [(k, float(f"{x:.{d}g}")) for k, x in zip(KNOTS, v)]
    print("\nb1    ", fmt(torch.exp(p["b1"]).tolist()))
    print("R     ", fmt((1 + torch.exp(p["rm1"])).tolist()))
    print("after ", fmt(torch.exp(p["after"]).tolist()))
    print("b3    ", fmt(torch.exp(p["b3"]).tolist()))
    print(f"p      {math.exp(p['log_p'].item()):.3f}")
    print("g     ", [(f, round(math.exp(v), 3)) for f, v in zip(G_KNOTS, (p["g"] - p["g"].mean()).tolist())])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/iowa_partials.json")
