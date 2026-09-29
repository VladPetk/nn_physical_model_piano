"""Fit the hammer spectrum (roll-off order per key, the second, high-frequency corner) to the attack
spectra of measured isolated notes. (A smooth correction of the body response was tried as well: it
came out below 2 dB up to 7 kHz, so the body prior is left alone.)

    python scripts/fit_spectra.py data/iowa_partials.json

Data: ``pianonn.calibration.partial_tables`` (early level of every partial below 10 kHz), mf and ff.
Model: the prompt-mode amplitude of each partial (hammer envelope x strike comb) times the 1/3-octave
smoothed body response at its frequency, both normalised per note by the power of the partials below
3 kHz, so the level of the note drops out and only the shape is fitted. Partials that the recording
does not show above its noise floor enter as upper bounds (a one-sided penalty).
Verify afterwards with scripts/compare_partials.py (renders and measures the model like a recording).
"""

import json
import math
import sys

import numpy as np
import torch
import torch.nn.functional as F

from pianonn.config import PianoConfig
from pianonn.physics import HAMMER_ORDER_MF, HAMMER_ORDER_VEL, PianoPhysics, hammer_velocity
from pianonn.room import _octave_smooth, soundboard_body

VEL = {"mf": 64, "ff": 110}
SMOOTH = 50.0  # curvature penalty on the per-key orders (5 gives the same C4 order, 1.41)
Q_KEYS = [0, 15, 27, 39, 51, 63, 75, 87]


def interp(xk, yk, x):
    xk = torch.as_tensor(xk, dtype=torch.float64)
    i = torch.searchsorted(xk, x).clamp(1, len(xk) - 1)
    w = ((x - xk[i - 1]) / (xk[i] - xk[i - 1])).clamp(0, 1)
    return yk[i - 1] + w * (yk[i] - yk[i - 1])


def load(path, phys, body_db_at):
    notes = []
    for r in json.load(open(path)):
        if r["dynamic"] not in VEL:
            continue
        e = np.array(r["early_db"], dtype=float)
        if np.isfinite(e).sum() < 3:
            continue
        k = r["pitch"] - 21
        u = VEL[r["dynamic"]] / 127
        with torch.no_grad():
            tc = phys.contact_time(torch.tensor([[k]]), torch.tensor([[u]]), torch.zeros(1, 1), torch.tensor([0])).item()
        f = np.array(r["f"])
        n = np.array(r["n"], dtype=float)
        x0 = phys.prior_strike[k].item()
        notes.append(dict(k=k, vh=hammer_velocity(torch.tensor(u)).item(), tc=tc, f=torch.tensor(f), n=n,
                          comb_db=torch.tensor(20 * np.log10(np.abs(np.sin(np.pi * n * x0)) + 1e-4)),
                          body_db=body_db_at(torch.tensor(f)), rec=torch.tensor(e - np.nanmax(e)),
                          ok=torch.tensor(np.isfinite(e)), ref=torch.tensor(np.isfinite(e) & (f < 3000)),
                          bound=torch.tensor(np.array(r["noise_db"]) + 10.0), pitch=r["pitch"], dyn=r["dynamic"]))
    return notes


def hammer_db(f, tc, q, x2, q2):
    x = f * tc
    return (-10 / math.log(10)) * (F.softplus(2 * q * torch.log(x / 0.59)) + F.softplus(2 * q2 * torch.log(x / x2)))


def predict(note, p):
    log_q = torch.cat([p["log_q"], p["_log_q_treble"]])
    q = torch.exp(interp(Q_KEYS, log_q, torch.tensor(float(note["k"]), dtype=torch.float64))) * (note["vh"] / 2.8) ** -HAMMER_ORDER_VEL
    lv = hammer_db(note["f"], note["tc"], q, torch.exp(p["log_x2"]), torch.exp(p["log_q2"])) + note["comb_db"] + note["body_db"]
    # same normalisation as the recording: re the loudest reliable partial, via the power of the reference set
    def lse(v):
        return 10 * torch.log10((10 ** (v[note["ref"]] / 10)).sum())
    return lv - lse(lv) + lse(note["rec"])


def fit(notes, p0, steps=1500):
    p = {k: v.clone().requires_grad_(True) for k, v in p0.items()}
    fixed = [k for k in p if k.startswith("_")]
    opt = torch.optim.Adam([v for k, v in p.items() if k not in fixed], lr=0.02)
    for it in range(steps):
        loss = 0.0
        cnt = 0
        for nt in notes:
            pr = predict(nt, p)
            d = pr[nt["ok"]] - nt["rec"][nt["ok"]]
            loss = loss + F.huber_loss(d, torch.zeros_like(d), delta=6.0, reduction="sum")
            cens = ~nt["ok"]
            loss = loss + (F.relu(pr[cens] - nt["bound"][cens]) ** 2).sum() * 0.5
            cnt += len(d)
        lq = torch.cat([p["log_q"], p["_log_q_treble"]])
        reg = SMOOTH * ((lq[2:] - 2 * lq[1:-1] + lq[:-2]) ** 2).sum()
        total = loss / cnt + reg / 100
        opt.zero_grad()
        total.backward()
        opt.step()
        if it % 250 == 0 or it == steps - 1:
            print(f"step {it}: data {loss.item() / cnt:.2f}  reg {reg.item():.2f}  q={[round(v, 2) for v in torch.exp(torch.cat([p['log_q'], p['_log_q_treble']])).tolist()]} "
                  f"x2={math.exp(p['log_x2'].item()):.2f} q2={math.exp(p['log_q2'].item()):.2f}", flush=True)
    return {k: v.detach() for k, v in p.items()}


def residuals(notes, p, label):
    rows = []
    with torch.no_grad():
        for nt in notes:
            pr = predict(nt, p)
            for j in torch.nonzero(nt["ok"]).flatten().tolist():
                rows.append((nt["pitch"], nt["f"][j].item(), (pr[j] - nt["rec"][j]).item()))
    a = np.array(rows)
    print(f"\n{label}: median model - recording (dB), by partial frequency;  A0-B2 | C3-B4 | C5+")
    for lo in [55 * 2**k for k in range(8)]:
        s = a[(a[:, 1] >= lo) & (a[:, 1] < 2 * lo)]
        g = lambda q: f"{np.median(q[:, 2]):6.1f} ({len(q):3d})" if len(q) >= 5 else "      -     "
        print(f"  {lo:6.0f}-{2 * lo:6.0f} Hz  {g(s[s[:, 0] < 48])} {g(s[(s[:, 0] >= 48) & (s[:, 0] < 72)])} {g(s[s[:, 0] >= 72])}")
    print(f"  rms {np.sqrt(np.mean(a[:, 2] ** 2)):.2f} dB, mean |err| {np.mean(np.abs(a[:, 2])):.2f} dB")


def main(path):
    cfg = PianoConfig()
    phys = PianoPhysics(cfg)
    sr = cfg.sample_rate
    h = soundboard_body(sr, cfg.body_seconds).double()
    nfft = 1 << 16
    H = torch.fft.rfft(h, nfft).abs() ** 2
    hz = torch.fft.rfftfreq(nfft, 1 / sr).double()
    body_db = 10 * torch.log10(_octave_smooth(H, hz).clamp(min=1e-20))

    def body_db_at(f):
        return interp(hz, body_db, f.double())

    notes = load(path, phys, body_db_at)
    print(f"{len(notes)} notes")
    q0 = torch.log(torch.tensor([dict(HAMMER_ORDER_MF).get(k, None) or float(np.interp(k, *zip(*HAMMER_ORDER_MF))) for k in Q_KEYS], dtype=torch.float64))
    # the top keys have 1-4 partials below 10 kHz: their order stays at the prior
    base = {"log_q": q0[:-2], "_log_q_treble": q0[-2:], "log_x2": torch.tensor(math.log(1e3), dtype=torch.float64),
            "log_q2": torch.tensor(0.0, dtype=torch.float64)}
    residuals(notes, base, "current prior (no second corner)")
    p0 = dict(base, log_x2=torch.tensor(math.log(7.0), dtype=torch.float64), log_q2=torch.tensor(math.log(2.0), dtype=torch.float64))
    p = fit(notes, p0)
    residuals(notes, p, "fitted")
    print("\nHAMMER_ORDER_MF =", [(k, round(math.exp(v), 2)) for k, v in zip(Q_KEYS, torch.cat([p["log_q"], p["_log_q_treble"]]).tolist())])
    print(f"HAMMER_X2 = {math.exp(p['log_x2'].item()):.2f}, HAMMER_Q2 = {math.exp(p['log_q2'].item()):.2f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/iowa_partials.json")
