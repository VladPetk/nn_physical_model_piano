"""Beat fluctuation of single notes (docs/physics_revamp.md 12.2): partials 1-6 of the note bench's notes that sound
freely for 1.6 s (key held or pedal down), the rms of the dB envelope around a quadratic in time over 0.25-1.6 s;
recordings against single-note renders at the same pitch, velocity, pedal and duration. Beats show as fluctuation, a
smooth double decay does not.

    python scripts/beat_fluctuation.py B_comp=runs/loss_compare/B_comp/train/last.pt coupled=<ckpt>[:key=value,...]
"""
import sys, json, numpy as np, torch, soundfile as sf
sys.path.insert(0, ".")
from pianonn.render import load_model, render_notes
from pianonn.config import year_to_condition
dev = torch.device("cuda"); torch.cuda.set_per_process_memory_fraction(0.2)
sr = 24000
b = json.load(open("runs/measurements/bench_2018.json"))
P = {p["id"]: p for p in json.load(open("data/maestro24k/index.json"))}
notes = [x for x in b["notes"] if x["clear_after"] >= 1.6 and (x["offset"] - x["onset"] >= 1.6 or x["pedal"] >= 64)]
W, HOP = 2048, 240
win = np.hanning(W)
def fluct(x, f0, B=0.0):
    fr = np.arange(0, len(x) - W, HOP)
    S = np.abs(np.fft.rfft(np.stack([x[i:i + W] * win for i in fr]), axis=1))
    t = (fr + W / 2) / sr
    fb = np.fft.rfftfreq(W, 1 / sr)
    s = (t > 0.25) & (t < 1.6)
    out = {}
    for n in range(1, 7):
        f = n * f0
        if f > 4000: break
        h = max(0.012 * f, 1.5 * sr / W)
        m = (fb > f - h) & (fb < f + h)
        e = 20 * np.log10(S[:, m].max(1) + 1e-9)[s]
        if e.max() - 20 * np.log10(S[s].max() + 1e-9) < -40: continue  # too weak to read
        A = np.c_[np.ones(s.sum()), t[s], t[s] ** 2]
        r = e - A @ np.linalg.lstsq(A, e, rcond=None)[0]
        out[n] = float(np.sqrt((r ** 2).mean()))
    return out
models = dict(a.split("=", 1) for a in sys.argv[1:])
res = {"recording": []}
for x in notes:
    a, _ = sf.read(f"data/maestro24k/{P[x['piece']]['audio']}", start=int(x["onset"] * sr), frames=int(1.8 * sr),
                   dtype="float32", always_2d=True)
    f0 = 440 * 2 ** ((x["pitch"] - 69) / 12)
    res["recording"].append(fluct(a.mean(1), f0))
for lab, ck in models.items():
    ck, *ov = ck.split(":")
    m = load_model(ck, device=dev, **{k: float(v) for k, v in (o.split("=") for o in ov)})
    m.strike_on = True
    res[lab] = []
    for x in notes:
        nt = np.array([[x["pitch"], 0.05, 0.05 + max(x["offset"] - x["onset"], 0.1), x["velocity"]]], float)
        ped = {f"{n}_{s}": np.zeros(0) for n in ("sostenuto", "soft") for s in ("t", "v")}
        ped["sustain_t"], ped["sustain_v"] = np.array([0.0]), np.array([float(x["pedal"])])
        a = render_notes(m, nt, ped, year_to_condition(2018), tail=1.9, residual=False)[:, int(0.05 * sr):]
        f0 = 440 * 2 ** ((x["pitch"] - 69) / 12)
        res[lab].append(fluct(a.mean(0), f0))
    del m; torch.cuda.empty_cache()
regs = sorted(set(x["register"] for x in notes))
print(f"{len(notes)} notes; median fluctuation (dB rms) over partials 1-6, per register [n readings]")
print("reg   " + "".join(f"{k:>14s}" for k in res))
for r in regs:
    row = []
    for k in res:
        v = [f for x, d in zip(notes, res[k]) if x["register"] == r for f in d.values()]
        row.append(f"{np.median(v):8.2f} [{len(v):3d}]" if v else "      -       ")
    print(f"{r:5s} " + "".join(row))
for n in range(1, 7):
    row = []
    for k in res:
        v = [d[n] for d in res[k] if n in d]
        row.append(f"{np.median(v):8.2f} [{len(v):3d}]" if v else "      -       ")
    print(f"n={n}   " + "".join(row))
json.dump({k: v for k, v in res.items()}, open("scratch/beat_fluctuation.json", "w"))
