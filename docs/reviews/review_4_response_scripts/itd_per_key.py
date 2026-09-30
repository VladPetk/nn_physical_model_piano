"""Inter-channel delay and level difference per key on the mined isolated notes (direct sound, first 60 ms), vs the model."""
import json, os, sys
import numpy as np, soundfile as sf, torch
sys.path.insert(0, ".")
root = "data/maestro24k"
idx = {p["id"]: p for p in json.load(open(f"{root}/index.json"))}
d = json.load(open("runs/measurements/mined_2018.json"))
sr = 24000
rows = []
for n in d["notes"]:
    if n["suspect"]:
        continue
    p = idx[n["piece"]]
    x, _ = sf.read(os.path.join(root, p["audio"]), start=int((n["onset"] - 0.005) * sr), frames=int(0.065 * sr), dtype="float64", always_2d=True)
    L, R = x[:, 0], x[:, 1]
    N = 4096
    XL, XR = np.fft.rfft(L * np.hanning(len(L)), N), np.fft.rfft(R * np.hanning(len(R)), N)
    f = np.fft.rfftfreq(N, 1 / sr)
    G = XL * np.conj(XR)
    G[(f < 150) | (f > 6000)] = 0
    cc = np.fft.irfft(G / (np.abs(G) + 1e-12), N)
    cc = np.concatenate([cc[-72:], cc[:73]])  # +-3 ms
    k = int(np.argmax(cc))
    frac = 0.0
    if 0 < k < len(cc) - 1:
        y0, y1, y2 = cc[k - 1], cc[k], cc[k + 1]
        frac = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
    lag_ms = (k - 72 + frac) / sr * 1000  # >0: left leads? (L correlates with R delayed)
    ild = 10 * np.log10((L ** 2).sum() / (R ** 2).sum())
    rows.append((n["pitch"], lag_ms, ild, float(cc[k])))
rows = np.array(rows)
print(f"{len(rows)} notes. Inter-channel lag (ms, GCC-PHAT on the first 60 ms, 150 Hz-6 kHz) and level L-R (dB), by register:")
for lo in range(21, 109, 12):
    r = rows[(rows[:, 0] >= lo) & (rows[:, 0] < lo + 12)]
    if len(r) >= 3:
        print(f"  MIDI {lo:3d}-{lo+11:3d}: n={len(r):3d}  lag median {np.median(r[:,1]):+.2f} ms (IQR {np.percentile(r[:,1],25):+.2f}..{np.percentile(r[:,1],75):+.2f})  L-R {np.median(r[:,2]):+.1f} dB  peak {np.median(r[:,3]):.2f}")
