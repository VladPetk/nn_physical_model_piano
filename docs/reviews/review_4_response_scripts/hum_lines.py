"""Mains hum in MAESTRO 2018: the strongest spectral lines near 60 and 120 Hz, in the leading silence and mid-piece."""
import json, os
import numpy as np, soundfile as sf

root = "data/maestro24k"
n_fft = 24000 * 4  # 0.25 Hz bins
S, M = [], []
for p in json.load(open(f"{root}/index.json")):
    first = float(np.load(os.path.join(root, p["midi"]))["notes"][:, 1].min())
    if first - 0.15 < 0.5:
        continue
    x, _ = sf.read(os.path.join(root, p["audio"]), start=1200, frames=int((first - 0.15) * 24000), dtype="float64", always_2d=True)
    S.append(np.abs(np.fft.rfft(x.mean(1) * np.hanning(len(x)), n_fft)) ** 2 / (np.hanning(len(x)) ** 2).sum())
    y, _ = sf.read(os.path.join(root, p["audio"]), start=int(p["duration"] / 2 * 24000), frames=24000 * 2, dtype="float64", always_2d=True)
    M.append(np.abs(np.fft.rfft(y.mean(1) * np.hanning(len(y)), n_fft)) ** 2 / (np.hanning(len(y)) ** 2).sum())
f = np.fft.rfftfreq(n_fft, 1 / 24000)
for name, X in (("silence", np.median(np.stack(S), 0)), ("mid-piece music", np.median(np.stack(M), 0))):
    for lo, hi in ((40, 70), (110, 200)):
        b = (f >= lo) & (f <= hi)
        top = np.argsort(X[b])[-4:][::-1]
        print(f"{name:16s} {lo}-{hi} Hz: strongest bins (Hz, dB re the band median)",
              [(round(float(f[b][i]), 2), round(float(10 * np.log10(X[b][i] / np.median(X[b]))), 1)) for i in top])
