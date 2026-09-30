"""Is the leading-silence low-frequency floor stationary and audible-band, or a start-of-file artefact?"""
import json, os, sys
import numpy as np, soundfile as sf, torch
sys.path.insert(0, ".")
root = "data/maestro24k"
idx = json.load(open(f"{root}/index.json"))
n_fft = 16384  # 1.46 Hz bins
w = np.hanning(4096)
spec_sil, spec_perf, tc = [], [], []
for p in idx:
    notes = np.load(os.path.join(root, p["midi"]))["notes"]
    first = float(notes[:, 1].min())
    path = os.path.join(root, p["audio"])
    if first - 0.15 >= 0.5:
        x, sr = sf.read(path, start=int(0.05 * 24000), frames=int((first - 0.15) * 24000), dtype="float64", always_2d=True)
        x = x.mean(1)
        # time course of <60 Hz energy in 100 ms frames
        fr = [x[i:i + 2400] for i in range(0, len(x) - 2400, 1200)]
        lp = []
        for f in fr:
            X = np.abs(np.fft.rfft(f * np.hanning(len(f)), n_fft)) ** 2
            fq = np.fft.rfftfreq(n_fft, 1 / 24000)
            lp.append(10 * np.log10(X[(fq > 20) & (fq < 60)].sum() + 1e-20))
        tc.append(np.array(lp) - lp[len(lp) // 2])
        X = np.abs(np.fft.rfft(x[:4096 * (len(x) // 4096)].reshape(-1, 4096) * w, n_fft, axis=1)) ** 2
        spec_sil.append(X.mean(0))
    # a performance window in the middle of the piece, for comparison
    x, sr = sf.read(path, start=int(p["duration"] / 2 * 24000), frames=4096 * 24, dtype="float64", always_2d=True)
    X = np.abs(np.fft.rfft(x.mean(1).reshape(-1, 4096)[:, :] * w, n_fft, axis=1)) ** 2
    spec_perf.append(X.mean(0))
fq = np.fft.rfftfreq(n_fft, 1 / 24000)
S, P = np.median(np.stack(spec_sil), 0), np.median(np.stack(spec_perf), 0)
print(f"{len(spec_sil)} pieces. Power spectral density, dB re 1 (same scale), median over pieces:")
for lo, hi in ((0, 5), (5, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 70), (70, 100), (100, 150), (150, 250), (250, 500), (500, 1000), (1000, 4000)):
    b = (fq >= lo) & (fq < hi)
    print(f"  {lo:5d}-{hi:5d} Hz: silence {10*np.log10(S[b].mean()):6.1f}   mid-piece music {10*np.log10(P[b].mean()):6.1f}   music - silence {10*np.log10(P[b].mean()/S[b].mean()):+5.1f} dB")
L = min(len(t) for t in tc)
T = np.median(np.stack([t[:L] for t in tc]), 0)
print("20-60 Hz energy over the leading silence, 100 ms steps, re its middle (median over pieces):", np.round(T, 1))
