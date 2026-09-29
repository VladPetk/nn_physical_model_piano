"""Does the float32 oscillator bank degrade for notes far from t=0 (long renders)?"""
import numpy as np, torch
from pianonn import NeuralPhysicalPiano, PianoConfig
from pianonn.diagnostics import _perf, _variant
from pianonn.render import load_model

torch.manual_seed(0)
model = load_model(use_noise=False, use_sympathetic=False, use_room=False)
sr = model.cfg.sample_rate


def note_at(t_on, pitch=96, vel=90, dur=1.0):
    n = int((t_on + dur) * sr)
    with torch.no_grad():
        y = model(_perf(model, n, [(pitch, t_on, t_on + dur, vel)]), n, block_seconds=2.0)["audio"][0].numpy()
    return y[int(t_on * sr):]


ref = note_at(0.0)
for t_on in (0.0, 30.0, 120.0, 300.0, 600.0):
    y = note_at(t_on)
    L = min(len(ref), len(y))
    a, b = ref[:L], y[:L]
    # spectrum comparison in the first 0.5 s: noise floor between the partials
    seg_a, seg_b = a[: sr // 2], b[: sr // 2]
    w = np.hanning(len(seg_a))
    A, Bs = np.abs(np.fft.rfft(seg_a * w)), np.abs(np.fft.rfft(seg_b * w))
    hz = np.fft.rfftfreq(len(seg_a), 1 / sr)
    # residual energy relative to the reference (time-domain, phase-sensitive)
    err_db = 10 * np.log10(np.mean((a - b) ** 2) / np.mean(a**2) + 1e-30)
    # spectral floor: median level between 5-11 kHz (C7 partials sit at 2.1, 4.3, 6.6, 9.0 kHz)
    band = (hz > 5000) & (hz < 11000)
    print(f"onset {t_on:5.0f}s: time-domain error {err_db:6.1f} dB re signal; median 5-11 kHz level {20*np.log10(np.median(Bs[band])/A.max()):6.1f} dB re peak (ref {20*np.log10(np.median(A[band])/A.max()):6.1f})")
