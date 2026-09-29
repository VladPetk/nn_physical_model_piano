"""A note struck and released (no pedal) 3 s before the window: how loud is it at the window start?"""
import numpy as np, torch
from pianonn.diagnostics import _perf
from pianonn.render import load_model
m = load_model(use_noise=False, use_sympathetic=False, use_room=False)
sr = m.cfg.sample_rate
n = 2 * sr
def e(notes, t0, t1):
    with torch.no_grad():
        y = m(_perf(m, n, notes), n)["audio"][0].numpy()
    return 10 * np.log10(np.mean(y[int(t0 * sr):int(t1 * sr)] ** 2) + 1e-30)
fresh = e([(48, 0.0, 2.0, 90)], 0.0, 0.2)
stale = e([(48, -3.0, -2.5, 90)], 0.0, 0.2)     # released 2.5 s before the window: should be ~silent
stale_late = e([(48, -3.0, -2.5, 90)], 0.3, 0.5)
held = e([(48, -3.0, 2.0, 90)], 0.0, 0.2)         # still held: legitimately rings
print(f"fresh note, first 200 ms: {fresh:.1f} dB")
print(f"held since -3 s, first 200 ms: {held:.1f} dB")
print(f"released at -2.5 s, first 200 ms: {stale:.1f} dB  (should be far below the held case)")
print(f"released at -2.5 s, 300-500 ms: {stale_late:.1f} dB")
