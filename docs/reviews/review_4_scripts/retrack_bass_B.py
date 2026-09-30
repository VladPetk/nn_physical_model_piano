"""Re-track the mined bass notes with a wider inharmonicity search range: does B move off the old lower bound?"""
import json, math, os, sys
import numpy as np, soundfile as sf
sys.path.insert(0, "D:/MyProjects/nn_physical_model_piano")
from pianonn.calibration import rigaud_B, track_partials

root = "data/maestro24k"
idx = {p["id"]: p for p in json.load(open(os.path.join(root, "index.json")))}
d = json.load(open("runs/measurements/mined_2018.json"))
notes = [n for n in d["notes"] if not n["suspect"] and n["B_reliable"] and 24 <= n["pitch"] <= 47]
rows = []
for n in notes:
    p = idx[n["piece"]]
    path = os.path.join(root, p["audio"])
    sr = sf.info(path).samplerate
    start = int((n["onset"] - 0.3) * sr)
    x, _ = sf.read(path, start=start, frames=int(1.3 * sr), dtype="float64", always_2d=True)
    f_nom = 440.0 * 2 ** ((n["pitch"] - 69) / 12)
    Br = rigaud_B(n["pitch"])
    out = {}
    for tag, rng in (("old x/4", (Br / 4, Br * 4)), ("wide x/32", (Br / 32, Br * 4))):
        tr = track_partials(x, sr, int(0.3 * sr), f_nom, rng, seconds=1.0)
        out[tag] = (tr["B"], tr["fit_rms_cents"], len(tr["n"])) if tr else None
    rows.append((n["pitch"], n["velocity"], Br / 4, out))
print("pitch vel | old-range B (rms cents, #partials) | wide-range B (rms, #) | B_wide / B_old | old lower bound")
for pitch, vel, lb, out in rows:
    o, w = out["old x/4"], out["wide x/32"]
    if o and w:
        print(f"{pitch:3d} {vel:3.0f} | {o[0]:.2e} ({o[1]:.1f}, {o[2]:2d}) | {w[0]:.2e} ({w[1]:.1f}, {w[2]:2d}) | {w[0]/o[0]:.2f} | {lb:.2e}")
ratios = [out["wide x/32"][0] / out["old x/4"][0] for _, _, _, out in rows if out["old x/4"] and out["wide x/32"]]
by_reg = {}
for (pitch, _, _, out), r in zip([x for x in rows if x[3]["old x/4"] and x[3]["wide x/32"]], ratios):
    by_reg.setdefault(pitch // 6 * 6, []).append((r, out["wide x/32"][0], out["wide x/32"][1]))
for k, v in sorted(by_reg.items()):
    print(f"MIDI {k}-{k+5}: n={len(v)} median B_wide/B_old {np.median([r for r,_,_ in v]):.2f}; median B_wide {np.median([b for _,b,_ in v]):.2e}; median rms wide {np.median([c for _,_,c in v]):.1f} cents")
