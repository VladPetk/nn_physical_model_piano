"""Noise floor measured on the leading silence of each MAESTRO 2018 piece vs the init (2nd percentile) and the fitted floor."""
import json, os, sys, math
import numpy as np, soundfile as sf, torch
sys.path.insert(0, ".")
from pianonn.config import PianoConfig, year_to_condition
from pianonn.synth import NeuralPhysicalPiano
from pianonn.render import load_weights
from pianonn.fit_init import _band_matrix
from pianonn.losses import _mag

root = "data/maestro24k"
state = torch.load("runs/round1_trial/main/best.pt", map_location="cpu")
cfg = PianoConfig.from_dict(state["cfg"])
model = load_weights(NeuralPhysicalPiano(cfg), state["model"], log=lambda *a: None).eval()
M = _band_matrix(model, 2048)
per_piece = []
for p in json.load(open(f"{root}/index.json")):
    first = float(np.load(os.path.join(root, p["midi"]))["notes"][:, 1].min())
    a, b = 0.05, first - 0.1
    if b - a < 0.3:
        continue
    x, sr = sf.read(os.path.join(root, p["audio"]), start=int(a * 24000), frames=int((b - a) * 24000), dtype="float32", always_2d=True)
    x = torch.from_numpy(x.T.copy())
    E = torch.einsum("kf,cft->ckt", M, _mag(x, 2048, 512) ** 2)  # [ch, bands, frames]
    per_piece.append(E.median(-1).values)  # [ch, bands]
E = torch.stack(per_piece)  # [pieces, ch, bands]
db = lambda v: 10 * torch.log10(v.clamp(min=1e-14))
med, p25, p75 = db(E.median(0).values), db(E.quantile(0.25, 0)), db(E.quantile(0.75, 0))
cond = year_to_condition(2018)
fit = model.room.floor_db.detach()[cond]
c = model.noise.centers
print(f"{len(per_piece)} pieces with >= 0.3 s of leading silence")
print("band Hz  | silence L (IQR over pieces)  | silence R | init (mean L/R) | fitted L")
init = {0: -40.3, 4: -49.2, 8: -47.2, 12: -43.0, 16: -52.8, 20: -67.0, 24: -81.2, 28: -85.8}
for i in range(0, 32):
    print(f"{float(c[i]):7.0f} | {float(med[0,i]):6.1f} ({float(p25[0,i]):6.1f},{float(p75[0,i]):6.1f}) | {float(med[1,i]):6.1f} | {init.get(i, float('nan')):6.1f} | {float(fit[0,i]):6.1f}")
