"""Noise ceiling of the loss, fitted noise floor vs init, gain oracle, and the residual budget's share term."""
import sys, math
import numpy as np, torch
sys.path.insert(0, "D:/MyProjects/nn_physical_model_piano")
from pianonn.config import PianoConfig, year_to_condition
from pianonn.data import MaestroSegments
from pianonn.synth import NeuralPhysicalPiano
from pianonn.render import load_weights
from pianonn.losses import MultiResolutionSTFTLoss, band_energies
from pianonn.train import fixed_batches

dev = torch.device("cuda")
torch.cuda.set_per_process_memory_fraction(0.6)
state = torch.load("runs/round1_trial/main/best.pt", map_location=dev)
cfg = PianoConfig.from_dict(state["cfg"])
model = load_weights(NeuralPhysicalPiano(cfg), state["model"]).to(dev).eval()
cond = year_to_condition(2018)
print("step", state["step"], "stage", state["stage"])
fl = model.room.floor_db[cond].detach().cpu()
print("fitted floor L, every 4th band:", [round(float(v), 1) for v in fl[0][::4]])
print("fitted floor R, every 4th band:", [round(float(v), 1) for v in fl[1][::4]])
print("init (mean L/R):               ", [-40.3, -49.2, -47.2, -43.0, -52.8, -67.0, -81.2, -85.8])

ds = MaestroSegments("data/maestro24k", "test", cfg, 2.0, 1.0, 12.0, length=24, deterministic=True, years=[2018], seed=11)
batches = fixed_batches(ds, 24, 8, dev)
recon = MultiResolutionSTFTLoss()
res_keys = list(recon.fft_sizes)
acc = {}
def add(k, l, per=None):
    acc.setdefault(k, []).append((l, per))
gains = np.arange(-6, 6.01, 0.5)
lvl_err, oracle_g = [], []
share_all, share_empty_frac, share_nonempty = [], [], []
with torch.no_grad():
    for b in batches:
        n, s = b["audio"].shape[-1], int(b["loss_start"][0])
        t = b["audio"][..., s:]
        o0 = model(b, n, residual=True, generator=torch.Generator(device=dev).manual_seed(0))
        o1 = model(b, n, residual=True, generator=torch.Generator(device=dev).manual_seed(1))
        y0, y1 = o0["audio"][..., s:], o1["audio"][..., s:]
        for k, (a, c) in {"model vs target": (y0, t), "render seed0 vs seed1": (y0, y1)}.items():
            l, per = recon(a, c, per_resolution=True); add(k, float(l), {r: float(v) for r, v in per.items()})
        fn = model.room.floor_noise(b["condition"], n, torch.Generator(device=dev).manual_seed(2))[..., s:]
        l, per = recon(t + fn, t, per_resolution=True); add("target vs target+learned floor", float(l), {r: float(v) for r, v in per.items()})
        # physics-only render vs target, for reference
        yp = model(b, n, residual=False, generator=torch.Generator(device=dev).manual_seed(0))["audio"][..., s:]
        l, per = recon(yp, t, per_resolution=True); add("physics vs target", float(l), {r: float(v) for r, v in per.items()})
        # per-excerpt broadband level error and gain oracle
        for i in range(t.shape[0]):
            e = 10 * math.log10(float(y0[i].pow(2).mean()) / float(t[i].pow(2).mean()))
            lvl_err.append(e)
            ls = [float(recon(y0[i:i+1] * 10 ** (g / 20), t[i:i+1])) for g in gains]
            j = int(np.argmin(ls)); oracle_g.append(gains[j]); add("gain oracle (per excerpt best gain)", ls[j])
            add("per-excerpt model vs target", float(recon(y0[i:i+1], t[i:i+1])))
        # residual budget share term
        M = model.noise.band_masks(512)
        e_res = band_energies(o0["noise_res"], M)
        e_dry = band_energies(o0["dry_phys"].mean(1), M)
        share = e_res / (e_res + e_dry + 1e-12)
        empty = e_dry < 1e-8
        share_all.append(float(share.mean())); share_empty_frac.append(float(empty.float().mean()))
        share_nonempty.append(float(share[~empty].mean()) if (~empty).any() else float("nan"))
for k, v in acc.items():
    ls = [x[0] for x in v]
    line = f"{k:40s} mean {np.mean(ls):.4f}"
    if v[0][1] is not None:
        line += "  per res " + " ".join(f"{r}:{np.mean([x[1][r] for x in v]):.3f}" for r in res_keys)
    print(line)
print(f"broadband level error (model - target) dB: median {np.median(lvl_err):+.2f} IQR ({np.percentile(lvl_err,25):+.2f},{np.percentile(lvl_err,75):+.2f})")
print(f"oracle gain: median {np.median(oracle_g):+.2f} dB, IQR ({np.percentile(oracle_g,25):+.2f},{np.percentile(oracle_g,75):+.2f})")
print(f"budget share: mean {np.mean(share_all):.4f}; fraction of (band,frame) cells with ~empty dry signal {np.mean(share_empty_frac):.4f}; share in non-empty cells {np.nanmean(share_nonempty):.5f}")
