"""Level bias of loss terms, per band, separated from the model's spectral-shape error.

For each (excerpt, band): the 'truth' gain is the one that matches the model's mean energy in that band to the
recording's (energy-weighted, the same STFT). A term's optimum is the gain that minimises that term restricted to
the band. bias = optimum - truth. An unbiased term has bias ~0 in every band whatever the model's shape errors.
Also reports the level error in several weightings, to separate "loss bias" from "which bins/frames count"."""
import sys, math
import numpy as np, torch
sys.path.insert(0, ".")
from pianonn.config import PianoConfig
from pianonn.data import MaestroSegments
from pianonn.synth import NeuralPhysicalPiano
from pianonn.render import load_weights
from pianonn.losses import _mag
from pianonn.train import fixed_batches

dev = torch.device("cuda")
torch.cuda.set_per_process_memory_fraction(0.6)
state = torch.load("runs/round1_trial/main/best.pt", map_location=dev)
cfg = PianoConfig.from_dict(state["cfg"])
model = load_weights(NeuralPhysicalPiano(cfg), state["model"], log=lambda *a: None).to(dev).eval()
ds = MaestroSegments("data/maestro24k", "test", cfg, 2.0, 1.0, 12.0, length=24, deterministic=True, years=[2018], seed=11)
batches = fixed_batches(ds, 24, 8, dev)
sr = cfg.sample_rate
EDGES = [40 * 2 ** (k / 2) for k in range(0, 17)]  # half-octave report bands 40 Hz .. 10 kHz
BANDS = list(zip(EDGES[:-1], EDGES[1:]))
G = torch.arange(-9, 9.01, 0.25, device=dev)  # dB
floor = lambda n: 10 ** (-80 / 20) * (0.375 * n) ** 0.5

Y, T = [], []
with torch.no_grad():
    for b in batches:
        n, s = b["audio"].shape[-1], int(b["loss_start"][0])
        Y.append(model(b, n, residual=True, generator=torch.Generator(device=dev).manual_seed(0))["audio"][..., s:])
        T.append(b["audio"][..., s:])
Y, T = torch.cat(Y), torch.cat(T)  # [24, 2, t]
N = Y.shape[0]

def per_bin_terms(n_fft):
    S, P = _mag(T.reshape(-1, T.shape[-1]), n_fft), _mag(Y.reshape(-1, Y.shape[-1]), n_fft)  # [N*2, F, frames]
    S, P = S.reshape(N, 2, *S.shape[1:]), P.reshape(N, 2, *P.shape[1:])
    f = torch.fft.rfftfreq(n_fft, 1 / sr).to(dev)
    e = floor(n_fft)
    out = {"logL1": np.zeros((N, len(BANDS))), "SC(L2 mag)": np.zeros((N, len(BANDS))), "truth": np.zeros((N, len(BANDS)))}
    for j, (lo, hi) in enumerate(BANDS):
        m = (f >= lo) & (f < hi)
        if m.sum() == 0:
            out["logL1"][:, j] = out["SC(L2 mag)"][:, j] = out["truth"][:, j] = np.nan
            continue
        s, p = S[:, :, m], P[:, :, m]  # [N, 2, bins, frames]
        g = 10 ** (G / 20)
        l1 = (torch.log(s[None] + e) - torch.log(g[:, None, None, None, None] * p[None] + e)).abs().mean((2, 3, 4))  # [G, N]
        l2 = ((s[None] - g[:, None, None, None, None] * p[None]) ** 2).sum((2, 3, 4))
        out["logL1"][:, j] = G[l1.argmin(0)].cpu().numpy()
        out["SC(L2 mag)"][:, j] = G[l2.argmin(0)].cpu().numpy()
        out["truth"][:, j] = (10 * torch.log10((s ** 2).sum((1, 2, 3)) / (p ** 2).sum((1, 2, 3)).clamp(min=1e-20))).cpu().numpy()
    return out

def band_energy_term(frame_s, n_fft=2048, width_oct=1 / 6):
    """log band energies: 1/6-octave raised-cosine-free (rectangular) bands, frames pooled to frame_s, L1 in log."""
    S, P = _mag(T.reshape(-1, T.shape[-1]), n_fft, 256) ** 2, _mag(Y.reshape(-1, Y.shape[-1]), n_fft, 256) ** 2
    f = torch.fft.rfftfreq(n_fft, 1 / sr).to(dev)
    k = max(1, int(round(frame_s * sr / 256)))
    out = {"logL1": np.zeros((N, len(BANDS))), "truth": np.zeros((N, len(BANDS))), "dof": np.zeros(len(BANDS))}
    for j, (lo, hi) in enumerate(BANDS):
        subs = [(lo * 2 ** (i * width_oct), lo * 2 ** ((i + 1) * width_oct)) for i in range(int(round(0.5 / width_oct)))]
        ls = torch.zeros(len(G), N, device=dev)
        es, ep = 0, 0
        for a, c in subs:
            m = (f >= a) & (f < c)
            if m.sum() == 0:
                m = (f - (a * c) ** 0.5).abs() == (f - (a * c) ** 0.5).abs().min()
            st = S[:, m].sum(1)  # [N*2, frames]
            pt = P[:, m].sum(1)
            st = torch.nn.functional.avg_pool1d(st[:, None], k, k)[:, 0].reshape(N, 2, -1)
            pt = torch.nn.functional.avg_pool1d(pt[:, None], k, k)[:, 0].reshape(N, 2, -1)
            e = 10 ** (-80 / 10) * 0.375 * n_fft * m.sum()
            g = 10 ** (G / 10)
            ls += (torch.log(st[None] + e) - torch.log(g[:, None, None, None] * pt[None] + e)).abs().mean((2, 3))
            es, ep = es + st.sum((1, 2)), ep + pt.sum((1, 2))
        out["logL1"][:, j] = G[ls.argmin(0)].cpu().numpy()
        out["truth"][:, j] = (10 * torch.log10(es / ep.clamp(min=1e-20))).cpu().numpy()
        bw = (hi - lo) / len(subs)
        out["dof"][j] = 2 * bw * max(frame_s, 256 / sr * 4 / 1.5)  # rough: 2 x time-bandwidth product per cell
    return out

print(f"model level re recording, broadband energy: median {np.median([10*math.log10(float(Y[i].pow(2).sum()/T[i].pow(2).sum())) for i in range(N)]):+.2f} dB")
cols = " ".join(f"{int(lo):>6d}" for lo, _ in BANDS)
print(f"{'band lower edge (Hz)':34s} {cols}")
rows = []
for n_fft in (4096, 1024, 256):
    o = per_bin_terms(n_fft)
    if n_fft == 4096:
        print(f"{'truth: model - recording (dB, med)':34s} " + " ".join(f"{-np.nanmedian(o['truth'][:, j]):+6.1f}" for j in range(len(BANDS))))
    for k in ("SC(L2 mag)", "logL1"):
        bias = o[k] - o["truth"]
        print(f"{'bias ' + k + ' ' + str(n_fft):34s} " + " ".join(f"{np.nanmedian(bias[:, j]):+6.1f}" for j in range(len(BANDS))) + f" | all {np.nanmedian(bias):+.2f}")
for fs in (0.02, 0.05, 0.1, 0.25, 0.5):
    o = band_energy_term(fs)
    bias = o["logL1"] - o["truth"]
    print(f"{f'bias logband 1/6oct {int(fs*1000)}ms':34s} " + " ".join(f"{np.nanmedian(bias[:, j]):+6.1f}" for j in range(len(BANDS))) + f" | all {np.nanmedian(bias):+.2f}")
