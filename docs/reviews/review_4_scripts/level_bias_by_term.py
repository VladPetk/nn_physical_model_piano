"""On real test excerpts: which loss term wants the model quieter than it is? Scan a broadband gain on the trained
model's render and report the optimum of the spectral-convergence term, the log-magnitude term, and log-mel."""
import sys, math
import numpy as np, torch
sys.path.insert(0, "D:/MyProjects/nn_physical_model_piano")
from pianonn.config import PianoConfig
from pianonn.data import MaestroSegments
from pianonn.synth import NeuralPhysicalPiano
from pianonn.render import load_weights
from pianonn.losses import MultiResolutionSTFTLoss, LogMelLoss, _mag
from pianonn.train import fixed_batches

dev = torch.device("cuda")
torch.cuda.set_per_process_memory_fraction(0.6)
state = torch.load("runs/round1_trial/main/best.pt", map_location=dev)
cfg = PianoConfig.from_dict(state["cfg"])
model = load_weights(NeuralPhysicalPiano(cfg), state["model"], log=lambda *a: None).to(dev).eval()
ds = MaestroSegments("data/maestro24k", "test", cfg, 2.0, 1.0, 12.0, length=24, deterministic=True, years=[2018], seed=11)
batches = fixed_batches(ds, 24, 8, dev)
recon = MultiResolutionSTFTLoss()
mel = LogMelLoss(cfg.sample_rate).to(dev)

def terms(pred, target):
    pred, target = pred.reshape(-1, pred.shape[-1]), target.reshape(-1, target.shape[-1])
    sc_t, lm_t = {}, {}
    for n in recon.fft_sizes:
        S, P = _mag(target, n), _mag(pred, n)
        eps = recon.eps(n)
        sc_t[n] = float((torch.linalg.norm(S - P, dim=(-2, -1)) / (torch.linalg.norm(S, dim=(-2, -1)) + eps)).mean())
        lm_t[n] = float((torch.log(S + eps) - torch.log(P + eps)).abs().mean())
    return sc_t, lm_t

gains = np.arange(-3, 5.01, 0.5)
SC = {g: {n: 0.0 for n in recon.fft_sizes} for g in gains}
LM = {g: {n: 0.0 for n in recon.fft_sizes} for g in gains}
ML = {g: 0.0 for g in gains}
lvl = []
with torch.no_grad():
    for b in batches:
        n, s = b["audio"].shape[-1], int(b["loss_start"][0])
        t = b["audio"][..., s:]
        y = model(b, n, residual=True, generator=torch.Generator(device=dev).manual_seed(0))["audio"][..., s:]
        for i in range(t.shape[0]):
            lvl.append(10 * math.log10(float(y[i].pow(2).mean()) / float(t[i].pow(2).mean())))
        for g in gains:
            sc, lm = terms(y * 10 ** (g / 20), t)
            for k in recon.fft_sizes:
                SC[g][k] += sc[k] / len(batches); LM[g][k] += lm[k] / len(batches)
            ML[g] += float(mel(y * 10 ** (g / 20), t)) / len(batches)
print(f"model level re target: median {np.median(lvl):+.2f} dB")
for name, T in (("spectral convergence", SC), ("log-magnitude L1", LM)):
    for k in recon.fft_sizes:
        vals = [T[g][k] for g in gains]
        print(f"{name:22s} {k:5d}: optimum gain {gains[int(np.argmin(vals))]:+.1f} dB  (at 0: {vals[list(gains).index(0.0)]:.4f}, min {min(vals):.4f})")
    tot = [np.mean([T[g][k] for k in recon.fft_sizes]) for g in gains]
    print(f"{name:22s}   all: optimum gain {gains[int(np.argmin(tot))]:+.1f} dB")
vals = [ML[g] for g in gains]
print(f"log-mel                : optimum gain {gains[int(np.argmin(vals))]:+.1f} dB (at 0: {vals[list(gains).index(0.0)]:.4f}, min {min(vals):.4f})")
