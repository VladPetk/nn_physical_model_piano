"""Does the loss prefer quieter bass keys when the target's bass fine structure is misaligned?
Target = the trained model with re-drawn unison mistuning or bass B x1.3; scan a gain on keys MIDI 21-50 only."""
import sys, math, copy
import numpy as np, torch
sys.path.insert(0, "D:/MyProjects/nn_physical_model_piano")
from pianonn.config import PianoConfig
from pianonn.data import MaestroSegments
from pianonn.synth import NeuralPhysicalPiano
from pianonn.render import load_weights
from pianonn.losses import MultiResolutionSTFTLoss, LogMelLoss
from pianonn.train import fixed_batches

dev = torch.device("cuda")
torch.cuda.set_per_process_memory_fraction(0.6)
state = torch.load("runs/round1_trial/main/best.pt", map_location=dev)
cfg = PianoConfig.from_dict(state["cfg"])
model = load_weights(NeuralPhysicalPiano(cfg), state["model"], log=lambda *a: None).to(dev).eval()
ds = MaestroSegments("data/maestro24k", "test", cfg, 2.0, 1.0, 12.0, length=16, deterministic=True, years=[2018], seed=11)
batches = fixed_batches(ds, 16, 8, dev)
recon = MultiResolutionSTFTLoss()
mel = LogMelLoss(cfg.sample_rate).to(dev)
K = 30  # keys MIDI 21-50

def perturbed(kind):
    m = copy.deepcopy(model)
    with torch.no_grad():
        if kind == "unison re-drawn":
            gg = torch.Generator().manual_seed(999)
            mag = 0.2 + 1.8 * torch.rand(88, cfg.n_modes - 1, generator=gg) ** 2
            sign = torch.where(torch.rand(88, cfg.n_modes - 1, generator=gg) < 0.5, -1.0, 1.0)
            m.physics.raw_unison.copy_((5.0 * torch.atanh(mag * sign / 5.0)).to(dev))
        elif kind == "bass B x1.3":
            m.physics.raw_log_B[:K] += 1.5 * torch.atanh(torch.tensor(math.log(1.3) / 1.5))
        elif kind == "bass B x1.1":
            m.physics.raw_log_B[:K] += 1.5 * torch.atanh(torch.tensor(math.log(1.1) / 1.5))
    return m
gains = np.arange(-6, 2.01, 0.5)
targets = {}
with torch.no_grad():
    for kind in ("none", "unison re-drawn", "bass B x1.1", "bass B x1.3"):
        tm = perturbed(kind)
        targets[kind] = [tm(b, b["audio"].shape[-1], residual=False, generator=torch.Generator(device=dev).manual_seed(7))["audio"] for b in batches]
    renders = {}
    for g in gains:
        m = copy.deepcopy(model)
        m.physics.gain_db[:K] += float(g)
        renders[g] = [m(b, b["audio"].shape[-1], residual=False, generator=torch.Generator(device=dev).manual_seed(8))["audio"] for b in batches]
    s = int(batches[0]["loss_start"][0])
    for kind, ts in targets.items():
        ls = [np.mean([float(recon(renders[g][i][..., s:], ts[i][..., s:])) for i in range(len(batches))]) for g in gains]
        lm = [np.mean([float(mel(renders[g][i][..., s:], ts[i][..., s:])) for i in range(len(batches))]) for g in gains]
        print(f"target = {kind:16s}: bass-gain optimum MR-STFT {gains[int(np.argmin(ls))]:+.1f} dB (min {min(ls):.4f}, at 0: {ls[int(np.argmin(np.abs(gains)))]:.4f}, at -3: {ls[list(gains).index(-3.0)]:.4f}); log-mel {gains[int(np.argmin(lm))]:+.1f} dB")
