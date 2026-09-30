"""(1) How much of each band is the learned stationary floor? (2) Is the loss biased towards a quiet model when
the target's fine structure (unison beating, bass inharmonicity) is misaligned? Synthetic: target = the trained
model with re-drawn unison mistuning / perturbed bass B; scan a broadband gain on the unperturbed model."""
import sys, math, copy
import numpy as np, torch
sys.path.insert(0, "D:/MyProjects/nn_physical_model_piano")
from pianonn.config import PianoConfig, year_to_condition
from pianonn.data import MaestroSegments
from pianonn.synth import NeuralPhysicalPiano
from pianonn.render import load_weights
from pianonn.losses import MultiResolutionSTFTLoss, LogMelLoss, band_energies
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
M = model.noise.band_masks(512)
centers = model.noise.centers.cpu().numpy()

# (1) band levels: target vs model with / without the floor
E = {"target": [], "with floor": [], "no floor": [], "floor only": []}
with torch.no_grad():
    for b in batches:
        n, s = b["audio"].shape[-1], int(b["loss_start"][0])
        t = b["audio"][..., s:]
        g = lambda k: torch.Generator(device=dev).manual_seed(k)
        yf = model(b, n, residual=True, generator=g(0))["audio"][..., s:]
        yn = model(b, n, residual=True, generator=g(0), floor=False)["audio"][..., s:]
        fl = model.room.floor_noise(b["condition"], n, g(3))[..., s:]
        for k, x in (("target", t), ("with floor", yf), ("no floor", yn), ("floor only", fl)):
            E[k].append(band_energies(x, M).mean((0, 1, 3)).cpu())  # [bands], mean over batch, ch, frames
E = {k: torch.stack(v).mean(0) for k, v in E.items()}
print("band (Hz) | model-target dB with floor | without floor | floor share of model %")
for i in range(0, 32, 2):
    print(f"{centers[i]:7.0f} | {10*math.log10(E['with floor'][i]/E['target'][i]):+6.1f} | {10*math.log10(E['no floor'][i]/E['target'][i]):+6.1f} | {100*E['floor only'][i]/E['with floor'][i]:5.1f}")

# (2) level bias of the loss under misalignment (synthetic target from the same model)
def perturbed(kind):
    m = copy.deepcopy(model)
    with torch.no_grad():
        if kind == "unison re-drawn":
            gg = torch.Generator().manual_seed(999)
            mag = 0.2 + 1.8 * torch.rand(88, cfg.n_modes - 1, generator=gg) ** 2
            sign = torch.where(torch.rand(88, cfg.n_modes - 1, generator=gg) < 0.5, -1.0, 1.0)
            m.physics.raw_unison.copy_((5.0 * torch.atanh(mag * sign / 5.0)).to(dev))
        elif kind == "bass B x1.3":
            m.physics.raw_log_B[:30] += 1.5 * torch.atanh(torch.tensor(math.log(1.3) / 1.5))
        elif kind == "none":
            pass
    return m
gains = np.arange(-4, 2.01, 0.25)
for kind in ("none", "unison re-drawn", "bass B x1.3"):
    tm = perturbed(kind)
    ls, lm = np.zeros(len(gains)), np.zeros(len(gains))
    with torch.no_grad():
        for b in batches:
            n, s = b["audio"].shape[-1], int(b["loss_start"][0])
            t = tm(b, n, residual=False, generator=torch.Generator(device=dev).manual_seed(7), floor=True)["audio"][..., s:]
            y = model(b, n, residual=False, generator=torch.Generator(device=dev).manual_seed(8), floor=True)["audio"][..., s:]
            for j, g in enumerate(gains):
                ls[j] += float(recon(y * 10 ** (g / 20), t)) / len(batches)
                lm[j] += float(mel(y * 10 ** (g / 20), t)) / len(batches)
    print(f"target = {kind:16s}: MR-STFT optimum at {gains[np.argmin(ls)]:+.2f} dB (loss {ls.min():.4f}, at 0 dB {ls[np.argmin(np.abs(gains))]:.4f});"
          f" log-mel optimum at {gains[np.argmin(lm)]:+.2f} dB")
