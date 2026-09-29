"""Benchmark one training step at the real config on the GPU, with a MAESTRO-like note density."""
import time, sys
import numpy as np, torch
from pianonn import NeuralPhysicalPiano, PianoConfig
from pianonn.data import _perf_from_notes, collate, PEDAL_CCS
from pianonn.losses import MultiResolutionSTFTLoss
from pianonn.train import param_groups

dev = torch.device("cuda")
symp = len(sys.argv) > 1 and sys.argv[1] == "symp"
cfg = PianoConfig(use_sympathetic=symp)
model = NeuralPhysicalPiano(cfg).to(dev)
warm, seg, lookback = 1.0, 2.0, 4.0
sr = cfg.sample_rate
n = int((warm + seg) * sr)
rng = np.random.default_rng(0)

def perf(density=8.0):  # notes per second (MAESTRO averages ~6-10 in dense music)
    T = warm + seg + lookback
    k = int(density * T)
    on = np.sort(rng.uniform(-lookback, warm + seg, k))
    notes = np.stack([rng.integers(30, 100, k), on, on + rng.uniform(0.05, 1.5, k), rng.integers(20, 110, k)], 1)
    pedals = {}
    for name in PEDAL_CCS:
        t = np.sort(rng.uniform(0, warm + seg, 6)) if name == "sustain" else np.zeros(0)
        pedals[name + "_t"], pedals[name + "_v"] = t, rng.choice([0.0, 127.0], len(t))
    p = _perf_from_notes(notes, pedals, 0.0, warm + seg, lookback, n // cfg.hop + 2, cfg)
    p["condition"] = torch.tensor(3)
    return p

B = int(sys.argv[2]) if len(sys.argv) > 2 else 4
batch = collate([perf() for _ in range(B)])
batch = {k: v.to(dev) for k, v in batch.items()}
print("notes per example:", batch["pitch"].shape[1])
target = torch.randn(B, n, device=dev) * 0.01
loss_fn = MultiResolutionSTFTLoss()
opt = torch.optim.Adam(param_groups(model, 1e-3))
s = int(warm * sr)
torch.cuda.reset_peak_memory_stats()
for i in range(4):
    torch.cuda.synchronize(); t0 = time.time()
    out = model(batch, n)["audio"]
    loss = loss_fn(out[:, s:], target[:, s:]) + model.physics.regularizer()
    opt.zero_grad(); loss.backward(); opt.step()
    torch.cuda.synchronize()
    print(f"step {i}: {time.time() - t0:.2f}s  peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
