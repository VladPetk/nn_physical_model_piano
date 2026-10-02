"""Where a training step's time goes: one batch of 8 x (1 s warm-up + 2 s) as phase 6 trained, the residual on.

Times each part of the forward pass (GPU synchronised), the loss, the backward pass and the data fetch, and counts
the oscillator bank's work (notes x oscillators x samples).

    .venv/Scripts/python.exe docs/reviews/review_6_scripts/step_profile.py
"""
import os
import sys
import time
import collections

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.6,max_split_size_mb:256")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
import torch  # noqa: E402

import pianonn.synth as synth  # noqa: E402
from pianonn.data import MaestroSegments  # noqa: E402
from pianonn.losses import OnsetLoss, PianoLoss  # noqa: E402
from pianonn.render import load_model  # noqa: E402
from pianonn.train import fixed_batches, onsets_of  # noqa: E402

CK, N_BATCH, B = "runs/phase6/train_run/train/last.pt", 4, 8
dev = torch.device("cuda")
torch.cuda.set_per_process_memory_fraction(0.6)
model = load_model(CK, device=dev)
model.train()
model.strike_on = False
cfg, sr = model.cfg, model.cfg.sample_rate

T = collections.defaultdict(float)
work = collections.defaultdict(float)


def sync():
    torch.cuda.synchronize()
    return time.perf_counter()


def timed(obj, name, label):
    f = getattr(obj, name)

    def g(*a, **k):
        t = sync()
        r = f(*a, **k)
        T[label] += sync() - t
        return r
    setattr(obj, name, g)


_osc = synth.osc_bank


def osc(freq, *a, **k):
    t = sync()
    y = _osc(freq, *a, **k)
    T["  of which the oscillator bank"] += sync() - t
    work["calls"] += 1
    work["elements"] += freq.shape[0] * freq.shape[1] * y.shape[-1]
    work["note-chunks"] += freq.shape[0]
    return y


synth.osc_bank = osc
import pianonn.oscbank as oscbank  # noqa: E402

_bw = oscbank.OscBank.backward


def bank_backward(ctx, gy):
    t = sync()
    r = _bw(ctx, gy)
    T["oscillator bank (backward)"] += sync() - t
    return r


oscbank.OscBank.backward = staticmethod(bank_backward)
timed(model, "render_strings", "strings (forward)")
timed(model, "render_noise", "noises (forward)")
timed(model, "render_impulses", "impulses (forward)")
timed(model.context, "forward", "residual network (forward)")
timed(model.symp, "forward", "sympathetic bank (forward)")
timed(model.physics, "modes", "physics tables -> modes (forward)")

t = time.perf_counter()
ds = MaestroSegments("data/maestro24k", "train", cfg, 2.0, 1.0, 12.0, length=N_BATCH * B, deterministic=True,
                     years=[2018], seed=3)
items = [ds[i] for i in range(B)]
T_data = (time.perf_counter() - t)
batches = fixed_batches(ds, N_BATCH * B, B, dev)
loss = PianoLoss(sr, level_weight=0.5).to(dev)
onset = OnsetLoss(sr).to(dev) if "OnsetLoss" in dir() else None
params = [p for p in model.parameters() if p.requires_grad]

tot = collections.defaultdict(float)
notes = []
for bi, b in enumerate(batches):
    for k in list(T):
        T[k] = 0.0
    for k in list(work):
        work[k] = 0.0
    n, s = b["audio"].shape[-1], int(b["loss_start"][0])
    t0 = sync()
    out = model(b, n, residual=True)
    t1 = sync()
    pred, tgt = out["audio"][..., s:].float(), b["audio"][..., s:].float()
    val, _ = loss(pred, tgt, *onsets_of(b, s, sr))
    t2 = sync()
    for p in params:
        p.grad = None
    val.backward()
    t3 = sync()
    if bi == 0:  # the first batch warms up cuDNN and the allocator
        continue
    tot["forward"] += t1 - t0
    tot["loss"] += t2 - t1
    tot["backward"] += t3 - t2
    for k, v in T.items():
        tot[k] += v
    for k, v in work.items():
        tot["work_" + k] += v
    notes.append(int(b["mask"].sum()))
    tot["mem"] = max(tot["mem"], torch.cuda.max_memory_allocated() / 2 ** 30)

m = N_BATCH - 1
print(f"checkpoint {CK}; batch {B} x {n / sr:.1f} s = {B * n / sr:.0f} s of audio; mean of {m} batches\n")
print(f"notes in a batch (sounding or ringing in from the 12 s before): {sum(notes) / m:.0f}")
print(f"oscillator bank: {tot['work_calls'] / m:.0f} calls, {tot['work_elements'] / m / 1e9:.2f} e9 oscillator-samples "
      f"(notes x oscillators x samples) per step")
n_par = sum(p.numel() for p in model.parameters())
print(f"parameters: {n_par / 1e6:.2f} million; peak GPU memory {tot['mem']:.1f} GB\n")
step = (tot["forward"] + tot["loss"] + tot["backward"]) / m
print(f"{'part':44s} s per step   share")
for k in ("forward", "loss", "backward"):
    print(f"{k:44s} {tot[k] / m:8.2f}   {100 * tot[k] / m / step:5.1f} %")
print(f"{'step (without the optimiser and the data)':44s} {step:8.2f}")
print("\ninside the forward pass:")
for k in sorted((k for k in tot if "forward)" in k or "of which" in k), key=lambda k: -tot[k]):
    print(f"   {k:41s} {tot[k] / m:8.2f}   {100 * tot[k] / tot['forward']:5.1f} % of the forward pass")
k = "oscillator bank (backward)"
print(f"\ninside the backward pass:\n   {k:41s} {tot[k] / m:8.2f}   {100 * tot[k] / tot['backward']:5.1f} % of the backward pass")
print(f"\ndata: {T_data / B * 1000:.0f} ms per excerpt on one CPU thread ({T_data:.2f} s per batch of {B} without workers)")
e = tot["work_elements"] / m
print(f"\noscillator-samples per second of step time: {e / step / 1e9:.1f} e9")
