"""(1) How old are the notes the loss weighs: per (octave group, 20 ms frame) cell of the loss window, the share of the
physics' expected energy from notes of each age. (2) How coherent is the gradient between independent training batches,
per parameter group (cosine between two batches' gradients ~ signal / (signal + noise) in power)."""
import os, sys, math, itertools
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.6,max_split_size_mb:256")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
import numpy as np
import torch
from pianonn.data import MaestroSegments
from pianonn.losses import PianoLoss
from pianonn.render import load_model
from pianonn.train import fixed_batches, onsets_of, ENV_PARAMS

CK = "runs/phase6/train_run/train/last.pt"
K, BS = int(sys.argv[1]) if len(sys.argv) > 1 else 12, 4
dev = torch.device("cuda")
torch.cuda.set_per_process_memory_fraction(0.7)
state = torch.load(CK, map_location="cpu")
model = load_model(CK, device=dev)
cfg = model.cfg
sr = cfg.sample_rate
model.train(); model.strike_on = False
train = MaestroSegments("data/maestro24k", "train", cfg, 2.0, 1.0, 12.0, length=K * BS, deterministic=True, years=[2018], seed=77)
gain = dict(zip(state["piece_gain"]["ids"], state["piece_gain"]["db"].tolist())) if "piece_gain" in state else {}
print("piece gains in the checkpoint:", len(gain), "sd", np.std(list(gain.values())) if gain else None)
batches = fixed_batches(train, K * BS, BS, dev)
loss = PianoLoss(sr, level_weight=0.5).to(dev)

captured = {}
orig = model.context.features
def hook(*a, **k):
    f = orig(*a, **k)
    captured["f"] = f
    return f
model.context.features = hook

GROUPS = {
    "residual (context.*)": lambda n: n.startswith("context."),
    "residual curve head": lambda n: n.startswith("context.curve_head"),
    "strings' decay (ENV_PARAMS)": lambda n: n.startswith(ENV_PARAMS),
    "per-key level gain_db": lambda n: n == "physics.gain_db",
    "partial_gain table": lambda n: n == "physics.partial_gain",
    "colour table": lambda n: n == "physics.color",
    "hammer (tc, order, strike)": lambda n: n.startswith(("physics.raw_log_tc", "physics.raw_order", "physics.raw_strike", "physics.raw_tc_vel")),
    "physics, all": lambda n: n.startswith("physics."),
    "body FIR": lambda n: n == "room.body",
    "room, all": lambda n: n.startswith("room."),
    "noise, all": lambda n: n.startswith("noise."),
}
for p in model.parameters():
    p.requires_grad_(True)
grads = {g: [] for g in GROUPS}
age_edges = [0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 99]
age_share = np.zeros(len(age_edges) - 1)   # mean over cells of the energy share from notes of each age
dom_age = np.zeros(len(age_edges) - 1)     # cells by the age of their strongest note
n_cells = 0
vals = []
lev = []
for bi, b in enumerate(batches):
    n, s = b["audio"].shape[-1], int(b["loss_start"][0])
    g = torch.Generator(device=dev).manual_seed(bi)
    model.zero_grad(set_to_none=True)
    out = model(b, n, residual=True, generator=g)
    audio = out["audio"]
    if gain:
        db = torch.tensor([gain.get(train.pieces[int(i)]["id"], 0.0) for i in b["piece"]], device=dev)
        audio = audio * (10 ** (db / 20))[:, None, None]
    pred, tgt = audio[..., s:].float(), b["audio"][..., s:].float()
    l, parts = loss(pred, tgt, *onsets_of(b, s, sr))
    l.backward()
    vals.append(float(l))
    lev.append(float(10 * torch.log10(pred.pow(2).mean() / tgt.pow(2).mean())))
    named = dict(model.named_parameters())
    for gname, sel in GROUPS.items():
        v = [p.grad.flatten() for nme, p in named.items() if sel(nme) and p.grad is not None]
        grads[gname].append(torch.cat(v).double().cpu() if v else None)
    # ages
    f = captured["f"]
    own, tau, sounding = f["own"], f["tau"], f["sounding"]  # [B,N,G,C], [B,N,C], [B,N,C]
    e = (10 ** (2 * own) - 1e-10).clamp(min=0) * sounding[:, :, None, :]
    Cn = tau.shape[-1]
    t = torch.arange(Cn, device=dev) * (cfg.res_control * cfg.hop / sr)
    inwin = t >= s / sr
    tot = e.sum(1)  # [B,G,C]
    ok = (tot > 0) & inwin[None, None, :]
    for i in range(len(age_edges) - 1):
        m = ((tau >= age_edges[i]) & (tau < age_edges[i + 1]))[:, :, None, :]
        share = (e * m).sum(1) / tot.clamp(min=1e-30)
        age_share[i] += float(share[ok].sum())
    idx = e.argmax(1)  # [B,G,C]
    dage = torch.stack([tau[bb].gather(0, idx[bb]) for bb in range(idx.shape[0])])
    for i in range(len(age_edges) - 1):
        dom_age[i] += float((((dage >= age_edges[i]) & (dage < age_edges[i + 1])) & ok).sum())
    n_cells += float(ok.sum())
    del out, audio, pred, tgt, l
    torch.cuda.empty_cache()

print(f"\nloss over {K} training batches of {BS}: mean {np.mean(vals):.4f} sd {np.std(vals):.4f}")
print("\nage of the notes in the loss window's cells (octave group x 20 ms; the physics' expected energies):")
print("   age (s)          share of the cells' energy     cells whose strongest note has this age")
for i in range(len(age_edges) - 1):
    print(f"   {age_edges[i]:>5g} .. {age_edges[i+1]:<5g}        {age_share[i] / n_cells:6.1%}                      {dom_age[i] / n_cells:6.1%}")

print(f"\ngradient coherence between independent batches of {BS} excerpts ({K} batches, {K*(K-1)//2} pairs):")
print("   group                              n params   mean cos    sd of cos   |mean grad| / mean |grad|   batches for SNR 1")
for gname, gs in grads.items():
    gs = [g for g in gs if g is not None]
    if len(gs) < 2:
        continue
    G = torch.stack(gs)
    norms = G.norm(dim=1)
    cs = [float(G[i] @ G[j] / (norms[i] * norms[j] + 1e-300)) for i, j in itertools.combinations(range(len(gs)), 2)]
    c = float(np.mean(cs))
    ratio = float(G.mean(0).norm() / norms.mean())
    need = (1 - c) / c if c > 0 else float("inf")
    print(f"   {gname:34s} {G.shape[1]:8d}   {c:+.4f}     {np.std(cs):.4f}      {ratio:.3f}                     {need:8.1f}")

lev = np.array(lev)
print("batch level error (model - recording, dB, energy):", np.round(lev, 2))
for gname in ("residual (context.*)", "residual curve head", "physics, all", "per-key level gain_db", "room, all"):
    G = torch.stack(grads[gname])
    U, S, Vh = torch.linalg.svd(G, full_matrices=False)
    proj = (G @ Vh[0]).numpy()
    share = float(S[0] ** 2 / (S ** 2).sum())
    r = float(np.corrcoef(proj, lev)[0, 1])
    R = G - (G @ Vh[0])[:, None] * Vh[0][None]
    nr = R.norm(dim=1)
    cs = [float(R[i] @ R[j] / (nr[i] * nr[j])) for i, j in itertools.combinations(range(len(R)), 2)]
    print(f"   {gname:28s} first direction holds {share:5.1%} of the gradients' power (uncentred); its sign vs the batch's level error: r = {r:+.2f}; "
          f"mean cos of the rest {np.mean(cs):+.4f} (sd {np.std(cs):.3f})")
