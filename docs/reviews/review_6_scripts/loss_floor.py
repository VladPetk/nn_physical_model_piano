"""Calibrating the training loss: what it reads for the model against the recording, for the model against itself
(another draw), for unrelated music, and where its mass sits (cell level, time since the last onset)."""
import os, sys, math, json
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.6,max_split_size_mb:256")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
import numpy as np
import torch
from pianonn.data import MaestroSegments
from pianonn.losses import PianoLoss, highpass
from pianonn.render import load_model
from pianonn.train import fixed_batches, onsets_of

CK = sys.argv[1] if len(sys.argv) > 1 else "runs/phase6/train_run/train/last.pt"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 96
dev = torch.device("cuda")
torch.cuda.set_per_process_memory_fraction(0.6)
model = load_model(CK, device=dev)
cfg = model.cfg
sr = cfg.sample_rate
val = MaestroSegments("data/maestro24k", "validation", cfg, 2.0, 1.0, 12.0, length=N, deterministic=True, years=[2018], seed=1)
batches = fixed_batches(val, N, 4, dev)
loss = PianoLoss(sr, level_weight=0.5).to(dev)
has_strike = model.strike_on
print("checkpoint", CK, "strike variation in config:", has_strike, "examples", N, "pieces", len(val.pieces))


@torch.no_grad()
def render(b, seed, strike):
    model.strike_on = strike and has_strike
    g = torch.Generator(device=dev).manual_seed(seed)
    n, s = b["audio"].shape[-1], int(b["loss_start"][0])
    return model(b, n, residual=False, generator=g)["audio"][..., s:].float()


def terms(pred, tgt, b):
    s = int(b["loss_start"][0])
    t = loss.terms(pred, tgt, *onsets_of(b, s, sr))
    return {k: v.cpu().numpy() for k, v in t.items()}


def cells(pred, tgt):
    """Per-cell signed log10 difference, the target's band power (per sample) and the floor: [B*ch, bands, frames]."""
    B = pred.shape[0]
    p = highpass(pred, sr).reshape(-1, pred.shape[-1])
    t = highpass(tgt, sr).reshape(-1, tgt.shape[-1])
    D, L, Fl, Lp = [], [], [], []
    for gi, n, n_bins in loss.groups:
        m = getattr(loss, f"mask{gi}")
        Ep, Et = loss._bands(p, n, m, n_bins, loss.hop), loss._bands(t, n, m, n_bins, loss.hop)
        eps = loss._eps(n, m)
        norm = 2 / (0.375 * n * n)
        D.append(torch.log10(Ep + eps) - torch.log10(Et + eps))
        L.append(10 * torch.log10(Et * norm + 1e-30))
        Lp.append(10 * torch.log10(Ep * norm + 1e-30))
        Fl.append((10 * torch.log10(eps * norm)).expand(Et.shape[0], -1, Et.shape[-1]))
    return torch.cat(D, 1), torch.cat(L, 1), torch.cat(Fl, 1), torch.cat(Lp, 1)


acc = {}
def add(name, t):
    for k, v in t.items():
        acc.setdefault(name, {}).setdefault(k, []).append(v)

rel_edges = [0, 10, 20, 30, 40, 50, 60, 200]
dt_edges = [0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 99]
fl_edges = [-200, 0, 10, 20, 30, 40, 200]
stat = {k: np.zeros((len(e) - 1, 3)) for k, e in (("rel", rel_edges), ("dt", dt_edges), ("fl", fl_edges))}  # n, sum|d|, sum d
band_dt = {}
piece_rows = []
all_T = []
for bi, b in enumerate(batches):
    s = int(b["loss_start"][0])
    T = b["audio"][..., s:].float()
    P0, P1 = render(b, 0, False), render(b, 1, False)
    Q0, Q1 = render(b, 0, True), render(b, 1, True)
    add("A model(var off) vs recording", terms(P0, T, b))
    add("B model vs model, other noise seed (var off)", terms(P1, P0, b))
    add("C model vs model, other draw (var on)", terms(Q1, Q0, b))
    add("D model(var on) vs recording", terms(Q0, T, b))
    sh = int(0.005 * sr)
    add("F recording vs itself 5 ms late", terms(torch.roll(T, sh, -1), T, b))
    add("G recording vs itself +1 dB", terms(T * 10 ** (1 / 20), T, b))
    add("H recording L/R swapped", terms(T.flip(1), T, b))
    all_T.append((T.cpu(), b))
    # where the band term's mass sits (A)
    d, L, Fl, Lp = cells(P0, T)
    ch = d.shape[0] // T.shape[0]
    rel = L.amax(1, keepdim=True) - L  # dB below the frame's loudest band
    over = L - Fl  # dB over the loss's floor
    t = torch.arange(d.shape[-1], device=dev) * loss.hop / sr
    on = (b["onset"] - s / sr)
    on = torch.where(b["mask"], on, torch.full_like(on, -1e9))
    dt = (t[None, None, :] - on[..., None])
    dt = torch.where(dt >= 0, dt, torch.full_like(dt, 1e9)).amin(1)  # [B, frames]
    dt = dt.repeat_interleave(ch, 0)[:, None, :].expand_as(d)
    for key, x, edges in (("rel", rel, rel_edges), ("dt", dt, dt_edges), ("fl", over, fl_edges)):
        for i in range(len(edges) - 1):
            mk = (x >= edges[i]) & (x < edges[i + 1])
            stat[key][i] += [float(mk.sum()), float(d.abs()[mk].sum()), float(d[mk].sum())]
    # per excerpt: broadband and octave-band level difference (model - recording, energy), for the per-piece spread
    cen = loss.centers.cpu().numpy()
    Ep = (10 ** (Lp / 10)).reshape(T.shape[0], ch, d.shape[1], -1).sum((1, 3))
    Et = (10 ** (L / 10)).reshape(T.shape[0], ch, d.shape[1], -1).sum((1, 3))
    for j in range(T.shape[0]):
        row = {"piece": int(b["piece"][j]), "broad": float(10 * torch.log10(Ep[j].sum() / Et[j].sum()))}
        for lo in (62.5, 125, 250, 500, 1000, 2000, 4000):
            sel = torch.as_tensor((cen >= lo) & (cen < 2 * lo), device=dev)
            row[f"o{lo:g}"] = float(10 * torch.log10(Ep[j][sel].sum() / Et[j][sel].sum()))
        piece_rows.append(row)

# E: the recording against another excerpt of the validation set (level-matched), an "unrelated piano music" anchor
Ts = torch.cat([t for t, _ in all_T])
perm = torch.roll(torch.arange(len(Ts)), 7)
k = 0
for T, b in all_T:
    other = Ts[perm[k: k + len(T)]].to(dev)
    T = T.to(dev)
    g = (T.pow(2).mean((1, 2), keepdim=True) / other.pow(2).mean((1, 2), keepdim=True).clamp(min=1e-12)).sqrt()
    add("E another excerpt, level-matched, vs recording", terms(other * g, T, b))
    k += len(T)
    # white noise at the recording's level against the recording, and against other white noise
    w1 = torch.randn_like(T) * T.pow(2).mean((1, 2), keepdim=True).sqrt()
    w2 = torch.randn_like(T) * T.pow(2).mean((1, 2), keepdim=True).sqrt()
    add("I white noise vs white noise (same level)", terms(w1, w2, b))

print("\nper-term means over excerpts (band/fine/attack/level in log10 power, 1 = 10 dB); total = band + .25 fine + .5 attack + .5 level")
for name, d in acc.items():
    m = {k: np.concatenate(v) for k, v in d.items()}
    tot = m["band"] + 0.25 * m["fine"] + 0.5 * m["attack"] + 0.5 * m["level"]
    print(f"  {name:52s} total {tot.mean():.4f} (sd over excerpts {tot.std():.3f}) | " + " ".join(f"{k} {v.mean():.4f}" for k, v in m.items()))

tot_n = stat["rel"][:, 0].sum()
tot_abs = stat["rel"][:, 1].sum()
for key, edges, label in (("rel", rel_edges, "dB below the frame's loudest band"), ("fl", fl_edges, "dB over the loss floor (target)"),
                          ("dt", dt_edges, "s since the last onset (any note)")):
    print(f"\nband term (A) by {label}: share of cells, mean |d| (dB), mean signed d (dB, model - recording), share of the band loss")
    for i in range(len(edges) - 1):
        n, sa, sd = stat[key][i]
        if n:
            print(f"   {edges[i]:>6g} .. {edges[i+1]:<6g}  cells {n / tot_n:6.1%}   |d| {10 * sa / n:5.2f}   signed {10 * sd / n:+5.2f}   loss share {sa / tot_abs:6.1%}")

import collections
by = collections.defaultdict(list)
for r in piece_rows:
    by[r["piece"]].append(r)
keys = [k for k in piece_rows[0] if k != "piece"]
print("\nper piece: mean level difference model - recording (dB, energy over the 2 s), n excerpts")
pm = []
for p, rows in sorted(by.items()):
    m = {k: np.mean([r[k] for r in rows]) for k in keys}
    pm.append(m)
    print(f"   piece {p:3d} n {len(rows):2d}  " + " ".join(f"{k} {m[k]:+5.1f}" for k in keys))
print("   sd across pieces:      " + " ".join(f"{k} {np.std([m[k] for m in pm]):5.2f}" for k in keys))
print("   sd across excerpts:    " + " ".join(f"{k} {np.std([r[k] for r in piece_rows]):5.2f}" for k in keys))
resid = {k: np.std([r[k] - np.mean([q[k] for q in by[r['piece']]]) for r in piece_rows]) for k in keys}
print("   sd within pieces:      " + " ".join(f"{k} {resid[k]:5.2f}" for k in keys))
