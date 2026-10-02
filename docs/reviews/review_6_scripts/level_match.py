"""How much of the validation distance is the recording's level per excerpt / per piece (a nuisance no MIDI predicts)."""
import os, sys, collections
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.6,max_split_size_mb:256")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
import numpy as np
import torch
from pianonn.data import MaestroSegments
from pianonn.losses import PianoLoss
from pianonn.render import load_model
from pianonn.train import fixed_batches, onsets_of

N = 96
dev = torch.device("cuda")
torch.cuda.set_per_process_memory_fraction(0.6)
rows = {}
for label, CK in (("phase4", "runs/phase4/train_run/train/last.pt"), ("phase5", "runs/phase5/train_run/train/last.pt"),
                  ("phase6", "runs/phase6/train_run/train/last.pt")):
    model = load_model(CK, device=dev)
    cfg, sr = model.cfg, model.cfg.sample_rate
    val = MaestroSegments("data/maestro24k", "validation", cfg, 2.0, 1.0, 12.0, length=N, deterministic=True, years=[2018], seed=1)
    batches = fixed_batches(val, N, 4, dev)
    loss = PianoLoss(sr, level_weight=0.5).to(dev)
    has = model.strike_on
    P, Q0, Q1, T, db, piece = [], [], [], [], [], []
    with torch.no_grad():
        for b in batches:
            n, s = b["audio"].shape[-1], int(b["loss_start"][0])
            def r(seed, strike):
                model.strike_on = strike and has
                return model(b, n, residual=False, generator=torch.Generator(device=dev).manual_seed(seed))["audio"][..., s:].float()
            p, t = r(0, False), b["audio"][..., s:].float()
            P.append(p); T.append(t); Q0.append(r(0, True)); Q1.append(r(1, True))
            db += (10 * torch.log10(p.pow(2).mean((1, 2)) / t.pow(2).mean((1, 2)))).tolist()
            piece += b["piece"].tolist()
        db, piece = np.array(db), np.array(piece)
        pm = {p: db[piece == p].mean() for p in set(piece)}
        g_ex = torch.tensor(10 ** (-db / 20), device=dev, dtype=torch.float32)
        g_pc = torch.tensor([10 ** (-pm[p] / 20) for p in piece], device=dev, dtype=torch.float32)

        def score(preds, gains=None, tgts=None):
            out, k = collections.defaultdict(list), 0
            for i, b in enumerate(batches):
                s = int(b["loss_start"][0])
                p = preds[i]
                if gains is not None:
                    p = p * gains[k: k + len(p), None, None]
                t = loss.terms(p, (tgts or T)[i], *onsets_of(b, s, sr))
                for kk, v in t.items():
                    out[kk] += v.tolist()
                k += len(p)
            m = {kk: np.array(v) for kk, v in out.items()}
            m["total"] = m["band"] + 0.25 * m["fine"] + 0.5 * m["attack"] + 0.5 * m["level"]
            return m
        res = {"as rendered": score(P), "level matched per piece (13 gains)": score(P, g_pc),
               "level matched per excerpt": score(P, g_ex), "model vs its own other draw (variation on)": score(Q1, None, Q0)}
    rows[label] = res
    print(f"\n{label}  ({CK})")
    for name, m in res.items():
        print(f"   {name:44s} total {m['total'].mean():.4f} | band {m['band'].mean():.4f} fine {m['fine'].mean():.4f} attack {m['attack'].mean():.4f} level {m['level'].mean():.4f}")
    del model
    torch.cuda.empty_cache()

print("\npaired differences on the 96 validation excerpts (mean +- 2 se):")
for a, b in (("phase5", "phase4"), ("phase6", "phase5"), ("phase6", "phase4")):
    for name in ("as rendered", "level matched per excerpt"):
        d = rows[a][name]["total"] - rows[b][name]["total"]
        print(f"   {a} - {b}, {name:28s} {d.mean():+.4f} +- {2 * d.std() / np.sqrt(len(d)):.4f}")
