"""Listening material: held-out excerpts rendered by several checkpoints, played in turn after the recording.

    python scripts/ab_render.py data/maestro24k --years 2018 --out runs/round2/listen \\
        --model trial=runs/round1_trial/main/best.pt:residual --model new=runs/round2/a_residual/last.pt:residual

Each ``--model`` is ``label=checkpoint:physics|residual``, with ``+symp`` (e.g. ``physics+symp``) to switch the
sympathetic string bank on. For each excerpt (the same ones as
``scripts/evaluate.py``: test split, seed 12) it writes the recording and every render, and ``<i>_ab.wav``:
the recording, then each model in the given order, with half a second of silence between. Nothing is
normalised: level is part of what is being judged. With ``--demo``, the MIDI file is rendered by each model too.

With ``--pick-from K`` the excerpts are chosen from K candidates, spread evenly over their mean velocity (soft
to loud), among candidates with at least 8 notes. ``--flac`` writes FLAC instead of WAV and skips ``<i>_ab``.
``manifest.json`` lists every excerpt (piece, start, mean velocity, pedal fraction, note count) and each
clip's level (dB RMS after a 20 Hz high-pass), for a player that switches between versions.
"""

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

from pianonn.config import year_to_condition  # noqa: E402
from pianonn.data import MaestroSegments, load_midi  # noqa: E402
from pianonn.losses import highpass  # noqa: E402
from pianonn.render import load_model, render_notes  # noqa: E402
from pianonn.train import collate, to_device  # noqa: E402


def describe(ds, item, cfg):
    """Piece, start (s), mean velocity, pedal fraction and note count of a MaestroSegments item."""
    n = item["audio"].shape[-1]
    m = (item["onset"] >= 0) & (item["onset"] < n / cfg.sample_rate)  # notes struck inside the excerpt
    H = int(item["hist_frames"])
    sus = item["sustain"][H: H + n // cfg.hop].float()
    return {"piece": ds.pieces[int(item["piece"])]["id"], "start_s": round(int(item["start"]) / cfg.sample_rate, 1),
            "velocity": round(float(item["velocity"][m].float().mean()), 1) if m.any() else None,
            "pedal": round(float((sus >= 0.5).float().mean()), 2), "notes": int(m.sum())}


def level_db(x, sr):
    return 10 * math.log10(max(1e-20, float(highpass(x[None], sr).pow(2).mean())))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data")
    ap.add_argument("--years", type=int, nargs="*", default=[2018])
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", action="append", required=True, help="label=checkpoint:physics|residual")
    ap.add_argument("--excerpts", type=int, default=2)
    ap.add_argument("--pick-from", type=int, default=0, help="choose the excerpts from this many candidates, soft to loud")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=12)
    ap.add_argument("--flac", action="store_true")
    ap.add_argument("--demo", help="also render this MIDI file with every model")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.6)
    os.makedirs(args.out, exist_ok=True)
    specs = []
    for m in args.model:
        label, rest = m.split("=", 1)
        ckpt, mode = rest.rsplit(":", 1)
        mode, *flags = mode.split("+")
        specs.append((label, ckpt + ("+symp" if "symp" in flags else ""), mode == "residual"))
    models = {}
    for label, key, _ in specs:  # key: the checkpoint, "+symp" with the sympathetic bank switched on
        if key not in models:
            symp = key.endswith("+symp")
            models[key] = load_model(key.removesuffix("+symp"), device=dev, **({"use_sympathetic": True} if symp else {}))
    cfg = next(iter(models.values())).cfg
    sr = cfg.sample_rate
    ext = "flac" if args.flac else "wav"
    n_cand = max(args.pick_from, args.excerpts)
    ds = MaestroSegments(args.data, args.split, cfg, args.seconds - 1.0, 1.0, 12.0, length=n_cand, deterministic=True,
                         years=args.years, seed=args.seed)
    cands = [(i, it, describe(ds, it, cfg)) for i, it in ((i, ds[i]) for i in range(n_cand))]
    if args.pick_from:
        ok = sorted((c for c in cands if c[2]["notes"] >= 8), key=lambda c: c[2]["velocity"])
        idx = np.unique(np.round(np.linspace(0, len(ok) - 1, args.excerpts)).astype(int))
        chosen = [ok[k] for k in idx]
    else:
        chosen = cands[: args.excerpts]
    order = ["recording"] + [label for label, _, _ in specs]
    gap = np.zeros((sr // 2, cfg.channels), dtype=np.float32)
    manifest = {"sample_rate": sr, "seconds": args.seconds, "split": args.split, "format": ext, "order": order,
                "models": {label: {"checkpoint": ckpt, "residual": res} for label, ckpt, res in specs}, "excerpts": []}
    for k, (i, item, info) in enumerate(chosen):
        b = to_device(collate([item]), dev)
        n = b["audio"].shape[-1]
        clips = [b["audio"][0]]
        for label, ckpt, residual in specs:
            with torch.no_grad():
                y = models[ckpt](b, n, residual=residual, block_seconds=4.0,
                                 generator=torch.Generator(device=dev).manual_seed(0))["audio"][0]
            clips.append(y.clamp(-1, 1))
        levels = {}
        for label, c in zip(order, clips):
            sf.write(os.path.join(args.out, f"{k}_{label}.{ext}"), c.T.cpu().numpy(), sr)
            levels[label] = round(level_db(c.float(), sr), 2)
        if not args.flac:
            sf.write(os.path.join(args.out, f"{k}_ab.wav"),
                     np.concatenate([x for c in clips for x in (c.T.cpu().numpy(), gap)]), sr)
        manifest["excerpts"].append({"index": k, **info, "level_db": levels})
        print(k, info, flush=True)
    if args.demo:
        notes, pedals = load_midi(args.demo)
        cond = year_to_condition(args.years[0])
        for label, ckpt, residual in specs:
            audio = render_notes(models[ckpt], notes, pedals, cond, residual=residual)
            sf.write(os.path.join(args.out, f"demo_{label}.{ext}"), np.clip(audio.T, -1, 1), sr)
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    with open(os.path.join(args.out, "README.md"), "w", encoding="utf-8") as f:
        f.write(f"# Listening: {len(chosen)} {args.split} excerpts of {args.seconds:.0f} s\n\n"
                + ("`<i>_ab.wav` plays, with half a second between: " + ", ".join(order) + ".\n\n" if not args.flac else "")
                + "\n".join(f"- {label}: `{ckpt}`, {'with' if res else 'without'} the residual" for label, ckpt, res in specs)
                + "\n\nExcerpts (manifest.json has the levels):\n\n"
                + "\n".join(f"{e['index']}. `{e['piece']}` at {e['start_s']:.0f} s: velocity {e['velocity']}, "
                            f"pedal {100 * e['pedal']:.0f} %, {e['notes']} notes" for e in manifest["excerpts"]) + "\n")
    print("wrote", args.out, "order:", ", ".join(order))


if __name__ == "__main__":
    main()
