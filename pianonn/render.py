"""Render a MIDI file with a trained (or untrained, physics-prior-only) model."""

import argparse

import numpy as np
import torch

from .config import PianoConfig, year_to_condition
from .data import _perf_from_notes, load_midi
from .synth import NeuralPhysicalPiano


def load_model(ckpt=None, **overrides):
    if ckpt:
        state = torch.load(ckpt, map_location="cpu")
        model = NeuralPhysicalPiano(PianoConfig(**{**state["cfg"], **overrides}))
        model.load_state_dict(state["model"])
    else:
        model = NeuralPhysicalPiano(PianoConfig(**overrides))
    return model.eval()


@torch.no_grad()
def render_notes(model, notes, pedals, condition=0, tail=3.0, block_seconds=2.0, seed=0):
    """``notes[N, 4]`` = (pitch, onset, offset, velocity) in seconds; returns float32 audio."""
    cfg = model.cfg
    duration = float(notes[:, 2].max()) + tail if len(notes) else tail
    n_samples = int(duration * cfg.sample_rate)
    perf = _perf_from_notes(notes, pedals, 0.0, duration, 0.0, model.n_frames(n_samples), cfg)
    perf = {k: v[None] for k, v in perf.items()}
    perf["mask"] = torch.ones_like(perf["pitch"], dtype=torch.bool)
    perf["condition"] = torch.tensor([condition])
    gen = torch.Generator().manual_seed(seed)
    return model(perf, n_samples, block_seconds=block_seconds, generator=gen)["audio"][0].numpy()


def main(argv=None):
    import soundfile as sf

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("midi")
    ap.add_argument("out")
    ap.add_argument("--ckpt")
    ap.add_argument("--year", type=int, default=2018, help="MAESTRO year = piano/hall/mic condition")
    ap.add_argument("--sr", type=int, help="override sample rate (untrained model only)")
    ap.add_argument("--normalize", action="store_true", help="scale the output to a -1 dBFS peak")
    args = ap.parse_args(argv)

    model = load_model(args.ckpt, **({"sample_rate": args.sr} if args.sr else {}))
    notes, pedals = load_midi(args.midi)
    audio = render_notes(model, notes, pedals, year_to_condition(args.year))
    peak = np.abs(audio).max()
    if peak > 0.99 or args.normalize:
        audio = audio * (0.89 / peak)
    sf.write(args.out, audio, model.cfg.sample_rate)
    print(f"wrote {args.out}: {len(audio) / model.cfg.sample_rate:.1f}s, peak {peak:.3f}")


if __name__ == "__main__":
    main()
