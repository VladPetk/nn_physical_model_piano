"""Render a MIDI file with a trained (or untrained, physics-prior-only) model."""

import argparse

import numpy as np
import torch

from .config import PianoConfig, year_to_condition
from .data import _perf_from_notes, load_midi
from .synth import NeuralPhysicalPiano


def load_weights(model, state_dict, log=print):
    """Load a checkpoint, tolerating parameters added since it was written (they keep their defaults)."""
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        log(f"checkpoint: {len(missing)} new parameter(s) at their defaults {missing[:6]}, "
            f"{len(unexpected)} unknown ignored {unexpected[:6]}")
    return model


def load_model(ckpt=None, device="cpu", **overrides):
    if ckpt:
        state = torch.load(ckpt, map_location=device)
        model = NeuralPhysicalPiano(PianoConfig.from_dict({**state["cfg"], **overrides}))
        load_weights(model, state["model"])
    else:
        model = NeuralPhysicalPiano(PianoConfig(**overrides))
    return model.to(device).eval()


@torch.no_grad()
def render_notes(model, notes, pedals, condition=0, tail=3.0, block_seconds=2.0, seed=0, residual=True, floor=False):
    """``notes[N, 4]`` = (pitch, onset, offset, velocity) in seconds; returns float32 audio ``[channels, T]``.

    ``residual``: include the learned residual (context net); ``floor``: add the recording's noise floor."""
    cfg = model.cfg
    device = next(model.parameters()).device
    duration = float(notes[:, 2].max()) + tail if len(notes) else tail
    n_samples = int(duration * cfg.sample_rate)
    perf = _perf_from_notes(notes, pedals, 0.0, duration, 0.0, model.n_frames(n_samples), cfg)
    perf = {k: v[None].to(device) for k, v in perf.items()}
    perf["mask"] = torch.ones_like(perf["pitch"], dtype=torch.bool)
    perf["condition"] = torch.tensor([condition], device=device)
    # the noise generator must live on the model's device (a CPU generator with CUDA tensors raises)
    gen = torch.Generator(device=device).manual_seed(seed)
    out = model(perf, n_samples, block_seconds=block_seconds, generator=gen, residual=residual, floor=floor)
    return out["audio"][0].cpu().numpy()


def main(argv=None):
    import soundfile as sf

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("midi")
    ap.add_argument("out")
    ap.add_argument("--ckpt")
    ap.add_argument("--year", type=int, default=2018, help="MAESTRO year = piano/hall/mic condition")
    ap.add_argument("--sr", type=int, help="override sample rate (untrained model only)")
    ap.add_argument("--normalize", action="store_true", help="scale the output to a -1 dBFS peak")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--physics-only", action="store_true", help="switch the learned residual off")
    ap.add_argument("--floor", action="store_true", help="add the recording's noise floor")
    args = ap.parse_args(argv)

    model = load_model(args.ckpt, device=args.device, **({"sample_rate": args.sr} if args.sr else {}))
    notes, pedals = load_midi(args.midi)
    audio = render_notes(model, notes, pedals, year_to_condition(args.year), residual=not args.physics_only, floor=args.floor)
    peak = np.abs(audio).max()
    if peak > 0.99 or args.normalize:
        audio = audio * (0.89 / peak)
    sf.write(args.out, audio.T, model.cfg.sample_rate)
    print(f"wrote {args.out}: {audio.shape[-1] / model.cfg.sample_rate:.1f}s, {audio.shape[0]} channels, peak {peak:.3f}")


if __name__ == "__main__":
    main()
