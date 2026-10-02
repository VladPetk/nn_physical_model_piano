"""Render a MIDI file with a trained (or untrained, physics-prior-only) model."""

import argparse
import math

import numpy as np
import torch

from .config import PianoConfig, year_to_condition
from .data import _perf_from_notes, load_midi
from .synth import NeuralPhysicalPiano


def load_weights(model, state_dict, log=print):
    """Load a checkpoint, tolerating parameters added since it was written (they keep their defaults) and parameters
    whose shape the config has changed (the aware residual's wider outputs; they keep their initialisation too)."""
    own = model.state_dict()
    reshaped = [k for k, v in state_dict.items() if k in own and own[k].shape != v.shape]
    if reshaped:
        state_dict = {k: v for k, v in state_dict.items() if k not in reshaped}
        log(f"checkpoint: {len(reshaped)} parameter(s) of another shape left at their initialisation {reshaped[:6]}")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        log(f"checkpoint: {len(missing)} new parameter(s) at their defaults {missing[:6]}, "
            f"{len(unexpected)} unknown ignored {unexpected[:6]}")
    return model


def load_model(ckpt=None, device="cpu", ema=True, **overrides):
    """A model from a checkpoint; with ``ema`` (default) its averaged weights where it has them (train.py --ema)."""
    if ckpt:
        state = torch.load(ckpt, map_location=device)
        model = NeuralPhysicalPiano(PianoConfig.from_dict({**state["cfg"], **overrides}))
        load_weights(model, state["ema"]["weights"] if ema and "ema" in state else state["model"])
    else:
        model = NeuralPhysicalPiano(PianoConfig(**overrides))
    return model.to(device).eval()


def _parse_value(v):
    if v.lower() in ("1", "true", "yes", "on"):
        return True
    if v.lower() in ("0", "false", "no", "off"):
        return False
    for kind in (int, float):
        try:
            return kind(v)
        except ValueError:
            pass
    return v


def load_variant(spec, device="cpu", log=print):
    """A model from ``label=checkpoint:physics|residual[:option,option,...]``; returns ``(label, model, residual)``.

    Options change the checkpoint's model without training it: ``key=value`` overrides a config field (e.g.
    ``bridge_end_comb=1``); ``mined=<file>`` re-applies per-key B from a ``scripts/mine_notes.py`` file (the tuning
    is left as it is); ``pedal_theta=<CC>`` moves the sustain value at which the dampers lift half-way (0-127,
    within the parameter's range, about 15-91)."""
    label, rest = spec.split("=", 1)
    parts = rest.split(":")
    i = max(j for j, p in enumerate(parts) if p in ("physics", "residual"))
    ckpt, mode = ":".join(parts[:i]), parts[i]
    overrides, mined, theta = {}, None, None
    for opt in filter(None, ":".join(parts[i + 1:]).split(",")):
        k, v = opt.split("=", 1)
        if k == "mined":
            mined = v
        elif k == "pedal_theta":
            theta = float(v)
        else:
            overrides[k] = _parse_value(v)
    model = load_model(ckpt, device=device, **overrides)
    if theta is not None:  # theta = 0.42 + 0.3 tanh(raw / 0.3) (physics.pedal_lift)
        with torch.no_grad():
            model.physics.raw_pedal_theta.fill_(0.3 * math.atanh(max(-0.999, min(0.999, (theta / 127 - 0.42) / 0.3))))
    if mined:
        import json

        from .fit_init import apply_mined_priors
        with open(mined) as f:
            apply_mined_priors(model, json.load(f), log=log, set_cents=False)
    return label, model, mode == "residual"


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
