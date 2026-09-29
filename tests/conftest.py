import pytest
import torch

from pianonn import NeuralPhysicalPiano, PianoConfig


def small_cfg(**kw):
    base = dict(sample_rate=8000, hop=40, n_partials=16, body_seconds=0.05, hall_seconds=0.2, ctx_hidden=32, synth_chunk=512,
                noise_bands=16, use_sympathetic=True)  # exercise the bank even though it is off by default
    return PianoConfig(**{**base, **kw})


def make_perf(model, n_samples, notes, sustain=0.0, soft=0.0, condition=3):
    """notes: list of (pitch, onset, offset, velocity)."""
    F = model.n_frames(n_samples)
    t = torch.tensor(notes, dtype=torch.float32)[None]

    def curve(v):
        return v(torch.arange(F) * model.cfg.hop / model.cfg.sample_rate)[None] if callable(v) else torch.full((1, F), v)

    return {"pitch": t[..., 0].long(), "onset": t[..., 1], "offset": t[..., 2], "velocity": t[..., 3],
            "mask": torch.ones(1, len(notes), dtype=torch.bool), "condition": torch.tensor([condition]),
            "sustain": curve(sustain), "soft": curve(soft), "sostenuto": curve(0.0)}


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


@pytest.fixture
def model():
    return NeuralPhysicalPiano(small_cfg())
