"""The interaction tables (pianonn/interactions.py): the identity at the start, zero mean over the factor each adds,
and a gradient for every table in a render."""

import torch

from pianonn.config import PianoConfig
from pianonn.interactions import TABLES, hat_weights
from pianonn.synth import NeuralPhysicalPiano
from tests.conftest import make_perf, small_cfg

NOTES = [(40, 0.05, 0.6, 30), (64, 0.1, 0.9, 70), (64, 0.5, 0.9, 120), (90, 0.2, 0.5, 100), (52, 0.3, 0.8, 90)]


def cfg(**kw):
    return PianoConfig(**{**small_cfg().to_dict(), **kw})


def test_hat_weights_sum_to_one_and_interpolate():
    x = torch.tensor([-0.3, 0.0, 0.1, 0.5, 0.99, 1.0, 1.7])
    w = hat_weights(x, 5, 0.0, 1.0)
    assert torch.allclose(w.sum(-1), torch.ones(len(x)))
    knots = torch.linspace(0, 1, 5)
    assert torch.allclose(w @ knots, x.clamp(0, 1), atol=1e-6)


def test_tables_start_at_identity():
    """With the tables at zero the render is the same as without them, for both string models."""
    for model in ("modes", "coupled"):
        torch.manual_seed(0)
        a = NeuralPhysicalPiano(cfg(string_model=model))
        torch.manual_seed(0)
        b = NeuralPhysicalPiano(cfg(string_model=model, interactions=True))
        missing, _ = b.load_state_dict(a.state_dict(), strict=False)
        assert all(k.startswith("physics.inter.") for k in missing)
        n = a.cfg.sample_rate
        perf = make_perf(a, n, NOTES, sustain=lambda t: (t > 0.25).float())
        with torch.no_grad():
            ya = a(perf, n, residual=False, generator=torch.Generator().manual_seed(1))["audio"]
            yb = b(perf, n, residual=False, generator=torch.Generator().manual_seed(1))["audio"]
        assert torch.allclose(ya, yb, atol=1e-7), model


def test_tables_hold_zero_mean_over_their_factor():
    m = NeuralPhysicalPiano(cfg(interactions=True))
    with torch.no_grad():
        for p in m.physics.inter.parameters():
            p.normal_(0, 1.0)
    for name, (cols, bound) in TABLES.items():
        t = m.physics.inter.table(name)
        assert t.mean(1).abs().max() < 1e-6, name
        assert t.abs().max() <= 2 * bound, name
        if cols == "strings":
            assert t.mean(2).abs().max() < 1e-6


def test_every_table_learns():
    """A coupled render with notes at several velocities, the pedal down for part of it: every table gets a gradient
    (the pedal table through the notes struck with the pedal lifted)."""
    m = NeuralPhysicalPiano(cfg(string_model="coupled", interactions=True))
    n = m.cfg.sample_rate
    perf = make_perf(m, n, NOTES, sustain=lambda t: (t > 0.25).float())
    out = m(perf, n, residual=False)
    (out["audio"].pow(2).sum() + 1e3 * out["strings"].pow(2).sum()).backward()
    for k, p in m.physics.inter.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, k
