import math

import torch

from pianonn.losses import MultiResolutionSTFTLoss, PianoLoss, highpass
from pianonn.metrics import parabolic


def shaped_noise(B, n, sr, seed):
    """Random signals with a piano-like falling spectrum and slow amplitude modulation (unpredictable detail)."""
    g = torch.Generator().manual_seed(seed)
    X = torch.fft.rfft(torch.randn(B, 2, n, generator=g))
    f = torch.fft.rfftfreq(n, 1 / sr)
    x = torch.fft.irfft(X / (1 + f / 300.0), n)
    t = torch.arange(n) / sr
    return 0.05 * x * (1 + 0.5 * torch.sin(2 * math.pi * 1.3 * t + torch.rand(B, 2, 1, generator=g) * 6.28))


def test_piano_loss_is_zero_on_identical_inputs_and_differentiable():
    sr, n = 16000, 16000
    loss = PianoLoss(sr)
    x = shaped_noise(2, n, sr, 0)
    on, om = torch.tensor([[0.1, 0.5], [0.3, 2.0]]), torch.tensor([[True, True], [True, False]])
    total, terms = loss(x, x.clone(), on, om)
    assert float(total) < 1e-6 and set(terms) == {"band", "fine", "attack"}
    y = (0.5 * x).requires_grad_()
    total, _ = loss(y, x, on, om)
    total.backward()
    assert torch.isfinite(y.grad).all() and y.grad.abs().sum() > 0


def test_band_term_level_is_unbiased_where_spectral_convergence_is_not():
    """Target and prediction are independent draws with the same spectrum: the right gain is 0 dB. The band term's
    optimum must be there; spectral convergence (a least-squares fit of magnitudes) wants the prediction ~2 dB quieter
    (docs/plan_round2.md, 2.1)."""
    sr, n = 16000, 32000
    t, p = shaped_noise(4, n, sr, 1), shaped_noise(4, n, sr, 2)
    gains = torch.arange(-4.0, 4.01, 0.5)
    band = PianoLoss(sr, weights=(1.0, 0.0, 0.0))
    sc = lambda a, b: torch.linalg.norm(a - b) / torch.linalg.norm(b)
    from pianonn.losses import _mag

    lb = [float(band(p * 10 ** (g / 20), t)[0]) for g in gains]
    ls = [float(sc(_mag(p.reshape(-1, n) * 10 ** (g / 20), 1024), _mag(t.reshape(-1, n), 1024))) for g in gains]
    assert abs(parabolic(gains.numpy(), lb)) < 0.5
    assert parabolic(gains.numpy(), ls) < -1.0


def test_highpass_removes_infrasound_only():
    sr, n = 16000, 32000
    t = torch.arange(n) / sr
    x = torch.sin(2 * math.pi * 5 * t) + torch.sin(2 * math.pi * 60 * t)
    y = highpass(x, sr)
    X = torch.fft.rfft(y).abs()
    assert X[10] < 1e-3 * X[120]  # 5 Hz gone (bins are 0.5 Hz), 60 Hz kept
    assert abs(float(X[120]) / float(torch.fft.rfft(x).abs()[120]) - 1) < 1e-3


def test_attack_term_only_sees_frames_near_onsets():
    sr, n = 16000, 16000
    loss = PianoLoss(sr, weights=(0.0, 0.0, 1.0))
    x = shaped_noise(1, n, sr, 3)
    on, om = torch.tensor([[0.2]]), torch.tensor([[True]])
    y = x.clone()
    y[..., int(0.6 * sr):] *= 3.0  # far from the onset
    assert float(loss(y, x, on, om)[0]) < 1e-6
    y = x.clone()
    y[..., int(0.19 * sr): int(0.25 * sr)] *= 3.0  # at the onset
    assert float(loss(y, x, on, om)[0]) > 0.05


def test_old_loss_per_example_matches_the_batch_mean():
    sr, n = 16000, 16000
    loss = MultiResolutionSTFTLoss(fft_sizes=(1024, 256))
    a, b = shaped_noise(3, n, sr, 4), shaped_noise(3, n, sr, 5)
    per = loss(a, b, per_example=True)
    assert per.shape == (3,) and abs(float(per.mean()) - float(loss(a, b))) < 1e-5
