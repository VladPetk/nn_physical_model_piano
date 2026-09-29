import numpy as np
import torch

from pianonn.dsp import fft_convolve, frames_to_samples, linear_recurrence


def naive(x, log_a, init):
    a = torch.exp(log_a)
    y, s = torch.zeros_like(x), init.clone()
    for t in range(x.shape[-1]):
        s = a[..., t] * s + x[..., t]
        y[..., t] = s
    return y


def test_linear_recurrence_matches_loop():
    T = 1000  # deliberately not a multiple of the chunk
    x = torch.randn(3, T, dtype=torch.complex128)
    decay = 0.001 + 0.05 * torch.rand(3, T, dtype=torch.float64)
    log_a = torch.complex(-decay, 0.3 * torch.ones_like(decay))
    init = torch.randn(3, dtype=torch.complex128)
    y, last = linear_recurrence(x, log_a, init, chunk=64)
    ref = naive(x, log_a, init)
    assert torch.allclose(y, ref, atol=1e-9)
    assert torch.allclose(last, ref[..., -1])


def test_linear_recurrence_blockwise_state_carry():
    x = torch.randn(2, 700, dtype=torch.complex128)
    log_a = torch.complex(-0.01 * torch.ones(2, 700, dtype=torch.float64), 0.2 * torch.ones(2, 700, dtype=torch.float64))
    full, _ = linear_recurrence(x, log_a, chunk=32)
    a, s = linear_recurrence(x[:, :300], log_a[:, :300], chunk=32)
    b, _ = linear_recurrence(x[:, 300:], log_a[:, 300:], s, chunk=32)
    assert torch.allclose(torch.cat([a, b], -1), full, atol=1e-10)


def test_fft_convolve():
    x, h = np.random.randn(2, 300), np.random.randn(2, 50)
    y = fft_convolve(torch.from_numpy(x), torch.from_numpy(h)).numpy()
    for i in range(2):
        assert np.allclose(y[i], np.convolve(x[i], h[i])[:300])


def test_frames_to_samples_linear():
    frames = torch.arange(10.0)[None]
    s = frames_to_samples(frames, 5, 20, hop=4)
    assert torch.allclose(s[0], (5 + torch.arange(20.0)) / 4)
