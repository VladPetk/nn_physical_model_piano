"""Differentiable DSP primitives."""

import torch


def bounded(x: torch.Tensor, r: float) -> torch.Tensor:
    """Identity near zero (slope 1), smoothly saturating at +-r."""
    return r * torch.tanh(x / r)


def fft_convolve(x: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
    """Linear convolution along the last dim, truncated to the length of ``x``."""
    T, L = x.shape[-1], ir.shape[-1]
    n = 1 << (T + L - 2).bit_length()
    y = torch.fft.irfft(torch.fft.rfft(x, n) * torch.fft.rfft(ir, n), n)
    return y[..., :T]


def linear_recurrence(x: torch.Tensor, log_a: torch.Tensor, init: torch.Tensor | None = None,
                      chunk: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve ``y[t] = exp(log_a[t]) * y[t-1] + x[t]`` for complex, time-varying ``a``.

    Within a chunk the recurrence has the closed form
    ``y_i = P_i * cumsum(x_j / P_j)`` with ``P_i = exp(cumsum(log_a))``; this is
    numerically safe as long as the decay accumulated over one chunk is modest
    (callers clamp decay rates). Chunks are then stitched with a short
    sequential carry, so the cost is O(T) work with only T/chunk serial steps.
    This is what lets damper-controlled resonators (time-varying decay) train
    on a GPU without a per-sample Python loop.

    Returns ``(y, final_state)`` so long signals can be processed in blocks.
    """
    *lead, T = x.shape
    pad = (-T) % chunk
    if pad:
        x = torch.cat([x, x.new_zeros(*lead, pad)], -1)
        log_a = torch.cat([log_a, log_a.new_zeros(*lead, pad)], -1)
    n_chunks = x.shape[-1] // chunk
    xs = x.reshape(*lead, n_chunks, chunk)
    cum = torch.cumsum(log_a.reshape(*lead, n_chunks, chunk), -1)
    growth = torch.exp(cum)
    z = growth * torch.cumsum(torch.exp(-cum) * xs, -1)

    carry = growth[..., -1]
    state = init if init is not None else x.new_zeros(*lead)
    incoming = []
    for c in range(n_chunks):
        incoming.append(state)
        state = z[..., c, -1] + carry[..., c] * state
    y = z + growth * torch.stack(incoming, -1)[..., None]
    y = y.reshape(*lead, -1)[..., :T]
    return y, y[..., -1]


def frames_to_samples(x: torch.Tensor, start: int, length: int, hop: int) -> torch.Tensor:
    """Linearly interpolate frame-rate controls ``[..., F]`` to samples ``[start, start+length)``."""
    n_frames = x.shape[-1]
    pos = torch.arange(start, start + length, device=x.device, dtype=torch.float64) / hop
    i0 = pos.floor().long().clamp(0, n_frames - 2)
    w = (pos - i0).clamp(0, 1).to(x.dtype)
    return x[..., i0] * (1 - w) + x[..., i0 + 1] * w


def sample_keyed(x: torch.Tensor, key: torch.Tensor, t: torch.Tensor, sr: int, hop: int) -> torch.Tensor:
    """Read per-key frame curves ``x[B, K, F]`` at per-note times ``t[B, N]`` (seconds) for keys ``key[B, N]``."""
    B, K, F = x.shape
    pos = (t * sr / hop).clamp(0, F - 1)
    i0 = pos.floor().long().clamp(max=F - 2)  # clamp the index, not pos: F - 1 - eps rounds to F - 1 in float32 for F > ~2000
    w = pos - i0
    flat = x.reshape(B, K * F)
    base = key * F + i0
    return flat.gather(1, base) * (1 - w) + flat.gather(1, base + 1) * w


def sample_curve(x: torch.Tensor, t: torch.Tensor, sr: int, hop: int) -> torch.Tensor:
    """Read frame curves ``x[B, F]`` at per-note times ``t[B, N]``."""
    return sample_keyed(x[:, None], torch.zeros_like(t, dtype=torch.long), t, sr, hop)
