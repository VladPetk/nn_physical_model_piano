"""The damped-sinusoid bank as one autograd function with an analytic backward.

For ``P`` sounding notes (flattened over the batch) with ``Q`` oscillators each, over one chunk of
``L`` samples starting at absolute time ``t0``:

    tau   = max(0, t0 + l / sr - onset)                      time since the strike
    ramp  = (1 - cos(pi min(tau / tc, 1))) / 2               build-up over the hammer contact
    D     = max(0, c_note - c_onset)                         damper-on time since the strike
    S     = sum_r clamp((tau - rs_delay_r) / w, 0, 1)        later strikes of the same key so far
    E_q   = exp(-alpha_q tau - alpha_d,q D - rs_nats S)
    phi_q = 2 pi frac(f_q (tau - tc / 2))                    phase re the centre of the force pulse
    y     = ramp * sum_q a_q E_q sin(phi_q)

Autograd through this graph stores (or, checkpointed, replays) a dozen ``[P, Q, L]`` tensors and
launches hundreds of kernels per chunk. Here the backward recomputes ``E sin(phi)`` and ``E cos(phi)``
once and reduces them with batched matrix products, which is several times faster.

The cycle count is formed in float64 and only its fractional part is kept, so the phase stays
exact minutes into a piece (review 3, F11).

Optionally each note's oscillators are split into ``G`` overlapping groups by frequency (weights ``W[P, G, Q]``,
summing to 1 over ``G``), and each group follows its own gain curve ``m[P, G, L]`` (the physics-aware residual's
time-varying spectral shape): ``y = ramp * sum_q a_q E_q sin(phi_q) * sum_g W_gq m_g``. Gradients reach ``m`` too.
"""

import math

import torch

RESTRIKE_RAMP = 0.002  # s over which a re-strike takes out the ringing vibration (about a contact time)


def _common(t0, sr, L, onset, tc, c_note, c_onset, rs_delay, dtype):
    dev = onset.device
    t = t0 + torch.arange(L, device=dev, dtype=torch.float64) / sr
    tau64 = (t[None] - onset.double()[:, None]).clamp(min=0)  # [P, L]
    tau = tau64.to(dtype)
    x = tau / tc[:, None]
    ramp = 0.5 - 0.5 * torch.cos(math.pi * x.clamp(max=1))
    Draw = c_note - c_onset[:, None]
    D = Draw.clamp(min=0)
    S = ((tau[:, None, :] - rs_delay[:, :, None]) / RESTRIKE_RAMP).clamp(0, 1).sum(1)
    return tau64, tau, x, ramp, Draw, D, S


def _modes(freq, alpha, adamp, rs_nats, tau64, tau, tc, D, S, want_cos):
    cyc = freq.double()[:, :, None] * (tau64 - 0.5 * tc.double()[:, None])[:, None, :]
    phi = (2 * math.pi) * (cyc - cyc.floor()).to(freq.dtype)
    del cyc
    E = torch.exp(-(alpha[:, :, None] * tau[:, None, :] + adamp[:, :, None] * D[:, None, :]) - (rs_nats[:, None] * S)[:, None, :])
    Ms = E * torch.sin(phi)
    Mc = E * torch.cos(phi) if want_cos else None
    return Ms, Mc


class OscBank(torch.autograd.Function):
    @staticmethod
    def forward(ctx, freq, alpha, amp, adamp, tc, c_note, c_onset, rs_nats, onset, rs_delay, t0, sr, W=None, m=None):
        L = c_note.shape[-1]
        tau64, tau, x, ramp, Draw, D, S = _common(t0, sr, L, onset, tc, c_note, c_onset, rs_delay, freq.dtype)
        Ms, _ = _modes(freq, alpha, adamp, rs_nats, tau64, tau, tc, D, S, want_cos=False)
        if m is None:
            y = torch.bmm(amp[:, None, :], Ms)[:, 0] * ramp
        else:
            y = (torch.bmm(W * amp[:, None, :], Ms) * m).sum(1) * ramp
        ctx.save_for_backward(freq, alpha, amp, adamp, tc, c_note, c_onset, rs_nats, onset, rs_delay, W, m)
        ctx.t0, ctx.sr = t0, sr
        return y

    @staticmethod
    def backward(ctx, gy):
        freq, alpha, amp, adamp, tc, c_note, c_onset, rs_nats, onset, rs_delay, W, m = ctx.saved_tensors
        L = c_note.shape[-1]
        tau64, tau, x, ramp, Draw, D, S = _common(ctx.t0, ctx.sr, L, onset, tc, c_note, c_onset, rs_delay, freq.dtype)
        Ms, Mc = _modes(freq, alpha, adamp, rs_nats, tau64, tau, tc, D, S, want_cos=True)
        G = gy * ramp
        if m is not None:
            return _grouped_backward(gy, G, ramp, x, tau, Draw, D, S, tc, freq, amp, adamp, W, m, Ms, Mc)
        rel = tau - 0.5 * tc[:, None]
        R = torch.bmm(Ms, torch.stack([G, G * tau, G * D], -1))  # [P, Q, 3]
        d_amp = R[..., 0]
        d_alpha = -amp * R[..., 1]
        d_adamp = -amp * R[..., 2]
        d_freq = (2 * math.pi) * amp * torch.bmm(Mc, (G * rel)[..., None])[..., 0]
        lhs = torch.bmm(torch.stack([amp, amp * adamp], 1), Ms)  # [P, 2, L]: sum_q a Ms, sum_q a adamp Ms
        ys, yd = lhs[:, 0], lhs[:, 1]
        yc = torch.bmm((amp * freq)[:, None, :], Mc)[:, 0]
        dramp_dtc = torch.where(x < 1, -0.5 * math.pi * torch.sin(math.pi * x.clamp(max=1)) * tau / tc[:, None] ** 2,
                                torch.zeros_like(x))
        d_tc = (gy * dramp_dtc * ys).sum(-1) - math.pi * (G * yc).sum(-1)
        d_c_note = -G * yd * (Draw > 0)
        d_c_onset = -d_c_note.sum(-1)
        d_rs = -(G * S * ys).sum(-1)
        return d_freq, d_alpha, d_amp, d_adamp, d_tc, d_c_note, d_c_onset, d_rs, None, None, None, None, None, None


def _grouped_backward(gy, G, ramp, x, tau, Draw, D, S, tc, freq, amp, adamp, W, m, Ms, Mc):
    """The backward with group gain curves: each oscillator's sample weight is ``M_q = sum_g W_gq m_g``, so every
    reduction over samples runs once per group (``Gm = G * m_g``) and is folded back over the groups with ``W``."""
    Gn = m.shape[1]
    Gm = G[:, None, :] * m  # [P, G, L]
    rel = tau - 0.5 * tc[:, None]
    WT = W.transpose(1, 2)  # [P, Q, G]
    R = torch.bmm(Ms, torch.cat([Gm, Gm * tau[:, None], Gm * D[:, None]], 1).transpose(1, 2))  # [P, Q, 3G]
    d_amp = (WT * R[..., :Gn]).sum(-1)
    d_alpha = -amp * (WT * R[..., Gn:2 * Gn]).sum(-1)
    d_adamp = -amp * (WT * R[..., 2 * Gn:]).sum(-1)
    d_freq = (2 * math.pi) * amp * (WT * torch.bmm(Mc, (Gm * rel[:, None]).transpose(1, 2))).sum(-1)
    Wa = W * amp[:, None, :]
    Y = torch.bmm(Wa, Ms)  # [P, G, L]: each group's sum before its gain
    ys = (Y * m).sum(1)
    yd = (torch.bmm(Wa * adamp[:, None, :], Ms) * m).sum(1)
    yc = (torch.bmm(Wa * freq[:, None, :], Mc) * m).sum(1)
    dramp_dtc = torch.where(x < 1, -0.5 * math.pi * torch.sin(math.pi * x.clamp(max=1)) * tau / tc[:, None] ** 2,
                            torch.zeros_like(x))
    d_tc = (gy * dramp_dtc * ys).sum(-1) - math.pi * (G * yc).sum(-1)
    d_c_note = -G * yd * (Draw > 0)
    d_c_onset = -d_c_note.sum(-1)
    d_rs = -(G * S * ys).sum(-1)
    d_m = G[:, None, :] * Y
    return d_freq, d_alpha, d_amp, d_adamp, d_tc, d_c_note, d_c_onset, d_rs, None, None, None, None, None, d_m


def osc_bank(freq, alpha, amp, adamp, tc, c_note, c_onset, rs_nats, onset, rs_delay, t0, sr, W=None, m=None):
    """``[P, Q]`` oscillators, ``[P]`` note scalars, ``c_note[P, L]``, ``rs_delay[P, R]`` -> ``y[P, L]``; optionally
    group weights ``W[P, G, Q]`` (no gradient) and group gain curves ``m[P, G, L]``."""
    return OscBank.apply(freq, alpha, amp, adamp, tc, c_note, c_onset, rs_nats, onset, rs_delay, t0, sr, W, m)


def group_weights(freq, centers):
    """Soft frequency groups ``[..., G, Q]`` for oscillators at ``freq[..., Q]``: raised-cosine bands one octave
    apart in log f (``centers`` must be octaves), summing to 1; flat below the first centre and above the last."""
    lf = torch.log2(freq.clamp(min=1.0))[..., None, :]
    c = torch.log2(torch.as_tensor(centers, dtype=freq.dtype, device=freq.device))[:, None]
    d = (lf - c).clamp(-1, 1)
    w = torch.cos(0.5 * math.pi * d) ** 2
    w[..., 0, :] = torch.where(lf[..., 0, :] < c[0], torch.ones_like(w[..., 0, :]), w[..., 0, :])
    w[..., -1, :] = torch.where(lf[..., 0, :] > c[-1], torch.ones_like(w[..., -1, :]), w[..., -1, :])
    return w
