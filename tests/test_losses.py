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


def _attacks(B, n, sr, onsets, amps, seed, low_hz=None, bg=0.002):
    """Shaped-noise background plus a 30 ms decaying burst at each onset's expected sound onset (pitch 60, velocity
    64: 7.17 ms after the MIDI onset), scaled by ``amps[B, N]``; ``low_hz``: a sine burst there instead."""
    from pianonn.losses import ONSET_DELAY_MS

    x = bg * shaped_noise(B, n, sr, seed) / 0.05
    src = shaped_noise(B, n, sr, seed + 100) / 0.05
    L = int(0.03 * sr)
    env = torch.exp(-torch.arange(L) / (0.008 * sr))
    for b in range(B):
        for k, t in enumerate(onsets[b]):
            i = int(round((t + 1e-3 * ONSET_DELAY_MS[0]) * sr))
            burst = torch.sin(2 * math.pi * low_hz * torch.arange(L) / sr) if low_hz else src[b, :, i: i + L]
            x[b, :, i: i + L] += amps[b][k] * env * burst
    return x


def _perf(onsets):
    on = torch.tensor(onsets, dtype=torch.float32)
    return {"onset": on, "pitch": torch.full(on.shape, 60), "velocity": torch.full(on.shape, 64.0),
            "mask": torch.ones(on.shape, dtype=torch.bool)}


def test_onset_loss_is_zero_on_identical_inputs_and_reads_a_level_change_exactly():
    from pianonn.losses import OnsetLoss

    sr, n = 16000, 24000
    ons = [[0.5, 0.9], [0.6, 1.2]]
    x = _attacks(2, n, sr, ons, [[0.05, 0.03], [0.04, 0.05]], 0)
    loss = OnsetLoss(sr)
    assert float(loss(x, x.clone(), _perf(ons), 0.4)) < 1e-6
    y = (0.5 * x).requires_grad_()
    v = loss(y, x, _perf(ons), 0.4)
    assert abs(float(v) - math.log10(4)) < 1e-3  # every counted band 6 dB down
    v.backward()
    assert torch.isfinite(y.grad).all() and y.grad.abs().sum() > 0


def test_onset_loss_windows_sit_at_the_expected_onset():
    from pianonn.losses import OnsetLoss

    sr, n = 16000, 24000
    ons = [[0.6]]
    x = _attacks(1, n, sr, ons, [[0.05]], 1)
    loss = OnsetLoss(sr)
    y = x.clone()
    y[..., int(0.72 * sr): int(0.9 * sr)] *= 3.0  # 110-300 ms after the onset
    assert float(loss(y, x, _perf(ons), 0.4)) < 1e-6
    y = x.clone()
    y[..., int(0.607 * sr): int(0.627 * sr)] *= 2.0  # the first 20 ms of the sound
    assert float(loss(y, x, _perf(ons), 0.4)) > 0.1
    assert float(loss(y, x, _perf(ons), 0.7)) == 0.0  # anchored before t_lo: not counted


def test_onset_loss_sees_a_low_attack_the_attack_term_misses():
    """A 150 Hz burst at the onset: the onset term reads it, the attack term (no band below ~400 Hz) barely does."""
    from pianonn.losses import OnsetLoss, PianoLoss

    sr, n = 16000, 24000
    ons = [[0.5, 1.0]]
    x = _attacks(1, n, sr, ons, [[0.03, 0.03]], 2)
    y = x + _attacks(1, n, sr, ons, [[0.05, 0.05]], 2, low_hz=150.0, bg=0.0)
    onset = float(OnsetLoss(sr)(y, x, _perf(ons), 0.4))
    s = int(0.4 * sr)
    on = torch.tensor(ons) - 0.4
    attack = float(PianoLoss(sr, weights=(0.0, 0.0, 1.0))(y[..., s:], x[..., s:], on, torch.ones_like(on, dtype=torch.bool))[0])
    assert onset > 0.1 and attack < 0.2 * onset, (onset, attack)


def test_onset_loss_optimum_is_the_energy_match():
    """The target's attacks vary by 8 dB (log-normal), the prediction's not at all: a per-window median would leave
    the prediction several dB under the energy match (0.115 sigma^2); summing the windows first puts the optimum
    there."""
    from pianonn.losses import OnsetLoss

    sr, n, B, N = 16000, 32000, 4, 6
    g = torch.Generator().manual_seed(3)
    ons = [[0.5 + 0.22 * k for k in range(N)] for _ in range(B)]
    t = _attacks(B, n, sr, ons, (0.03 * 10 ** (8.0 * torch.randn(B, N, generator=g) / 20)).tolist(), 4)
    p = _attacks(B, n, sr, ons, [[0.03] * N] * B, 5)
    loss = OnsetLoss(sr)
    rows, times, _ = loss.anchors(_perf(ons), 0.4, 2.0)
    pp, pt = loss.powers(p, rows, times), loss.powers(t, rows, times)
    c = loss.cells(pp, pt) > 0
    has = c.any(0)
    match = torch.median(10 * torch.log10((pt["attack"] * c).sum(0)[has] / (pp["attack"] * c).sum(0)[has]))
    per_window = torch.median(10 * torch.log10(pt["attack"][c] / pp["attack"][c]))
    gains = torch.arange(-4.0, 14.0, 0.1)
    vals = torch.tensor([float(loss(p * 10 ** (gg / 20), t, _perf(ons), 0.4)) for gg in gains])
    best = float(gains[vals.argmin()])
    assert abs(best - float(match)) < 0.5, (best, float(match))
    assert abs(float(per_window) - float(match)) > 3.0, (float(per_window), float(match))


def _tones(B, n, sr, onsets, amp, seed, seconds=0.25):
    """A sustained, slowly decaying broadband tone from each onset's expected sound onset (pitch 60, velocity 64)."""
    from pianonn.losses import ONSET_DELAY_MS

    x = torch.zeros(B, 2, n)
    src = shaped_noise(B, n, sr, seed) / 0.05
    L = int(seconds * sr)
    env = torch.exp(-torch.arange(L) / (0.3 * sr))
    for b in range(B):
        for t in onsets[b]:
            i = int(round((t + 1e-3 * ONSET_DELAY_MS[0]) * sr))
            x[b, :, i: i + L] += amp * env * src[b, :, i: i + L]
    return x


def test_onset_loss_relative_reads_the_attack_re_the_tone():
    """``relative``: a level change of the whole note (attack and tone alike) reads nothing, where the absolute form
    reads 6 dB; a change of the attack alone reads in both."""
    from pianonn.losses import OnsetLoss

    sr, n = 16000, 24000
    ons = [[0.5, 1.0]]
    bg = 0.0003 * shaped_noise(1, n, sr, 7) / 0.05
    burst = _attacks(1, n, sr, ons, [[0.05, 0.05]], 6, bg=0.0)
    tone = _tones(1, n, sr, ons, 0.02, 8)
    x = bg + burst + tone
    rel, ab = OnsetLoss(sr, relative=True), OnsetLoss(sr)
    whole = bg + 0.5 * (burst + tone)
    assert float(rel(whole, x, _perf(ons), 0.4)) < 0.01
    assert abs(float(ab(whole, x, _perf(ons), 0.4)) - math.log10(4)) < 0.02
    att = (bg + 0.5 * burst + tone).requires_grad_()
    v = rel(att, x, _perf(ons), 0.4)
    assert float(v) > 0.15 and float(ab(att, x, _perf(ons), 0.4)) > 0.15
    v.backward()
    assert torch.isfinite(att.grad).all() and att.grad.abs().sum() > 0


def test_onset_loss_running_pool_settles_at_the_energy_match_of_the_stream():
    """A stream of batches of one example with two attacks each; the target's attacks vary by 8 dB, the prediction's
    not at all. A level offset trained with the batch's own pool settles near the median of the batches' ratios,
    several dB under the energy match; with the running pool it settles at the energy match of the whole stream."""
    from pianonn.losses import OnsetLoss

    sr, n, M = 16000, 20000, 40
    ons = [[0.5, 0.9]]
    g = torch.Generator().manual_seed(11)
    tgt = [_attacks(1, n, sr, ons, (0.03 * 10 ** (8.0 * torch.randn(1, 2, generator=g) / 20)).tolist(), 20 + k)
           for k in range(M)]
    prd = [_attacks(1, n, sr, ons, [[0.03, 0.03]], 100 + k) for k in range(M)]
    probe = OnsetLoss(sr)
    rows, times, _ = probe.anchors(_perf(ons), 0.4, 1.5)
    num, den = 0.0, 0.0
    for t, p in zip(tgt, prd):
        pt, pp = probe.powers(t, rows, times), probe.powers(p, rows, times)
        c = probe.cells(pp, pt)
        num, den = num + (pt["attack"] * c).sum(0), den + (pp["attack"] * c).sum(0)
    match = float(torch.median(10 * torch.log10(num[den > 0] / den[den > 0])))

    def settle(loss):
        delta = torch.zeros((), requires_grad=True)
        order = torch.randperm(M, generator=torch.Generator().manual_seed(0)).tolist()
        steps = 30 * M
        for s in range(steps):
            k = order[s % M]
            v = loss(prd[k] * 10 ** (delta / 20), tgt[k], _perf(ons), 0.4)
            (grad,) = torch.autograd.grad(v, delta)
            with torch.no_grad():
                delta -= 20.0 * 0.02 ** (s / steps) * grad  # 2 dB per step down to 0.04
        return float(delta)

    batch, pool = settle(OnsetLoss(sr)), settle(OnsetLoss(sr, pool_decay=0.95))
    assert abs(pool - match) < 1.0, (pool, match)
    assert abs(batch - match) > 2.0, (batch, match)


def test_level_term_optimum_is_the_energy_match_where_the_band_term_is_not():
    """The recording's energy comes in bursts (decaying events), the prediction's is steady, the totals are equal: the
    right gain is 0 dB. The per-frame band term is median-seeking over frames and wants the prediction quieter; the
    whole-excerpt level term's optimum is the energy match."""
    sr, n = 16000, 32000
    t, p = shaped_noise(4, n, sr, 1), shaped_noise(4, n, sr, 2)
    time = torch.arange(n) / sr
    env = torch.exp(-((time % 0.25) / 0.03))  # an event every 250 ms, decaying with a 30 ms time constant
    t = t * env
    t = t * (p.pow(2).sum(-1, keepdim=True) / t.pow(2).sum(-1, keepdim=True)).sqrt()  # same energy per channel
    gains = torch.arange(-6.0, 6.01, 0.5)
    band = PianoLoss(sr, weights=(1.0, 0.0, 0.0))
    level = PianoLoss(sr, weights=(0.0, 0.0, 0.0), level_weight=1.0)
    lb = [float(band(p * 10 ** (g / 20), t)[0]) for g in gains]
    ll = [float(level(p * 10 ** (g / 20), t)[0]) for g in gains]
    assert parabolic(gains.numpy(), lb) < -1.5
    assert abs(parabolic(gains.numpy(), ll)) < 0.3
    assert set(level.terms(p, t)) == {"band", "level"}
    assert "level" not in PianoLoss(sr).terms(p, t)  # off by default: the evaluation's tables keep their terms
