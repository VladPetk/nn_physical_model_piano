"""The stereo measures (pianonn.stereo) recover known answers on synthetic signals (docs/physics_revamp.md 14)."""

import numpy as np

from pianonn import stereo as S

SR = 24000


def _delay(x, tau):
    """``x`` delayed by ``tau`` s (circular, fractional: a phase ramp)."""
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    return np.fft.irfft(X * np.exp(-2j * np.pi * f * tau), len(x))


def _fir(rng, n=600):
    """A random decaying FIR (a room-like filter, 25 ms)."""
    return rng.standard_normal(n) * np.exp(-np.arange(n) / 120)


def test_identical_and_delayed():
    rng = np.random.default_rng(0)
    s = rng.standard_normal(SR * 4)
    same = S.band_stereo(np.stack([s, s], 1), SR)
    assert np.allclose(same["msc"], 1, atol=1e-6) and np.allclose(same["phase"], 0, atol=1e-6)
    tau = 1e-4  # left lags by 0.1 ms
    d = np.stack([_delay(s, tau), s], 1)
    st = S.band_stereo(d, SR)
    assert np.all(st["msc"] > 0.98)
    centres = np.array([np.sqrt(lo * hi) for lo, hi in S.BANDS])
    want = -360 * centres * tau
    assert np.all(np.abs(np.median(st["phase"], 0)[:3] - want[:3]) < 10)  # bands up to 2 kHz: no wrap
    lag, h = S.gcc_phat(d, SR)
    assert abs(lag - 0.1) < 0.01 and h > 0.9


def test_independent_noise_known_coherence():
    rng = np.random.default_rng(1)
    s = rng.standard_normal(SR * 30)
    for snr_db in (0.0, 6.0):
        g = 10 ** (-snr_db / 20)
        x = np.stack([s + g * rng.standard_normal(len(s)), s + g * rng.standard_normal(len(s))], 1)
        snr = 10 ** (snr_db / 10)
        want = (snr / (1 + snr)) ** 2
        got = S.band_stereo(x, SR, seg=10.0, hop=10.0)["msc"]  # long segments: bias ~0.01
        assert np.all(np.abs(got - want) < 0.03), (snr_db, got, want)
    # independent channels in the measure's own 0.5 s segments read its bias
    x = rng.standard_normal((SR * 20, 2))
    got = np.median(S.band_stereo(x, SR)["msc"])
    assert abs(got - S.msc_bias(SR)) < 0.03, (got, S.msc_bias(SR))


def test_stationary_through_two_filters_is_coherent():
    """Partials through two different random filters: coherent, with no motion of their left/right relation."""
    rng = np.random.default_rng(2)
    t = np.arange(SR * 2) / SR
    f0 = 196.0
    freqs = f0 * np.arange(1, 9)
    x = sum(np.cos(2 * np.pi * f * t + rng.uniform(0, 6.3)) / k for k, f in enumerate(freqs, 1)) * np.exp(-t)
    st = np.stack([np.convolve(x, _fir(rng))[: len(x)], np.convolve(x, _fir(rng))[: len(x)]], 1)
    coh = S.partial_coherence(st, SR, 0.3, 0.8, freqs)
    assert np.all(coh > 0.99), coh
    lev, ph, snr = S.lr_motion(st, SR, 0.3, 0.8, freqs)
    assert np.all(lev < 0.05) and np.all(ph < 1.0), (lev, ph)
    assert np.all(snr > 30)


def test_beating_components_move_the_relation():
    """Two components 2 Hz apart reaching the microphones in different proportions: the level and phase difference
    move as in the closed form, and the partial's coherence drops below one."""
    t = np.arange(SR * 2) / SR
    f, df = 600.0, 2.0
    gL = np.array([1.0, 0.5 * np.exp(1j * 1.2)])  # each component's complex gain to each microphone
    gR = np.array([0.6 * np.exp(1j * 0.3), 1.0])
    comps = np.array([np.exp(2j * np.pi * f * t), 0.8 * np.exp(2j * np.pi * (f + df) * t)])
    x = np.stack([np.real(gL @ comps), np.real(gR @ comps)], 1)
    lev, ph, _ = S.lr_motion(x, SR, 0.3, 1.3, np.array([f + df / 2]))
    tt = np.arange(0.3, 1.3, 0.01)
    beat = np.exp(2j * np.pi * df * tt)
    r = (gL[0] + 0.8 * gL[1] * beat) / (gR[0] + 0.8 * gR[1] * beat)
    want_lev = np.std(20 * np.log10(np.abs(r)))
    u = np.exp(1j * np.angle(r))
    want_ph = np.degrees(np.sqrt(-2 * np.log(abs(u.mean()))))
    assert abs(lev[0] - want_lev) < 0.1 * want_lev, (lev, want_lev)
    assert abs(ph[0] - want_ph) < 0.1 * want_ph, (ph, want_ph)
    coh = S.partial_coherence(x, SR, 0.3, 1.3, np.array([f + df / 2]))
    assert coh[0] < 0.9
