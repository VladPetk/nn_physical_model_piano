"""Every measure of pianonn.measures recovers a known answer on synthetic notes (docs/tone_measures.md, principle 5)."""

import math

import numpy as np
import pytest

from pianonn import measures as M

SR = 24000


def synth(dur=1.5, t_on=0.7, f0=220.0, B=3e-4, slope=-6.0, alpha=3.0, alpha_n=0.0, ramp=0.002, gain=1.0, knock=0.0,
          knock_tau=0.006, precursor=0.0, background=0.0, release=None, rel_rate=300.0, room=0.0, am=None, glide=0.0,
          glide_tau=0.3, extra=(), extra_bg=(), seed=0):
    """A stereo note: inharmonic partials with a slope (dB/oct over partial number) and decays alpha + alpha_n (n-1)
    (1/s, amplitude), a raised-cosine ramp, optional knock (high-passed noise burst), precursor (low noise burst
    120-40 ms before), a ringing earlier note, a release (extra decay in dB/s from ``release`` s), a pitch glide
    (``glide`` cents at the onset, decaying with ``glide_tau`` s), extra tones ``(Hz, amplitude)`` with the note
    (``extra``) or throughout (``extra_bg``) and a -90 dB floor."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * SR)) / SR
    tau = t - t_on
    on = tau >= 0
    ramp_env = (0.5 - 0.5 * np.cos(np.pi * np.clip(tau / ramp, 0, 1))) * on
    y = np.zeros(len(t))
    n = np.arange(1, 60)
    f = n * f0 * np.sqrt(1 + B * n * n)
    warp = tau + math.log(2) / 1200 * glide * glide_tau * (1 - np.exp(-np.clip(tau, 0, None) / glide_tau))
    for k, fk in zip(n, f):
        if fk > 0.45 * SR:
            break
        a = gain * 10 ** (slope * math.log2(k) / 20)
        y += a * np.exp(-(alpha + alpha_n * (k - 1)) * np.clip(tau, 0, None)) * np.sin(2 * np.pi * fk * warp + rng.uniform(0, 2 * np.pi))
    for fx, ax in extra:
        y += gain * ax * np.exp(-alpha * np.clip(tau, 0, None)) * np.sin(2 * np.pi * fx * tau + 1.0)
    y *= ramp_env
    for fx, ax in extra_bg:
        y += gain * ax * np.sin(2 * np.pi * fx * t + 2.0)
    if release is not None:
        held = y.copy()
        y *= 10 ** (-rel_rate * np.clip(tau - release, 0, None) / 20)
        if room:  # a hall's tail of the note, room x the direct sound, decaying at 37 dB/s (T60 1.6 s) after the release
            y += room * held * 10 ** (-37 * np.clip(tau - release, 0, None) / 20)
    if knock:
        w = rng.standard_normal(len(t))
        W = np.fft.rfft(w)
        W[np.fft.rfftfreq(len(t), 1 / SR) < 500] = 0
        y += knock * np.fft.irfft(W, len(t)) * np.exp(-np.clip(tau, 0, None) / knock_tau) * on
    if precursor:
        w = rng.standard_normal(len(t))
        W = np.fft.rfft(w)
        W[np.fft.rfftfreq(len(t), 1 / SR) > 800] = 0
        y += precursor * np.fft.irfft(W, len(t)) * ((tau > -0.12) & (tau < -0.04))
    if background:
        for k in range(1, 12):  # a note struck 0.5 s before the clip, ringing
            y += background * k ** -1.0 * np.exp(-1.5 * (t + 0.5)) * np.sin(2 * np.pi * 311.1 * k * t + k)
    if am is not None:
        y *= 1 + 0.8 * np.sin(2 * np.pi * am * t)
    y += 10 ** (-90 / 20) * rng.standard_normal(len(t))
    return np.stack([y, 0.8 * y], 1)


def partials(f0, B=3e-4, n=48):
    k = np.arange(1, n + 1)
    return k * f0 * np.sqrt(1 + B * k * k)


@pytest.mark.parametrize("f0", [55.0, 220.0, 880.0])
def test_onset_follows_a_shift_and_ignores_ringing(f0):
    a = M.onset(synth(f0=f0, background=0.3), SR, 0.7, f0)
    b = M.onset(synth(f0=f0, background=0.3, t_on=0.705), SR, 0.7, f0)
    assert abs(a - 0.7) < 0.003
    assert abs((b - a) - 0.005) < 0.0005


def test_onset_fails_cleanly_without_a_note():
    x = synth(gain=0.0, background=0.3)
    assert math.isnan(M.onset(x, SR, 0.7, 220.0))


def test_level_scales_with_gain():
    a = M.note_measures(synth(), SR, 0.7, partials(220.0))
    b = M.note_measures(synth(gain=2.0), SR, 0.7, partials(220.0))
    assert abs(b["N1 level"] - a["N1 level"] - 20 * math.log10(2)) < 0.1


@pytest.mark.parametrize("slope", [-3.0, -9.0])
def test_spectral_slope(slope):
    m = M.note_measures(synth(slope=slope), SR, 0.7, partials(220.0))
    assert abs(m["N2 slope"] - slope) < 1.0
    assert abs(m["N2 partial 5"] - slope * math.log2(5)) < 1.5


def test_early_decay():
    m = M.note_measures(synth(alpha=5.0), SR, 0.7, partials(220.0))
    expected = 20 * math.log10(math.e) * 5.0 * 0.385
    assert abs(m["N8 drop 1-4"] - expected) < 0.7


@pytest.mark.parametrize("f0", [440.0, 1000.0])
def test_rise_time(f0):
    # 10 -> 90 % of a raised-cosine ramp's power takes 0.474 of its length; resolution is about one period
    rise = lambda ramp: np.mean([M.note_measures(synth(ramp=ramp, f0=f0, seed=s), SR, 0.7, partials(f0))["N4 rise"] for s in range(3)])
    slow, fast = rise(0.02), rise(0.002)
    assert abs((slow - fast) - 0.474 * 18) < 2.0
    assert abs(slow - 0.474 * 20) < 1.5
    assert "N4 rise" not in M.note_measures(synth(f0=110.0), SR, 0.7, partials(110.0))  # below the resolution limit


def test_knock():
    ref = M.note_measures(synth(f0=440.0, knock=1.0), SR, 0.7, partials(440.0))
    loud = M.note_measures(synth(f0=440.0, knock=math.sqrt(10)), SR, 0.7, partials(440.0))
    bright = M.note_measures(synth(f0=440.0, knock=1.0, slope=-4.0), SR, 0.7, partials(440.0))
    for c in (2000, 4000, 8000):
        assert abs(loud[f"N6 knock {c}"] - ref[f"N6 knock {c}"] - 10) < 1.5, c
        assert abs(bright[f"N6 knock {c}"] - ref[f"N6 knock {c}"]) < 2.0, c
        assert loud[f"N4 percussive {c}"] > ref[f"N4 percussive {c}"] + 5, c
    assert loud["N6 sustain 2000"] < -15  # the knock is gone by 100 ms


def test_phantoms():
    f0, B = 110.0, 3e-4
    x = synth(f0=f0, B=B)
    t = np.arange(len(x)) / SR - 0.7
    env = np.exp(-3.0 * np.clip(t, 0, None)) * (t >= 0)
    ps = partials(f0, B, 96)
    y = x.copy()
    for j in range(2, 24):
        y[:, 0] += 0.1 / (2 * j) * env * np.sin(2 * np.pi * 2 * ps[j - 1] * t + j)  # 20 dB under partial 2j
    y[:, 1] = 0.8 * y[:, 0]
    with_ph = M.note_measures(y, SR, 0.7, ps)
    without = M.note_measures(x, SR, 0.7, ps)
    assert with_ph["N9 n"] >= 5
    assert abs(with_ph["N9 phantom"] + 20) < 2.5
    assert with_ph["N9 control"] < -40 and without["N9 phantom"] < -40


def test_pre_onset():
    quiet = M.note_measures(synth(background=0.3), SR, 0.7, partials(220.0))
    touch = M.note_measures(synth(background=0.3, precursor=0.3), SR, 0.7, partials(220.0))  # ~4 dB over the ringing
    assert quiet["N7 pre-onset low"] < 1.5  # the ringing's decay is taken out
    assert touch["N7 pre-onset low"] > 3.0


def test_texture_modulation():
    steady = M.texture_stats(synth(background=0.0, knock=0.0, slope=-3, alpha=0.0, dur=4.0)[SR:], SR)
    wobble = M.texture_stats(synth(background=0.0, knock=0.0, slope=-3, alpha=0.0, dur=4.0, am=4.0)[SR:], SR)
    assert wobble["P3 mod 1000 4-8"] > steady["P3 mod 1000 4-8"] + 6
    assert wobble["P3 cv 1000"] > steady["P3 cv 1000"] + 0.2


def test_onset_flux():
    sharp = M.onset_flux(synth(knock=0.3, ramp=0.001), SR, [0.7])
    soft = M.onset_flux(synth(ramp=0.03), SR, [0.7])
    assert sharp[0, 3] > soft[0, 3] + 3  # 2 kHz band


def test_isolated_and_bench_strata():
    on = np.array([0.0, 1.0, 1.1, 3.0, 3.5, 5.0])
    assert list(M.isolated(on, 0.3, 0.65)) == [0, 4, 5]  # 3.5 s: 0.5 s clear before, 1.5 s after
    assert M.register_of(28) == "R1" and M.register_of(29) == "R2" and M.register_of(89) == "R7"
    assert M.bin_of(64, M.VELOCITY) == "mf-f" and M.bin_of(30, M.PEDAL) == "up"


T60 = {125: 2.4, 250: 2.2, 500: 2.0, 1000: 1.9, 2000: 1.6, 4000: 1.2, 8000: 0.8}


def room_stop(direct=3.0, t_stop=0.6, dur=2.0, floor_db=-80.0, seed=0):
    """Octave bands of noise held until ``t_stop``, then a reverberant field decaying with ``T60`` per band plus a
    direct part ``direct`` times its power that stops within ~3 ms; a stationary floor."""
    rng = np.random.default_rng(seed)
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = np.fft.rfftfreq(n, 1 / SR)
    tau = np.clip(t - t_stop, 0, None)
    y = np.zeros((n, 2))
    for c, t60 in T60.items():
        for ch in range(2):
            W = np.fft.rfft(rng.standard_normal(n))
            W[(f < c / math.sqrt(2)) | (f >= c * math.sqrt(2))] = 0
            w = np.fft.irfft(W, n)
            w /= w.std()
            power = 1e-2 * (10 ** (-6 * tau / t60) + direct * np.exp(-tau / 0.003))
            y[:, ch] += w * np.sqrt(power)
    floor = 10 ** (floor_db / 20) * rng.standard_normal((n, 2))
    return y + floor, floor


def test_room_decay():
    outs = []
    for seed in range(4):  # one stop's noise is a few dB either way in the low bands; the bench takes medians
        x, floor = room_stop(seed=seed)
        outs.append(M.room_measures(x, SR, 0.6, 2.0, M.band_floor(floor, SR, tuple(T60)), bands=tuple(T60)))
    assert all(-20 < o["P4 knee ms"] < 5 for o in outs)  # the 10 ms smoothing and the walk back to half the slope
    for c, t60 in T60.items():
        t = np.median([o[f"P4 T60 {c}"] for o in outs])
        e = np.median([o[f"P4 early {c}"] for o in outs])
        assert abs(t / t60 - 1) < 0.1, (c, t)
        # at 8 kHz the band's smooth lower edge takes in some of the 4 kHz band, which decays 1.5x slower here (a
        # real hall's T60 changes more gently): the line comes out a little flat and the early level ~1.5 dB high
        assert abs(e - 10 * math.log10(4.0)) < (2.0 if c == 8000 else 1.0), (c, e)


def test_room_early_follows_the_direct_part():
    a = M.room_measures(room_stop(direct=1.0)[0], SR, 0.6, 2.0, bands=(1000,))
    b = M.room_measures(room_stop(direct=9.0)[0], SR, 0.6, 2.0, bands=(1000,))
    assert abs((b["P4 early 1000"] - a["P4 early 1000"]) - 10 * math.log10(10 / 2)) < 1.0


def test_channel_measures():
    rng = np.random.default_rng(3)
    n, t_on, d = int(0.6 * SR), 0.1, int(0.001 * SR)
    direct = rng.standard_normal(n + d) * ((np.arange(n + d) / SR >= t_on) & (np.arange(n + d) / SR < t_on + 0.04))
    x = np.stack([2 * direct[d:], direct[:-d]], 1)  # channel 1 hears it 1 ms later, 6 dB weaker
    late = (np.arange(n) / SR >= t_on + 0.12)[:, None]
    x = x + late * rng.standard_normal((n, 2))  # an uncorrelated field
    out = M.channel_measures(x, SR, t_on)
    for c in (500, 1000, 4000):
        assert out[f"P4 iacc early {c}"] > 0.95
        assert out[f"P4 iacc late {c}"] < 0.3
    y = x.copy()
    y[:, 0] *= 2
    assert abs(M.channel_measures(y, SR, t_on)["P4 lr 1000"] - M.channel_measures(x, SR, t_on)["P4 lr 1000"] - 6.02) < 0.05


def test_free_decays():
    notes = np.array([[60, 1.5, 2.0, 64], [64, 2.1, 2.5, 64], [62, 4.0, 4.5, 64], [65, 5.3, 5.6, 64], [96, 7.0, 7.2, 64]], float)
    pedals = {"sustain_t": np.array([0.0, 4.2, 5.0]), "sustain_v": np.array([0.0, 100.0, 0.0]),
              "sostenuto_t": np.zeros(0), "sostenuto_v": np.zeros(0)}
    # 2.5: both keys up, 1.5 s to the next onset; 5.0 (the pedal holds the 4.5 release) is 0.3 s before the next;
    # 5.6: 1.4 s; 7.2: an undamped key rings on
    assert M.free_decays(notes, pedals, 12.0) == [(2.5, 4.0), (5.6, 7.0)]


def test_repeat_runs():
    notes = np.array([[60, 2.0, 2.1, 50], [60, 2.3, 2.4, 60], [60, 2.6, 2.7, 70],  # a run under the pedal
                      [60, 6.0, 6.1, 50], [60, 6.3, 6.4, 50],  # the pedal lifts between these two
                      [64, 3.0, 3.1, 50], [64, 3.9, 4.0, 50]], float)  # 0.9 s apart: not a re-strike run
    pedals = {"sustain_t": np.array([0.0, 1.5, 6.2]), "sustain_v": np.array([0.0, 100.0, 0.0])}
    runs = M.repeat_runs(notes, pedals)
    assert [(r["pitch"], r["times"], r["velocities"]) for r in runs] == [(60, [2.0, 2.3, 2.6], [50, 60, 70])]


def test_restrike_levels_follow_a_build_up():
    f0, times = 220.0, [10.0, 10.3, 10.6]
    a = M.restrike_levels(synth(f0=f0, dur=2.0, gain=1.0), SR, 0.7, times, partials(f0))
    b = M.restrike_levels(synth(f0=f0, dur=2.0, gain=2.0), SR, 0.7, times, partials(f0))
    assert all(abs((bb - aa) - 6.02) < 0.1 for aa, bb in zip(a, b))
    assert a[1] < a[0] - 3  # no new strike in the synthetic note: the level falls between the strike times


def test_release_tracks():
    f0 = 261.6
    ps = partials(f0)
    kw = dict(f0=f0, dur=2.0, alpha=1.0, release=0.5, rel_rate=400.0, room=0.7)
    a = M.release_tracks(synth(**kw), SR, 0.7 + 0.5, ps)
    b = M.release_tracks(synth(**{**kw, "release": 0.52}), SR, 0.7 + 0.5, ps)
    # the breakpoint sits where the falling direct sound meets the room's field: ~18 ms into a 400 dB/s fall
    assert a["N10 damped"] > 0.7 and 5 < a["N10 delay"] < 30, a
    assert abs(b["N10 delay"] - a["N10 delay"] - 20) < 4, (a, b)
    assert a["N10 step"] < -2 and a["N10 slope after"] < a["N10 slope before"] - 15


def test_release_tracks_skip_shared_partials():
    ps = partials(261.6)
    x = synth(f0=261.6, release=0.5)
    assert M.release_tracks(x, SR, 1.2, ps, others=[ps])["N10 n"] == 0  # the same note doubled: nothing is clear
    out = M.release_tracks(x, SR, 1.2, ps, others=[ps[1::2]])  # every even partial is taken: odd ones remain
    assert out["N10 n"] == 7 and out["N10 damped"] > 0.7  # partials 1, 3, ..., 13 (15 is above 4 kHz)


def test_release_events():
    notes = np.array([[60, 1.0, 2.0, 64],  # C4: G4 is struck 0.1 s after its release, so no
                      [64, 1.5, 2.6, 64],  # E4: released while G4 rings, so an event with G4 as the other
                      [67, 2.1, 2.9, 64],  # G4: released alone, an event
                      [72, 4.0, 5.0, 64]], float)  # C5: released under the pedal, so no
    pedals = {"sustain_t": np.array([0.0, 4.5]), "sustain_v": np.array([0.0, 100.0]),
              "sostenuto_t": np.zeros(0), "sostenuto_v": np.zeros(0)}
    ev = M.release_events(notes, pedals)
    assert [(i, o) for i, o in ev] == [(1, [67]), (2, [])]


@pytest.mark.parametrize("f0", [55.0, 110.0, 330.0])
def test_glide(f0):
    win = max(0.04, 4.0 / f0)
    ce = 0.01 + win / 2 + np.linspace(0, 0.1, 50)
    cl = 0.62 - win / 2 - np.linspace(0, 0.1, 50)
    expected = 3.0 * (np.exp(-ce / 0.3).mean() - np.exp(-cl / 0.3).mean())
    x = synth(f0=f0, B=1e-4, alpha=1.0, glide=3.0)
    m = M.glide(x, SR, 0.7, partials(f0, 1e-4))
    assert m["N3 n"] == 8
    assert abs(m["N3 glide"] - expected) < 0.25, (m, expected)
    assert abs(M.glide(synth(f0=f0, B=1e-4, alpha=1.0), SR, 0.7, partials(f0, 1e-4))["N3 glide"]) < 0.1
    # an earlier note ringing near one partial moves that partial, not the median
    assert abs(M.glide(synth(f0=f0, B=1e-4, alpha=1.0, background=1.0), SR, 0.7, partials(f0, 1e-4))["N3 glide"]) < 0.3


def test_extra_peaks():
    f0, B = 880.0, 5e-4
    ps = partials(f0, B, 12)
    extra = [(1.02 * ps[1], 10 ** (-30 / 20)), (0.985 * ps[3], 10 ** (-35 / 20))]
    kw = dict(f0=f0, B=B, alpha=2.0, extra_bg=[(5000.0, 10 ** (-30 / 20))])
    m = M.extra_peaks(synth(**kw), SR, 0.7, ps)
    assert m["N12 n"] == 0, m["N12 peaks"]  # the steady tone was there before the note
    m = M.extra_peaks(synth(extra=extra, **kw), SR, 0.7, ps)
    assert m["N12 n between"] == 2 and m["N12 n"] == 2, m["N12 peaks"]
    amps = np.array([10 ** (-6 * math.log2(k) / 20) for k in range(1, 13) if ps[k - 1] < 8000])
    expected = 10 * math.log10((10 ** -3 + 10 ** -3.5) / (amps ** 2).sum())
    assert abs(m["N12 level between"] - expected) < 1.5, (m["N12 level between"], expected)
    got = sorted(m["N12 peaks"])
    assert abs(got[0][0] - extra[0][0]) < 1 and abs(got[1][0] - extra[1][0]) < 1
    assert abs(got[0][1] - 1200 * math.log2(1.02)) < 3 and abs(got[1][1] - 1200 * math.log2(0.985)) < 3
    # a tone 15 cents from a partial is "near"; one at 2 f_3 (36 Hz under partial 6) is a phantom
    m = M.extra_peaks(synth(extra=[(ps[2] * 2 ** (15 / 1200), 0.03), (2 * ps[2], 0.03)], **kw), SR, 0.7, ps)
    assert m["N12 n near"] == 1 and m["N12 n phantom"] == 1, m["N12 peaks"]


def test_partial_profile_reads_two_stage_decay_beating_and_the_attack():
    """Known answers: partial 1 decays in two stages (a 60 dB/s part, 0.7 of the amplitude, and a 5 dB/s part: the
    knee at ~0.15 s), partial 2 at a steady 10 dB/s and beats at 6 Hz (a second string 6 Hz up at 0.3 of its amplitude:
    +-2.3 dB), partial 3 is steady."""
    sr, t_on, dur = SR, 0.6, 2.0
    t = np.arange(int(dur * sr)) / sr
    tau = np.clip(t - t_on, 0, None)
    on = (t >= t_on).astype(float)
    p1 = 0.7 * np.exp(-tau * 60 / 8.686) + 0.3 * np.exp(-tau * 5 / 8.686)  # dB/s / 8.686 = nepers/s
    x = on * p1 * np.sin(2 * np.pi * 150 * t)
    x = x + on * np.exp(-tau * 10 / 8.686) * 0.5 * (np.sin(2 * np.pi * 301 * t) + 0.3 * np.sin(2 * np.pi * 307 * t))
    x = x + on * 0.2 * np.sin(2 * np.pi * 452 * t)
    x = x + 1e-6 * np.random.default_rng(0).standard_normal(len(t))
    x = np.stack([x, x], 1)
    pr = M.partial_profile(x, sr, t_on, np.array([150.0, 301.0, 452.0]), 1.2)
    assert pr["early"][0] < -12 and abs(pr["late"][0] + 5) < 1.5  # the knee: steep, then slow
    assert abs(pr["early"][1] + 10) < 1.5 and abs(pr["late"][1] + 10) < 1.5
    assert abs(pr["beat"][1] - 6.0) < 0.3 and pr["fluct"][1] > 1.0 and pr["periodic"][1] > 0.7
    assert pr["fluct"][2] < 0.2 and abs(pr["early"][2]) < 0.5
    assert 0 <= pr["t_peak"][0] < 0.03  # a decaying partial peaks as the 40 ms window fills
    # irregular fluctuation: three weak strings at incommensurate offsets
    y = on * np.exp(-tau * 10 / 8.686) * 0.5 * (np.sin(2 * np.pi * 301 * t) + 0.25 * np.sin(2 * np.pi * 302.3 * t)
                                                + 0.25 * np.sin(2 * np.pi * 305.1 * t + 1.0) + 0.2 * np.sin(2 * np.pi * 308.9 * t + 2.0))
    y = np.stack([y, y], 1) + 1e-6
    pi = M.partial_profile(y, sr, t_on, np.array([301.0]), 1.2)
    assert pi["periodic"][0] < pr["periodic"][1] - 0.2
    nt = M.non_tonal(x, sr, t_on, np.array([150.0, 301.0, 452.0, 603.0]))
    assert nt[((0.1, 0.4), 1000)] < -60  # nothing but partials


def _onset_note(ramp_lo=0.01, ramp_hi=0.002, delay_hi=0.0, knock=0.0, t_on=0.5, f0=150.0, seed=0):
    """Partials of f0 (amplitude 1/n, decaying 26 dB/s), each rising as a raised cosine over ``ramp_lo`` s below 1 kHz
    and ``ramp_hi`` above (those starting ``delay_hi`` s late); ``knock``: a 2-8 kHz noise burst decaying in 3 ms."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(0.8 * SR)) / SR
    x = np.zeros_like(t)
    for n in range(1, 60):
        f = n * f0 * np.sqrt(1 + 1e-4 * n * n)
        if f > 10000:
            break
        T, d = (ramp_lo, 0.0) if f < 1000 else (ramp_hi, delay_hi)
        tau = t - t_on - d
        env = np.where(tau < 0, 0, np.where(tau < T, 0.5 * (1 - np.cos(np.pi * np.clip(tau, 0, T) / T)), 1.0))
        x += env * np.exp(-np.clip(tau, 0, None) * 3) / n * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    if knock:
        hz = np.fft.rfftfreq(len(t), 1 / SR)
        nz = np.fft.irfft(np.fft.rfft(rng.standard_normal(len(t))) * ((hz >= 2000) & (hz < 8000)), len(t))
        x += knock * (t >= t_on) * np.exp(-np.clip(t - t_on, 0, None) / 0.003) * nz / nz.std()
    x += 1e-5 * rng.standard_normal(len(t))
    return np.stack([x, x], 1)


def test_onset_profile_reads_known_delays_ramps_and_a_knock():
    """Known changes to a synthetic note: the high partials 3 ms late, the low partials' rise 10 -> 30 ms (10 % of a
    raised cosine's power is reached at 0.38 of its length: +7.6 ms), a 2-8 kHz burst."""
    bands = list(M.KNOCK_BANDS)
    i500, i2k, i4k, i8k = (bands.index(c) for c in (500, 2000, 4000, 8000))
    prof = lambda **kw: M.onset_profile(_onset_note(**kw), SR, 0.5, 150.0)
    base = prof()
    late = prof(delay_hi=0.003)
    assert abs(late["arrival"][i2k] - base["arrival"][i2k] - 0.003) < 0.001
    assert abs(late["arrival"][i4k] - base["arrival"][i4k] - 0.003) < 0.001
    assert abs(late["arrival"][i500] - base["arrival"][i500]) < 0.0005
    assert late["win"][0, i4k] < base["win"][0, i4k] - 3  # the first 5 ms lose most of the band
    slow = prof(ramp_lo=0.03)
    assert abs(slow["arrival"][i500] - base["arrival"][i500] - 0.0076) < 0.002
    assert slow["rise"][i500] > base["rise"][i500] + 0.005
    assert slow["win"][0, i500] < base["win"][0, i500] - 8 and abs(slow["win"][3, i500] - base["win"][3, i500]) < 1.0
    knock = prof(knock=0.3)
    assert knock["win"][0, i8k] > base["win"][0, i8k] + 4 and knock["win"][0, i4k] > base["win"][0, i4k] + 2
    assert abs(knock["win"][0, i500] - base["win"][0, i500]) < 0.3
    # a steady note settles: 20-40 ms sits at its 50-100 ms level plus the decay in between (26 dB/s x ~55 ms)
    assert np.all(np.abs(base["win"][3, 1:] - 1.4) < 0.6)
