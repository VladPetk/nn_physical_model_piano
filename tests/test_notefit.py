import math

import numpy as np
import torch

from pianonn import notefit as NF


def test_band_masks_sum_to_one_between_the_end_centres():
    m = NF.band_masks(8192, 24000)
    f = np.fft.rfftfreq(8192, 1 / 24000)
    inside = (f >= NF.BAND_CENTERS[0]) & (f <= NF.BAND_CENTERS[-1])
    assert torch.allclose(m.sum(0)[inside], torch.ones(int(inside.sum())), atol=1e-5)
    assert m.sum(0)[f < NF.BAND_CENTERS[0] / 1.3].max() < 1e-6


def test_window_levels_read_a_sine_in_its_band_at_any_window_length():
    sr, a = 24000, 0.05
    t = torch.arange(sr) / sr
    x = (a * torch.sin(2 * math.pi * 1000.0 * t)).repeat(1, 2, 1)  # both channels
    for length in (1680, 7200):
        n_fft = 1 << math.ceil(math.log2(length))
        lv = NF.window_levels(x, torch.tensor([2000]), length, NF.band_masks(n_fft, sr), n_fft)[0]
        k = int(np.argmin(np.abs(np.log2(NF.BAND_CENTERS / 1000.0))))
        assert abs(float(lv[k]) - 10 * math.log10(a**2)) < 0.3
        assert float(lv[k]) - float(lv[k + 3]) > 40  # an octave away: nothing


def test_windows_follow_each_clips_own_onset():
    """Two clips with the same tone burst at different times read the same levels when each is placed at its own
    onset, and the burst stands over the background, so its cells count."""
    sr = 24000
    term = NF.NoteTerm(sr, "cpu")
    t = torch.arange(int(1.2 * sr)) / sr
    x = []
    for on in (0.5, 0.537):
        env = torch.exp(-(t - on).clamp(min=0) / 0.3) * (t >= on)
        x.append((0.1 * env * torch.sin(2 * math.pi * 440.0 * t) + 1e-4 * torch.randn_like(t)).repeat(2, 1))
    x = torch.stack(x)
    lv = term.levels(x, np.array([0.5, 0.537]))
    k = int(np.argmin(np.abs(np.log2(NF.BAND_CENTERS / 440.0))))
    for w in NF.WINDOWS:  # the two bands either side of the tone (the others hold the two clips' different noise)
        assert torch.allclose(lv[w][0, k: k + 2], lv[w][1, k: k + 2], atol=0.1)
    cells = term.cells(lv, np.array([440.0, 440.0]))
    assert cells["early"][:, k].all() and cells["sustain"][:, k].all()
    assert not cells["early"][:, 0].any()  # 100 Hz: below 0.7 f0
    mis = term.levels(x, np.array([0.5, 0.5]))  # the second clip placed 37 ms early reads a different part of the decay
    assert abs(float(mis["early"][1, k]) - float(lv["early"][1, k])) > 0.5


def test_gap_windows_see_a_knock_and_not_the_partials():
    """A harmonic tone (f0 330 Hz) with and without a 10 ms noise burst at its onset: the energy between the partials
    in the attack window moves by the burst, the attack-re-early level moves too, the early window does not."""
    sr, f0, on = 24000, 330.0, 0.5
    t = torch.arange(int(1.2 * sr)) / sr
    g = torch.Generator().manual_seed(0)
    tone = sum(0.1 / k * torch.sin(2 * math.pi * k * f0 * t) for k in range(1, 13)) * (t >= on) * torch.exp(-(t - on).clamp(min=0) / 0.5)
    knock = 0.1 * torch.randn(len(t), generator=g) * ((t >= on) & (t < on + 0.01))
    floor = 1e-5 * torch.randn(len(t), generator=g)
    x = torch.stack([(tone + floor).repeat(2, 1), (tone + knock + floor).repeat(2, 1)])
    win = {"att": (-0.003, 0.033, {"flat": 0.7, "rel": "early", "all_bands": True}),
           "gaps": (-0.003, 0.033, {"flat": 0.7, "gaps": True}), "early": (0.03, 0.10)}
    term = NF.NoteTerm(sr, "cpu", win)
    partials = np.tile(f0 * np.arange(1, 40), (2, 1))
    lv = term.levels(x, np.array([on, on]), partials)
    k = int(np.argmin(np.abs(np.log2(NF.BAND_CENTERS / 2000.0))))
    # the tone's own abrupt onset leaks between its partials, but far under the partials themselves
    assert float(lv["att:raw"][0, k] - lv["gaps"][0, k]) > 20
    assert float(lv["gaps"][1, k] - lv["gaps"][0, k]) > 5  # the burst, between the partials
    assert float(lv["att"][1, k] - lv["att"][0, k]) > 1  # the attack's excess over the tone grows
    assert abs(float(lv["early"][1, k] - lv["early"][0, k])) < 0.1
    cells = term.cells(lv, np.array([f0, f0]))
    assert cells["gaps"][1, k] > 0 and cells["att"][1, k] > 0


def test_cells_count_where_either_side_stands_out():
    """A band where only the model's note stands over its background counts (the model's excess is seen); one where
    neither does is left out."""
    term = NF.NoteTerm(24000, "cpu")
    nb = len(NF.BAND_CENTERS)
    bg = torch.full((1, nb), -80.0)
    rec = {"bg": bg, "early": torch.full((1, nb), -78.0), "sustain": torch.full((1, nb), -78.0)}
    mod = {"bg": bg.clone(), "early": torch.full((1, nb), -78.0), "sustain": torch.full((1, nb), -78.0)}
    mod["early"][0, 10] = -60.0  # the model alone has something at band 10
    c_rec, c_sym = term.cells(rec, np.array([100.0])), term.cells(rec, np.array([100.0]), mod)
    assert c_rec["early"][0, 10] == 0 and c_sym["early"][0, 10] == 1
    assert c_sym["early"][0, 9] == 0 and c_sym["sustain"].sum() == 0
