import math

import torch

from pianonn.partial_view import PartialView, note_sizes

SR = 24000


def _note(f0, n_partials=12, seconds=2.0, amp_db=None, decay=3.0):
    t = torch.arange(int(seconds * SR)) / SR
    x = torch.zeros_like(t)
    for n in range(1, n_partials + 1):
        a = 0.1 / n * (10 ** (amp_db[n] / 20) if amp_db and n in amp_db else 1.0)
        x = x + a * torch.sin(2 * math.pi * f0 * n * t) * torch.exp(-decay * t)
    return x


def _notes(f0, n_partials=12):
    f = (f0 * torch.arange(1, n_partials + 1, dtype=torch.float32))[None, None]  # [1, 1, P]
    amp = (0.1 / torch.arange(1, n_partials + 1, dtype=torch.float32))[None, None, :, None]
    return {"freq": f, "amp": amp, "alpha": torch.full_like(amp, 3.0), "onset": torch.zeros(1, 1),
            "mask": torch.ones(1, 1, dtype=torch.bool)}


def test_sizes_hold_four_periods():
    s = note_sizes(torch.tensor([27.5, 110.0, 440.0, 4000.0]), SR)
    assert s.tolist() == [4096, 1024, 256, 256]


def test_a_level_change_reads_its_size():
    """+6 dB on everything reads log10(4) = 0.60 (a slow decay: every reading stays over the floors); the same
    signal reads 0."""
    x = _note(220.0, decay=0.5)[None, None].repeat(1, 2, 1)
    v = PartialView(SR)
    same = v(x, x, _notes(220.0))
    assert same["partials"].abs().item() < 1e-6 and same["between"].abs().item() < 1e-6
    d = v(2 * x, x, _notes(220.0))
    assert abs(d["partials"].item() - math.log10(4)) < 0.03


def test_one_partial_is_seen_and_the_gaps_are_not():
    """Partial 3 +10 dB: the partial view reads it (about 1 of 12 tracks x 1.0); noise between the partials moves the
    between view much more than the partial view."""
    v = PartialView(SR)
    x = _note(220.0)[None, None].repeat(1, 2, 1)
    y = _note(220.0, amp_db={3: 10.0})[None, None].repeat(1, 2, 1)
    d = v(y, x, _notes(220.0))
    assert 0.5 / 12 < d["partials"].item() < 1.5 / 12
    g = torch.Generator().manual_seed(0)
    noise = 3e-4 * torch.randn(1, 2, x.shape[-1], generator=g)
    dn = v(x + noise, x, _notes(220.0))
    assert dn["between"].item() > 5 * dn["partials"].item()


def test_noise_between_the_partials_pools():
    """The noise between the partials +6 dB reads log10(4) = 0.60 pooled; the same signal reads 0."""
    v = PartialView(SR)
    x = _note(220.0, decay=0.5)[None, None].repeat(1, 2, 1)
    g = torch.Generator().manual_seed(0)
    noise = 0.3 * torch.randn(1, 2, x.shape[-1], generator=g)  # over the partials' leakage into the gaps
    same = v(x + noise, x + noise, _notes(220.0))
    assert same["between_pooled"].abs().item() < 1e-6
    d = v(x + 2 * noise, x + noise, _notes(220.0))
    assert abs(d["between_pooled"].item() - math.log10(4)) < 0.05


def test_exposure_is_the_notes_share():
    """A quiet note under a loud one an octave up: its odd partials are its own (share 1), its even partials sit on the
    loud note's (share (0.1/0.5)^2 / (1 + that) at the start, the same decay on both)."""
    v = PartialView(SR)
    f = torch.stack([220.0 * torch.arange(1, 13), 440.0 * torch.arange(1, 13)])[None].float()
    amp = torch.tensor([0.1, 0.5])[None, :, None, None] / torch.arange(1, 13, dtype=torch.float32)[None, None, :, None]
    notes = {"freq": f, "amp": amp, "alpha": torch.full_like(amp, 3.0), "onset": torch.zeros(1, 2),
             "mask": torch.ones(1, 2, dtype=torch.bool)}
    n_idx = torch.tensor([0, 0, 1])
    p_idx = torch.tensor([0, 1, 0])  # 220 Hz (alone), 440 Hz (on the loud note's fundamental), that fundamental
    fr = f[0, n_idx, p_idx]
    t = torch.arange(10) * 0.01
    share = v.exposure(notes, 0, n_idx, p_idx, fr, torch.tensor([512, 512, 512]), torch.zeros(3), t)
    assert torch.allclose(share[0], torch.ones(10))
    r = (0.1 / 2) ** 2 / 0.5 ** 2
    assert torch.allclose(share[1], torch.full((10,), r / (1 + r)), rtol=1e-3)
    assert torch.allclose(share[2], torch.full((10,), 1 / (1 + r)), rtol=1e-3)


def test_a_small_mistuning_is_tolerated():
    """A partial a third of a bin off its expected frequency reads within 1 dB: the view does not train tuning."""
    v = PartialView(SR)
    x = _note(220.0)[None, None].repeat(1, 2, 1)
    y = _note(220.0 * 2 ** (5 / 1200))[None, None].repeat(1, 2, 1)  # 5 cents sharp
    d = v(y, x, _notes(220.0))
    assert d["partials"].item() < 0.1


def test_scored_frames_read_context_not_a_mirror():
    """``extend_scored`` + ``crop_frames``: as many frames as a centred STFT of the scored slice, the same readings in
    the slice's interior, and an attack 20 ms before the slice's end not counted again through its mirror (reflect
    padding puts a copy 20 ms after the end, which the last frames' 8192-point windows read: ~1.6 times the energy)."""
    from pianonn.losses import PianoLoss, extend_scored
    s, T = SR, 2 * SR
    full = torch.zeros(1, 1, s + T)
    t0 = s + T - int(0.02 * SR)
    full[..., t0: t0 + 48] = torch.hann_window(48)  # a click
    piano = PianoLoss(SR, weights=(1.0, 0.0, 0.0))
    old = piano.band_list(full[..., s:])
    x, crop = extend_scored(full, s)
    new = piano.band_list(x, crop)
    assert [e.shape for e in old] == [e.shape for e in new]
    ratio = (old[0].sum() / new[0].sum()).item()  # the 8192-point group (bands below 200 Hz)
    assert 1.4 < ratio < 2.0, ratio
    # a steady tone: the interior frames agree (to -50 dB re the loudest band: the edges' high-pass ringing differs)
    tt = torch.arange(s + T) / SR
    tone = (0.1 * torch.sin(2 * math.pi * 220 * tt))[None, None]
    a, b = piano.band_list(tone[..., s:]), piano.band_list(*extend_scored(tone, s))
    for ea, eb in zip(a, b):
        mid = slice(40, -40)
        assert torch.allclose(ea[..., mid], eb[..., mid], rtol=1e-3, atol=1e-5 * float(ea.max()))


def test_sound_onsets_follow_the_delay_law():
    from pianonn.composite import sound_onsets
    batch = {"pitch": torch.tensor([[60, 30, 108]]), "velocity": torch.tensor([[64, 64, 64]])}
    on = sound_onsets(batch, torch.zeros(1, 3))
    assert torch.allclose(on, torch.tensor([[0.00717, 0.00717 + 0.00819, 0.0]]), atol=1e-6)


def test_lowest_key_fundamental_is_read():
    f = torch.tensor([[[27.5, 55.0]]])
    notes = {"freq": f, "amp": torch.ones(1, 1, 2, 1), "alpha": torch.ones(1, 1, 2, 1), "onset": torch.zeros(1, 1),
             "mask": torch.ones(1, 1, dtype=torch.bool)}
    n_idx, p_idx, fr, _ = PartialView(SR).select(notes, 0, 2.0)
    assert 27.5 in fr.tolist()
