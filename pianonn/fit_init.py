"""Initialise a condition's recording chain from its recordings, before gradient descent.

Spectral-loss gradients are informative only close to the answer: a sinusoid's frequency only
within about an FFT bin, an onset's timing only within a window hop. So what gradient descent is
bad at finding from far away is estimated directly, by comparing renders of the untrained prior
with the recordings of the same MIDI:

1. latency: cross-correlation of onset-strength curves (mic distance + MIDI/audio offset);
2. tuning: cross-correlation of log-frequency spectra (reference pitch of that year's piano);
3. level and long-term spectrum per channel: the microphone gain, and a smooth EQ folded into
   the body FIR (third-octave, +-20 dB);
4. the stationary noise floor per channel and band, and the mains hum, measured on the silence before
   each piece's first note (:func:`floor_from_silence`). In continuous music the quietest windows are
   quiet piano, not floor: the trial's low-percentile estimate was 6-16 dB high between 300 Hz and 1.6 kHz
   and 15 dB low at 40 Hz (docs/plan_round2.md, 2.2).

Each is logged, so the per-year recording chain is also a measurement.
"""

import math

import numpy as np
import torch

from .dsp import bounded
from .losses import _mag
from .room import _minimum_phase, _octave_smooth


def _onset_strength(x, n_fft=1024, hop=48):
    S = torch.log(_mag(x, n_fft, hop) + 1e-4)
    return (S[..., 1:] - S[..., :-1]).clamp(min=0).sum(-2)  # [N, frames]


def _xcorr_lag(a, b, max_lag):
    """Lag (in frames) by which ``b`` must be delayed to line up with ``a``, summed over rows, with parabolic refinement."""
    a = a - a.mean(-1, keepdim=True)
    b = b - b.mean(-1, keepdim=True)
    lags = torch.arange(-max_lag, max_lag + 1)
    T = a.shape[-1]
    score = torch.stack([(a[..., max_lag: T - max_lag] * b[..., max_lag - l: T - max_lag - l]).sum() for l in lags.tolist()])
    i = int(score.argmax())
    frac = 0.0
    if 0 < i < len(lags) - 1:
        y0, y1, y2 = score[i - 1], score[i], score[i + 1]
        den = y0 - 2 * y1 + y2
        frac = float(0.5 * (y0 - y2) / den) if den != 0 else 0.0
    return float(lags[i]) + frac, score


def _log_f_spectrum(x, sr, f_lo=150.0, f_hi=4000.0, n_fft=16384):
    """Welch power spectrum on a 1-cent log-frequency grid, detrended over +-50 cents: ``[N, cents]``."""
    P = (_mag(x, n_fft, n_fft // 4) ** 2).mean(-1)  # [N, bins]
    f = torch.arange(P.shape[-1], device=x.device) * sr / n_fft
    grid = f_lo * 2 ** (torch.arange(int(1200 * math.log2(f_hi / f_lo)), device=x.device) / 1200)
    pos = grid / (sr / n_fft)
    i0 = pos.floor().long()
    w = pos - i0
    L = torch.log(P + 1e-12)
    y = L[..., i0] * (1 - w) + L[..., i0 + 1] * w
    k = torch.ones(1, 1, 101, device=x.device) / 101
    trend = torch.nn.functional.conv1d(y[:, None], k, padding=50)[:, 0]
    return y - trend


def _band_matrix(model, n_fft):
    """The noise bank's band masks on an ``n_fft``-point grid, normalised to white-equivalent power."""
    masks = model.noise.band_masks(n_fft)  # [bands, bins]
    return masks / (masks.sum(-1, keepdim=True) * 0.375 * n_fft)


@torch.no_grad()
def floor_from_silence(model, cond, clips, log=print, hum_halfwidth=2.5):
    """Set the condition's noise floor and hum from recorded silence: ``clips`` is a list of ``[ch, n]`` tensors
    (e.g. :meth:`pianonn.data.MaestroSegments.silence_clips`). Per clip, a periodogram of the 20 Hz high-passed
    signal: the hum lines' power is read off (above the local floor) and the lines are replaced by that floor;
    the band levels are then the white-equivalent power in the noise bank's bands. Medians over clips."""
    from .losses import highpass
    from .room import HUM_LINES, MAINS_HZ

    sr = model.cfg.sample_rate
    dev = model.room.floor_ref_db.device
    bands, hums = [], []
    for x in clips:
        x = highpass(x.to(dev).float(), sr)
        n = x.shape[-1]
        w = torch.hann_window(n, device=dev)
        P = torch.fft.rfft(x * w).abs() ** 2 / w.pow(2).sum()  # white noise of variance s2 -> E[P] = s2
        f = torch.fft.rfftfreq(n, 1 / sr).to(dev)
        width = max(hum_halfwidth, 4.0 * sr / n)  # the Hann main lobe is +-2 bins
        line_p = []
        for k in range(1, HUM_LINES + 1):
            fk = MAINS_HZ * k
            line = (f - fk).abs() <= width
            near = ((f - fk).abs() > width) & ((f - fk).abs() <= width + 15.0)
            local = P[:, near].median(-1).values[:, None]  # [ch, 1]
            line_p.append(2.0 * (P[:, line] - local).clamp(min=0).sum(-1) / n)  # RMS power of the line
            P[:, line] = local.expand(-1, int(line.sum()))
        M = model.noise.band_masks(n) * (f >= 25.0)  # the high-passed bins would bias the lowest band down
        bands.append((P @ M.T) / M.sum(-1))  # [ch, bands]
        hums.append(torch.stack(line_p, -1))  # [ch, lines]
    E, H = torch.stack(bands).median(0).values, torch.stack(hums).median(0).values
    model.room.floor_ref_db[cond] = 10 * torch.log10(E.clamp(min=1e-14))
    model.room.hum_ref_db[cond] = 10 * torch.log10(H.clamp(min=1e-20))
    model.room.raw_floor.data[cond] = 0.0
    model.room.raw_hum.data[cond] = 0.0
    fl = model.room.floor_ref_db[cond].mean(0)
    est = {"floor_db": [round(float(v), 1) for v in fl[::4]],
           "hum_db": [[round(float(v), 1) for v in c] for c in model.room.hum_ref_db[cond]], "silence_clips": len(clips)}
    log(f"init: noise floor from {len(clips)} silences (white-equivalent dBFS, every 4th band) {est['floor_db']}; "
        f"hum at {MAINS_HZ:.0f}/{2 * MAINS_HZ:.0f}/{3 * MAINS_HZ:.0f} Hz (RMS dBFS per channel) {est['hum_db']}")
    return est


@torch.no_grad()
def initialise_from_data(model, batches, log=print, max_latency=0.06, max_cents=60, tuning=True, silence=None):
    """``batches``: collated examples of one condition, on the model's device. Returns a dict of estimates.
    ``tuning=False`` keeps the tuning (e.g. when it came from :func:`apply_mined_priors`). ``silence``: clips of
    that condition's recorded silence for the floor (without them the floor is left as it is)."""
    cfg = model.cfg
    sr = cfg.sample_rate
    conds = torch.cat([b["condition"] for b in batches]).unique()
    assert len(conds) == 1, "initialise one recording condition at a time (one MAESTRO year per call)"
    cond = int(conds[0])
    s = int(batches[0]["loss_start"][0])

    def render():
        return [model(b, b["audio"].shape[-1], residual=False, floor=False)["audio"][..., s:] for b in batches]

    targets = [b["audio"][..., s:] for b in batches]
    mono = lambda xs: torch.cat([x.mean(1) for x in xs])
    est = {}

    # 1. latency: shift the body FIRs (all channels) so the onsets line up
    hop = 48
    lag, _ = _xcorr_lag(_onset_strength(mono(targets)), _onset_strength(mono(render())), int(max_latency * sr / hop))
    shift = int(round(lag * hop))
    body = model.room.body.data[cond]
    if shift > 0:
        body.copy_(torch.cat([body.new_zeros(*body.shape[:-1], shift), body[..., :-shift]], -1))
    elif shift < 0:
        body.copy_(torch.cat([body[..., -shift:], body.new_zeros(*body.shape[:-1], -shift)], -1))
    est["latency_ms"] = 1000 * shift / sr
    log(f"init: latency {est['latency_ms']:+.1f} ms (model {'delayed' if shift > 0 else 'advanced'} to match)")

    # 2. tuning: one offset in cents for the condition
    lt, lm = _log_f_spectrum(mono(targets), sr), _log_f_spectrum(mono(render()), sr)
    cents, _ = _xcorr_lag(lt, lm, max_cents)
    cur = float(bounded(model.physics.cond_cents.data[cond], 30.0))
    if tuning:
        new = max(-29.0, min(29.0, cur + cents))
        model.physics.cond_cents.data[cond] = 30.0 * math.atanh(new / 30.0)
        est["tuning_cents"] = new
        log(f"init: tuning {new:+.1f} cents re the prior's A440 stretch")
    else:
        est["tuning_residual_cents"] = cents
        log(f"init: tuning kept (spectra agree within {cents:+.1f} cents)")

    # 3. level and long-term spectrum per channel -> mic gain + body EQ
    n_fft = 2048
    tgt, pred = torch.cat(targets), torch.cat(render())
    Pt = torch.stack([(_mag(tgt[:, c], n_fft) ** 2).mean((0, -1)) for c in range(cfg.channels)])
    Pm = torch.stack([(_mag(pred[:, c], n_fft) ** 2).mean((0, -1)) for c in range(cfg.channels)])
    freqs = torch.fft.rfftfreq(n_fft, 1 / sr).to(Pt.device).double()
    band = (freqs >= 100) & (freqs <= 5000)
    est["level_db"], est["eq_db"] = [], []
    for c in range(cfg.channels):
        ratio = _octave_smooth(Pt[c].double(), freqs) / _octave_smooth(Pm[c].double(), freqs).clamp(min=1e-30)
        eq_db = 10 * torch.log10(ratio.clamp(min=1e-30))
        level = float(eq_db[band].mean())
        eq_db = (eq_db - level).clamp(-20, 20)
        eq_db = torch.where(freqs < 30, eq_db[(freqs >= 30).nonzero()[0, 0]], eq_db)
        eq_db = torch.where(freqs > 11000, eq_db[(freqs <= 11000).nonzero()[-1, 0]], eq_db)
        # min-phase EQ on the body's FFT grid, applied by convolution (truncated to the body length)
        Lb = body.shape[-1]
        nb = 2 * Lb
        fb = torch.fft.rfftfreq(nb, 1 / sr).to(Pt.device).double()
        g = torch.from_numpy(np.interp(fb.cpu().numpy(), freqs.cpu().numpy(), eq_db.cpu().numpy())).to(Pt.device)
        H = _minimum_phase(g * math.log(10) / 20)
        body[c] = torch.fft.irfft(torch.fft.rfft(body[c].double(), nb) * H, nb)[:Lb].to(body.dtype)
        model.room.mic_gain_db.data[cond, c] += level
        est["level_db"].append(level)
        est["eq_db"].append({int(f): round(float(eq_db[(freqs - f).abs().argmin()]), 1) for f in (63, 125, 250, 500, 1000, 2000, 4000, 8000)})
    log(f"init: mic gain {['%+.1f dB' % v for v in est['level_db']]}; body EQ (dB) {est['eq_db']}")

    # 4. noise floor and hum, from recorded silence
    if silence:
        est.update(floor_from_silence(model, cond, silence, log=log))
    else:
        log("init: no silence clips, noise floor left as it is")
    return est


REGISTERS = ((21, 36), (36, 48), (48, 60), (60, 72), (72, 84), (84, 96), (96, 109))


@torch.no_grad()
def apply_mined_priors(model, mined, log=print, min_notes=5, min_reliable=3, b_per_key=True, b_window=1,
                       b_max_window=6, set_cents=True):
    """Start inharmonicity and stretch from values tracked on isolated notes of the same recordings
    (``scripts/mine_notes.py``). Frequencies are what spectral gradients cannot find from far away (an error in B
    of 2x puts the high bass partials tens of Hz off), so they come from measurement and are then only refined.

    B, with ``b_per_key``: per key, the median ratio to the prior over the reliable notes within ``b_window`` keys
    (widened up to ``b_max_window`` until ``min_notes`` are in), interpolated across keys without any, flat beyond
    the measured keys. The piano's B is flat through the wound bass and rises over a few keys where the plain strings
    begin (2018: ~5e-5 up to MIDI 42, ~1.3e-4 by 49-51); per-register medians put that knee between register centres
    and left the round-2 models' B 15-20 % low from MIDI 43 to 63. Without ``b_per_key``: per-register medians
    interpolated over the keys (rounds 1-2).

    The stretch: per-register medians, into the per-key cents offsets (shared by all conditions: right for a
    single-year fit); ``set_cents=False`` leaves them (re-applying B to a trained checkpoint).
    """
    from .physics import LOWEST_MIDI, N_KEYS, key_curve

    ph = model.physics
    notes = [n for n in mined["notes"] if not n["suspect"]]
    b_pts, c_pts = [], []
    for lo, hi in REGISTERS:
        ns = [n for n in notes if lo <= n["pitch"] < hi]
        if len(ns) >= min_notes:
            centre = float(np.median([n["pitch"] for n in ns])) - LOWEST_MIDI
            prior_c = float(np.interp(centre, np.arange(N_KEYS), ph.prior_cents.cpu().numpy()))
            c_pts.append((centre, float(np.median([n["cents"] for n in ns])) - prior_c))  # offset from the prior
            rel = [n for n in ns if n["B_reliable"]]
            if len(rel) >= min_reliable:
                centre_b = float(np.median([n["pitch"] for n in rel])) - LOWEST_MIDI
                k = int(round(centre_b))
                prior = float(torch.exp(ph.prior_log_B[k]))
                b_pts.append((centre_b, math.log(float(np.median([n["B"] for n in rel])) / prior)))

    def curve(pts):
        pts = sorted(pts)
        if pts[0][0] > 0:
            pts = [(0.0, pts[0][1])] + pts
        if pts[-1][0] < N_KEYS - 1:
            pts = pts + [(float(N_KEYS - 1), pts[-1][1])]
        if len(pts) == 1:
            pts = [(0.0, pts[0][1]), (float(N_KEYS - 1), pts[0][1])]
        return key_curve(pts).to(ph.raw_log_B.device)

    rel = [n for n in notes if n["B_reliable"]]
    if b_per_key and rel:
        prior = torch.exp(ph.prior_log_B).cpu().numpy().astype(np.float64)
        pk = np.array([n["pitch"] - LOWEST_MIDI for n in rel])
        lr = np.log([n["B"] for n in rel]) - np.log(prior[pk.clip(0, N_KEYS - 1)])
        ratio = np.full(N_KEYS, np.nan)
        for k in range(N_KEYS):
            for w in range(b_window, b_max_window + 1):
                sel = np.abs(pk - k) <= w
                if sel.sum() >= min_notes:
                    ratio[k] = np.median(lr[sel])
                    break
        ok = np.isfinite(ratio)
        if ok.any():
            keys = np.arange(N_KEYS)
            b_pts = [(float(k), float(ratio[k])) for k in keys[ok]]
            log_ratio = torch.tensor(np.interp(keys, keys[ok], ratio[ok]), dtype=torch.float32).clamp(-1.4, 1.4)
            ph.raw_log_B.copy_(1.5 * torch.atanh(log_ratio.to(ph.raw_log_B.device) / 1.5))
    elif b_pts:
        log_ratio = curve(b_pts).clamp(-1.4, 1.4)
        ph.raw_log_B.copy_(1.5 * torch.atanh(log_ratio / 1.5))
    if c_pts and set_cents:  # offsets interpolated, held beyond the measured registers: the prior's curvature at the ends stays
        off = curve(c_pts).clamp(-29, 29)
        ph.raw_cents.copy_(30.0 * torch.atanh(off / 30.0))
    shown = b_pts[:: max(1, len(b_pts) // 12)] if b_per_key else b_pts
    log(f"mined priors from {len(notes)} notes: B x " + " ".join(f"{LOWEST_MIDI + k:.0f}:{math.exp(v):.2f}" for k, v in shown)
        + " | cents re prior " + " ".join(f"{LOWEST_MIDI + k:.0f}:{v:+.1f}" for k, v in c_pts))
    return {"B_ratio": [(LOWEST_MIDI + k, math.exp(v)) for k, v in b_pts],
            "cents_re_prior": [(LOWEST_MIDI + k, v) for k, v in c_pts]}


def partial_envelopes(model, u, cond, times):
    """Each partial's energy envelope ``[V, 88, P, T]`` for every key struck at velocities ``u[V]`` under condition
    ``cond`` (an int), summed incoherently over its modes (beats ignored): ``sum_m |A_m|^2 exp(-2 alpha_m t)``. The
    coupled model's vertical (with its radiation tie) and in-plane buses together; keys held, no pedal."""
    dev = times.device
    V = u.shape[0]
    ki = torch.arange(88, device=dev)[None].expand(V, -1)
    m = model.physics.modes(ki, u[:, None].expand(-1, 88), torch.zeros(V, 88, device=dev),
                            torch.full((V,), cond, device=dev), phantoms=False)
    if "bus_amp" in m:
        a2 = (m["bus_amp"][..., :2, :].double() ** 2).sum((-1, -2))
    else:
        a2 = m["amp"].double() ** 2
    al = m["alpha"].double()
    return (a2[..., None] * torch.exp(-2 * al[..., None] * times.double())).sum(-2)


@torch.no_grad()
def longitudinal_energy(model, u, cond, seconds=0.5, keys_per_slice=4):
    """The coupled model's longitudinal force (``NeuralPhysicalPiano._longitudinal``, its scale included): energy over
    the first ``seconds`` of every key held at velocity ``u``, ``[88]``."""
    from .oscbank import bus_bank

    dev = next(model.parameters()).device
    sr = model.cfg.sample_rate
    L = int(seconds * sr)
    out = []
    for k0 in range(0, 88, keys_per_slice):
        ki = torch.arange(k0, min(88, k0 + keys_per_slice), device=dev)[None]
        K = ki.shape[1]
        m = model.physics.modes(ki, torch.full((1, K), u, device=dev), torch.zeros(1, K, device=dev),
                                torch.tensor([cond], device=dev), phantoms=True)
        f, al = m["freq"][0].flatten(1), m["alpha"][0].flatten(1)
        ba = m["bus_amp"][0][..., 2:, :].flatten(1, 2)  # the two longitudinal sums
        z = torch.zeros(K, device=dev)
        Y = bus_bank(f, al, ba, torch.zeros_like(f), m["tc"][0], torch.zeros(K, L, device=dev), z, z,
                     torch.zeros(K, dtype=torch.float64, device=dev), torch.full((K, 1), math.inf, device=dev), 0.0, sr)
        long = {k: (v[0] if torch.is_tensor(v) and v.dim() >= 2 else v) for k, v in m["long"].items()}
        J = long["lm_f"].shape[-1]
        st = {"hp": torch.zeros(K, 2, device=dev), "lm": torch.zeros(K, J, dtype=torch.complex64, device=dev)}
        out.append(model._longitudinal(Y, long, torch.arange(K, device=dev), st).double().pow(2).sum(-1))
    return torch.cat(out)


@torch.no_grad()
def phantom_energy(teacher, u, cond, seconds=0.5):
    """A mode model's phantom partials: energy over the first ``seconds`` of every key held at velocity ``u``, ``[88]``
    (in samples' units, as ``longitudinal_energy``)."""
    dev = next(teacher.parameters()).device
    sr = teacher.cfg.sample_rate
    ki = torch.arange(88, device=dev)[None]
    m = teacher.physics.modes(ki, torch.full((1, 88), u, device=dev), torch.zeros(1, 88, device=dev),
                              torch.tensor([cond], device=dev), phantoms=True)
    if "ph_amp" not in m:
        return torch.zeros(88, dtype=torch.float64, device=dev)
    a2, al = m["ph_amp"][0].double() ** 2, m["ph_alpha"][0].double()
    ok = m["ph_freq"][0] < 0.48 * sr
    return (a2 / 2 * -torch.expm1(-2 * al * seconds) / (2 * al) * sr * ok).sum(-1)


@torch.no_grad()
def calibrate_longitudinal(model, teacher, cond, u=0.6, below_db=20.0, log=print):
    """Set the coupled model's longitudinal level per key (``coupled.long_cal_db``) so that the force's energy over
    0.5 s at velocity ``u`` is ``below_db`` under the teacher's phantom partials' (docs/physics_revamp.md 4); keys where
    either is silent get -60 dB. Below, because at equal energy the force (every sum and difference tone of every
    mode, and a spike at the attack) scored far worse than the phantoms (scratch check: composite 0.83 against 0.66
    without it, B_comp 0.62); its trainable level (+-20 dB) reaches the phantoms' if the score asks for it. Returns
    the offsets (dB)."""
    cp = model.physics.coupled
    cp.long_cal_db.zero_()
    e_l, e_p = longitudinal_energy(model, u, cond), phantom_energy(teacher, u, cond)
    ok = (e_l > 0) & (e_p > 0)
    cal = torch.where(ok, 10 * torch.log10(e_p.clamp(min=1e-300) / e_l.clamp(min=1e-300)) - below_db,
                      torch.full_like(e_l, -60.0))
    cp.long_cal_db.copy_(cal.float().clamp(-60.0, 60.0))
    if log:
        q = cal[ok]
        log(f"longitudinal level calibrated to {below_db:.0f} dB under the phantoms at u={u}: offsets {float(q.min()):+.1f} to {float(q.max()):+.1f}"
            f" dB (median {float(q.median()):+.1f}) over {int(ok.sum())} keys")
    return cal


def distill_coupled(model, teacher, cond, steps=400, lr=0.03, log=print, frozen=()):
    """Fit the coupled strings' own parameters (``physics.coupled``) so that each partial's energy envelope matches a
    mode model's (``teacher``; docs/physics_revamp.md 10): every key at four velocities, on a log time grid to 8 s,
    in dB with a floor 80 dB under the note's strike, each partial weighted by its share of the teacher's energy at
    that time, plus the note's total. The coupled model's start from a mode model's weights (``init_coupled``) keeps
    its levels at the strike but not its decays: the in-plane polarisation and the aftersound follow from the
    coupling, the mode model's from free per-key levels. Then the longitudinal force's level per key is set to the
    teacher's phantom partials' (``calibrate_longitudinal``). Returns the envelope losses before and after."""
    dev = next(model.parameters()).device
    times = torch.cat([torch.zeros(1), torch.logspace(math.log10(0.05), math.log10(8.0), 16)]).to(dev)
    u = torch.tensor([0.25, 0.5, 0.75, 0.95], device=dev)
    with torch.no_grad():
        target = partial_envelopes(teacher, u, cond, times)
    sounds = (target.sum(-2)[..., 0] > 0).double()[..., None, None]  # notes with a partial below Nyquist
    floor = target.sum(-2)[..., :1, None].clamp(min=1e-30) * 1e-8  # [V, 88, 1, 1]
    w = (target + floor) / (target + floor).sum(-2, keepdim=True) * sounds
    t_db = 10 * torch.log10(target + floor)
    t_tot = 10 * torch.log10(target.sum(-2) + floor[..., 0])
    params = [p for k, p in model.physics.coupled.named_parameters() if p.requires_grad and not k.startswith(tuple(frozen))]
    opt = torch.optim.Adam(params, lr=lr)

    def loss_fn():
        e = partial_envelopes(model, u, cond, times)
        d = 10 * torch.log10(e + floor) - t_db
        tot = 10 * torch.log10(e.sum(-2) + floor[..., 0]) - t_tot
        return (w * d ** 2).sum(-2).mean() + (sounds[..., 0] * tot ** 2).mean()

    with torch.no_grad():
        first = float(loss_fn())
    for step in range(steps):
        for g in opt.param_groups:  # down to a tenth: the fit settles (constant, repeat runs ended 1.0-2.5 dB rms)
            g["lr"] = lr * (1 - 0.9 * step / steps)
        opt.zero_grad(set_to_none=True)
        loss = loss_fn() + model.physics.coupled.regularizer()
        loss.backward()
        opt.step()
        if log and (step + 1) % max(1, steps // 8) == 0:
            log(f"distil coupled strings: step {step + 1}/{steps}, envelope loss {float(loss):.2f} dB^2")
    with torch.no_grad():
        last = float(loss_fn())
    if log:
        log(f"distil coupled strings: envelope loss {first:.2f} -> {last:.2f} dB^2 (rms {math.sqrt(first):.1f} -> "
            f"{math.sqrt(last):.1f} dB)")
    calibrate_longitudinal(model, teacher, cond, log=log)
    return first, last
