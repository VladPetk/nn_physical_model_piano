"""The full differentiable piano: strings -> sympathetic resonance -> noise -> soundboard body + hall.

Signal flow for one example (``B`` examples of ``n_samples`` at the model rate, stereo):

* Control curves (pedals, key rolls, the damper integral) cover a *history* of ``hist_frames``
  frames before sample 0 as well as the rendered window, so dampers, sostenuto latches and
  re-strikes of notes struck before the window are exact (review 3, F8).
* Strings: closed-form damped sinusoids (transverse partials x coupled modes, plus phantom
  partials), chunked and skipped once inaudible (the activity test includes the dampers, F12).
* Knock impulse, mechanical noises, and (with ``residual=True``) the residual stack:
  R1 per-note corrections of the modal parameters, R2 a learned attack-noise component,
  R3 frame-rate band gains on the dry signal plus a filtered-noise path (review 3, section 9.2).
* Body + hall per microphone channel, then the stationary noise floor.
"""

import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .config import PianoConfig
from .dsp import bounded, fft_convolve, frames_to_samples, linear_recurrence, sample_curve, sample_keyed
from .physics import LOWEST_MIDI, N_KEYS, PianoPhysics, hammer_velocity, key_curve, log_f_bumps
from .oscbank import osc_bank
from .room import Room

class ContextNet(nn.Module):
    """The learned residual: a causal GRU over the performance (piano roll + pedals, including the
    history before the window) that predicts bounded corrections to the physical model.

    Per note (R1, modal language; R2, attack): gain, brightness (via the contact time), decay and its
    tilt over frequency, a smooth spectral correction (log-f bumps), the knock's level and spectrum, and
    a slower learned attack-noise component (level, spectrum, decay time).
    Per frame (R3): band gains on the dry signal (+-6 dB) and a filtered-noise path.

    Output layers are zero-initialised, so switching the residual on starts from pure physics, and
    every output is bounded, so it can reshape what the physics renders but cannot replace it.
    """

    NOTE = {"gain_db": (1, 6.0), "log_fc": (1, 0.3), "log_decay": (1, 0.3), "decay_tilt": (1, 0.15),
            "spec": (6, 0.46), "log_knock": (1, 1.0), "knock_spec": (8, 1.0),
            "att_level": (1, 3.0), "att_spec": (8, 1.5), "att_log_tau": (1, 1.0)}
    FRAME = {"band_gain": (16, 0.69), "noise_level": (16, 4.0)}

    def __init__(self, cfg: PianoConfig):
        super().__init__()
        self.cfg = cfg
        H = cfg.ctx_hidden
        self.inp = nn.Linear(2 * N_KEYS + 3, H)
        self.gru = nn.GRU(H, H, batch_first=True)
        self.key_emb = nn.Embedding(N_KEYS, 16)
        self.cond_emb = nn.Embedding(cfg.n_conditions, 16)
        n_note = sum(n for n, _ in self.NOTE.values())
        n_frame = sum(n for n, _ in self.FRAME.values())
        self.head = nn.Sequential(nn.Linear(H + 33, H), nn.GELU(), nn.Linear(H, n_note))
        self.frame_head = nn.Sequential(nn.Linear(H + 16, 64), nn.GELU(), nn.Linear(64, n_frame))
        for head in (self.head, self.frame_head):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    @staticmethod
    def _split(z, spec):
        out, i = {}, 0
        for name, (n, s) in spec.items():
            v = s * torch.tanh(z[..., i: i + n] / s)
            out[name] = v[..., 0] if n == 1 else v
            i += n
        return out

    def forward(self, onset_roll, key_down, pedals, ki, u, onset_frame, cond, hist_frames):
        x = torch.cat([onset_roll, key_down, pedals], 1).transpose(1, 2)
        h, _ = self.gru(torch.tanh(self.inp(x)))
        F, H = h.shape[1], h.shape[2]
        idx = torch.ceil(onset_frame).clamp(0, F - 1).long()
        h_note = h.gather(1, idx[..., None].expand(-1, -1, H))
        cemb = self.cond_emb(cond)
        feats = torch.cat([h_note, self.key_emb(ki), cemb[:, None].expand(-1, ki.shape[1], -1), u[..., None]], -1)
        note = self._split(self.head(feats), self.NOTE)
        hf = h[:, hist_frames:]
        frame = self._split(self.frame_head(torch.cat([hf, cemb[:, None].expand(-1, hf.shape[1], -1)], -1)), self.FRAME)
        return note, {k: v.transpose(1, 2) for k, v in frame.items()}  # frame outputs [B, bands, F]


class SympatheticBank(nn.Module):
    """Every string on the instrument as a resonator driven by the bridge.

    With the dampers down only the undamped treble strings respond; lift the
    pedal and the whole bank rings. Each key is driven by the bridge signal
    minus its own strings (those are already modelled). The damper state is
    time-varying, so this is a linear recurrence with time-varying poles, solved
    by the chunked parallel scan in :func:`linear_recurrence`.

    Drive coupling follows reciprocity: a string receives energy through the
    bridge at the rate it loses energy to it, ``kappa = alpha_prompt -
    alpha_after`` (the bridge part of its decay). The resonant gain is then
    ``G * kappa / alpha``: the fraction of the string's losses that go through
    the bridge, not an accident of the register.
    """

    def __init__(self, cfg: PianoConfig):
        super().__init__()
        self.cfg = cfg
        self.log_gain = nn.Parameter(torch.log(0.262 * key_curve([(0, 0.08), (12, 0.12), (30, 0.15), (67, 0.15), (87, 0.10)])))

    def _block(self, drive, es, freq, alpha, alpha_damp, gin, state):
        sr = self.cfg.sample_rate
        decay = (alpha[..., None] + es[:, :, None, :] * alpha_damp[..., None]).clamp(max=1000.0) / sr
        omega = (2 * math.pi / sr) * freq[..., None].expand_as(decay)
        log_a = torch.complex(-decay, omega)
        x = drive[:, :, None, :] * gin[..., None]
        y, state = linear_recurrence(torch.complex(x, torch.zeros_like(x)), log_a, state, self.cfg.rec_chunk)
        return y.real.sum((1, 2)), state

    def forward(self, bridge, own, key_modes, engagement, start, state=None):
        """``bridge[B,L]`` total string signal, ``own[B,88,L]`` per-key share, for frame-grid samples ``[start, start+L)``."""
        cfg = self.cfg
        S = cfg.symp_partials
        freq = key_modes["freq"][..., :S, 0]
        alpha = key_modes["alpha"][..., :S, 0]
        kappa = (alpha - key_modes["alpha"][..., :S, 1]).clamp(min=0)
        alpha_damp = key_modes["alpha_damp"][..., :S]
        valid = (freq < 0.45 * cfg.sample_rate).to(freq.dtype)
        gin = self.log_gain.exp()[None, :, None] * kappa / cfg.sample_rate * valid
        drive = bridge[:, None, :] - own
        es = frames_to_samples(engagement, start, bridge.shape[-1], cfg.hop)
        args = (drive, es, freq, alpha, alpha_damp, gin, state)
        if cfg.checkpoint and torch.is_grad_enabled():
            return checkpoint(self._block, *args, use_reentrant=False)
        return self._block(*args)


def _key_bottom_delay(v):
    """Key-bottom contact relative to hammer-string contact (s) vs hammer speed (m/s):
    p +12 ms, mf +0.5 ms, f -2.5 ms, ff -5 ms (Askenfelt & Jansson, Five Lectures, Figs. 4-5)."""
    vs = torch.tensor([1.0, 2.8, 5.5, 9.0], device=v.device, dtype=v.dtype)
    ds = torch.tensor([0.012, 0.0005, -0.0025, -0.005], device=v.device, dtype=v.dtype)
    i = torch.searchsorted(vs, v.clamp(vs[0], vs[-1]).contiguous()).clamp(1, len(vs) - 1)
    w = ((v.clamp(vs[0], vs[-1]) - vs[i - 1]) / (vs[i] - vs[i - 1]))
    return ds[i - 1] + w * (ds[i] - ds[i - 1])


def _dark_bands(centers, corner, base):
    """Log-amplitude band levels: flat up to ``corner`` Hz, then -12 dB/oct."""
    return base - 2 * math.log(2) * torch.log2(centers / corner).clamp(min=0)


def _interp_bands(x, n_out):
    """Linearly resample band values ``[..., n_in, F]`` (band axis -2) to ``n_out`` bands."""
    n_in = x.shape[-2]
    pos = torch.linspace(0, n_in - 1, n_out, device=x.device)
    i0 = pos.floor().long().clamp(max=n_in - 2)
    w = (pos - i0)[:, None]
    return x[..., i0, :] * (1 - w) + x[..., i0 + 1, :] * w


class NoiseBank(nn.Module):
    """Mechanical noises: hammer/soundboard knock at note-on, key-bottom thump
    (velocity-dependent timing), damper noise at release, and pedal-mechanism
    noise when the dampers move. Initial levels (docs/physical_parameters.md, section 5): knock
    -25 dB (ff) to -12 dB (pp) re the tone, damper noise ~-40 dB and pedal noise ~-35 dB re an mf note.

    Rendered in the time domain: white noise is split into log-spaced bands
    (overlap-save, so long pieces render blockwise without seams) and each band
    is scaled by a sample-accurate envelope, so a 5 ms knock really lasts 5 ms
    and nothing sounds before the hammer. Band levels are spectral densities
    (log amplitude): equal values give white noise. Every decay time is bounded
    (x/e around its prior), so no noise can grow into a stationary floor (review 3, F1).

    The residual's attack component (R2) is a second, slower noise event per note with a
    learned level, spectrum and decay time.
    """

    MARGIN = 4096  # overlap-save context; longer than the lowest band filter's ringing
    EVENT_WINDOW = 1.0  # s after an event during which its noise is rendered (> 9 x the longest bounded tau, 0.11 s)

    def __init__(self, cfg: PianoConfig):
        super().__init__()
        self.cfg = cfg
        nb = cfg.noise_bands
        centers = torch.logspace(math.log10(40), math.log10(cfg.sample_rate / 2), nb)
        self.register_buffer("centers", centers)
        self.knock = nn.Parameter(_dark_bands(centers, 600.0, -4.38).repeat(N_KEYS, 1))
        self.knock_vel = nn.Parameter(torch.full((N_KEYS,), 4.2))  # noise grows ~13 dB less than the tone pp -> ff
        self.register_buffer("prior_knock_log_tau", torch.log(key_curve([(0, 0.010), (40, 0.007), (87, 0.005)])))
        self.raw_knock_tau = nn.Parameter(torch.zeros(N_KEYS))
        self.thump_log_gain = nn.Parameter(torch.tensor(math.log(0.7)))
        self.raw_thump_tau = nn.Parameter(torch.tensor(0.0))  # 8 ms x/e
        self.release = nn.Parameter(_dark_bands(centers, 1500.0, -9.88).repeat(N_KEYS, 1))
        self.raw_release_tau = nn.Parameter(torch.zeros(N_KEYS))  # 5 ms x/e
        self.pedal = nn.Parameter(_dark_bands(centers, 800.0, -9.09))
        self.raw_pedal_tau = nn.Parameter(torch.tensor(0.0))  # 30 ms x/e
        # R2 attack noise: starts 40 dB under the knock (nearly neutral), 40 ms decay (x/e from the context net)
        self.att = nn.Parameter(_dark_bands(centers, 600.0, -4.38 - 4.6).repeat(N_KEYS, 1))
        self._masks = {}

    def taus(self, ki):
        return {"knock": torch.exp(self.prior_knock_log_tau[ki] + bounded(self.raw_knock_tau[ki], 1.0)),
                "thump": 0.008 * torch.exp(bounded(self.raw_thump_tau, 1.0)),
                "release": 0.005 * torch.exp(bounded(self.raw_release_tau[ki], 1.0)),
                "pedal": 0.03 * torch.exp(bounded(self.raw_pedal_tau, 1.0))}

    def band_masks(self, n):
        """Raised-cosine (in log f) band-split masks for an ``n``-point rFFT; they sum to one."""
        if n not in self._masks:
            lf = torch.log2(torch.fft.rfftfreq(n, 1 / self.cfg.sample_rate).clamp(min=1.0))
            c = torch.log2(self.centers.cpu())
            masks = []
            for i in range(len(c)):
                m = torch.ones_like(lf)
                if i > 0:
                    m = torch.where(lf < c[i], torch.sin(0.5 * math.pi * ((lf - c[i - 1]) / (c[i] - c[i - 1])).clamp(0, 1)) ** 2, m)
                if i < len(c) - 1:
                    m = torch.where(lf > c[i], torch.cos(0.5 * math.pi * ((lf - c[i]) / (c[i + 1] - c[i])).clamp(0, 1)) ** 2, m)
                masks.append(m)
            if len(self._masks) > 8:
                self._masks.clear()
            self._masks[n] = torch.stack(masks).to(self.centers.device)
        return self._masks[n]

    def band_split(self, x, start, length):
        """Band-split ``x[..., T]`` into ``[..., bands, length]`` for samples ``[start, start+length)``.

        The segment always has MARGIN samples of context on both sides (zeros beyond the signal's ends), so
        the zero-phase band filters act as a linear convolution: no wrap-around of the window's start into
        its end, and a single block gives the same result as many."""
        M = self.MARGIN
        lo, hi = max(0, start - M), min(x.shape[-1], start + length + M)
        seg = torch.nn.functional.pad(x[..., lo:hi], (M - (start - lo), M - (hi - start - length)))
        n = seg.shape[-1]
        bands = torch.fft.irfft(torch.fft.rfft(seg, n)[..., None, :] * self.band_masks(n), n)
        return bands[..., M: M + length]

    def _env(self, t, t_event, tau):
        """Power envelope ``[B,N,L]`` of exponentially decaying events at ``t_event``."""
        d = t - t_event[..., None]
        return torch.exp(-2 * d.clamp(min=0) / tau[..., None]) * (d >= 0)

    def event_power(self, ki, u, onset, release, weight_on, weight_off, ctx, start, length, residual):
        """Per-band power ``[B, bands, L]`` of note events: (physical, residual attack)."""
        cfg = self.cfg
        t = (start + torch.arange(length, device=ki.device, dtype=torch.float64)).to(u.dtype) / cfg.sample_rate
        taus = self.taus(ki)
        thump_at = onset + _key_bottom_delay(hammer_velocity(u))
        zero = torch.zeros_like(u)
        knock = self.knock[ki] + (self.knock_vel[ki] * (u - 0.6) + ctx.get("log_knock", zero))[..., None]
        if "knock_spec" in ctx:
            knock = knock + log_f_bumps(ctx["knock_spec"], self.centers, hi=cfg.sample_rate / 2)
        env_on = (self._env(t, onset, taus["knock"])
                  + self.thump_log_gain.exp() ** 2 * self._env(t, thump_at, taus["thump"].expand_as(onset)))
        env_off = self._env(t, release, taus["release"])
        power = (torch.einsum("bnk,bnl->bkl", torch.exp(2 * knock) * weight_on[..., None], env_on)
                 + torch.einsum("bnk,bnl->bkl", torch.exp(2 * self.release[ki]) * weight_off[..., None], env_off))
        if not residual:
            return power, None
        att = self.att[ki] + (self.knock_vel[ki] * (u - 0.6) + ctx.get("att_level", zero))[..., None]
        if "att_spec" in ctx:
            att = att + log_f_bumps(ctx["att_spec"], self.centers, hi=cfg.sample_rate / 2)
        tau_att = 0.04 * torch.exp(ctx.get("att_log_tau", zero))
        res = torch.einsum("bnk,bnl->bkl", torch.exp(2 * att) * weight_on[..., None], self._env(t, onset, tau_att))
        return power, res

    def pedal_envelope(self, lift):
        """Frame-rate power envelope ``[B, F]`` of the pedal mechanism, driven by how fast the damper rail moves."""
        cfg = self.cfg
        rate = torch.cat([lift.new_zeros(lift.shape[0], 1), (lift[:, 1:] - lift[:, :-1]).abs()], 1) * cfg.sample_rate / cfg.hop
        drive = (rate / 20.0).clamp(max=1.0) ** 2  # a full press in ~50 ms saturates
        n = torch.arange(lift.shape[-1], device=lift.device) * cfg.hop / cfg.sample_rate
        kernel = torch.exp(-2 * n / self.taus(torch.zeros(1, dtype=torch.long, device=lift.device))["pedal"])
        return fft_convolve(drive, kernel).clamp(min=0)  # FFT round-off can go slightly negative


# log amplitude (white-equivalent, dry domain) of the R3 noise path at zero output: ~-104 dB, i.e. below the loss
# floor after the room, so switching the residual on starts (nearly) neutral; the context net can raise it by 35 dB
R3_NOISE_BASE = -12.0


class NeuralPhysicalPiano(nn.Module):
    """MIDI performance -> audio ``[B, channels, n_samples]``.

    ``perf`` is a dict of tensors:
      pitch [B,N] (MIDI, long), onset/offset [B,N] (seconds re sample 0, may be negative),
      velocity [B,N] (1..127), mask [B,N] (bool), condition [B] (long),
      sustain/soft/sostenuto [B,F] in [0,1], and optionally hist_frames (int, same for the batch).
      Frame f of the control curves is at time ``(f - hist_frames) * hop / sr``, and
      ``F >= hist_frames + n_samples // hop + 2``.
    """

    def __init__(self, cfg: PianoConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or PianoConfig()
        self.physics = PianoPhysics(cfg)
        self.context = ContextNet(cfg)
        self.symp = SympatheticBank(cfg)
        self.noise = NoiseBank(cfg)
        self.room = Room(cfg)

    def n_frames(self, n_samples: int) -> int:
        return n_samples // self.cfg.hop + 2

    def key_rolls(self, ki, onset, release, u, mask, F):
        """Per-key damper-lifted curve ``key_down[B,88,F]`` and onset-velocity roll ``[B,88,F]``.

        Times are relative to frame 0. Frame f holds the fraction of ``[f, f+1)`` (in frames) the
        key is down, so the release edge is fractional and the damper integral is differentiable in
        the release time (the damper delay is learned per condition)."""
        cfg = self.cfg
        B = ki.shape[0]
        on_f = torch.ceil(onset * cfg.sample_rate / cfg.hop).clamp(0, F).long()
        pos = (release * cfg.sample_rate / cfg.hop).clamp(0, F - 1)
        r0 = pos.detach().floor()
        frac = pos - r0
        r0 = torch.maximum(r0.long(), on_f).clamp(max=F - 1)
        w = mask.to(frac.dtype)
        base = ki * (F + 1)
        counts = torch.zeros(B, N_KEYS * (F + 1), device=ki.device, dtype=frac.dtype)
        counts = counts.scatter_add(1, base + on_f, w)
        counts = counts.scatter_add(1, base + r0, -w * (1 - frac))
        counts = counts.scatter_add(1, base + r0 + 1, -w * frac)
        key_down = counts.view(B, N_KEYS, F + 1).cumsum(-1)[..., :F].clamp(0, 1)
        roll = torch.zeros(B, N_KEYS * F, device=ki.device)
        valid = (mask & (onset >= 0) & (on_f < F)).float()
        roll = roll.scatter_add(1, ki * F + on_f.clamp(max=F - 1), u * valid)
        return key_down, roll.view(B, N_KEYS, F)

    @staticmethod
    def sostenuto_latch(key_down, sostenuto, threshold=0.5):
        """Keys held when the sostenuto pedal goes down keep their dampers up until it comes up."""
        B, K, F = key_down.shape
        on = sostenuto[:, :F] > threshold
        rising = on & ~torch.cat([on.new_zeros(B, 1), on[:, :-1]], 1)
        frames = torch.arange(F, device=key_down.device).expand(B, F)
        last = torch.where(rising, frames, torch.full_like(frames, -1)).cummax(1).values
        held = key_down.gather(2, last.clamp(min=0)[:, None, :].expand(-1, K, -1))
        return held * (on & (last >= 0))[:, None, :].float()

    @staticmethod
    def next_strikes(ki, onset, mask, R=8):
        """Delays ``[B,N,R]`` (s) from each note to the next ``R`` strikes of the same key (inf if none)."""
        B, N = ki.shape
        same = (ki[:, :, None] == ki[:, None, :]) & mask[:, :, None] & mask[:, None, :]
        later = same & (onset[:, None, :] > onset[:, :, None] + 1e-3)
        d = torch.where(later, onset[:, None, :] - onset[:, :, None], torch.full_like(later, math.inf, dtype=onset.dtype))
        d = d.topk(min(R, N), dim=-1, largest=False).values
        if d.shape[-1] < R:
            d = torch.cat([d, d.new_full((B, N, R - d.shape[-1]), math.inf)], -1)
        return d

    @staticmethod
    def _flat_sets(modes):
        """Oscillator sets ``(freq, alpha, amp, alpha_damp)`` ``[B,N,Q]``: transverse (partial-major) and phantoms."""
        M = modes["freq"].shape[-1]
        tr = (modes["freq"].flatten(2), modes["alpha"].flatten(2), modes["amp"].flatten(2),
              modes["alpha_damp"][..., None].expand(*modes["alpha_damp"].shape, M).flatten(2))
        sets = [tr]
        if "ph_amp" in modes:
            sets.append((modes["ph_freq"], modes["ph_alpha"], modes["ph_amp"], modes["ph_alpha_damp"]))
        return sets

    def render_strings(self, modes, ki, onset, mask, C, hist, rs_delay, pan, start, length, per_key=False):
        """String (bridge-force) signal ``[B, ch, L]`` for samples ``[start, start+length)``, optionally also
        per key (mono) ``[B,88,L]``. ``hist`` = samples of control history before sample 0 on the frame grid.

        Per chunk, only the (example, note) pairs that are still audible are rendered: the activity test
        bounds each note's envelope at the chunk start with its dampers and pedals (review 3, F12). Pairs are
        sorted by how many oscillators they need (treble notes need few) and rendered in slices that keep
        the ``[pairs, oscillators, samples]`` work under ``cfg.bank_elements``."""
        cfg = self.cfg
        sr, hop = cfg.sample_rate, cfg.hop
        B, N = ki.shape
        ch = pan.shape[-1]
        BN = B * N
        flat = lambda x: x.reshape(BN, *x.shape[2:])
        sets = [tuple(flat(x) for x in st) for st in self._flat_sets(modes)]
        tc, on, msk = flat(modes["tc"]), flat(onset), flat(mask)
        rs_nats, rsd, kf, pan_f = flat(modes["restrike"]), flat(rs_delay), flat(ki), flat(pan)
        bi = torch.arange(B, device=ki.device).repeat_interleave(N)
        row = bi * N_KEYS + kf
        C_rows = C.reshape(B * N_KEYS, -1)
        c_onset = sample_keyed(C, ki, onset.clamp(min=-hist / sr) + hist / sr, sr, hop).reshape(BN)
        with torch.no_grad():
            log_amps = [torch.log(st[2].abs() + 1e-30) for st in sets]
            thresh = log_amps[0].amax(-1) - cfg.activity_db * math.log(10) / 20
        out, keys = [], []
        for s0 in range(start, start + length, cfg.synth_chunk):
            L = min(cfg.synth_chunk, start + length - s0)
            with torch.no_grad():  # exact activity: the envelope bound at the chunk start, dampers included
                tau0 = (s0 / sr - on).clamp(min=0)
                c0 = frames_to_samples(C_rows, s0 + hist, 1, hop)[:, 0][row]
                d0 = (c0 - c_onset).clamp(min=0)
                n_valid = []  # per note: oscillators up to the last one still above the threshold (partials are
                for (freq, alpha, amp, adamp), la in zip(sets, log_amps):  # ordered by frequency, and decay with it)
                    act = (la - alpha * tau0[:, None] - adamp * d0[:, None]) > thresh[:, None]
                    n_valid.append(torch.where(act.any(-1), act.shape[-1] - act.flip(-1).int().argmax(-1), 0))
                idx = (msk & (on < (s0 + L) / sr) & (n_valid[0] > 0)).nonzero().squeeze(1)
            if idx.numel() == 0:
                out.append(C.new_zeros(B, ch, L))
                keys.append(C.new_zeros(B, N_KEYS, L) if per_key else None)
                continue
            P = idx.numel()
            c_note = frames_to_samples(C_rows.index_select(0, row[idx]), s0 + hist, L, hop)  # [P, L]
            y = C.new_zeros(P, L)
            for si, (freq, alpha, amp, adamp) in enumerate(sets):
                nv = n_valid[si][idx]
                order = torch.argsort(nv, descending=True)
                nv_sorted = nv[order].tolist()
                i = 0
                while i < P and nv_sorted[i] > 0:
                    hi = nv_sorted[i]
                    cnt = max(1, min(P - i, cfg.bank_elements // (hi * L)))
                    sl = order[i: i + cnt]
                    pid = idx[sl]
                    sel = lambda x: x.index_select(0, pid)
                    ys = osc_bank(sel(freq)[:, :hi], sel(alpha)[:, :hi], sel(amp)[:, :hi], sel(adamp)[:, :hi], sel(tc),
                                  c_note.index_select(0, sl), sel(c_onset), sel(rs_nats), sel(on), sel(rsd), s0 / sr, sr)
                    y = y.index_add(0, sl, ys)
                    i += cnt
            out.append(C.new_zeros(B, ch, L).index_add(0, bi[idx], y[:, None, :] * pan_f[idx][:, :, None]))
            if per_key:
                keys.append(C.new_zeros(B * N_KEYS, L).index_add(0, row[idx], y).view(B, N_KEYS, L))
        return torch.cat(out, -1), (torch.cat(keys, -1) if per_key else None)

    def render_impulses(self, amp, tc, onset, mask, pan, n_samples):
        """Knock impulse: a raised-cosine force pulse of the contact time at every strike, ``[B, ch, n]``."""
        sr = self.cfg.sample_rate
        B, N = onset.shape
        W = int(min(256, math.ceil(float(tc.detach().max()) * sr) + 2)) if N else 1
        s = torch.floor(onset.detach() * sr).long()[..., None] + torch.arange(W, device=onset.device)  # [B,N,W]
        d = s.double() / sr - onset.double()[..., None]
        x = (d / tc.double()[..., None]).to(amp.dtype)
        val = amp[..., None] * 0.5 * (1 - torch.cos(2 * math.pi * x.clamp(0, 1))) * ((x >= 0) & (x <= 1))
        ok = (s >= 0) & (s < n_samples) & mask[..., None]
        val = val * ok
        idx = s.clamp(0, n_samples - 1).reshape(B, 1, -1).expand(-1, pan.shape[-1], -1)
        src = (val[:, None] * pan.permute(0, 2, 1)[..., None]).reshape(B, pan.shape[-1], -1)
        return amp.new_zeros(B, pan.shape[-1], n_samples).scatter_add(2, idx, src)

    def render_noise(self, ki, u, onset, release, mask, pedal_env, w_off, ctx, frame_ctx, n_samples, block,
                     generator, residual, white=None, white_res=None):
        """Mechanical noise ``[B, n]``, the residual noise (R2 attack + R3 path) ``[B, n]`` or None, and the two white
        noises they were shaped from (pass them back in to render the same realisation again)."""
        cfg = self.cfg
        sr, hop = cfg.sample_rate, cfg.hop
        w_on = mask.float()
        if white is None:
            white = torch.randn(ki.shape[0], n_samples, generator=generator, device=ki.device)
        if white_res is None and residual:
            white_res = torch.randn(ki.shape[0], n_samples, generator=generator, device=ki.device)
        pedal_bands = torch.exp(2 * self.noise.pedal)[None, :, None]
        r3 = None
        if residual and frame_ctx is not None:
            r3 = torch.exp(2 * (R3_NOISE_BASE + _interp_bands(frame_ctx["noise_level"], cfg.noise_bands)))
        out, res_out = [], []
        for s0 in range(0, n_samples, block):
            L = min(block, n_samples - s0)
            t0, t1 = s0 / sr, (s0 + L) / sr
            power = pedal_bands * frames_to_samples(pedal_env, s0, L, hop)[:, None]
            res = frames_to_samples(r3, s0, L, hop) if r3 is not None else None
            sel = ((onset < t1) & (torch.maximum(onset, release) + NoiseBank.EVENT_WINDOW > t0) & mask).any(0).nonzero().squeeze(1)
            if sel.numel():
                sub = {k: v[:, sel] for k, v in ctx.items()}
                p, r = self.noise.event_power(ki[:, sel], u[:, sel], onset[:, sel], release[:, sel], w_on[:, sel],
                                              w_off[:, sel], sub, s0, L, residual)
                power = power + p
                if r is not None:
                    res = r if res is None else res + r
            # sqrt has an infinite gradient at 0 (and power is exactly 0 in silent bands/samples),
            # which would poison the backward pass with NaNs; the floor keeps the amplitude gradient finite.
            out.append((self.noise.band_split(white, s0, L) * (power.clamp(min=0) + 1e-12).sqrt()).sum(1))
            if residual:
                res = torch.zeros_like(power) if res is None else res
                res_out.append((self.noise.band_split(white_res, s0, L) * (res.clamp(min=0) + 1e-12).sqrt()).sum(1))
        return torch.cat(out, -1), (torch.cat(res_out, -1) if residual else None), white, white_res

    def apply_band_gains(self, x, gains, block):
        """R3: frame-rate band gains (log amplitude, ``[B, 16, F]``) on ``x[B, ch, T]``, blockwise overlap-save."""
        cfg = self.cfg
        g = _interp_bands(gains, cfg.noise_bands)
        out = []
        for s0 in range(0, x.shape[-1], block):
            L = min(block, x.shape[-1] - s0)
            bands = self.noise.band_split(x, s0, L)  # [B, ch, bands, L]
            out.append((bands * torch.exp(frames_to_samples(g, s0, L, cfg.hop))[:, None]).sum(2))
        return torch.cat(out, -1)

    def forward(self, perf, n_samples, block_seconds=None, generator=None, residual=True, floor=None, extras=()):
        """``extras``: ``"residual_out"`` adds ``out["noise_res_out"]``, the residual noise at the microphones (the
        budget measures it against the output); ``"texture_view"`` adds ``out["audio_texture"]``, numerically the
        same audio, in which only the noise bank's and the residual's parameters are differentiable: the noise is
        rendered a second time from the same white noise with every physical input (damper timing, pedal, levels)
        detached, and the strings, knock impulse, room and floor enter detached. That is what the GAN may change
        (docs/plan_round2.md, A10)."""
        cfg = self.cfg
        sr, hop = cfg.sample_rate, cfg.hop
        pitch, mask, cond = perf["pitch"], perf["mask"], perf["condition"]
        onset, offset = perf["onset"], perf["offset"]
        B, N = pitch.shape
        H = int(perf["hist_frames"].reshape(-1)[0]) if "hist_frames" in perf else 0
        t_hist = H * hop / sr
        ki = (pitch - LOWEST_MIDI).clamp(0, N_KEYS - 1)
        u = perf["velocity"] / 127.0
        F = H + self.n_frames(n_samples)
        pedals = torch.stack([perf["sustain"][:, :F], perf["soft"][:, :F], perf["sostenuto"][:, :F]], 1)
        assert pedals.shape[-1] == F, f"pedal curves need {F} frames, got {pedals.shape[-1]}"
        residual = residual and cfg.use_context

        release = offset + self.physics.damper_delay(cond)[:, None]
        key_down, onset_roll = self.key_rolls(ki, onset + t_hist, offset + t_hist, u, mask, F)
        damper_off, _ = self.key_rolls(ki, onset + t_hist, release + t_hist, u, mask, F)
        lift = self.physics.pedal_lift(pedals[:, 0])
        pedal_damping = self.physics.pedal_damping(lift)
        latch = self.sostenuto_latch(damper_off, pedals[:, 2])
        engagement = (1 - damper_off) * pedal_damping[:, None] * (1 - latch)
        C = torch.cat([engagement.new_zeros(B, N_KEYS, 1), torch.cumsum(engagement[..., :-1], -1) * hop / sr], -1)

        soft_on = sample_curve(pedals[:, 1], onset.clamp(min=-t_hist) + t_hist, sr, hop)
        ctx, frame_ctx = {}, None
        if residual:
            ctx, frame_ctx = self.context(onset_roll, key_down, pedals, ki, u, (onset + t_hist) * sr / hop, cond, H)
        modes = self.physics.modes(ki, u, soft_on, cond, ctx, phantoms=cfg.n_phantoms > 0)
        m = mask[..., None]
        modes["amp"] = modes["amp"] * m[..., None]
        if "ph_amp" in modes:
            modes["ph_amp"] = modes["ph_amp"] * m
        rs_delay = self.next_strikes(ki, onset, mask)
        pan = self.room.pan_gains(ki, cond)

        if cfg.use_sympathetic:
            all_keys = torch.arange(N_KEYS, device=pitch.device).expand(B, N_KEYS)
            half = torch.full((B, N_KEYS), 0.6, device=pitch.device)
            key_modes = self.physics.modes(all_keys, half, torch.zeros_like(half), cond, phantoms=False)
        block = int(block_seconds * sr) if block_seconds else n_samples
        strings, symp, state = [], [], None
        for s0 in range(0, n_samples, block):
            L = min(block, n_samples - s0)
            s, own = self.render_strings(modes, ki, onset, mask, C, H * hop, rs_delay, pan, s0, L, per_key=cfg.use_sympathetic)
            strings.append(s)
            if cfg.use_sympathetic:
                y, state = self.symp(own.sum(1), own, key_modes, engagement, s0 + H * hop, state)
                symp.append(y)
        out = {"strings": torch.cat(strings, -1)}
        tonal = out["strings"]
        if cfg.use_sympathetic:
            out["symp"] = torch.cat(symp, -1)
            tonal = tonal + out["symp"][:, None]
        texture = torch.zeros_like(tonal)  # stochastic and attack components
        if cfg.use_impulse:
            out["impulse"] = self.render_impulses(modes["impulse"], modes["tc"], onset, mask, pan, n_samples)
            texture = texture + out["impulse"]
        if cfg.use_noise:
            damp_at_release = sample_curve(pedal_damping, release.clamp(min=-t_hist) + t_hist, sr, hop)
            latched = sample_keyed(latch, ki, release.clamp(min=-t_hist) + t_hist, sr, hop)
            w_off = mask.float() * self.physics.damper_strength[ki] * damp_at_release * (1 - latched)
            pedal_env = self.noise.pedal_envelope(lift)[:, H:]
            white = torch.randn(B, n_samples, generator=generator, device=pitch.device)
            white_res = torch.randn(B, n_samples, generator=generator, device=pitch.device) if residual else None
            args = (ki, u, onset, release, mask, pedal_env, w_off, ctx, frame_ctx, n_samples, block, None, residual,
                    white, white_res)
            if "texture_view" in extras and torch.is_grad_enabled():  # the GAN's memory has to come from somewhere
                noise, res = checkpoint(lambda *a: self.render_noise(*a)[:2], *args, use_reentrant=False)
            else:
                noise, res, _, _ = self.render_noise(*args)
            out["noise"] = noise
            texture = texture + noise[:, None]
            out["dry_phys"] = tonal + texture  # before the residual
            if res is not None:
                out["noise_res"] = res
                texture = texture + res[:, None]
        gains = frame_ctx["band_gain"] if residual and frame_ctx is not None else None
        dry = tonal + texture
        if gains is not None:
            dry = self.apply_band_gains(dry, gains, block)
        out["dry"] = dry
        ir = self.room(cond) if cfg.use_room else None
        audio = fft_convolve(dry, ir) if cfg.use_room else dry
        fl = None
        if cfg.use_room and (cfg.use_floor if floor is None else floor):
            fl = self.room.floor_noise(cond, n_samples, generator)
            audio = audio + fl
        out["audio"] = audio
        if "residual_out" in extras and "noise_res" in out:
            r = out["noise_res"][:, None].expand(-1, tonal.shape[1], -1)
            out["noise_res_out"] = fft_convolve(r, ir.detach()) if cfg.use_room else r
        if "texture_view" in extras:
            view = (tonal + out["impulse"]).detach() if "impulse" in out else tonal.detach()
            if cfg.use_noise:
                # recomputed in the backward pass: the per-note envelopes ([B, notes, samples]) would double the
                # noise path's memory
                nv, rv = checkpoint(lambda *a: self.render_noise(*a)[:2], ki, u, onset, release.detach(), mask,
                                    pedal_env.detach(), w_off.detach(), ctx, frame_ctx, n_samples, block, None, residual,
                                    white, white_res, use_reentrant=False)
                view = view + nv[:, None] + (rv[:, None] if rv is not None else 0)
            if gains is not None:
                view = self.apply_band_gains(view, gains.detach(), block)
            view = fft_convolve(view, ir.detach()) if cfg.use_room else view
            out["audio_texture"] = view + fl.detach() if fl is not None else view
        out["ctx"], out["frame_ctx"] = ctx, frame_ctx
        return out
