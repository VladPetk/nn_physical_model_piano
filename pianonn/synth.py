"""The full differentiable piano: strings -> sympathetic resonance -> noise -> soundboard body + hall."""

import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .config import PianoConfig
from .dsp import fft_convolve, frames_to_samples, linear_recurrence, sample_curve, sample_keyed
from .physics import LOWEST_MIDI, N_KEYS, PianoPhysics, hammer_velocity, key_curve
from .room import Room


def _osc_bank(freq, alpha, amp, alpha_damp, tc, onset, c_note, c_onset, t):
    """Closed-form damped-sinusoid bank for one time chunk; returns per-note output ``[B,N,L]``.

    Decay is ``alpha * (t - onset) + alpha_damp * D(t)``, where ``D`` is the time
    the key's damper has been on the string since the onset, so key releases,
    pedalling and half-pedalling are all exact (no per-sample recursion) and
    every chunk is independent of the others. Partials ramp in over the hammer
    contact time ``tc`` with phases referenced to the centre of the force pulse.
    """
    tau = (t[None, None, :] - onset[..., None]).clamp(min=0)  # [B,N,L]
    ramp = 0.5 - 0.5 * torch.cos(math.pi * (tau / tc[..., None]).clamp(max=1))
    damped = (c_note - c_onset[..., None]).clamp(min=0)[:, :, None, None, :]
    log_env = -alpha[..., None] * tau[:, :, None, None, :] - alpha_damp[..., None, None] * damped
    cycles = freq[..., None] * (tau - 0.5 * tc[..., None])[:, :, None, None, :]
    phase = 2 * math.pi * (cycles - cycles.detach().floor())
    return (amp[..., None] * torch.exp(log_env) * torch.sin(phase)).sum((2, 3)) * ramp


def _partial_groups(P):
    """(lo, hi] partial-count buckets, so treble notes don't pay for the bass's partials."""
    edges = sorted({0, P // 6, P // 3, 2 * P // 3, P})
    return list(zip(edges[:-1], edges[1:]))


class ContextNet(nn.Module):
    """Causal GRU over the performance (piano roll + pedals) that predicts small,
    bounded per-note corrections to the physical model. Zero-initialised, so
    training starts from pure physics and the network only learns what the
    physics leaves unexplained."""

    OUTPUTS = {"gain_db": 6.0, "log_fc": 0.5, "log_decay": 0.5, "log_knock": 1.0}

    def __init__(self, cfg: PianoConfig):
        super().__init__()
        self.cfg = cfg
        H = cfg.ctx_hidden
        self.inp = nn.Linear(2 * N_KEYS + 3, H)
        self.gru = nn.GRU(H, H, batch_first=True)
        self.key_emb = nn.Embedding(N_KEYS, 16)
        self.cond_emb = nn.Embedding(cfg.n_conditions, 16)
        self.head = nn.Sequential(nn.Linear(H + 33, H), nn.GELU(), nn.Linear(H, len(self.OUTPUTS)))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, onset_roll, key_down, pedals, ki, u, onset, cond):
        cfg = self.cfg
        x = torch.cat([onset_roll, key_down, pedals], 1).transpose(1, 2)
        h, _ = self.gru(torch.tanh(self.inp(x)))
        F, H = h.shape[1], h.shape[2]
        idx = torch.ceil(onset * cfg.sample_rate / cfg.hop).clamp(0, F - 1).long()
        h_note = h.gather(1, idx[..., None].expand(-1, -1, H))
        feats = torch.cat([h_note, self.key_emb(ki), self.cond_emb(cond)[:, None].expand(-1, ki.shape[1], -1),
                           u[..., None]], -1)
        z = self.head(feats)
        return {name: s * torch.tanh(z[..., i] / s) for i, (name, s) in enumerate(self.OUTPUTS.items())}


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
        """``bridge[B,L]`` total string signal, ``own[B,88,L]`` per-key share, for samples ``[start, start+L)``."""
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


class NoiseBank(nn.Module):
    """Mechanical noises: hammer/soundboard knock at note-on, key-bottom thump
    (velocity-dependent timing), damper noise at release, and pedal-mechanism
    noise when the dampers move. Initial levels (docs/physical_parameters.md, section 6): knock
    -25 dB (ff) to -12 dB (pp) re the tone, damper noise ~-40 dB and pedal noise ~-35 dB re an mf note.

    Rendered in the time domain: white noise is split into log-spaced bands
    (overlap-save, so long pieces render blockwise without seams) and each band
    is scaled by a sample-accurate envelope, so a 5 ms knock really lasts 5 ms
    and nothing sounds before the hammer. Band levels are spectral densities
    (log amplitude): equal values give white noise.
    """

    MARGIN = 4096  # overlap-save context; longer than the lowest band filter's ringing

    def __init__(self, cfg: PianoConfig):
        super().__init__()
        self.cfg = cfg
        nb = cfg.noise_bands
        centers = torch.logspace(math.log10(40), math.log10(cfg.sample_rate / 2), nb)
        self.register_buffer("centers", centers)
        self.knock = nn.Parameter(_dark_bands(centers, 600.0, -4.38).repeat(N_KEYS, 1))
        self.knock_vel = nn.Parameter(torch.full((N_KEYS,), 4.2))  # noise grows ~13 dB less than the tone pp -> ff
        self.knock_log_tau = nn.Parameter(torch.log(key_curve([(0, 0.010), (40, 0.007), (87, 0.005)])))
        self.thump_log_gain = nn.Parameter(torch.tensor(math.log(0.7)))
        self.thump_log_tau = nn.Parameter(torch.tensor(math.log(0.008)))
        self.release = nn.Parameter(_dark_bands(centers, 1500.0, -9.88).repeat(N_KEYS, 1))
        self.release_log_tau = nn.Parameter(torch.full((N_KEYS,), math.log(0.005)))
        self.pedal = nn.Parameter(_dark_bands(centers, 800.0, -9.09))
        self.pedal_log_tau = nn.Parameter(torch.tensor(math.log(0.03)))
        self._masks = {}

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
            self._masks[n] = torch.stack(masks)
        return self._masks[n].to(self.centers.device)

    def band_noise(self, white, start, length):
        """Band-split white noise ``[B, bands, length]`` for samples ``[start, start+length)``."""
        lo, hi = max(0, start - self.MARGIN), min(white.shape[-1], start + length + self.MARGIN)
        seg = white[:, lo:hi]
        n = seg.shape[-1]
        bands = torch.fft.irfft(torch.fft.rfft(seg, n)[:, None] * self.band_masks(n), n)
        return bands[..., start - lo: start - lo + length]

    def _env(self, t, t_event, tau):
        """Power envelope ``[B,N,L]`` of exponentially decaying events at ``t_event``."""
        d = t - t_event[..., None]
        return torch.exp(-2 * d.clamp(min=0) / tau[..., None]) * (d >= 0)

    def event_power(self, ki, u, onset, release, weight_on, weight_off, log_knock, start, length):
        """Per-band power ``[B, bands, L]`` of note events."""
        cfg = self.cfg
        t = (start + torch.arange(length, device=ki.device, dtype=torch.float64)).to(u.dtype) / cfg.sample_rate
        thump_at = onset + _key_bottom_delay(hammer_velocity(u))
        p_on = torch.exp(2 * (self.knock[ki] + (self.knock_vel[ki] * (u - 0.6) + log_knock)[..., None]))
        env_on = (self._env(t, onset, self.knock_log_tau[ki].exp())
                  + self.thump_log_gain.exp() ** 2 * self._env(t, thump_at, self.thump_log_tau.exp().expand_as(onset)))
        env_off = self._env(t, release, self.release_log_tau[ki].exp())
        p_off = torch.exp(2 * self.release[ki])
        return (torch.einsum("bnk,bnl->bkl", p_on * weight_on[..., None], env_on)
                + torch.einsum("bnk,bnl->bkl", p_off * weight_off[..., None], env_off))

    def pedal_envelope(self, lift):
        """Frame-rate power envelope ``[B, F]`` of the pedal mechanism, driven by how fast the damper rail moves."""
        cfg = self.cfg
        rate = torch.cat([lift.new_zeros(lift.shape[0], 1), (lift[:, 1:] - lift[:, :-1]).abs()], 1) * cfg.sample_rate / cfg.hop
        drive = (rate / 20.0).clamp(max=1.0) ** 2  # a full press in ~50 ms saturates
        n = torch.arange(lift.shape[-1], device=lift.device) * cfg.hop / cfg.sample_rate
        kernel = torch.exp(-2 * n / self.pedal_log_tau.exp())
        return fft_convolve(drive, kernel).clamp(min=0)  # FFT round-off can go slightly negative


class NeuralPhysicalPiano(nn.Module):
    """MIDI performance -> audio.

    ``perf`` is a dict of tensors:
      pitch [B,N] (MIDI, long), onset/offset [B,N] (seconds, onset may be < 0),
      velocity [B,N] (1..127), mask [B,N] (bool), condition [B] (long),
      sustain/soft/sostenuto [B,F] in [0,1] at frame times ``f * hop / sr``
      with ``F >= n_samples // hop + 2``.
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

    def key_rolls(self, ki, onset, offset, u, mask, F):
        """Per-key ``key_down[B,88,F]`` and onset-velocity roll ``[B,88,F]``."""
        cfg = self.cfg
        B = ki.shape[0]
        on_f = torch.ceil(onset * cfg.sample_rate / cfg.hop).clamp(0, F).long()
        off_f = torch.maximum(torch.ceil(offset * cfg.sample_rate / cfg.hop).clamp(0, F).long(), on_f)
        w = mask.float()
        counts = torch.zeros(B, N_KEYS * (F + 1), device=ki.device)
        counts.scatter_add_(1, ki * (F + 1) + on_f, w)
        counts.scatter_add_(1, ki * (F + 1) + off_f, -w)
        key_down = (counts.view(B, N_KEYS, F + 1).cumsum(-1)[..., :F] > 0.5).float()
        roll = torch.zeros(B, N_KEYS * F, device=ki.device)
        valid = (mask & (onset >= 0) & (on_f < F)).float()
        roll.scatter_add_(1, ki * F + on_f.clamp(max=F - 1), u * valid)
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

    @torch.no_grad()
    def _ring_end(self, modes, onset, floor=1e-5, max_seconds=60.0):
        amp, alpha = modes["amp"].abs(), modes["alpha"]
        t = torch.log(amp.clamp(min=1e-30) / floor) / alpha
        t = torch.where(amp > floor, t, torch.zeros_like(t)).amax((2, 3))
        return onset + t.clamp(0, max_seconds)

    def render_strings(self, modes, ki, onset, ring_end, C, start, length, per_key=False):
        """String (bridge-force) signal for samples ``[start, start+length)``; optionally also per key ``[B,88,L]``."""
        cfg = self.cfg
        sr = cfg.sample_rate
        B = ki.shape[0]
        out, keys = [], []
        for s0 in range(start, start + length, cfg.synth_chunk):
            L = min(cfg.synth_chunk, start + length - s0)
            active = (onset < (s0 + L) / sr) & (ring_end > s0 / sr)
            sel = active.any(0).nonzero().squeeze(1)
            if sel.numel() == 0:
                out.append(C.new_zeros(B, L))
                keys.append(C.new_zeros(B, N_KEYS, L) if per_key else None)
                continue
            m = {k: v[:, sel] for k, v in modes.items()}
            k_sel, on = ki[:, sel], onset[:, sel]
            c_note = frames_to_samples(C, s0, L, cfg.hop).gather(1, k_sel[..., None].expand(-1, -1, L))
            c_onset = sample_keyed(C, k_sel, on.clamp(min=0), sr, cfg.hop)
            t = (s0 + torch.arange(L, device=ki.device, dtype=torch.float64)) / sr
            # partials above Nyquist carry zero amplitude: only compute as many as each group of notes needs
            n_valid = (m["amp"] != 0).any(-1).any(0).flip(-1).int().argmax(-1)  # trailing zero partials per note
            n_valid = torch.where((m["amp"] != 0).any(-1).any(0).any(-1), m["amp"].shape[2] - n_valid, 0)
            notes = C.new_zeros(B, sel.numel(), L)
            for lo, hi in _partial_groups(m["amp"].shape[2]):
                grp = ((n_valid > lo) & (n_valid <= hi)).nonzero().squeeze(1)
                if grp.numel() == 0:
                    continue
                args = (m["freq"][:, grp, :hi], m["alpha"][:, grp, :hi], m["amp"][:, grp, :hi],
                        m["alpha_damp"][:, grp, :hi], m["tc"][:, grp], on[:, grp], c_note[:, grp], c_onset[:, grp],
                        t.to(C.dtype))
                if cfg.checkpoint and torch.is_grad_enabled():
                    y = checkpoint(_osc_bank, *args, use_reentrant=False)
                else:
                    y = _osc_bank(*args)
                notes = notes.index_add(1, grp, y)
            out.append(notes.sum(1))
            if per_key:
                keys.append(notes.new_zeros(B, N_KEYS, L).scatter_add(1, k_sel[..., None].expand(-1, -1, L), notes))
        return torch.cat(out, -1), (torch.cat(keys, -1) if per_key else None)

    def render_noise(self, ki, u, onset, offset, mask, lift, pedal_damping, log_knock, n_samples, block, generator):
        cfg = self.cfg
        sr, hop = cfg.sample_rate, cfg.hop
        release = offset + cfg.damper_delay
        damp_at_release = sample_curve(pedal_damping, release.clamp(min=0), sr, hop)
        w_on = mask.float()
        w_off = mask.float() * self.physics.damper_strength[ki] * damp_at_release
        white = torch.randn(ki.shape[0], n_samples, generator=generator, device=ki.device)
        pedal_env = self.noise.pedal_envelope(lift)
        pedal_bands = torch.exp(2 * self.noise.pedal)[None, :, None]
        out = []
        for s0 in range(0, n_samples, block):
            L = min(block, n_samples - s0)
            t0, t1 = s0 / sr, (s0 + L) / sr
            power = pedal_bands * frames_to_samples(pedal_env, s0, L, hop)[:, None]
            sel = ((onset < t1) & (torch.maximum(onset, release) + 0.5 > t0) & mask).any(0).nonzero().squeeze(1)
            if sel.numel():
                power = power + self.noise.event_power(ki[:, sel], u[:, sel], onset[:, sel], release[:, sel],
                                                       w_on[:, sel], w_off[:, sel], log_knock[:, sel], s0, L)
            out.append((self.noise.band_noise(white, s0, L) * power.clamp(min=0).sqrt()).sum(1))
        return torch.cat(out, -1)

    def forward(self, perf, n_samples, block_seconds=None, generator=None):
        cfg = self.cfg
        sr, hop = cfg.sample_rate, cfg.hop
        pitch, mask, cond = perf["pitch"], perf["mask"], perf["condition"]
        onset, offset = perf["onset"], perf["offset"]
        B, N = pitch.shape
        ki = (pitch - LOWEST_MIDI).clamp(0, N_KEYS - 1)
        u = perf["velocity"] / 127.0
        F = self.n_frames(n_samples)
        pedals = torch.stack([perf["sustain"][:, :F], perf["soft"][:, :F], perf["sostenuto"][:, :F]], 1)
        assert pedals.shape[-1] == F, f"pedal curves need {F} frames, got {pedals.shape[-1]}"

        key_down, onset_roll = self.key_rolls(ki, onset, offset, u, mask, F)
        damper_off, _ = self.key_rolls(ki, onset, offset + cfg.damper_delay, u, mask, F)
        lift = self.physics.pedal_lift(pedals[:, 0])
        pedal_damping = self.physics.pedal_damping(lift)
        latch = self.sostenuto_latch(damper_off, pedals[:, 2])
        engagement = (1 - damper_off) * pedal_damping[:, None] * (1 - latch)
        C = torch.cat([engagement.new_zeros(B, N_KEYS, 1), torch.cumsum(engagement[..., :-1], -1) * hop / sr], -1)

        soft_on = sample_curve(pedals[:, 1], onset.clamp(min=0), sr, hop)
        zeros = torch.zeros_like(u)
        ctx = (self.context(onset_roll, key_down, pedals, ki, u, onset, cond) if cfg.use_context
               else {name: zeros for name in ContextNet.OUTPUTS})
        modes = self.physics.modes(ki, u, soft_on, cond, ctx)
        modes["amp"] = modes["amp"] * mask[..., None, None]
        ring_end = self._ring_end(modes, onset)

        if cfg.use_sympathetic:
            all_keys = torch.arange(N_KEYS, device=pitch.device).expand(B, N_KEYS)
            half = torch.full((B, N_KEYS), 0.6, device=pitch.device)
            key_modes = self.physics.modes(all_keys, half, torch.zeros_like(half), cond)
        block = int(block_seconds * sr) if block_seconds else n_samples
        strings, symp, state = [], [], None
        for s0 in range(0, n_samples, block):
            L = min(block, n_samples - s0)
            s, own = self.render_strings(modes, ki, onset, ring_end, C, s0, L, per_key=cfg.use_sympathetic)
            strings.append(s)
            if cfg.use_sympathetic:
                y, state = self.symp(s, own, key_modes, engagement, s0, state)
                symp.append(y)
        out = {"strings": torch.cat(strings, -1)}
        dry = out["strings"]
        if cfg.use_sympathetic:
            out["symp"] = torch.cat(symp, -1)
            dry = dry + out["symp"]
        if cfg.use_noise:
            out["noise"] = self.render_noise(ki, u, onset, offset, mask, lift, pedal_damping, ctx["log_knock"],
                                             n_samples, block, generator)
            dry = dry + out["noise"]
        out["dry"] = dry
        out["audio"] = fft_convolve(dry, self.room(cond)) if cfg.use_room else dry
        return out
