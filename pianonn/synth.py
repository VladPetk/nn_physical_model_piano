"""The full differentiable piano: strings -> sympathetic resonance -> noise -> soundboard/room."""

import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .config import PianoConfig
from .dsp import fft_convolve, frames_to_samples, linear_recurrence, sample_curve, sample_keyed
from .physics import LOWEST_MIDI, N_KEYS, PianoPhysics


def _osc_bank(freq, alpha, amp, alpha_damp, onset, c_note, c_onset, t):
    """Closed-form damped-sinusoid bank for one time chunk.

    Decay is ``alpha * (t - onset) + alpha_damp * D(t)``, where ``D`` is the time
    the key's damper has been on the string since the onset, so key releases,
    pedalling and half-pedalling are all exact (no per-sample recursion) and
    every chunk is independent of the others.
    """
    tau = t[None, None, :] - onset[..., None]  # [B,N,L]
    active = (tau >= 0).to(freq.dtype)
    tau = tau.clamp(min=0)[:, :, None, None, :]
    damped = (c_note - c_onset[..., None]).clamp(min=0)[:, :, None, None, :]
    log_env = -alpha[..., None] * tau - alpha_damp[..., None, None] * damped
    cycles = freq[..., None] * tau
    phase = 2 * math.pi * (cycles - cycles.detach().floor())
    y = (amp[..., None] * torch.exp(log_env) * torch.sin(phase)).sum((2, 3))
    return (y * active).sum(1)


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
    """Every string on the instrument as a resonator driven by the bridge signal.

    With the dampers down only the undamped treble strings respond; lift the
    pedal and the whole bank rings. The damper state is time-varying, so this
    is a linear recurrence with time-varying poles, solved by the chunked
    parallel scan in :func:`linear_recurrence`.
    """

    def __init__(self, cfg: PianoConfig):
        super().__init__()
        self.cfg = cfg
        self.log_gain = nn.Parameter(torch.full((N_KEYS,), math.log(0.05)))

    def _block(self, bridge, es, freq, alpha, alpha_damp, gin, state):
        sr = self.cfg.sample_rate
        decay = (alpha[..., None] + es[:, :, None, :] * alpha_damp[..., None]).clamp(max=1000.0) / sr
        omega = (2 * math.pi / sr) * freq[..., None].expand_as(decay)
        log_a = torch.complex(-decay, omega)
        drive = bridge[:, None, None, :] * gin[..., None]
        x = torch.complex(drive, torch.zeros_like(drive))
        y, state = linear_recurrence(x, log_a, state, self.cfg.rec_chunk)
        return y.real.sum((1, 2)), state

    def forward(self, bridge, key_modes, engagement, block=None):
        cfg = self.cfg
        S = cfg.symp_partials
        freq = key_modes["freq"][..., :S, 0]
        alpha = key_modes["alpha"][..., :S, 0]
        alpha_damp = key_modes["alpha_damp"][..., :S]
        valid = (freq < 0.45 * cfg.sample_rate).to(freq.dtype)
        # scale drive by (1 - |a|) so the steady-state gain at resonance is exp(log_gain)
        gin = self.log_gain.exp()[None, :, None] * alpha / cfg.sample_rate * valid
        T = bridge.shape[-1]
        block = block or T
        state, out = None, []
        for s0 in range(0, T, block):
            L = min(block, T - s0)
            es = frames_to_samples(engagement, s0, L, cfg.hop)
            args = (bridge[:, s0:s0 + L], es, freq, alpha, alpha_damp, gin, state)
            if cfg.checkpoint and torch.is_grad_enabled():
                y, state = checkpoint(self._block, *args, use_reentrant=False)
            else:
                y, state = self._block(*args)
            out.append(y)
        return torch.cat(out, -1)


class NoiseBank(nn.Module):
    """Hammer knock / key thump at note-on and damper noise at note-off, as
    band-shaped noise with per-key spectra and exponential envelopes."""

    def __init__(self, cfg: PianoConfig):
        super().__init__()
        self.cfg = cfg
        nb, n_bins = cfg.noise_bands, cfg.noise_fft // 2 + 1
        pos = torch.linspace(0, 1, nb)
        self.knock = nn.Parameter((-3.0 - 1.5 * pos).repeat(N_KEYS, 1))
        self.knock_vel = nn.Parameter(torch.full((N_KEYS,), 2.0))
        self.knock_log_tau = nn.Parameter(torch.full((N_KEYS,), math.log(0.02)))
        self.release = nn.Parameter((-4.0 - 1.5 * pos).repeat(N_KEYS, 1))
        self.release_log_tau = nn.Parameter(torch.full((N_KEYS,), math.log(0.04)))
        centers = torch.log(torch.logspace(math.log10(40), math.log10(cfg.sample_rate / 2), nb))
        bins = torch.log(torch.linspace(0, cfg.sample_rate / 2, n_bins).clamp(min=40))
        idx = torch.searchsorted(centers, bins).clamp(1, nb - 1)
        w = ((bins - centers[idx - 1]) / (centers[idx] - centers[idx - 1])).clamp(0, 1)
        W = torch.zeros(nb, n_bins)
        W[idx - 1, torch.arange(n_bins)] = 1 - w
        W[idx, torch.arange(n_bins)] += w
        self.register_buffer("band_to_bin", W)
        self.register_buffer("window", torch.hann_window(cfg.noise_fft))

    def magnitude(self, ki, u, onset, offset, weight_on, weight_off, log_knock, f_start, n_frames):
        cfg = self.cfg
        tf = (f_start + torch.arange(n_frames, device=ki.device)) * cfg.hop / cfg.sample_rate
        d_on = tf - onset[..., None]
        env_on = torch.exp(-2 * d_on.clamp(min=0) / self.knock_log_tau[ki].exp()[..., None]) * (d_on >= 0)
        d_off = tf - offset[..., None]
        env_off = torch.exp(-2 * d_off.clamp(min=0) / self.release_log_tau[ki].exp()[..., None]) * (d_off >= 0)
        p_on = torch.exp(2 * (self.knock[ki] + (self.knock_vel[ki] * (u - 0.6) + log_knock)[..., None]))
        p_off = torch.exp(2 * self.release[ki])
        power = (torch.einsum("bnk,bnf->bkf", p_on * weight_on[..., None], env_on)
                 + torch.einsum("bnk,bnf->bkf", p_off * weight_off[..., None], env_off))
        return torch.sqrt(torch.einsum("bkf,kq->bqf", power, self.band_to_bin) + 1e-12)

    def synth(self, mag, n_samples, generator=None):
        phase = 2 * math.pi * torch.rand(mag.shape, generator=generator, device=mag.device)
        return torch.istft(torch.polar(mag, phase), self.cfg.noise_fft, self.cfg.hop,
                           window=self.window, length=n_samples)


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
        L = int(cfg.ir_seconds * cfg.sample_rate)
        g = torch.Generator().manual_seed(0)
        t = torch.arange(L) / cfg.sample_rate
        ir = 0.02 * torch.randn(cfg.n_conditions, L, generator=g) * torch.exp(-6.9 * t / 0.8)
        ir[:, 0] = 1.0
        self.ir = nn.Parameter(ir)

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

    @torch.no_grad()
    def _ring_end(self, modes, onset, floor=1e-5, max_seconds=60.0):
        amp, alpha = modes["amp"], modes["alpha"]
        t = torch.log(amp.clamp(min=1e-30) / floor) / alpha
        t = torch.where(amp > floor, t, torch.zeros_like(t)).amax((2, 3))
        return onset + t.clamp(0, max_seconds)

    def render_strings(self, modes, ki, onset, ring_end, C, start, length):
        cfg = self.cfg
        sr = cfg.sample_rate
        B = ki.shape[0]
        out = []
        for s0 in range(start, start + length, cfg.synth_chunk):
            L = min(cfg.synth_chunk, start + length - s0)
            active = (onset < (s0 + L) / sr) & (ring_end > s0 / sr)
            sel = active.any(0).nonzero().squeeze(1)
            if sel.numel() == 0:
                out.append(C.new_zeros(B, L))
                continue
            m = {k: v[:, sel] for k, v in modes.items()}
            k_sel, on = ki[:, sel], onset[:, sel]
            c_note = frames_to_samples(C, s0, L, cfg.hop).gather(1, k_sel[..., None].expand(-1, -1, L))
            c_onset = sample_keyed(C, k_sel, on.clamp(min=0), sr, cfg.hop)
            t = (s0 + torch.arange(L, device=ki.device, dtype=torch.float64)) / sr
            args = (m["freq"], m["alpha"], m["amp"], m["alpha_damp"], on, c_note, c_onset, t.to(C.dtype))
            if cfg.checkpoint and torch.is_grad_enabled():
                out.append(checkpoint(_osc_bank, *args, use_reentrant=False))
            else:
                out.append(_osc_bank(*args))
        return torch.cat(out, -1)

    def render_noise(self, ki, u, onset, offset, mask, lift, log_knock, F, n_samples, block_frames, generator):
        cfg = self.cfg
        sr, hop = cfg.sample_rate, cfg.hop
        lift_off = sample_curve(lift, offset.clamp(min=0), sr, hop)
        w_on = mask.float()
        w_off = mask.float() * self.physics.has_damper[ki] * (1 - lift_off)
        mags = []
        for f0 in range(0, F, block_frames):
            nf = min(block_frames, F - f0)
            t0, t1 = f0 * hop / sr, (f0 + nf) * hop / sr
            sel = ((onset < t1) & (torch.maximum(onset, offset) + 1.0 > t0) & mask).any(0).nonzero().squeeze(1)
            if sel.numel() == 0:
                mags.append(u.new_zeros(ki.shape[0], cfg.noise_fft // 2 + 1, nf))
                continue
            mags.append(self.noise.magnitude(ki[:, sel], u[:, sel], onset[:, sel], offset[:, sel], w_on[:, sel],
                                             w_off[:, sel], log_knock[:, sel], f0, nf))
        return self.noise.synth(torch.cat(mags, -1), n_samples, generator)

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
        lift = self.physics.pedal_lift(pedals[:, 0])
        engagement = (1 - key_down) * (1 - lift[:, None])
        C = torch.cat([engagement.new_zeros(B, N_KEYS, 1), torch.cumsum(engagement[..., :-1], -1) * hop / sr], -1)

        soft_on = sample_curve(pedals[:, 1], onset.clamp(min=0), sr, hop)
        zeros = torch.zeros_like(u)
        ctx = (self.context(onset_roll, key_down, pedals, ki, u, onset, cond) if cfg.use_context
               else {name: zeros for name in ContextNet.OUTPUTS})
        modes = self.physics.modes(ki, u, soft_on, cond, ctx)
        modes["amp"] = modes["amp"] * mask[..., None, None]

        strings = self.render_strings(modes, ki, onset, self._ring_end(modes, onset), C, 0, n_samples)
        out = {"strings": strings}
        dry = strings
        block = int(block_seconds * sr) if block_seconds else None
        if cfg.use_sympathetic:
            all_keys = torch.arange(N_KEYS, device=pitch.device).expand(B, N_KEYS)
            half = torch.full((B, N_KEYS), 0.6, device=pitch.device)
            key_modes = self.physics.modes(all_keys, half, torch.zeros_like(half), cond)
            out["symp"] = self.symp(strings, key_modes, engagement, block)
            dry = dry + out["symp"]
        if cfg.use_noise:
            block_frames = block // hop if block else F
            out["noise"] = self.render_noise(ki, u, onset, offset, mask, lift, ctx["log_knock"], F, n_samples,
                                             block_frames, generator)
            dry = dry + out["noise"]
        out["dry"] = dry
        out["audio"] = fft_convolve(dry, self.ir[cond])
        return out
