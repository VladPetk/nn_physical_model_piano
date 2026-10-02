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
from .dsp import bounded, fft_convolve, frames_to_samples, interp_bands, linear_recurrence, sample_curve, sample_keyed
from .physics import LOWEST_MIDI, N_KEYS, PianoPhysics, hammer_velocity, key_curve, log_f_bumps
from .oscbank import RESTRIKE_RAMP, group_weights, osc_bank, partial_group_weights
from .residual import AwareResidual
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

    A cheaper bank (docs/tone_measures.md 14): only the keys ``symp_lo_midi``..``symp_hi_midi`` respond, only their
    partials below ``symp_max_hz`` (0: all up to 0.45 sr), and with ``symp_decimate`` = D > 1 the resonators run at
    sr / D: the drive is low-passed (Kaiser FIR, causal) and decimated, the response interpolated back with the same
    filter. The filters' histories carry across blocks, so blocks of a multiple of D samples render as one pass.
    """

    TAPS = 128  # the decimation filter: ~60 dB stopband, ~0.7 kHz transition at 24 kHz

    def __init__(self, cfg: PianoConfig):
        super().__init__()
        self.cfg = cfg
        self.log_gain = nn.Parameter(torch.log(0.262 * key_curve([(0, 0.08), (12, 0.12), (30, 0.15), (67, 0.15), (87, 0.10)])))
        self.register_buffer("keys", torch.arange(cfg.symp_lo_midi - LOWEST_MIDI, cfg.symp_hi_midi - LOWEST_MIDI + 1),
                             persistent=False)
        D = cfg.symp_decimate
        if D > 1:
            nyq = cfg.sample_rate / (2 * D)
            fc = 0.5 * (cfg.symp_max_hz + nyq) if 0 < cfg.symp_max_hz < nyq else 0.9 * nyq
            n = torch.arange(self.TAPS, dtype=torch.float64) - (self.TAPS - 1) / 2
            h = 2 * fc / cfg.sample_rate * torch.sinc(2 * fc / cfg.sample_rate * n)
            h = h * torch.kaiser_window(self.TAPS, periodic=False, beta=5.65, dtype=torch.float64)
            self.register_buffer("lowpass", (h / h.sum()).float(), persistent=False)

    def _decimate(self, x, hist):
        """Low-pass and keep every D-th sample of ``x[..., L]`` (causal FIR; ``hist``: the previous TAPS - 1 inputs)."""
        D, T = self.cfg.symp_decimate, self.TAPS
        lead = x.shape[:-1]
        xx = torch.cat([hist if hist is not None else x.new_zeros(*lead, T - 1), x], -1)
        y = torch.nn.functional.conv1d(xx.reshape(-1, 1, xx.shape[-1]), self.lowpass.flip(-1)[None, None], stride=D)
        return y.reshape(*lead, -1), xx[..., -(T - 1):]

    def _interpolate(self, u, carry, length):
        """Back to the full rate: zero-stuff by D and filter with D x the low-pass; ``carry`` is the previous block's
        overhang."""
        D = self.cfg.symp_decimate
        y = torch.nn.functional.conv_transpose1d(u[:, None], (D * self.lowpass)[None, None], stride=D)[:, 0]
        if carry is not None:
            n = min(carry.shape[-1], y.shape[-1])
            y = torch.cat([y[:, :n] + carry[:, :n], y[:, n:]], -1)
            if carry.shape[-1] > n:
                y = torch.cat([y, carry[:, n:]], -1)
        if y.shape[-1] < length:
            y = torch.nn.functional.pad(y, (0, length - y.shape[-1]))
        return y[:, :length], y[:, length:]

    def _block(self, drive, es, freq, alpha, alpha_damp, gin, state):
        sr = self.cfg.sample_rate / self.cfg.symp_decimate
        decay = (alpha[..., None] + es[:, :, None, :] * alpha_damp[..., None]).clamp(max=1000.0) / sr
        omega = (2 * math.pi / sr) * freq[..., None].expand_as(decay)
        log_a = torch.complex(-decay, omega)
        x = drive[:, :, None, :] * gin[..., None]
        y, state = linear_recurrence(torch.complex(x, torch.zeros_like(x)), log_a, state, self.cfg.rec_chunk)
        return y.real.sum((1, 2)), state

    def forward(self, bridge, own, key_modes, engagement, start, state=None):
        """``bridge[B,L]`` total string signal, ``own[B,88,L]`` per-key share, for frame-grid samples ``[start, start+L)``.
        Returns ``(y[B, L], state)``; pass the state to the next block."""
        cfg = self.cfg
        S, D, k = cfg.symp_partials, cfg.symp_decimate, self.keys
        state = state or {}
        sr = cfg.sample_rate / D
        freq = key_modes["freq"][:, k, :S, 0]
        alpha = key_modes["alpha"][:, k, :S, 0]
        kappa = (alpha - key_modes["alpha"][:, k, :S, 1]).clamp(min=0)
        alpha_damp = key_modes["alpha_damp"][:, k, :S]
        top = min(cfg.symp_max_hz, 0.45 * sr) if cfg.symp_max_hz > 0 else 0.45 * sr
        valid = (freq < top).to(freq.dtype)
        gin = self.log_gain.exp()[k][None, :, None] * kappa / sr * valid
        drive = bridge[:, None, :] - own[:, k]
        L = bridge.shape[-1]
        if D > 1:
            pad = (-L) % D  # only the last block may be ragged
            drive, state["hist"] = self._decimate(torch.nn.functional.pad(drive, (0, pad)), state.get("hist"))
            es = frames_to_samples(engagement[:, k], start // D, drive.shape[-1], cfg.hop / D)
        else:
            es = frames_to_samples(engagement[:, k], start, L, cfg.hop)
        args = (drive, es, freq, alpha, alpha_damp, gin, state.get("rec"))
        if cfg.checkpoint and torch.is_grad_enabled():
            y, state["rec"] = checkpoint(self._block, *args, use_reentrant=False)
        else:
            y, state["rec"] = self._block(*args)
        if D > 1:
            y, state["carry"] = self._interpolate(y, state.get("carry"), L)
        return y, state


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


# the attack's per-register knots (key indices: MIDI 21, 65, 108) for ``attack_model="parts"``
ATTACK_KNOTS = (0, 44, 87)


def knot_weights(knots, n=N_KEYS):
    """``[n, len(knots)]`` piecewise-linear (hat) weights over the key index: each row sums to one."""
    return torch.stack([key_curve([(k, float(i == j)) for j, k in enumerate(knots)], n) for i in range(len(knots))], 1)


def _span(raw, lo, hi):
    """A value in ``(lo, hi)``, log-uniform in a sigmoid of ``raw``."""
    return lo * (hi / lo) ** torch.sigmoid(raw)


def _span_inv(v, lo, hi):
    x = (torch.log(torch.as_tensor(v, dtype=torch.float32) / lo) / math.log(hi / lo)).clamp(1e-4, 1 - 1e-4)
    return torch.log(x / (1 - x))


_interp_bands = interp_bands


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

    ``cfg.attack_model = "parts"`` (docs/tone_measures.md 10.4 item 1, review 5 section 3.2) replaces the knock noise
    and the thump, which shared one spectrum and one step-onset envelope per key, with three parts, each an envelope
    per band (a smooth rise, then an exponential decay) and per register (knots ``ATTACK_KNOTS``):
    - the knock noise, its per-key spectrum rolled off above ``knock_max_hz`` (-40 dB/oct): the fitted knock's 4-8
      kHz is the bench's percussive excess (10.3), and nothing in the piano's attack puts broadband noise there;
    - the key-bottom thump (structure-borne), its own spectrum up to ``thump_max_hz``, a level per register and a
      decay per register and band (the treble board's low modes ring 0.2-0.5 s, Bank);
    - the string-borne precursor (the longitudinal wave, Askenfelt 1993): a burst of ~1-2 ms up to
      ``precursor_max_hz``, with its own velocity slope (it grows with the blow faster than the tone).
    The rise per band is at least half a period of the band's centre: a step onset, applied after the band split,
    spread every band's onset over all frequencies (12.5). Each kernel is scaled to the energy of the step-onset
    exponential with the same decay, so the levels keep their meaning. Rendered as event trains per band and register
    convolved with the kernels (FFT), so long decays cost no per-note envelopes.
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
        # attack_model="parts" (used only then; init_parts() sets them from the knock and thump above)
        R = len(ATTACK_KNOTS)
        self.register_buffer("knot_w", knot_weights(ATTACK_KNOTS))
        self.register_buffer("rise_floor", (0.5 / centers).clamp(min=2.5e-4))
        self.knock_raw_tau = nn.Parameter(_span_inv(0.008, *self.SPANS["knock"]).repeat(R, nb))
        self.knock_raw_rise = nn.Parameter(torch.full((nb,), -2.5))  # ~1.1 x the floor
        self.thump_spec = nn.Parameter(_dark_bands(centers, 300.0, -4.38 + math.log(0.7)))
        self.thump_reg = nn.Parameter(torch.zeros(R))
        self.thump_vel = nn.Parameter(torch.tensor(4.2))
        self.thump_raw_tau = nn.Parameter(_span_inv(0.008, *self.SPANS["thump"]).repeat(R, nb))
        self.thump_raw_rise = nn.Parameter(torch.full((nb,), -2.5))
        self.prec_spec = nn.Parameter(_dark_bands(centers, 2000.0, -4.38))
        self.prec_reg = nn.Parameter(torch.zeros(R))
        self.prec_vel = nn.Parameter(torch.tensor(8.4))
        self.prec_raw_tau = nn.Parameter(_span_inv(0.001, *self.SPANS["precursor"]).reshape(()))

    # decay time constants (amplitude, s) of the parts: (lo, hi)
    SPANS = {"knock": (0.001, 0.1), "thump": (0.002, 0.5), "precursor": (0.0003, 0.005)}
    KERNEL_MAX = {"knock": 0.5, "thump": 2.0, "precursor": 0.03}  # s

    @torch.no_grad()
    def init_parts(self):
        """Set the parts (``attack_model="parts"``) from the fitted knock noise and thump: the same energy per band and
        decay; the thump's spectrum is the knock's mean over the keys (a level per register), the precursor starts at the
        knock's mean level at 1 kHz, flat from 1 kHz up and falling at 12 dB/oct below (the "bite" above 1 kHz, F1),
        with the knock's velocity slope + 20 dB/u (the tone's is ~40) and a 1 ms decay; the rises start near their
        floors."""
        dev = self.knock.device
        taus = self.taus(torch.arange(N_KEYS, device=dev))
        knots = list(ATTACK_KNOTS)
        self.knock_raw_tau.copy_(_span_inv(taus["knock"][knots].cpu(), *self.SPANS["knock"]).to(dev)[:, None]
                                 .expand_as(self.knock_raw_tau))
        mean = self.knock.mean(0)
        low = self.centers <= self.cfg.thump_max_hz
        self.thump_spec.copy_(mean + self.thump_log_gain)
        self.thump_reg.copy_((self.knock[knots][:, low] - mean[low]).mean(1))
        self.thump_vel.copy_(self.knock_vel.mean())
        self.thump_raw_tau.fill_(float(_span_inv(float(taus["thump"]), *self.SPANS["thump"])))
        at_1k = mean[int(torch.argmin((torch.log2(self.centers / 1000.0)).abs()))]
        self.prec_spec.copy_(at_1k - 2 * math.log(2) * torch.log2(1000.0 / self.centers).clamp(min=0))
        self.prec_reg.copy_((self.knock[knots] - mean).mean(1))
        self.prec_vel.copy_(self.knock_vel.mean() + 20 / 20 * math.log(10))
        self.knock_raw_rise.fill_(-2.5)
        self.thump_raw_rise.fill_(-2.5)
        self.prec_raw_tau.fill_(float(_span_inv(0.001, *self.SPANS["precursor"])))

    def _cap(self, spec, max_hz):
        """Log amplitudes rolled off above ``max_hz`` at -40 dB per octave."""
        return spec - (40 / 20 * math.log(10)) * torch.log2(self.centers / max_hz).clamp(min=0)

    def part_kernels(self, name):
        """Power envelopes ``[R or 1, bands, Lk]`` of one part, each scaled to the energy of a step-onset exponential
        with the same decay (``tau / 2``)."""
        cfg = self.cfg
        if name == "precursor":
            tau_d = _span(self.prec_raw_tau, *self.SPANS[name]).reshape(1, 1).expand(1, len(self.centers))
            tau_r = self.rise_floor[None]
        else:
            raw_tau, raw_rise = ((self.knock_raw_tau, self.knock_raw_rise) if name == "knock"
                                 else (self.thump_raw_tau, self.thump_raw_rise))
            tau_d = _span(raw_tau, *self.SPANS[name])
            tau_r = (self.rise_floor * torch.exp(1.5 + bounded(raw_rise, 1.5)))[None]
        Lk = int(math.ceil(min(self.KERNEL_MAX[name], float(3.5 * tau_d.max() + 6 * tau_r.max())) * cfg.sample_rate))
        d = torch.arange(Lk, device=tau_d.device, dtype=tau_d.dtype) / cfg.sample_rate
        a = (1 - torch.exp(-d / tau_r[..., None])) ** 2 * torch.exp(-d / tau_d[..., None])
        p = a ** 2
        return p * (0.5 * tau_d / (p.sum(-1) / cfg.sample_rate).clamp(min=1e-12))[..., None]

    def attack_power(self, ki, u, onset, thump_at, w_on, ctx, start, length):
        """Per-band power ``[B, bands, length]`` of the parts (``attack_model="parts"``) for samples ``[start,
        start + length)``: event trains per band and register convolved with ``part_kernels``."""
        cfg = self.cfg
        sr, K = cfg.sample_rate, len(self.centers)
        zero = torch.zeros_like(u)
        base = ctx.get("log_knock", zero)  # log amplitude per note (the context net's and the strike's)
        reg = lambda v: self.knot_w[ki] @ v  # per-register values at each note's key
        knock = self._cap(self.knock[ki], cfg.knock_max_hz) + (self.knock_vel[ki] * (u - 0.6) + base)[..., None]
        if "knock_spec" in ctx:
            knock = knock + log_f_bumps(ctx["knock_spec"], self.centers, hi=sr / 2)
        thump = (self._cap(self.thump_spec, cfg.thump_max_hz)
                 + (reg(self.thump_reg) + self.thump_vel * (u - 0.6) + base)[..., None])
        prec = (self._cap(self.prec_spec, cfg.precursor_max_hz)
                + (reg(self.prec_reg) + self.prec_vel * (u - 0.6) + base)[..., None])
        out = torch.zeros(ki.shape[0], K, length, device=ki.device, dtype=u.dtype)
        for name, level, t_ev, max_hz, per_reg in (("knock", knock, onset, cfg.knock_max_hz, True),
                                                   ("thump", thump, thump_at, cfg.thump_max_hz, True),
                                                   ("precursor", prec, onset, cfg.precursor_max_hz, False)):
            kb = int((self.centers <= 2 * max_hz).sum())  # bands above are 40 dB down or more
            kern = self.part_kernels(name)[:, :kb]  # [R', kb, Lk]
            Lk = kern.shape[-1]
            T = -(-(Lk + length) // 4096) * 4096  # the train starts T - length before ``start``: no wrap-around
            idx = torch.round(t_ev.detach() * sr).long() - (start + length - T)  # [B, N]
            ok = (idx >= 0) & (idx < T)
            amp = torch.exp(2 * level[..., :kb]) * (w_on * ok)[..., None]  # [B, N, kb]
            amp = amp[:, :, None, :] * (self.knot_w[ki][..., None] if per_reg else 1.0)  # [B, N, R', kb]
            B, N, Rp = amp.shape[:3]
            pos = idx.clamp(0, T - 1)[:, None, :].expand(B, Rp * kb, N)
            train = amp.new_zeros(B, Rp * kb, T).scatter_add(2, pos, amp.reshape(B, N, Rp * kb).transpose(1, 2))
            y = torch.fft.irfft(torch.fft.rfft(train.view(B, Rp, kb, T), T) * torch.fft.rfft(kern, T), T)
            out[:, :kb] = out[:, :kb] + y[..., T - length:].sum(1)
        return out

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

    def event_power(self, ki, u, onset, release, weight_on, weight_off, ctx, start, length, residual, speed=None):
        """Per-band power ``[B, bands, L]`` of note events: (physical, residual attack). ``speed``: hammer speeds
        (m/s) from the condition's velocity map (default: the prior map)."""
        cfg = self.cfg
        t = (start + torch.arange(length, device=ki.device, dtype=torch.float64)).to(u.dtype) / cfg.sample_rate
        taus = self.taus(ki)
        thump_at = onset + _key_bottom_delay(hammer_velocity(u) if speed is None else speed)
        zero = torch.zeros_like(u)
        knock = self.knock[ki] + (self.knock_vel[ki] * (u - 0.6) + ctx.get("log_knock", zero))[..., None]
        if "knock_spec" in ctx:
            knock = knock + log_f_bumps(ctx["knock_spec"], self.centers, hi=cfg.sample_rate / 2)
        env_off = self._env(t, release, taus["release"])
        power = torch.einsum("bnk,bnl->bkl", torch.exp(2 * self.release[ki]) * weight_off[..., None], env_off)
        if cfg.attack_model != "parts":  # the parts are rendered by attack_power
            env_on = (self._env(t, onset, taus["knock"])
                      + self.thump_log_gain.exp() ** 2 * self._env(t, thump_at, taus["thump"].expand_as(onset)))
            power = power + torch.einsum("bnk,bnl->bkl", torch.exp(2 * knock) * weight_on[..., None], env_on)
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
LATENT_SEED = 1_000_003  # the aware residual's random inputs: drawn from the render's seed plus this

# per-strike variation (config ``strike_*``, docs/tone_measures.md 12.7): the dimensions, the MIDI pitches of their
# per-register knots (R2..R6 centres) and the clip of the normal draws
STRIKE_DIMS = ("level_db", "log_fc", "knock_db", "log_decay", "decay_tilt", "onset_ms")
# drawn per partial (and per aftersound mode), after the per-note dimensions, so that those keep their draws
STRIKE_PARTIAL_DIMS = ("partial_decay", "after")
STRIKE_KNOTS = (37.5, 53.0, 65.5, 77.5, 86.0)
STRIKE_CLIP = 2.5


def strike_sd_table(cfg, dims=STRIKE_DIMS):
    """``[dims, 88]``: the sd of each per-strike dimension per key, or None when every one is off."""
    rows = []
    for d in dims:
        v = getattr(cfg, "strike_" + d)
        if isinstance(v, str):
            v = [float(x) for x in v.split("/") if x.strip()]
        v = [float(v)] if isinstance(v, (int, float)) else [float(x) for x in v]
        if len(v) <= 1:
            rows.append(torch.full((N_KEYS,), v[0] if v else 0.0))
        else:
            assert len(v) == len(STRIKE_KNOTS), f"strike_{d}: one value or one per register ({len(STRIKE_KNOTS)})"
            rows.append(key_curve([(k - LOWEST_MIDI, s) for k, s in zip(STRIKE_KNOTS, v)]))
    table = torch.stack(rows).clamp(min=0)
    return table if bool(table.any()) else None


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
        frame_spec = dict(ContextNet.FRAME, noise_level=(cfg.res_noise_bands, ContextNet.FRAME["noise_level"][1]))
        self.context = (AwareResidual(cfg, ContextNet.NOTE, frame_spec) if cfg.residual_kind == "aware"
                        else ContextNet(cfg))
        self.symp = SympatheticBank(cfg)
        self.noise = NoiseBank(cfg)
        self.room = Room(cfg)
        sd = strike_sd_table(cfg)
        self.strike_on = sd is not None
        self.register_buffer("strike_sd", sd if sd is not None else torch.zeros(len(STRIKE_DIMS), N_KEYS), persistent=False)
        psd = strike_sd_table(cfg, STRIKE_PARTIAL_DIMS)
        self.strike_partial_on = psd is not None
        self.strike_on = self.strike_on or self.strike_partial_on
        self.register_buffer("strike_partial_sd", psd if psd is not None else torch.zeros(len(STRIKE_PARTIAL_DIMS), N_KEYS),
                             persistent=False)
        # the aware residual's gain curves: per octave group of partials (0), or one group per partial for this many
        # partials (config res_curve_partials; scripts/fit_passages.py sets it for its free outputs)
        self.curve_partials = cfg.res_curve_partials

    def n_frames(self, n_samples: int) -> int:
        return n_samples // self.cfg.hop + 2

    def strike_offsets(self, ki, generator=None):
        """Per-strike random offsets ``{dim: [B, N]}`` (config ``strike_*``), or None when they are off. Drawn afresh
        at every call: a new realisation of the same performance, as the piano never strikes a key twice alike."""
        if not self.strike_on:
            return None
        z = torch.randn(*ki.shape, len(STRIKE_DIMS), generator=generator, device=ki.device)
        x = z.clamp(-STRIKE_CLIP, STRIKE_CLIP) * self.strike_sd.T[ki]
        out = {d: x[..., i] for i, d in enumerate(STRIKE_DIMS)}
        if self.strike_partial_on:  # [B, N, P] and [B, N, P, M - 1]
            P, M = self.cfg.n_partials, self.cfg.n_modes
            zd = torch.randn(*ki.shape, P, generator=generator, device=ki.device).clamp(-STRIKE_CLIP, STRIKE_CLIP)
            za = torch.randn(*ki.shape, P, M - 1, generator=generator, device=ki.device).clamp(-STRIKE_CLIP, STRIKE_CLIP)
            sd = self.strike_partial_sd.T[ki]
            out["partial_decay"] = zd * sd[..., 0, None]
            sa = sd[..., 1, None, None]  # the random part's sd re the key's aftersound; the mean power is kept
            out["after"] = (1 + za * sa) / torch.sqrt(1 + sa ** 2)
        return out

    @staticmethod
    def with_strike(ctx, var, zero):
        """The per-note corrections the physics and the noise bank see: the context net's plus the strike's."""
        if var is None:
            return ctx
        out = dict(ctx)
        db = math.log(10) / 20  # dB -> the noise bank's log amplitude
        out["gain_db"] = ctx.get("gain_db", zero) + var["level_db"]
        out["log_knock"] = ctx.get("log_knock", zero) + db * (var["level_db"] + var["knock_db"])
        out["impulse_db"] = var["knock_db"]  # the level reaches the impulse through gain_db
        for d in ("log_fc", "log_decay", "decay_tilt"):  # level-neutral: PianoPhysics.modes keeps the early energy
            out["strike_" + d] = var[d]
        for d in STRIKE_PARTIAL_DIMS:
            if d in var:
                out["strike_" + d] = var[d]
        return out

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

    def render_strings(self, modes, ki, onset, mask, C, hist, rs_delay, pan, start, length, per_key=False, curves=None):
        """String (bridge-force) signal ``[B, ch, L]`` for samples ``[start, start+length)``, optionally also
        per key (mono) ``[B,88,L]``. ``hist`` = samples of control history before sample 0 on the frame grid.

        Per chunk, only the (example, note) pairs that are still audible are rendered: the activity test
        bounds each note's envelope at the chunk start with its dampers and pedals (review 3, F12). Pairs are
        sorted by how many oscillators they need (treble notes need few) and rendered in slices that keep
        the ``[pairs, oscillators, samples]`` work under ``cfg.bank_elements``. ``curves``: the aware residual's
        log gain per note and octave group of partials ``[B, N, G, C]`` at its control frames."""
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
        if curves is not None:
            c_flat, c_hop = curves.reshape(BN, *curves.shape[2:]), cfg.res_control * hop
            centers = None if self.curve_partials else self.context.centers
            f_part = flat(modes["freq"][..., 0]) if self.curve_partials else None  # [BN, partials]: the groups' centres
            M = modes["freq"].shape[-1]
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
            if curves is not None:  # the control frames around this chunk, as log gains (the bank interpolates them)
                c_lo, c_hi = s0 // c_hop, min(c_flat.shape[-1], (s0 + L - 1) // c_hop + 2)
                m_note = c_flat[..., c_lo:c_hi].index_select(0, idx)
            else:
                m_note = None
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
                    if curves is None:
                        grp = ()
                    elif self.curve_partials:
                        grp = (partial_group_weights(si, sel(freq)[:, :hi].detach(), f_part.index_select(0, pid), M,
                                                     self.curve_partials), m_note.index_select(0, sl), s0 - c_lo * c_hop,
                               c_hop)
                    else:
                        grp = (group_weights(sel(freq)[:, :hi].detach(), centers), m_note.index_select(0, sl),
                               s0 - c_lo * c_hop, c_hop)
                    ys = osc_bank(sel(freq)[:, :hi], sel(alpha)[:, :hi], sel(amp)[:, :hi], sel(adamp)[:, :hi], sel(tc),
                                  c_note.index_select(0, sl), sel(c_onset), sel(rs_nats), sel(on), sel(rsd), s0 / sr, sr,
                                  *grp)
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
                     generator, residual, white=None, white_res=None, speed=None):
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
            if "note_noise" in frame_ctx:  # the aware residual's per-note noise, summed over the notes
                r3 = r3 + frame_ctx["note_noise"]
        out, res_out = [], []
        for s0 in range(0, n_samples, block):
            L = min(block, n_samples - s0)
            t0, t1 = s0 / sr, (s0 + L) / sr
            power = pedal_bands * frames_to_samples(pedal_env, s0, L, hop)[:, None]
            res = frames_to_samples(r3, s0, L, hop) if r3 is not None else None
            if cfg.attack_model == "parts":
                thump_at = onset + _key_bottom_delay(hammer_velocity(u) if speed is None else speed)
                power = power + self.noise.attack_power(ki, u, onset, thump_at, w_on, ctx, s0, L)
            sel = ((onset < t1) & (torch.maximum(onset, release) + NoiseBank.EVENT_WINDOW > t0) & mask).any(0).nonzero().squeeze(1)
            if sel.numel():
                sub = {k: v[:, sel] for k, v in ctx.items()}
                p, r = self.noise.event_power(ki[:, sel], u[:, sel], onset[:, sel], release[:, sel], w_on[:, sel],
                                              w_off[:, sel], sub, s0, L, residual,
                                              None if speed is None else speed[:, sel])
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
        ctx, frame_ctx, curves = {}, None, None
        if residual and cfg.residual_kind == "aware":
            with torch.no_grad():  # what the physics alone would play: the residual's view, not steered through
                modes0 = self.physics.modes(ki, u, soft_on, cond, None, phantoms=cfg.n_phantoms > 0)
                modes0["amp"] = modes0["amp"] * mask[..., None, None]
                if "ph_amp" in modes0:
                    modes0["ph_amp"] = modes0["ph_amp"] * mask[..., None]
                rs0 = self.next_strikes(ki, onset, mask)
            latent = None
            if cfg.res_latent:  # its own stream: the strike, noise and floor draws of a seed stay as without it
                g = (None if generator is None else
                     torch.Generator(device=generator.device).manual_seed(generator.initial_seed() + LATENT_SEED))
                latent = torch.randn(B, N, cfg.res_latent, generator=g, device=pitch.device)
            ctx, frame_ctx, curves = self.context(onset_roll, key_down, pedals, ki, u, onset, offset, mask, cond, H,
                                                  n_samples, self._flat_sets(modes0), modes0["restrike"], rs0, C,
                                                  engagement, latent=latent)
        elif residual:
            ctx, frame_ctx = self.context(onset_roll, key_down, pedals, ki, u, (onset + t_hist) * sr / hop, cond, H)
        # per-strike variation: the keys and the context net follow the MIDI, the sound starts at the jittered onset;
        # ``out["ctx"]`` stays the context net's own (the residual budget penalises it)
        var = self.strike_offsets(ki, generator)
        note_ctx = self.with_strike(ctx, var, torch.zeros_like(u))
        if var is not None:
            onset = onset + var["onset_ms"] / 1000
        modes = self.physics.modes(ki, u, soft_on, cond, note_ctx, phantoms=cfg.n_phantoms > 0)
        if cfg.use_room and self.room.ring_on:  # the board's ring-up keeps the partials' levels (config body_ring_ms)
            modes["amp"] = modes["amp"] * self.room.ring_gain(modes["freq"][..., :1]).rsqrt()
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
            s, own = self.render_strings(modes, ki, onset, mask, C, H * hop, rs_delay, pan, s0, L, per_key=cfg.use_sympathetic,
                                         curves=curves)
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
            speed = self.physics.hammer_speed(u, cond)
            args = (ki, u, onset, release, mask, pedal_env, w_off, note_ctx, frame_ctx, n_samples, block, None, residual,
                    white, white_res, speed)
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
                                    pedal_env.detach(), w_off.detach(), note_ctx, frame_ctx, n_samples, block, None, residual,
                                    white, white_res, speed.detach(), use_reentrant=False)
                view = view + nv[:, None] + (rv[:, None] if rv is not None else 0)
            if gains is not None:
                view = self.apply_band_gains(view, gains.detach(), block)
            view = fft_convolve(view, ir.detach()) if cfg.use_room else view
            out["audio_texture"] = view + fl.detach() if fl is not None else view
        out["ctx"], out["frame_ctx"] = ctx, frame_ctx
        if curves is not None:
            out["curves"] = curves
        # what pianonn.partial_view reads: each note's partials as rendered, its sound onset (s re the window), and what
        # else shapes its envelope (oscbank: exp(-alpha tau - alpha_damp damp - restrike)) at the control frames from
        # sample 0: the damper-on time since its onset (s) and the re-strikes of its key so far (nats)
        with torch.no_grad():
            Fr = F - H
            c_key = C.gather(1, ki[..., None].expand(-1, -1, F))[..., H:]  # [B, N, Fr]
            c_on = sample_keyed(C, ki, onset.clamp(min=-t_hist) + t_hist, sr, hop)
            t_k = torch.arange(Fr, device=pitch.device, dtype=onset.dtype) * hop / sr
            S = ((t_k[None, None, None] - onset[..., None, None] - rs_delay[..., None]) / RESTRIKE_RAMP).clamp(0, 1).sum(2)
        out["partials"] = {"freq": modes["freq"][..., 0].detach(), "amp": modes["amp"].detach(),
                           "alpha": modes["alpha"].detach(), "onset": onset.detach(), "mask": mask,
                           "alpha_damp": modes["alpha_damp"].detach(), "damp": (c_key - c_on[..., None]).clamp(min=0),
                           "restrike": modes["restrike"].detach()[..., None] * S, "ctrl_hop": hop / sr}
        return out
