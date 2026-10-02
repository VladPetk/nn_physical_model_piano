"""The physics-aware residual: a learned correction that sees what the physics renders.

The first residual (``synth.ContextNet``) read only the MIDI, fixed each note's corrections at its onset and changed
the sound over time only through band gains shared by every note; per excerpt, free outputs in its own language
could take 0.136 off the test loss where it took 0.006 (docs/tone_measures.md 13.1). This one, per control frame
(``res_control`` frames, 20 ms by default) over the rendered window:

* sees, for every sounding note, its own expected energy in ``G`` octave groups of partials (from the physics' modal
  parameters without the residual: amplitudes, decay rates, the damper integral, re-strikes; no oscillators are
  rendered for this), the same summed over all notes (what the strings play at that moment), the note's age, whether
  its key is held, its damper, the pedals, its key and velocity, and a GRU's summary of the MIDI history (12 s);
* lets the sounding notes attend to each other at each frame, and each note to its own past (a GRU over frames);
* outputs per note a gain curve per octave group of partials (``curves``, applied inside the oscillator bank: the
  note's spectral shape over time), the onset-time corrections of ``ContextNet.NOTE`` (knock, attack noise, ...)
  read at the note's first frame, and the per-frame mix outputs of ``ContextNet.FRAME``.

Wider outputs (config ``res_*``, review 6, 8.2; off by default): a gain curve per partial (``res_curve_partials``)
instead of per octave group; the per-frame noise path in more bands (``res_noise_bands``); a noise path per note
(``res_note_noise`` bands), its power a level re the note's own expected energy at each control frame, so it follows
the note's decay, dampers and re-strikes (energy between its partials that comes and goes with it), starting at the
earliest one control step after the onset (it ramps in over the next, never before the hammer); and random inputs per
note (``res_latent``, drawn afresh at every render), so the corrections can vary from strike to strike as the takes do.

Every output layer starts at zero, so switching it on starts from the physics (the per-note noise at
``NOTE_NOISE_BASE``, 60 dB under its note). The features are computed without gradient: the residual cannot steer the
physics through them.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from .dsp import frames_to_samples, interp_bands, sample_keyed
from .oscbank import RESTRIKE_RAMP, group_weights
from .physics import N_KEYS

NOTE_NOISE_BASE = -60 / 20 * math.log(10)  # the per-note noise at zero output: -60 dB re its note (log amplitude)
NOTE_NOISE_BOUND = 5.5  # its range around that (log amplitude, +-48 dB)


def group_centers(n):
    """Octave group centres (Hz): 62.5, 125, ... (8 groups reach 8 kHz)."""
    return [62.5 * 2.0 ** k for k in range(n)]


def _bounded_split(z, spec):
    out, i = {}, 0
    for name, (n, s) in spec.items():
        v = s * torch.tanh(z[..., i: i + n] / s)
        out[name] = v[..., 0] if n == 1 else v
        i += n
    return out


class _Layer(nn.Module):
    """Attention across the sounding notes at each frame, an MLP, then each note's own past (a GRU over frames)."""

    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.ln1, self.ln2, self.ln3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.time = nn.GRU(d, d, batch_first=True)

    def forward(self, x, allowed):
        # x [B, N, C, d]; allowed [B, C, N, N] (query, key)
        B, N, C, d = x.shape
        h = self.ln1(x).permute(0, 2, 1, 3).reshape(B * C, N, d)
        q, k, v = self.qkv(h).view(B * C, N, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed.reshape(B * C, 1, N, N))
        a = self.proj(a.transpose(1, 2).reshape(B * C, N, d)).view(B, C, N, d).permute(0, 2, 1, 3)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        t, _ = self.time(self.ln3(x).reshape(B * N, C, d))
        return x + t.view(B, N, C, d)


class AwareResidual(nn.Module):
    N_SCALARS = 12  # per-note scalar features besides the energies, see ``forward``

    def __init__(self, cfg, note_spec, frame_spec):
        super().__init__()
        self.cfg, self.note_spec, self.frame_spec = cfg, note_spec, frame_spec
        H, d, G = cfg.ctx_hidden, cfg.res_dim, cfg.res_groups
        self.register_buffer("centers", torch.tensor(group_centers(G)), persistent=False)
        self.midi_in = nn.Linear(2 * N_KEYS + 3, H)
        self.midi_gru = nn.GRU(H, H, batch_first=True)
        self.key_table = nn.Embedding(N_KEYS, 16)
        self.cond_table = nn.Embedding(cfg.n_conditions, 16)
        self.tok_in = nn.Sequential(nn.Linear(3 * G + self.N_SCALARS + 32 + H + cfg.res_latent, d), nn.GELU(),
                                    nn.Linear(d, d))
        self.layers = nn.ModuleList(_Layer(d, cfg.res_heads) for _ in range(cfg.res_layers))
        self.curve_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, cfg.res_curve_partials or G))
        n_note = sum(n for n, _ in note_spec.values())
        n_frame = sum(n for n, _ in frame_spec.values())
        self.onset_head = nn.Sequential(nn.Linear(d + H + 33, d), nn.GELU(), nn.Linear(d, n_note))
        self.mix_head = nn.Sequential(nn.Linear(d + H + G + 16, d), nn.GELU(), nn.Linear(d, n_frame))
        heads = [self.curve_head, self.onset_head, self.mix_head]
        if cfg.res_note_noise:
            self.noise_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, cfg.res_note_noise))
            heads.append(self.noise_head)
        for head in heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def n_control(self, n_samples):
        """Control frames covering ``n_samples`` (one more than needed, for the interpolation at the end)."""
        return n_samples // (self.cfg.res_control * self.cfg.hop) + 2

    @torch.no_grad()
    def features(self, sets, restrike, rs_delay, onset, offset, ki, mask, C, engagement, pedals, H, n_samples):
        """Energies per octave group of partials, per note and summed, at the control frames; which notes sound in
        the window. Returns a dict of ``[B, N, C, ...]`` / ``[B, C, ...]`` tensors."""
        cfg = self.cfg
        sr, hop, K = cfg.sample_rate, cfg.hop, cfg.res_control
        B, N = ki.shape
        Cn = self.n_control(n_samples)
        t = torch.arange(Cn, device=onset.device, dtype=onset.dtype) * (K * hop / sr)  # re the window's first sample
        tau = (t[None, None] - onset[..., None]).clamp(min=0)  # [B, N, C]
        started = (t[None, None] >= onset[..., None]) & mask[..., None]
        f_idx = (H + torch.arange(Cn, device=onset.device) * K).clamp(max=C.shape[-1] - 1)
        Ck = C[..., f_idx]  # [B, 88, C]
        Cn_note = Ck.gather(1, ki[..., None].expand(-1, -1, Cn))
        c_on = sample_keyed(C, ki, onset.clamp(min=-H * hop / sr) + H * hop / sr, sr, hop)
        D = (Cn_note - c_on[..., None]).clamp(min=0)
        S = ((tau[..., None, :] - rs_delay[..., None]) / RESTRIKE_RAMP).clamp(0, 1).sum(-2)  # [B, N, C]
        e = 0.0
        for freq, alpha, amp, adamp in sets:  # [B, N, Q]
            W = group_weights(freq, self.centers)  # [B, N, G, Q]
            expo = -2 * (alpha[..., None] * tau[..., None, :] + adamp[..., None] * D[..., None, :]
                         + (restrike[..., None] * S)[..., None, :])  # [B, N, Q, C]
            e = e + torch.einsum("bngq,bnqc->bngc", W * amp[..., None, :] ** 2, torch.exp(expo))
        e = e * started[..., None, :]  # [B, N, G, C]
        ref = e.sum(2).amax((1, 2)).clamp(min=1e-20)  # per example: the loudest note's peak energy
        floor = 1e-9 * ref[:, None, None]  # -90 dB re that
        sounding = started & (e.sum(2) > floor)
        keep = sounding.any(-1)
        db = lambda x: torch.log10(x / ref.view(-1, *([1] * (x.dim() - 1))) + 1e-10) / 2  # ~[-5, 0]
        mix = e.sum(1)  # [B, G, C]
        key_held = (offset[..., None] > t[None, None]).float()
        eng = engagement[..., f_idx].gather(1, ki[..., None].expand(-1, -1, Cn))
        ped = pedals[..., f_idx]  # [B, 3, C]
        return {"own": db(e), "mix": db(mix), "tau": tau, "started": started, "sounding": sounding, "keep": keep,
                "held": key_held, "damper": eng, "pedals": ped, "S": S, "level": torch.log10(ref) / 10,
                "energy": e.sum(2), "settled": (t[None, None] - K * hop / sr >= onset[..., None]) & mask[..., None]}

    def forward(self, onset_roll, key_down, pedals, ki, u, onset, offset, mask, cond, hist_frames, n_samples, sets,
                restrike, rs_delay, C, engagement, latent=None):
        """Returns ``(note, frame, curves)``: ``ContextNet.NOTE`` outputs per note, the frame outputs (``frame_spec``)
        ``[B, bands, F]`` over the window's frames, and log-amplitude gain curves ``[B, N, G, C]`` per note and octave
        group of partials (or partial, ``res_curve_partials``) at the control frames (``res_control * hop`` samples
        apart, from the window's first). With ``res_note_noise``, ``frame["note_noise"]`` is the per-note noise summed
        over the notes: power ``[B, noise_bands, F]`` (the noise bank's bands, its units). ``latent``: the random inputs
        ``[B, N, res_latent]``."""
        cfg = self.cfg
        H = hist_frames
        B, N = ki.shape
        G, K = cfg.res_groups, cfg.res_control
        f = self.features(sets, restrike, rs_delay, onset, offset, ki, mask, C, engagement, pedals, H, n_samples)
        Cn = f["tau"].shape[-1]

        x = torch.cat([onset_roll, key_down, pedals], 1).transpose(1, 2)
        hm, _ = self.midi_gru(torch.tanh(self.midi_in(x)))  # [B, F, Hd]
        f_idx = (H + torch.arange(Cn, device=ki.device) * K).clamp(max=hm.shape[1] - 1)
        h_ctrl = hm[:, f_idx]  # [B, C, Hd]

        # the notes that sound in the window, packed to the front
        keep = f["keep"]
        Np = max(int(keep.sum(1).max()), 1)
        order = torch.argsort((~keep).to(torch.int8), dim=1, stable=True)[:, :Np]  # [B, Np]
        valid = keep.gather(1, order)
        take = lambda x: x.gather(1, order.view(B, Np, *([1] * (x.dim() - 2))).expand(B, Np, *x.shape[2:]))
        own = take(f["own"])  # [B, Np, G, C]
        mixe = f["mix"][:, None].expand(-1, Np, -1, -1)
        kin, un = take(ki), take(u)
        cemb = self.cond_table(cond)
        scal = torch.stack([
            torch.log1p(take(f["tau"]) / 0.01) / 5, take(f["started"]).float(), take(f["held"]), take(f["damper"]),
            *[f["pedals"][:, i][:, None].expand(-1, Np, -1) for i in range(3)], take(f["S"]).clamp(max=3) / 3,
            un[..., None].expand(-1, -1, Cn), (kin.float()[..., None] / (N_KEYS - 1)).expand(-1, -1, Cn),
            f["level"].view(B, 1, 1).expand(-1, Np, Cn), valid.float()[..., None].expand(-1, -1, Cn)], -1)
        assert scal.shape[-1] == self.N_SCALARS
        extra = [take(latent)[:, :, None].expand(-1, -1, Cn, -1)] if cfg.res_latent else []
        tok = torch.cat([own.transpose(2, 3), mixe.transpose(2, 3), (own - mixe).transpose(2, 3), scal,
                         self.key_table(kin)[:, :, None].expand(-1, -1, Cn, -1),
                         cemb[:, None, None].expand(-1, Np, Cn, -1),
                         h_ctrl[:, None].expand(-1, Np, -1, -1), *extra], -1)  # [B, Np, C, ...]
        z = self.tok_in(tok)
        active = take(f["sounding"]) & valid[..., None]  # [B, Np, C]
        act_k = active.permute(0, 2, 1)  # [B, C, Np] as keys
        eye = torch.eye(Np, dtype=torch.bool, device=ki.device)
        allowed = act_k[:, :, None, :] | eye  # every token sees the sounding notes and itself
        for layer in self.layers:
            z = layer(z, allowed)

        bound = cfg.res_curve_db / (20 / math.log(10))  # dB -> log amplitude
        # not gated at the onset: the strings are silent before it, and a gate would ramp the gain in over the
        # first control step of every note
        curves_p = bound * torch.tanh(self.curve_head(z) / bound)  # [B, Np, C, G]
        Gc = curves_p.shape[-1]
        curves = curves_p.new_zeros(B, N, Cn, Gc).scatter(
            1, order[..., None, None].expand(-1, -1, Cn, Gc), curves_p * valid[..., None, None].float())
        curves = curves.transpose(2, 3)  # [B, N, G, C]

        # onset-time outputs from each kept note's first frame in the window, with the MIDI state at its onset
        c_first = (onset.clamp(min=0) / (K * cfg.hop / cfg.sample_rate)).ceil().long().clamp(max=Cn - 1)
        z_on = z.gather(2, take(c_first).view(B, Np, 1, 1).expand(-1, -1, 1, z.shape[-1]))[:, :, 0]  # [B, Np, d]
        fr = ((onset + H * cfg.hop / cfg.sample_rate) * cfg.sample_rate / cfg.hop).ceil().clamp(0, hm.shape[1] - 1)
        h_on = hm.gather(1, take(fr.long())[..., None].expand(-1, -1, hm.shape[-1]))
        feats = torch.cat([z_on, h_on, self.key_table(kin), cemb[:, None].expand(-1, Np, -1), un[..., None]], -1)
        raw = self.onset_head(feats) * valid[..., None].float()
        raw_full = raw.new_zeros(B, N, raw.shape[-1]).scatter(1, order[..., None].expand(-1, -1, raw.shape[-1]), raw)
        note = _bounded_split(raw_full, self.note_spec)

        # the mix outputs per frame: the sounding notes' mean, the MIDI state, what the strings play
        w = active.float()
        pooled = (z * w[..., None]).sum(1) / w.sum(1).clamp(min=1)[..., None]  # [B, C, d]
        m = self.mix_head(torch.cat([pooled, h_ctrl, f["mix"].transpose(1, 2), cemb[:, None].expand(-1, Cn, -1)], -1))
        Fw = hm.shape[1] - H
        m = F.interpolate(m.transpose(1, 2), size=(Cn - 1) * K, mode="linear", align_corners=False)[..., :Fw]
        if m.shape[-1] < Fw:
            m = torch.cat([m, m[..., -1:].expand(-1, -1, Fw - m.shape[-1])], -1)
        frame = {k: v.transpose(1, 2) for k, v in _bounded_split(m.transpose(1, 2), self.frame_spec).items()}

        if cfg.res_note_noise:  # each note's noise: a level per band re its own expected energy, summed over the notes
            lv = NOTE_NOISE_BOUND * torch.tanh(self.noise_head(z) / NOTE_NOISE_BOUND)  # [B, Np, C, bands]
            lv = interp_bands(lv.transpose(2, 3), cfg.noise_bands)  # [B, Np, noise_bands, C]
            on = (take(f["settled"]) & valid[..., None]).float() * take(f["energy"])  # [B, Np, C]
            pw = (torch.exp(2 * (NOTE_NOISE_BASE + lv)) * on[:, :, None]).sum(1)  # [B, noise_bands, C]
            frame["note_noise"] = frames_to_samples(pw, 0, Fw, K)  # control steps -> frames, exactly (frame j at j / K)
        return note, frame, curves
