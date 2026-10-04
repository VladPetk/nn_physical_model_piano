"""The coupled unison: every string of a key in both polarisations, coupled through the bridge (docs/physics_revamp.md 1).

Weinreich (1977, JASA 62): the strings of a unison share one bridge of complex admittance, so they are one dynamical
system. Per key and partial, with x the complex amplitudes of the strings' vertical and horizontal motion in a frame
rotating at the partial's nominal frequency,

    dx/dt = i Omega x,   Omega = diag(d^V, d^H) + zeta_V J_VV + zeta_H J_HH + zeta_X (J_VH + J_HV)  (+ i a I)

with d the strings' own detunings (rad/s), zeta = i kappa y the bridge's effect on one string (Im: its loss through
the bridge, Re: its frequency shift; y the admittance, normalised), J the all-ones blocks (the bridge moves under
every string alike), and a the strings' internal and air losses (a multiple of the identity: it shifts every
eigenvalue by i a and is added after the decomposition). The normal modes of Omega give each mode's frequency offset
(Re lambda) and amplitude decay (Im lambda); a struck unison starts at rest with velocities e, x(0) = -i e (the real
signal is then a sine, as the oscillators render), and mode m reaches the vertical bus with (1_V . r_m)(l_m . x(0)),
the horizontal one with (1_H . r_m)(l_m . x(0)).

The decomposition is per recording condition, key and partial (the admittance is the condition's); per note only the
excitation is projected. Computed in float64: near an exceptional point (mistuning ~ bridge loss, resistive coupling)
two modes have large amplitudes of opposite sign that cancel.
"""

import math

import torch
from torch import nn

from .dsp import bounded

N_KEYS = 88
MAX_STRINGS = 3
SLOTS = 2 * MAX_STRINGS  # modes per partial: strings x polarisations


def string_counts():
    """Strings per key: 1 (A0-E1), 2 (F1-A#2), 3 above (as ``PianoPhysics.n_strings``)."""
    k = torch.arange(N_KEYS)
    return torch.where(k < 8, 1, torch.where(k < 26, 2, 3))


def slot_index(n_strings):
    """``[K, SLOTS]`` gather indices into ``cat(e_V[3], e_H[3], 0)`` laying out a key's state as ``(v_1..v_N, h_1..h_N,
    pad)``; the pad points at the zero (index 6)."""
    rows = []
    for N in n_strings.tolist():
        idx = list(range(N)) + [MAX_STRINGS + i for i in range(N)]
        rows.append(idx + [SLOTS] * (SLOTS - len(idx)))
    return torch.tensor(rows, dtype=torch.long)


def coupling_matrix(d_v, d_h, zeta_v, zeta_h, zeta_x):
    """``Omega [..., 2N, 2N]`` (complex) from the detunings ``d_v, d_h [..., N]`` (rad/s) and the couplings ``zeta_*
    [...]`` (complex, 1/s). The internal loss is left out (it commutes)."""
    N = d_v.shape[-1]
    ones = torch.ones(N, N, dtype=zeta_v.dtype, device=zeta_v.device)
    top = torch.cat([zeta_v[..., None, None] * ones, zeta_x[..., None, None] * ones], -1)
    bot = torch.cat([zeta_x[..., None, None] * ones, zeta_h[..., None, None] * ones], -1)
    om = torch.cat([top, bot], -2)
    return om + torch.diag_embed(torch.cat([d_v, d_h], -1).to(om.dtype))


EIG_GAP = 1e-5  # rad/s: eigenvalue gaps below this are regularised in the backward (exact degeneracy only)


class _Eig(torch.autograd.Function):
    """``torch.linalg.eig`` with PyTorch's backward (linalg_eig_backward: gA = V^-H (diag(gL) + (V^H gV - V^H V
    diag(Re V^H gV)) / E*) V^H, E_ij = L_j - L_i), but 1 / E regularised to E* / (|E|^2 + EIG_GAP^2) and without its
    check that the loss ignores each eigenvector's phase. Degenerate pairs are structural here (at the start the
    in-plane polarisation has the vertical's detunings, so their bridge-silent combinations coincide), and the loss is
    invariant to the eigenvectors' scale by construction (``normal_modes``)."""

    @staticmethod
    def forward(ctx, A):
        L, V = torch.linalg.eig(A)
        ctx.save_for_backward(L, V)
        return L, V

    @staticmethod
    def backward(ctx, gL, gV):
        L, V = ctx.saved_tensors
        Vh = V.mH
        n = L.shape[-1]
        if gV is None:
            ret = torch.zeros(*L.shape, n, dtype=V.dtype, device=V.device)
        else:
            VhgV = Vh @ gV
            ret = VhgV - Vh @ (V * VhgV.diagonal(dim1=-2, dim2=-1).real[..., None, :])
            Ec = L.conj()[..., None, :] - L.conj()[..., :, None]
            ret = ret * Ec.conj() / (Ec.abs() ** 2 + EIG_GAP ** 2)
        diag = gL if gL is not None else torch.zeros_like(L)
        ret = ret - torch.diag_embed(ret.diagonal(dim1=-2, dim2=-1)) + torch.diag_embed(diag)
        return torch.linalg.solve(Vh, ret @ Vh)


def normal_modes(omega):
    """``(lam [..., 2N], s_v [..., 2N], s_h [..., 2N], rinv [..., 2N, 2N])``: eigenvalues, each mode's share of the
    vertical and horizontal sums (1_V . r_m, 1_H . r_m) and the inverse of the eigenvector matrix (rows: left
    eigenvectors normalised so that l_m . r_m = 1).

    Omega is complex symmetric (``coupling_matrix``), so the left eigenvectors are the right ones transposed:
    l_m = r_m^T / (r_m^T r_m), no inverse needed. Every product s_m (l_m . x) is then invariant to any rescaling of r_m,
    its arbitrary phase included, exactly in the graph: ``torch.linalg.eig``'s backward checks that the loss does not
    depend on that phase, and through ``inv(R)`` rounding on an ill-conditioned R tripped the check. r_m^T r_m -> 0 at
    an exceptional point, where the modes' amplitudes diverge (physics, not rounding)."""
    N = omega.shape[-1] // 2
    lam, R = _Eig.apply(omega)
    rinv = R.transpose(-1, -2) / (R * R).sum(-2)[..., :, None]
    return lam, R[..., :N, :].sum(-2), R[..., N:, :].sum(-2), rinv


def two_string_check(eps, eta, t):
    """Weinreich's two strings, one polarisation, purely resistive coupling (zeta = i eta), mistuning 2 eps between
    them, struck alike: the relative power into the bridge R(t) = |x_1 + x_2|^2 / |2 b|^2 (his eqs. 13, 19-22): with
    nu = sqrt(eta^2 - eps^2) (imaginary above eps = eta: a beat), the sum's amplitude is
    ((eta + nu) e^{-(eta + nu) t} - (eta - nu) e^{-(eta - nu) t}) / (2 nu), the slow part (eta - nu)/(eta + nu) of the
    fast one. For the tests."""
    nu = torch.sqrt(torch.tensor(eta ** 2 - eps ** 2, dtype=torch.complex128))
    t = t.to(torch.complex128)
    a = ((eta + nu) * torch.exp(-(eta + nu) * t) - (eta - nu) * torch.exp(-(eta - nu) * t)) / (2 * nu)
    return a.abs() ** 2


class CoupledStrings(nn.Module):
    """The per-key parameters of the coupled unison, the board admittance per condition (build 3), the longitudinal
    path (build 4), the knock's structural resonances (build 5) and the pitch glide (build 6). Created by
    ``PianoPhysics`` when ``cfg.string_model == "coupled"``; every new quantity starts where the mode model was, or at
    zero effect (docs/physics_revamp.md)."""

    ADM_MODES = 16
    ADM_KEY_STEP = 4
    REACT_KNOTS = 10  # one per octave from 27.5 Hz
    LM_MODES = 4
    KNOCK_RES = 4
    KNOCK_RES_HZ = (250.0, 120.0, 1200.0, 1700.0)  # shank, treble low ring, capo bar / plate (docs 5)
    KNOCK_RES_S = (0.03, 0.3, 0.05, 0.05)

    def __init__(self, cfg):
        super().__init__()
        C = cfg.n_conditions
        n = string_counts()
        self.register_buffer("n_strings", n, persistent=False)
        self.register_buffer("slots", slot_index(n), persistent=False)
        live = (torch.arange(MAX_STRINGS)[None] < n[:, None]).float()
        self.register_buffer("live", live, persistent=False)  # [K, 3] strings that exist
        # unison: per string pitch offset (cents), inharmonicity offset (fraction of B), blow evenness (log), strike point
        # offset (fraction of x0); each zero-mean over the key's strings
        self.raw_cents = nn.Parameter(torch.zeros(N_KEYS, MAX_STRINGS))
        self.raw_boff = nn.Parameter(torch.zeros(N_KEYS, MAX_STRINGS))
        self.raw_even = nn.Parameter(torch.zeros(N_KEYS, MAX_STRINGS))
        self.raw_xoff = nn.Parameter(torch.zeros(N_KEYS, MAX_STRINGS))
        # polarisations: horizontal share of the blow, horizontal / vertical conductance, cross-coupling, detuning
        self.raw_sh = nn.Parameter(torch.zeros(N_KEYS))
        self.raw_rh = nn.Parameter(torch.zeros(N_KEYS))
        self.raw_rx = nn.Parameter(torch.zeros(N_KEYS))
        self.raw_dh = nn.Parameter(torch.zeros(N_KEYS))
        # una corda: the fraction of string 1's blow that the shifted hammer misses (trichords ~0.9, bichords ~0.5)
        self.register_buffer("uc_prior", torch.where(n == 3, 2.2, torch.where(n == 2, 0.0, -9.0)).float(),
                             persistent=False)
        self.raw_uc = nn.Parameter(torch.zeros(N_KEYS))
        # the admittance (build 3): reactive part re the conductance per octave, board modes with shapes over the keys,
        # and the radiation's tie to the admittance
        self.raw_react = nn.Parameter(torch.zeros(C, self.REACT_KNOTS))
        f0 = torch.logspace(math.log10(50.0), math.log10(900.0), self.ADM_MODES)
        self.adm_log_f = nn.Parameter(torch.log(f0).repeat(C, 1))
        self.adm_raw_q = nn.Parameter(torch.zeros(C, self.ADM_MODES))  # Q = 30 x/e 1.5
        nk = N_KEYS // self.ADM_KEY_STEP + 1
        self.adm_raw_shape = nn.Parameter(torch.full((C, self.ADM_MODES, nk), -6.0))
        self.raw_gamma = nn.Parameter(torch.zeros(()))
        # longitudinal (build 4): level re the note's mf level (dB), LM1 / f1 (x/e 0.3 around the phantom prior's 15),
        # LM decay (0.15 s x/e), LM gains per register (dB re the direct path), the vertical bus's share (dB)
        self.raw_long_db = nn.Parameter(torch.zeros(N_KEYS))
        self.raw_lm1 = nn.Parameter(torch.zeros(N_KEYS))
        self.raw_lm_tau = nn.Parameter(torch.zeros(N_KEYS))
        self.lm_gain_db = nn.Parameter(torch.full((3, self.LM_MODES), -40.0))
        self.raw_long_v = nn.Parameter(torch.zeros(()))
        # the level's calibration per key (dB): set when the strings are distilled from a mode model, to its phantom
        # partials' energy (fit_init.calibrate_longitudinal); a buffer, so the square law stays the physics'
        self.register_buffer("long_cal_db", torch.zeros(N_KEYS))
        # knock resonances (build 5), per register: frequency (x/e 0.5 around the prior), decay (x/e 1), level (dB re
        # the knock impulse)
        self.kr_raw_f = nn.Parameter(torch.zeros(3, self.KNOCK_RES))
        self.kr_raw_tau = nn.Parameter(torch.zeros(3, self.KNOCK_RES))
        self.kr_db = nn.Parameter(torch.full((3, self.KNOCK_RES), -60.0))
        self.register_buffer("kr_f0", torch.tensor(self.KNOCK_RES_HZ), persistent=False)
        self.register_buffer("kr_tau0", torch.tensor(self.KNOCK_RES_S), persistent=False)
        # pitch glide (build 6): relative frequency rise at the note's mf level, x 10^((level - mf)/10)
        self.raw_glide = nn.Parameter(torch.zeros(N_KEYS))

    # ------------------------------------------------------------------ per-key quantities
    def _zero_mean(self, x):
        live = self.live
        return (x - (x * live).sum(-1, keepdim=True) / live.sum(-1, keepdim=True)) * live

    def string_cents(self):
        return self._zero_mean(bounded(self.raw_cents, 5.0))

    def string_boff(self):
        return self._zero_mean(bounded(self.raw_boff, 0.05))

    def string_even(self):
        return self._zero_mean(bounded(self.raw_even, 0.7))

    def string_xoff(self):
        return self._zero_mean(bounded(self.raw_xoff, 0.05))

    def h_share(self):
        return 10 ** ((-20.0 + bounded(self.raw_sh, 20.0)) / 20)

    def h_ratio(self):
        return 0.1 * torch.exp(bounded(self.raw_rh, 2.3))

    def x_ratio(self):
        return 0.05 * torch.exp(bounded(self.raw_rx, 2.3))

    def una_corda(self):
        return torch.sigmoid(self.uc_prior + self.raw_uc)

    def gamma(self):
        return bounded(self.raw_gamma, 1.5)

    def admittance(self, f, cond, g_smooth):
        """Normalised admittance ``(y, y_smooth) [Cu, K, P]`` (complex) at partial frequencies ``f [Cu, K, P]`` (all 88
        keys) for conditions ``cond [Cu]``: ``g_smooth [Cu, K, P]`` (the condition's conductance curve, real) times
        (1 + i reactive ratio), plus the board modes, each its resonance times its shape at the key."""
        f = f.double()
        lf = torch.log2(f.clamp(min=1.0) / 27.5)
        react = bounded(self.raw_react[cond], 2.0).double()  # [Cu, knots]
        pos = lf.clamp(0, self.REACT_KNOTS - 1)
        i0 = pos.floor().long().clamp(max=self.REACT_KNOTS - 2)
        w = pos - i0
        beta = react.gather(1, i0.flatten(1)).view_as(w) * (1 - w) + react.gather(1, (i0 + 1).flatten(1)).view_as(w) * w
        y = g_smooth.double() * torch.complex(torch.ones_like(beta), beta)
        fm = torch.exp(self.adm_log_f[cond]).double()  # [Cu, M]
        q = 30.0 * torch.exp(bounded(self.adm_raw_q[cond], 1.5)).double()
        nk = self.adm_raw_shape.shape[-1]
        kp = (torch.arange(N_KEYS, device=f.device, dtype=torch.float64) / self.ADM_KEY_STEP).clamp(0, nk - 1)
        k0 = kp.floor().long().clamp(max=nk - 2)
        wk = kp - k0
        shape = torch.exp(self.adm_raw_shape[cond]).double()  # [Cu, M, nk]
        a = shape[..., k0] * (1 - wk) + shape[..., k0 + 1] * wk  # [Cu, M, K]
        om, omm = 2 * math.pi * f[:, None], 2 * math.pi * fm[..., None, None]  # [Cu,1,K,P], [Cu,M,1,1]
        damp = om * omm / q[..., None, None]
        res = torch.complex(torch.zeros_like(damp), damp) / torch.complex(omm ** 2 - om ** 2, damp)  # peak 1 at omm
        return y + (a[..., None] * res).sum(1), y

    def modes(self, f_n, B, kappa, y, valid):
        """The normal modes per condition, key and partial. ``f_n [Cu, K, P]`` nominal partial frequencies, ``B [K]``,
        ``kappa [K]`` the single string's bridge-loss scale (1/s; the in-phase mode of N strings loses N kappa Re y),
        ``y [Cu, K, P]`` the admittance, ``valid [Cu, K, P]`` partials below Nyquist. Returns ``lam, s_v, s_h [Cu, K, P,
        SLOTS]`` (complex; padded slots have s = 0) and ``rinv [Cu, K, P, SLOTS, SLOTS]``."""
        Cu, K, P = y.shape
        dev = y.device
        f64 = f_n.double()
        n = torch.arange(1, P + 1, device=dev, dtype=torch.float64)
        Bn2 = B.double()[:, None] * n ** 2
        d = 2 * math.pi * f64[..., None] * (self.string_cents().double()[:, None, :] * math.log(2) / 1200
                                            + (Bn2 / (2 * (1 + Bn2)))[..., None] * self.string_boff().double()[:, None, :])
        dh = d + 2 * math.pi * f64[..., None] * (bounded(self.raw_dh, 1.0).double()[:, None, None] * math.log(2) / 1200)
        # d, dh [Cu, K, P, 3]
        zv = 1j * kappa.double()[None, :, None] * y  # [Cu, K, P]
        yh = torch.complex(y.real * self.h_ratio().double()[None, :, None], y.imag)
        zh = 1j * kappa.double()[None, :, None] * yh
        zx = zv * self.x_ratio().double()[None, :, None]
        lam = torch.zeros(Cu, K, P, SLOTS, dtype=torch.complex128, device=dev)
        s_v, s_h = torch.zeros_like(lam), torch.zeros_like(lam)
        rinv = torch.zeros(Cu, K, P, SLOTS, SLOTS, dtype=torch.complex128, device=dev)
        for N in (1, 2, 3):
            keys = (self.n_strings == N).nonzero().squeeze(1)
            if keys.numel() == 0:
                continue
            sub = lambda x: x.index_select(1, keys)  # noqa: E731
            om = coupling_matrix(sub(d)[..., :N], sub(dh)[..., :N], sub(zv), sub(zh), sub(zx))
            ok = sub(valid)  # partials above Nyquist: a diagonal stand-in, zeroed after
            eye = torch.eye(2 * N, dtype=om.dtype, device=dev) * (1j + torch.arange(2 * N, device=dev))
            om = torch.where(ok[..., None, None], om, eye)
            l, sv, sh, ri = normal_modes(om)
            okc = ok[..., None].to(l.dtype)
            lam[:, keys, :, :2 * N] = l * okc
            s_v[:, keys, :, :2 * N] = sv * okc
            s_h[:, keys, :, :2 * N] = sh * okc
            rinv[:, keys, :, :2 * N, :2 * N] = ri * okc[..., None]
        return lam, s_v, s_h, rinv

    def regularizer(self):
        def smooth(x):
            return ((x[2:] - 2 * x[1:-1] + x[:-2]) ** 2).mean()
        reg = sum(smooth(t) for t in (self.raw_cents, self.raw_boff, self.raw_even, self.raw_xoff)) * 0.1
        reg = reg + sum(smooth(t) for t in (self.raw_sh / 20, self.raw_rh, self.raw_rx, self.raw_dh, self.raw_uc,
                                            self.raw_long_db / 20, self.raw_lm1, self.raw_lm_tau, self.raw_glide))
        reg = reg + ((self.adm_raw_shape[..., 2:] - 2 * self.adm_raw_shape[..., 1:-1] + self.adm_raw_shape[..., :-2]) ** 2).mean()
        return reg
