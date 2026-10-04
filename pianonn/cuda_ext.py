"""The fused CUDA kernels (``pianonn/csrc``): the string bank and the sympathetic resonators, with analytic backwards.

Built on first use with ``torch.utils.cpp_extension.load`` into ``pianonn/csrc/build`` (a rebuild only when the
sources change; not torch's default directory, which the Windows Store Python redirects where ninja cannot see it). On Windows the MSVC environment is taken from Visual Studio's ``vcvars64.bat`` when ``cl``
is not already on the PATH. If anything fails, ``get()`` returns None and the model renders with the PyTorch
reference (``pianonn.oscbank``, ``pianonn.dsp.linear_recurrence``), which stays the definition: tests/test_cuda.py
compares the two. ``PIANONN_FUSED=0`` switches the kernels off.
"""

import glob
import os
import shutil
import subprocess
import warnings

import torch

_EXT = None
_TRIED = False
SRC = os.path.join(os.path.dirname(__file__), "csrc")


def enabled():
    return os.environ.get("PIANONN_FUSED", "1") != "0"


def _msvc_env():
    """Import vcvars64.bat's environment (Windows) so that nvcc and ninja find cl.exe."""
    if os.name != "nt" or shutil.which("cl"):
        return
    vswhere = os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
                           "Microsoft Visual Studio", "Installer", "vswhere.exe")
    roots = []
    if os.path.exists(vswhere):
        out = subprocess.run([vswhere, "-all", "-products", "*", "-requires",
                              "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"],
                             capture_output=True, text=True).stdout
        roots = [r.strip() for r in out.splitlines() if r.strip()]
    bats = [os.path.join(r, "VC", "Auxiliary", "Build", "vcvars64.bat") for r in roots]
    bats += glob.glob(r"C:\Program Files*\Microsoft Visual Studio\*\*\VC\Auxiliary\Build\vcvars64.bat")
    bat = next((b for b in bats if os.path.exists(b)), None)
    if bat is None:
        raise RuntimeError("no MSVC (vcvars64.bat) found")
    out = subprocess.run(f'cmd /s /c ""{bat}" >nul && set"', capture_output=True, text=True, shell=True).stdout
    for line in out.splitlines():
        k, sep, v = line.partition("=")
        if sep and k:
            os.environ[k] = v


def get():
    """The compiled extension, or None (no CUDA, switched off, or the build failed: a warning says why)."""
    global _EXT, _TRIED
    if _TRIED or not enabled() or not torch.cuda.is_available():
        return _EXT
    _TRIED = True
    try:
        _msvc_env()
        from torch.utils.cpp_extension import load

        if "TORCH_CUDA_ARCH_LIST" not in os.environ:
            major, minor = torch.cuda.get_device_capability()
            os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        build = os.path.join(SRC, "build")
        os.makedirs(build, exist_ok=True)
        _EXT = load(name="pianonn_cuda", build_directory=build, sources=[os.path.join(SRC, "bindings.cpp"), os.path.join(SRC, "kernels.cu")],
                    extra_cflags=["/O2"] if os.name == "nt" else ["-O3"],
                    extra_cuda_cflags=["-O3", "-allow-unsupported-compiler"], verbose=False)
    except Exception as e:  # the PyTorch reference renders the same, slower
        warnings.warn(f"pianonn: fused CUDA kernels unavailable, using the PyTorch reference ({e})")
        _EXT = None
    return _EXT


class StringBank(torch.autograd.Function):
    """The string bank over the whole window: ``y[P, L]`` for ``P`` active notes. Differentiable in the oscillators
    (freq, alpha, amp, adamp ``[P, Q]``), the notes' contact time ``tc``, damper integral at the onset ``c_onset``
    and re-strike loss ``rs`` ``[P]``, the damper integral rows ``C[rows, F]`` and the group curves' log gains
    ``m[P, G, CM]``. ``fixed``: the rest, passed through to the kernels (see ``kernels.cu``, ``Bank``)."""

    @staticmethod
    def forward(ctx, freq, alpha, amp, adamp, tc, c_onset, rs, C, m, fixed):
        ext = get()
        g0, w0, g1, w1, onset, rsd, row, nv, ints, sr = fixed
        args = (freq.contiguous(), alpha.contiguous(), amp.contiguous(), adamp.contiguous(), g0, w0, g1, w1, onset,
                tc.contiguous(), c_onset.contiguous(), rs.contiguous(), rsd, C.contiguous(), row, m.contiguous(), nv,
                *ints, sr)
        ctx.template = tuple(None if isinstance(a, torch.Tensor) else a for a in args)
        ctx.save_for_backward(*[a for a in args if isinstance(a, torch.Tensor)])
        return ext.osc_forward(*args)

    @staticmethod
    def backward(ctx, gy):
        saved = iter(ctx.saved_tensors)
        args = tuple(next(saved) if a is None else a for a in ctx.template)
        d = get().osc_backward(gy.contiguous(), *args)
        d_freq, d_alpha, d_amp, d_adamp, d_tc, d_conset, d_rs, dC, dm = d
        return d_freq, d_alpha, d_amp, d_adamp, d_tc, d_conset, d_rs, dC, dm, None


class BusBank(torch.autograd.Function):
    """The coupled strings' bank over the whole window (``oscbank.bus_bank``) for ``P`` active notes: oscillators
    ``[P, Q]`` with bus amplitudes ``amp[P, Q, NB, 2]`` (sine, cosine), the modes (the first ``Q0``) gliding with
    ``ge``, ``gb`` ``[P]``. Returns the first ``NM`` buses summed per example ``[B, NM, L]``, the other buses per note
    ``[P, NB - NM, L]`` and, with ``keys``, the mean of the first ``NM / 2`` buses per key row ``[rows, L]`` (else
    empty). Differentiable as ``StringBank``, and in the amplitudes and the glide."""

    @staticmethod
    def forward(ctx, freq, alpha, amp, adamp, tc, c_onset, rs, ge, gb, C, m, fixed):
        ext = get()
        g0, w0, g1, w1, onset, rsd, row, nv, ints, sr, bidx, NM, n_ex, n_rows, keys = fixed
        args = (freq.contiguous(), alpha.contiguous(), amp.contiguous(), adamp.contiguous(), g0, w0, g1, w1, onset,
                tc.contiguous(), c_onset.contiguous(), rs.contiguous(), rsd, C.contiguous(), row, m.contiguous(), nv,
                *ints, sr)
        extra = (ge.contiguous(), gb.contiguous(), bidx, NM)
        ctx.template = tuple(None if isinstance(a, torch.Tensor) else a for a in args + extra)
        ctx.save_for_backward(*[a for a in args + extra if isinstance(a, torch.Tensor)])
        ctx.keys, ctx.n_ex = keys, n_ex
        return tuple(ext.bus_forward(*args, *extra, n_ex, n_rows, keys))

    @staticmethod
    def backward(ctx, gmix, glon, gkeys):
        saved = iter(ctx.saved_tensors)
        args = tuple(next(saved) if a is None else a for a in ctx.template)
        bank, extra = args[:-4], args[-4:]
        L = bank[-2]
        P, NB = bank[2].shape[0], bank[2].shape[2]
        NM = extra[-1]
        opts = dict(device=bank[0].device, dtype=torch.float32)
        gmix = torch.zeros(ctx.n_ex, NM, L, **opts) if gmix is None else gmix.contiguous()
        glon = torch.zeros(P, NB - NM, L, **opts) if glon is None else glon.contiguous()
        if not ctx.keys or gkeys is None:
            gkeys = torch.zeros(0, **opts)
        d = get().bus_backward(gmix, glon, gkeys.contiguous(), *bank, *extra, ctx.keys and gkeys.numel() > 0)
        d_freq, d_alpha, d_amp, d_adamp, d_tc, d_conset, d_rs, d_ge, d_gb, dC, dm = d
        return d_freq, d_alpha, d_amp, d_adamp, d_tc, d_conset, d_rs, d_ge, d_gb, dC, dm, None


class Longitudinal(torch.autograd.Function):
    """The longitudinal force per note (``NeuralPhysicalPiano._longitudinal`` before its scale) from the two
    longitudinal sums ``lon[P, 2, L]``: the DC-blocked F_even + F_odd plus the free longitudinal modes ``sum_j c_j Re
    z_j`` (``c``, ``omega`` = 2 pi f / sr ``[P, J]``, decay ``alpha[P]``), from the state ``hp0[P, 2]`` (last input,
    output of the DC blocker) and ``z0[P, J, 2]``. Returns ``(out[P, L], hp1, z1)``, the final state without gradient
    (no gradient flows from one block to the next)."""

    @staticmethod
    def forward(ctx, lon, c, alpha, omega, hp0, z0, sr, r_hp):
        a = tuple(x.contiguous().float() for x in (lon, c, alpha, omega, hp0, z0))
        out, hp1, z1 = get().long_forward(*a, float(sr), float(r_hp))
        ctx.save_for_backward(a[0], a[1], a[2], a[3], a[5])
        ctx.sr, ctx.r_hp = float(sr), float(r_hp)
        ctx.mark_non_differentiable(hp1, z1)
        return out, hp1, z1

    @staticmethod
    def backward(ctx, gout, _hp1, _z1):
        lon, c, alpha, omega, z0 = ctx.saved_tensors
        d_lon, d_c, d_alpha, d_omega = get().long_backward(gout.contiguous(), lon, c, alpha, omega, z0, ctx.sr, ctx.r_hp)
        return d_lon, d_c, d_alpha, d_omega, None, None, None, None


class Resonators(torch.autograd.Function):
    """``z_t = exp(-min(alpha + es_t adamp, max_decay) / sr + i 2 pi freq / sr) z_{t-1} + gin drive_t`` per resonator
    ``[B, K, S]``, from ``z0``; returns ``(out[B, L], zr_last, zi_last)`` with ``out`` the real parts summed over the
    resonators. ``drive``, ``es``: ``[B, K, L]``."""

    @staticmethod
    def forward(ctx, drive, es, alpha, adamp, freq, gin, z0r, z0i, sr, max_decay):
        ext = get()
        d_t, e_t = drive.transpose(1, 2).contiguous(), es.transpose(1, 2).contiguous()  # [B, L, K]
        a = (alpha.contiguous(), adamp.contiguous(), freq.contiguous(), gin.contiguous(), z0r.contiguous(),
             z0i.contiguous())
        zr, zi = ext.reson_forward(d_t, e_t, *a, float(sr), float(max_decay))
        ctx.save_for_backward(d_t, e_t, *a, zr, zi)
        ctx.sr, ctx.max_decay, ctx.K, ctx.S = float(sr), float(max_decay), alpha.shape[1], alpha.shape[2]
        B, L = zr.shape[:2]
        return zr.reshape(B, L, -1).sum(-1), zr[:, -1].clone(), zi[:, -1].clone()

    @staticmethod
    def backward(ctx, gy, gzr, gzi):
        d_t, e_t, alpha, adamp, freq, gin, z0r, z0i, zr, zi = ctx.saved_tensors
        gy = torch.zeros(zr.shape[:2], device=zr.device) if gy is None else gy.contiguous()
        gzr = torch.zeros_like(alpha) if gzr is None else gzr.contiguous()
        gzi = torch.zeros_like(alpha) if gzi is None else gzi.contiguous()
        d = get().reson_backward(gy, gzr, gzi, d_t, e_t, alpha, adamp, freq, gin, z0r, z0i, zr, zi, ctx.sr,
                                 ctx.max_decay)
        d_alpha, d_adamp, d_freq, d_gin, d_z0r, d_z0i, d_drive, d_es = d
        d_drive = d_drive.sum(-1).transpose(1, 2)  # [B, L, K, S] -> [B, K, L]
        d_es = d_es.sum(-1).transpose(1, 2)
        return d_drive, d_es, d_alpha, d_adamp, d_freq, d_gin, d_z0r, d_z0i, None, None
