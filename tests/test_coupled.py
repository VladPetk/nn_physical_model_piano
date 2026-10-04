"""The coupled unison (pianonn.coupled) against Weinreich's (1977) closed forms."""

import math

import torch

from pianonn.coupled import coupling_matrix, normal_modes, two_string_check

ETA = 1.0  # single-string bridge loss, 1/s (Weinreich's mid-range value)


def _vsum(eps, eta, x0, t, zh=0.0, zx=0.0):
    """|sum of the vertical amplitudes|(t) of two strings mistuned by 2 eps (one at +2 eps), resistive coupling i eta;
    the horizontal block decoupled unless ``zh``, ``zx``."""
    eps = torch.as_tensor(eps, dtype=torch.float64)
    d_v = torch.stack([2 * eps, torch.zeros_like(eps)])
    d_h = d_v + 50.0
    z = torch.as_tensor(1j * eta, dtype=torch.complex128)
    om = coupling_matrix(d_v, d_h, z, torch.as_tensor(zh, dtype=torch.complex128), torch.as_tensor(zx, dtype=torch.complex128))
    lam, s_v, _, rinv = normal_modes(om)
    c = rinv @ x0.to(torch.complex128)
    return (s_v * c * torch.exp(1j * lam * t.to(torch.complex128)[:, None])).sum(-1)


def test_two_strings_follow_weinreich():
    """Beat regime (eps > eta), double decay (eps < eta), and near the coalescence (eps = eta): the power into the
    bridge matches his eqs. 19-22 to 1e-9."""
    t = torch.linspace(0, 5, 200, dtype=torch.float64)
    x0 = torch.tensor([1.0, 1.0, 0.0, 0.0])
    for eps in (0.0, 0.3, 0.9, 0.999, 1.001, 2.0, 6.0):
        r = _vsum(eps, ETA, x0, t).abs() ** 2 / 4
        ref = two_string_check(eps, ETA, t)
        assert torch.allclose(r, ref, atol=1e-9), (eps, (r - ref).abs().max())


def test_aftersound_fraction_and_una_corda():
    """At eps < eta the slow component is (eta - nu)/(eta + nu) of the fast one. Struck on one string only (una
    corda), half the blow goes into the antisymmetric motion: with a slight mistuning the aftersound is several times
    stronger than when both strings are struck, the attack half as strong (at exact unison the antisymmetric mode
    exerts no force on the bridge and is silent)."""
    eps, t = 0.6, torch.tensor([20.0], dtype=torch.float64)
    nu = math.sqrt(ETA ** 2 - eps ** 2)
    late = _vsum(eps, ETA, torch.tensor([1.0, 1.0, 0.0, 0.0]), t).abs() / 2  # the slow part alone survives
    expect = (ETA - nu) / (2 * nu) * math.exp(-(ETA - nu) * 20.0)
    assert abs(late.item() - expect) < 1e-9
    t2 = torch.tensor([0.0, 20.0], dtype=torch.float64)
    both = _vsum(0.1, ETA, torch.tensor([1.0, 1.0, 0.0, 0.0]), t2).abs()
    one = _vsum(0.1, ETA, torch.tensor([1.0, 0.0, 0.0, 0.0]), t2).abs()
    assert abs(one[0] / both[0] - 0.5) < 1e-9
    assert one[1] / both[1] > 5, (one[1] / both[1]).item()


def test_trace_rule_and_no_coupling():
    """The decay rates sum to the trace's imaginary part (N x each block's loss); without coupling the modes are the
    strings themselves."""
    d = torch.tensor([3.0, -1.0, 0.5], dtype=torch.float64)
    zv, zh, zx = (torch.tensor(v, dtype=torch.complex128) for v in (0.2 + 1.3j, 0.2 + 0.13j, 0.05 + 0.06j))
    lam, *_ = normal_modes(coupling_matrix(d, d + 0.1, zv, zh, zx))
    assert abs(lam.imag.sum().item() - 3 * (1.3 + 0.13)) < 1e-9
    z0 = torch.tensor(0j, dtype=torch.complex128)
    lam, s_v, s_h, rinv = normal_modes(coupling_matrix(d, d + 0.1, z0, z0, z0))
    assert torch.allclose(torch.sort(lam.real).values, torch.sort(torch.cat([d, d + 0.1])).values)
    assert lam.imag.abs().max() < 1e-12


def test_gradients_finite_through_the_coalescence():
    """Through eps = eta, where the two eigenvalues coalesce, the rendered amplitude and its gradient stay finite."""
    t = torch.tensor([0.5, 2.0], dtype=torch.float64)
    for d in (1e-3, 1e-6, 0.0, -1e-6):
        eps = torch.tensor(ETA + d, dtype=torch.float64, requires_grad=True)
        y = _vsum(eps, ETA, torch.tensor([1.0, 0.8, 0.1, 0.0]), t, zh=0.05j, zx=0.02j).abs().sum()
        y.backward()
        assert torch.isfinite(y) and torch.isfinite(eps.grad), (d, y, eps.grad)
