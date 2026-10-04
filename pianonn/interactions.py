"""Interaction tables (docs/physics_revamp.md 7): where two factors meet that the physics does not join, a smooth
bounded table over key x a second factor (velocity, or the sustain pedal's lift at the strike) x log partial number,
interpolated linearly between knots. Each starts at zero (the identity: the model renders as without it) and is held
at zero mean over the factor it adds, so it can only carry the interaction, never a main effect that another
parameter already owns (a key's decay, its level, its brightness at mf).

    decay_vel      log decay rate of every mode            x velocity, per partial
    decay_lift     log decay rate of every mode            x pedal lift at the strike, per partial
    damp_vel       log damper rate                         x velocity, per partial
    restrike_vel   log re-strike loss                      x velocity
    spec_vel       log amplitude (the strike's spectrum)   x velocity, per partial
    even_vel       log blow per unison string (coupled)    x velocity, per string (also zero mean over the strings)

Config ``interactions`` (off by default: a checkpoint without the tables renders as before)."""

import torch
from torch import nn

from .dsp import bounded

KEY_KNOTS = 8                 # over the 88 keys, evenly
FACTOR_KNOTS = 5              # over velocity or lift in [0, 1]
PARTIAL_KNOTS = 7             # log2 n = 0 .. 6 (partials 1 .. 64; flat above)
TABLES = {  # name: (columns of the last axis: "partials", "strings" or 1; bound in nats)
    "decay_vel": ("partials", 0.7),
    "decay_lift": ("partials", 0.7),
    "damp_vel": ("partials", 0.7),
    "restrike_vel": (1, 0.7),
    "spec_vel": ("partials", 0.7),
    "even_vel": ("strings", 0.5),
}


def hat_weights(x, n_knots, lo, hi):
    """``[..., n_knots]`` piecewise-linear weights of ``x`` over knots evenly from ``lo`` to ``hi`` (flat outside)."""
    pos = ((x.float() - lo) / (hi - lo)).clamp(0, 1) * (n_knots - 1)
    i0 = pos.floor().clamp(max=n_knots - 2)
    w1 = pos - i0
    k = torch.arange(n_knots, device=x.device, dtype=pos.dtype)
    return (1 - w1)[..., None] * (k == i0[..., None]) + w1[..., None] * (k == (i0 + 1)[..., None])


class InteractionTables(nn.Module):
    def __init__(self):
        super().__init__()
        for name, (cols, _) in TABLES.items():
            c = PARTIAL_KNOTS if cols == "partials" else 3 if cols == "strings" else 1
            setattr(self, "raw_" + name, nn.Parameter(torch.zeros(KEY_KNOTS, FACTOR_KNOTS, c)))

    def table(self, name):
        """The bounded table ``[key knots, factor knots, columns]``, zero mean over the factor (and the strings)."""
        cols, bound = TABLES[name]
        t = bounded(getattr(self, "raw_" + name), bound)
        t = t - t.mean(1, keepdim=True)
        if cols == "strings":
            t = t - t.mean(2, keepdim=True)
        return t

    def forward(self, name, ki, x, n=None):
        """The table's value for keys ``ki[B, K]`` at factor ``x[B, K]`` (in [0, 1]): ``[B, K, P]`` for a table per
        partial (``n[P]`` the partial numbers), ``[B, K, 3]`` per string, ``[B, K]`` otherwise."""
        cols, _ = TABLES[name]
        t = self.table(name)
        wk = hat_weights(ki, KEY_KNOTS, 0, 87)  # [B, K, KK]
        wx = hat_weights(x, FACTOR_KNOTS, 0.0, 1.0)  # [B, K, FK]
        v = torch.einsum("bki,bkj,ijc->bkc", wk, wx, t)
        if cols == "partials":
            wn = hat_weights(torch.log2(n.float()), PARTIAL_KNOTS, 0.0, PARTIAL_KNOTS - 1.0)  # [P, NK]
            return v @ wn.T
        return v if cols == "strings" else v[..., 0]

    def regularizer(self):
        """Smoothness over the key knots (second differences), as the per-key curves elsewhere."""
        reg = 0.0
        for name in TABLES:
            t = self.table(name)
            reg = reg + ((t[2:] - 2 * t[1:-1] + t[:-2]) ** 2).mean()
        return reg
