"""Spectral reconstruction loss and an optional multi-resolution spectrogram discriminator."""

import torch
import torch.nn.functional as F
from torch import nn


def _mag(x, n_fft):
    window = torch.hann_window(n_fft, device=x.device)
    return torch.stft(x, n_fft, n_fft // 4, window=window, return_complex=True).abs()


class MultiResolutionSTFTLoss(nn.Module):
    """Spectral convergence + log-magnitude L1 over several resolutions.

    The 4096-point resolution matters for pianos: bass partials are ~30 Hz
    apart and unison beating lives in sub-Hz detail.
    """

    def __init__(self, fft_sizes=(4096, 2048, 1024, 512, 256, 128), eps=1e-5):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.eps = eps

    def forward(self, pred, target):
        loss = 0.0
        for n in self.fft_sizes:
            S, P = _mag(target, n), _mag(pred, n)
            sc = torch.linalg.norm(S - P, dim=(-2, -1)) / (torch.linalg.norm(S, dim=(-2, -1)) + self.eps)
            lm = (torch.log(S + self.eps) - torch.log(P + self.eps)).abs().mean((-2, -1))
            loss = loss + (sc + lm).mean()
        return loss / len(self.fft_sizes)


class SpecDiscriminator(nn.Module):
    def __init__(self, n_fft, ch=32):
        super().__init__()
        self.n_fft = n_fft
        self.convs = nn.ModuleList([
            nn.Conv2d(1, ch, (3, 9), padding=(1, 4)),
            nn.Conv2d(ch, ch, (3, 9), stride=(1, 2), padding=(1, 4)),
            nn.Conv2d(ch, ch, (3, 9), stride=(1, 2), padding=(1, 4)),
            nn.Conv2d(ch, ch, (3, 3), padding=(1, 1)),
        ])
        self.out = nn.Conv2d(ch, 1, (3, 3), padding=(1, 1))

    def forward(self, x):
        h = torch.log(_mag(x, self.n_fft) + 1e-5)[:, None]  # [B,1,F,T]
        feats = []
        for conv in self.convs:
            h = F.leaky_relu(conv(h), 0.1)
            feats.append(h)
        return self.out(h), feats


class MultiResolutionDiscriminator(nn.Module):
    """Adversarial critic on log spectrograms. Spectral losses alone average away
    the attack transients and noise texture that make a piano sound real."""

    def __init__(self, fft_sizes=(512, 1024, 2048)):
        super().__init__()
        self.discs = nn.ModuleList([SpecDiscriminator(n) for n in fft_sizes])

    def forward(self, x):
        return [d(x) for d in self.discs]


def discriminator_loss(disc, real, fake):
    loss = 0.0
    for (lr, _), (lf, _) in zip(disc(real), disc(fake.detach())):
        loss = loss + ((lr - 1) ** 2).mean() + (lf**2).mean()
    return loss


def generator_adv_loss(disc, real, fake):
    adv, fm = 0.0, 0.0
    with torch.no_grad():
        real_out = disc(real)
    for (lf, ff), (_, fr) in zip(disc(fake), real_out):
        adv = adv + ((lf - 1) ** 2).mean()
        fm = fm + sum((a - b).abs().mean() for a, b in zip(ff, fr)) / len(ff)
    return adv, fm
