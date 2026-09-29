import torch
from torch.utils.data import DataLoader

from pianonn import NeuralPhysicalPiano
from pianonn.data import SyntheticPerformances, collate
from pianonn.losses import MultiResolutionDiscriminator, MultiResolutionSTFTLoss, discriminator_loss, generator_adv_loss
from pianonn.train import param_groups, perturb_physics

from .conftest import small_cfg


def test_student_moves_towards_teacher():
    cfg = small_cfg(use_noise=False)  # noise is stochastic; keep the check deterministic
    teacher = perturb_physics(NeuralPhysicalPiano(cfg), scale=0.3, seed=1).eval()
    student = NeuralPhysicalPiano(cfg)
    data = SyntheticPerformances(cfg, seconds=0.5, max_notes=4, length=160)
    loader = DataLoader(data, batch_size=2, collate_fn=collate)
    loss_fn = MultiResolutionSTFTLoss(fft_sizes=(1024, 256, 64))
    # 3e-3 (training uses 1e-3): at 1e-2 the fast prompt decays and wide hammer-order bounds make 32 steps diverge
    opt = torch.optim.Adam(param_groups(student, 3e-3))
    n = int(0.5 * cfg.sample_rate)
    fixed = collate([data[i] for i in range(100, 104)])

    def eval_loss():
        with torch.no_grad():
            return loss_fn(student(fixed, n)["audio"], teacher(fixed, n)["audio"]).item()

    before = eval_loss()
    for batch in loader:
        with torch.no_grad():
            target = teacher(batch, n)["audio"]
        loss = loss_fn(student(batch, n)["audio"], target)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert eval_loss() < 0.9 * before


def test_adversarial_losses_run():
    disc = MultiResolutionDiscriminator(fft_sizes=(256, 512))
    real, fake = torch.randn(2, 4000), torch.randn(2, 4000, requires_grad=True)
    d = discriminator_loss(disc, real, fake)
    adv, fm = generator_adv_loss(disc, real, fake)
    (d + adv + fm).backward()
    assert fake.grad is not None and torch.isfinite(fake.grad).all()
