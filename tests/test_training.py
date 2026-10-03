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


def test_gradients_finite_with_noise():
    """A full step with the noise bank on must not produce NaN/Inf gradients.

    sqrt(power) has an infinite gradient where the band power is exactly 0 (silent
    bands/samples), which used to poison the whole backward pass with NaNs after the
    first step. The student-vs-teacher test runs with use_noise=False, so this path
    needs its own guard.
    """
    cfg = small_cfg(use_noise=True)
    teacher = perturb_physics(NeuralPhysicalPiano(cfg), scale=0.3, seed=1).eval()
    student = NeuralPhysicalPiano(cfg)
    data = SyntheticPerformances(cfg, seconds=0.5, max_notes=4, length=8)
    loss_fn = MultiResolutionSTFTLoss(fft_sizes=(1024, 256, 64))
    n = int(0.5 * cfg.sample_rate)
    batch = collate([data[i] for i in range(4)])

    with torch.no_grad():
        target = teacher(batch, n)["audio"]
    loss = loss_fn(student(batch, n)["audio"], target) + student.physics.regularizer()
    loss.backward()
    assert torch.isfinite(loss)
    bad = [name for name, p in student.named_parameters()
           if p.grad is not None and not torch.isfinite(p.grad).all()]
    assert not bad, f"non-finite gradients in: {bad}"


def test_adversarial_losses_run():
    disc = MultiResolutionDiscriminator(fft_sizes=(256, 512))
    real, fake = torch.randn(2, 4000), torch.randn(2, 4000, requires_grad=True)
    d = discriminator_loss(disc, real, fake)
    adv, fm = generator_adv_loss(disc, real, fake)
    (d + adv + fm).backward()
    assert fake.grad is not None and torch.isfinite(fake.grad).all()


def test_wide_critic_sees_the_whole_spectrum():
    """Each output of the wide critic depends on the lowest and the highest bin of its frame (the patch critic's ~11
    bins could not compare registers: runs/gan_check/check1), and the recordings' spectra computed once give the same
    losses as computed inside."""
    from pianonn.losses import WideMultiResolutionDiscriminator

    torch.manual_seed(0)
    disc = WideMultiResolutionDiscriminator(fft_sizes=(256, 512), amp=False)
    real, fake = torch.randn(2, 4000), torch.randn(2, 4000, requires_grad=True)
    spec = disc.spectra(real)
    d = discriminator_loss(disc, real, fake, spec)
    adv, fm = generator_adv_loss(disc, real, fake, spec)
    assert torch.allclose(d, discriminator_loss(disc, real, fake))
    assert torch.allclose(adv, generator_adv_loss(disc, real, fake)[0])
    (d + adv + fm).backward()
    assert fake.grad is not None and torch.isfinite(fake.grad).all()
    for sub in disc.discs:
        s = sub.spectrum(real[:1]).detach().requires_grad_(True)
        out, _ = sub(spec=s)
        assert out.shape[2] == 1 and out.shape[3] == s.shape[3]
        t = out.shape[3] // 2
        g, = torch.autograd.grad(out[0, 0, 0, t], s)
        col = g[0, 0, :, t].abs()
        assert col[0] > 0 and col[-1] > 0


def test_critic_step_with_r1_and_mono():
    """The critic's step with R1 runs and penalises (a positive penalty on the steps it is computed, none between); the
    mono critic judges a stereo pair and its left-right swap alike."""
    from pianonn.losses import WideMultiResolutionDiscriminator, critic_step

    torch.manual_seed(0)
    disc = WideMultiResolutionDiscriminator(fft_sizes=(256,), amp=False, mono=True)
    opt = torch.optim.Adam(disc.parameters(), lr=1e-3)
    real, fake = torch.randn(2, 2, 3000), torch.randn(2, 2, 3000)
    _, pen0 = critic_step(disc, opt, real, fake, disc.spectra(real), r1=1.0, r1_every=2, step=0)
    _, pen1 = critic_step(disc, opt, real, fake, disc.spectra(real), r1=1.0, r1_every=2, step=1)
    assert pen0 > 0 and pen1 == 0
    with torch.no_grad():
        a, b = disc(real)[0][0], disc(real.flip(1))[0][0]
    assert torch.allclose(a, b, atol=1e-6)


def test_loss_is_not_dominated_by_the_noise_floor():
    """A recording has a noise floor and the model's piano renders silence between notes. With the loss floor, the
    same piano must score far better than a different one against a -60 dBFS floor (review 3, section 4.1)."""
    from pianonn.diagnostics import _perf
    from pianonn.render import load_model

    torch.manual_seed(0)
    m = load_model(**small_cfg(sample_rate=16000, n_partials=32, use_sympathetic=False).to_dict())
    n = 3 * m.cfg.sample_rate
    notes = [(60, 0.1, 1.5, 80), (64, 0.6, 2.0, 70), (67, 1.1, 2.8, 90), (48, 0.2, 2.9, 85)]
    with torch.no_grad():
        clean = m(_perf(m, n, notes), n, generator=torch.Generator().manual_seed(0), residual=False)["audio"]
        other = m(_perf(m, n, [(p, a, b, v - 10) for p, a, b, v in notes]), n, generator=torch.Generator().manual_seed(1),
                  residual=False)["audio"]
    peak = clean.abs().max()
    loss = MultiResolutionSTFTLoss(fft_sizes=(2048, 512, 128))
    floor = lambda x, seed: x / peak * 0.5 + torch.randn(x.shape, generator=torch.Generator().manual_seed(seed)) * 1e-3
    target = floor(clean, 5)
    same, diff = loss(floor(clean, 6), target).item(), loss(floor(other, 6), target).item()
    assert same < 0.5 * diff, (same, diff)


def test_gradients_point_back_to_the_teacher():
    """Perturb one physical parameter of a teacher, render through the room with a noise floor, and check that the
    loss gradient on the student's copy points towards the teacher (review 3, section 5, test 1)."""
    from pianonn.diagnostics import _perf

    cfg = small_cfg(sample_rate=16000, n_partials=24, use_noise=False, use_sympathetic=False, use_floor=True)
    notes = [(48, 0.05, 1.2, 90), (55, 0.1, 1.2, 80), (60, 0.15, 1.2, 85), (64, 0.2, 1.2, 75)]
    for name, delta in (("raw_log_B", 0.1), ("raw_log_b1", 0.4), ("raw_log_tc", -0.22)):
        torch.manual_seed(0)
        student = NeuralPhysicalPiano(cfg)
        teacher = NeuralPhysicalPiano(cfg)
        teacher.load_state_dict(student.state_dict())
        with torch.no_grad():
            student.room.floor_ref_db.fill_(-100.0)
            teacher.room.floor_ref_db.fill_(-100.0)
            getattr(teacher.physics, name).add_(delta)
            n = int(1.3 * cfg.sample_rate)
            target = teacher(_perf(teacher, n, notes), n, residual=False, generator=torch.Generator().manual_seed(1))["audio"]
        pred = student(_perf(student, n, notes), n, residual=False, generator=torch.Generator().manual_seed(2))["audio"]
        loss = MultiResolutionSTFTLoss(fft_sizes=(4096, 1024, 256))(pred, target)
        loss.backward()
        g = getattr(student.physics, name).grad
        keys = torch.tensor([p - 21 for p, *_ in notes])
        assert (g[keys].sum() * delta) < 0, (name, g[keys])


def test_stages_freeze_the_residual_first():
    from pianonn.train import set_stage

    m = NeuralPhysicalPiano(small_cfg())
    opt = torch.optim.Adam(param_groups(m, 1e-3))
    set_stage(m, opt, 1)
    frozen = {n for n, p in m.named_parameters() if not p.requires_grad}
    assert {"physics.partial_gain", "physics.color", "noise.att", "context.head.2.weight"} <= frozen
    assert "physics.raw_log_B" not in frozen and "room.raw_floor" not in frozen
    set_stage(m, opt, 2, physics_lr=0.3)
    assert all(p.requires_grad for p in m.parameters())
    lrs = {g["name"]: g["lr"] / g["base_lr"] for g in opt.param_groups}
    assert abs(lrs["physics.raw_log_B"] - 0.3) < 1e-9 and lrs["context.head.2.weight"] == 1.0


def test_budget_share_is_not_degenerate_where_the_physics_is_silent():
    """Review 4, section 5: measured against the physics alone, a residual noise 60 dB under the floor counted as a
    100 % share wherever the physics was silent. Measured against the output (floor included), it must count ~0."""
    from pianonn.train import residual_budget

    m = NeuralPhysicalPiano(small_cfg(sample_rate=16000))
    g = torch.Generator().manual_seed(0)
    floor = 1e-3 * torch.randn(1, 2, 16000, generator=g)
    res = 1e-6 * torch.randn(1, 2, 16000, generator=g)
    out = {"noise_res_out": res, "audio": floor + res}
    assert float(residual_budget(m, out, torch.ones(1, 1, dtype=torch.bool))["additive"]) < 1e-4
    out = {"noise_res_out": floor, "audio": floor + 1e-6 * res}  # the residual is all there is
    assert float(residual_budget(m, out, torch.ones(1, 1, dtype=torch.bool))["additive"]) > 0.4


def test_texture_view_keeps_the_critic_away_from_the_physics():
    """The GAN sees ``audio_texture``: numerically the output, but only the noise bank and the residual are
    differentiable in it, so the critic cannot bend inharmonicity, decays, damper timing, the body or the floor."""
    from pianonn.diagnostics import _perf

    cfg = small_cfg(sample_rate=16000, use_sympathetic=False, use_floor=True)
    m = NeuralPhysicalPiano(cfg)
    n = 16000
    perf = _perf(m, n, [(60, 0.1, 0.6, 80), (48, 0.3, 0.9, 90)])
    out = m(perf, n, residual=True, extras=("texture_view", "residual_out"), generator=torch.Generator().manual_seed(0))
    assert torch.allclose(out["audio_texture"], out["audio"], atol=1e-6)
    assert out["noise_res_out"].shape == out["audio"].shape
    out["audio_texture"].pow(2).mean().backward()
    grads = {k: p.grad for k, p in m.named_parameters()}
    for name, g in grads.items():
        if not name.startswith(("noise.", "context.")):
            assert g is None or g.abs().sum() == 0, name
    for name in ("noise.knock", "noise.release", "context.frame_head.2.weight"):
        assert grads[name] is not None and grads[name].abs().sum() > 0, name


def test_piano_loss_gradients_point_back_to_the_teacher():
    """Round-2 loss: a perturbed decay, contact time, damper delay or velocity curve in the teacher must pull the
    student's parameter the right way (review 3, section 5, test 1; review 4, 6: the re-strike and delays)."""
    from pianonn.diagnostics import _perf
    from pianonn.losses import PianoLoss

    cfg = small_cfg(sample_rate=16000, n_partials=24, use_noise=False, use_sympathetic=False, use_floor=False)
    notes = [(48, 0.05, 0.5, 90), (55, 0.1, 0.7, 80), (60, 0.15, 0.4, 85), (64, 0.2, 0.9, 75), (60, 0.8, 1.1, 70)]
    cases = (("raw_log_b1", 0.4, slice(None)), ("raw_log_tc", -0.22, slice(None)), ("cond_damper_delay", 0.5, 9),
             ("cond_vel_curve", 0.6, 9))
    for name, delta, idx in cases:
        torch.manual_seed(0)
        student = NeuralPhysicalPiano(cfg)
        teacher = NeuralPhysicalPiano(cfg)
        teacher.load_state_dict(student.state_dict())
        with torch.no_grad():
            getattr(teacher.physics, name)[idx] += delta
            n = int(1.3 * cfg.sample_rate)
            target = teacher(_perf(teacher, n, notes), n, residual=False)["audio"]
        onsets = torch.tensor([[o for _, o, _, _ in notes]])
        pred = student(_perf(student, n, notes), n, residual=False)["audio"]
        loss, _ = PianoLoss(cfg.sample_rate)(pred, target, onsets, torch.ones_like(onsets, dtype=torch.bool))
        loss.backward()
        g = getattr(student.physics, name).grad[idx]
        if name == "raw_log_b1" or name == "raw_log_tc":
            g = g[torch.tensor([p - 21 for p, *_ in notes])]
        assert (g.sum() * delta) < 0, (name, g)
