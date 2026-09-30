import torch

from pianonn import NeuralPhysicalPiano
from pianonn.data import collate
from pianonn.fit_init import floor_from_silence, initialise_from_data

from .conftest import make_perf, small_cfg


def test_floor_and_hum_are_recovered_from_silence():
    cfg = small_cfg(sample_rate=16000, use_floor=True)
    truth = NeuralPhysicalPiano(cfg)
    with torch.no_grad():
        c = torch.linspace(0, 1, cfg.noise_bands)
        truth.room.floor_ref_db[3] = (-50 - 30 * c)[None]  # falling floor
        truth.room.hum_ref_db[3] = torch.tensor([-55.0, -48.0, -70.0])[None]
    g = torch.Generator().manual_seed(0)
    clips = [truth.room.floor_noise(torch.tensor([3]), 16000, g)[0] for _ in range(12)]
    est = NeuralPhysicalPiano(cfg)
    floor_from_silence(est, 3, clips, log=lambda *a: None)
    err = (est.room.floor_ref_db[3] - truth.room.floor_ref_db[3]).abs()
    assert float(err[:, 1:].max()) < 1.5, err  # band 0 straddles the 20 Hz high-pass
    assert float((est.room.hum_ref_db[3, :, :2] - truth.room.hum_ref_db[3, :, :2]).abs().max()) < 1.5
    assert float(est.room.raw_floor.abs().max()) == 0.0


def test_initialisation_finds_latency_sign_and_level():
    """The recordings are the model itself, 4 ms later and 6 dB louder: the init must delay the model's body by
    ~4 ms (not advance it) and raise the mic gain by ~6 dB."""
    cfg = small_cfg(sample_rate=16000, use_noise=False, use_sympathetic=False)
    torch.manual_seed(0)
    model = NeuralPhysicalPiano(cfg)
    teacher = NeuralPhysicalPiano(cfg)
    teacher.load_state_dict(model.state_dict())
    shift = 64  # 4 ms
    with torch.no_grad():
        body = teacher.room.body.data[3]
        body.copy_(torch.cat([body.new_zeros(*body.shape[:-1], shift), body[..., :-shift]], -1))
        teacher.room.mic_gain_db.data[3] += 6.0
    n = 2 * cfg.sample_rate
    songs = [[(60, 0.1, 0.8, 80), (64, 0.5, 1.2, 70), (48, 1.0, 1.8, 90)],
             [(55, 0.2, 1.0, 85), (67, 0.7, 1.5, 75), (72, 1.2, 1.9, 60)]]
    items = []
    for notes in songs:
        perf = make_perf(model, n, notes)
        with torch.no_grad():
            perf["audio"] = teacher(perf, n, residual=False)["audio"]
        perf["loss_start"] = torch.tensor(0)
        items.append({k: v[0] if torch.is_tensor(v) and v.dim() > 0 and k != "loss_start" else v for k, v in perf.items()})
    batch = collate(items)
    est = initialise_from_data(model, [batch], log=lambda *a: None, tuning=False)
    assert 2.5 < est["latency_ms"] < 5.5, est["latency_ms"]
    assert all(abs(v - 6.0) < 1.5 for v in est["level_db"]), est["level_db"]
