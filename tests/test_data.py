import csv
import json

import numpy as np
import pretty_midi
import soundfile as sf
import torch

from pianonn.data import MaestroSegments, collate, load_midi, pedal_frames, prepare_piece
from pianonn.render import load_model, render_notes
from pianonn.synth import NeuralPhysicalPiano

from .conftest import small_cfg


def test_pedal_frames_step_hold():
    v = pedal_frames(np.array([0.5, 1.0]), np.array([127.0, 64.0]), 0.0, 6, hop=2, sr=8)
    assert np.allclose(v, [0, 0, 1, 1, 64 / 127, 64 / 127])


def _fake_maestro(tmp_path):
    root = tmp_path / "maestro"
    (root / "2018").mkdir(parents=True)
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0)
    for i, p in enumerate([60, 64, 67, 72]):
        inst.notes.append(pretty_midi.Note(velocity=80, pitch=p, start=0.5 * i, end=0.5 * i + 0.4))
    inst.control_changes.append(pretty_midi.ControlChange(64, 127, 1.0))
    inst.control_changes.append(pretty_midi.ControlChange(64, 0, 2.0))
    pm.instruments.append(inst)
    pm.write(str(root / "2018" / "piece.midi"))
    sf.write(root / "2018" / "piece.wav", np.random.randn(44100 * 3, 2).astype(np.float32) * 0.1, 44100)
    row = {"split": "train", "year": "2018", "midi_filename": "2018/piece.midi", "audio_filename": "2018/piece.wav",
           "duration": "3.0"}
    with open(root / "maestro-v3.0.0.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        w.writeheader()
        w.writerow(row)
    return root, row


def test_prepare_and_sample_segments(tmp_path):
    root, row = _fake_maestro(tmp_path)
    out = tmp_path / "prepared"
    (out / "audio").mkdir(parents=True)
    (out / "midi").mkdir()
    cfg = small_cfg()
    info = prepare_piece(str(root), row, str(out), cfg.sample_rate)
    assert abs(info["duration"] - 3.0) < 1e-3
    with open(out / "index.json", "w") as f:
        json.dump([info], f)

    notes, pedals = load_midi(root / "2018" / "piece.midi")
    assert notes.shape == (4, 4) and list(pedals["sustain_v"]) == [127, 0]

    ds = MaestroSegments(str(out), "train", cfg, segment=1.0, warmup=0.5, lookback=1.0, length=4, deterministic=True)
    batch = collate([ds[i] for i in range(4)])
    n = batch["audio"].shape[-1]
    assert n == int(1.5 * cfg.sample_rate) and batch["audio"].shape[1] == 2  # both channels kept
    H = int(batch["hist_frames"][0])
    assert H == int(1.0 * cfg.sample_rate / cfg.hop)
    assert batch["sustain"].shape[-1] == H + n // cfg.hop + 2
    assert batch["condition"].tolist() == [9] * 4
    model = NeuralPhysicalPiano(cfg)
    assert model(batch, n)["audio"].shape == batch["audio"].shape


def test_render_notes_untrained():
    model = load_model(**small_cfg().to_dict())
    notes = np.array([[60, 0.0, 0.5, 80], [67, 0.25, 0.75, 70]], dtype=np.float64)
    pedals = {f"{k}_{s}": np.zeros(0) for k in ("sustain", "sostenuto", "soft") for s in "tv"}
    audio = render_notes(model, notes, pedals, tail=0.5)
    assert audio.shape == (2, int(1.25 * model.cfg.sample_rate)) and np.isfinite(audio).all()
    assert torch.tensor(audio).abs().max() > 1e-3


def test_prepare_from_zip(tmp_path):
    """The MAESTRO archive can be read without unpacking it."""
    import zipfile

    root, row = _fake_maestro(tmp_path)
    zpath = tmp_path / "maestro-v3.0.0.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        for f in root.rglob("*"):
            if f.is_file():
                z.write(f, "maestro-v3.0.0/" + f.relative_to(root).as_posix())
    out = tmp_path / "prepared"
    (out / "audio").mkdir(parents=True)
    (out / "midi").mkdir()
    info = prepare_piece(str(zpath), row, str(out), 8000)
    assert info["channels"] == 2 and abs(info["duration"] - 3.0) < 1e-3
