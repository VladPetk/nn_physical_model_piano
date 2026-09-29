"""MAESTRO preparation, segment sampling and batching."""

import json
import os
from math import gcd

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import PianoConfig, year_to_condition

PEDAL_CCS = {"sustain": 64, "sostenuto": 66, "soft": 67}


def load_midi(path):
    """Return ``notes[N, 4]`` = (pitch, onset, offset, velocity) and raw pedal events."""
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(str(path))
    notes, ccs = [], {cc: [] for cc in PEDAL_CCS.values()}
    for inst in pm.instruments:
        notes += [(n.pitch, n.start, n.end, n.velocity) for n in inst.notes]
        for c in inst.control_changes:
            if c.number in ccs:
                ccs[c.number].append((c.time, c.value))
    notes = np.array(sorted(notes, key=lambda n: n[1]), dtype=np.float64).reshape(-1, 4)
    pedals = {}
    for name, cc in PEDAL_CCS.items():
        ev = np.array(sorted(ccs[cc]), dtype=np.float64).reshape(-1, 2)
        pedals[name + "_t"], pedals[name + "_v"] = ev[:, 0], ev[:, 1]
    return notes, pedals


def pedal_frames(times, values, t0, n_frames, hop, sr):
    """Step-hold pedal curve in [0, 1] at frame times ``t0 + f * hop / sr``."""
    tf = t0 + np.arange(n_frames) * hop / sr
    idx = np.searchsorted(times, tf, side="right") - 1
    return np.where(idx >= 0, values[np.clip(idx, 0, None)] if len(values) else 0.0, 0.0) / 127.0


def prepare_piece(maestro_dir, row, out_dir, sr):
    """Resample one MAESTRO recording to mono ``sr`` FLAC and cache its MIDI as npz."""
    import soundfile as sf
    from scipy.signal import resample_poly

    stem = os.path.splitext(os.path.basename(row["midi_filename"]))[0]
    audio_rel, midi_rel = f"audio/{stem}.flac", f"midi/{stem}.npz"
    audio_out, midi_out = os.path.join(out_dir, audio_rel), os.path.join(out_dir, midi_rel)
    if not os.path.exists(audio_out):
        x, in_sr = sf.read(os.path.join(maestro_dir, row["audio_filename"]), dtype="float32", always_2d=True)
        x = x.mean(1)
        g = gcd(int(in_sr), sr)
        x = resample_poly(x, sr // g, int(in_sr) // g).astype(np.float32)
        sf.write(audio_out, np.clip(x, -1, 1), sr, subtype="PCM_16")
    if not os.path.exists(midi_out):
        notes, pedals = load_midi(os.path.join(maestro_dir, row["midi_filename"]))
        np.savez(midi_out, notes=notes, **pedals)
    info = sf.info(audio_out)
    return {"id": stem, "split": row["split"], "year": int(row["year"]), "duration": info.frames / info.samplerate,
            "audio": audio_rel, "midi": midi_rel}


def _perf_from_notes(notes, pedals, t0, window, lookback, n_frames, cfg):
    """Build the model's per-example performance dict for a window starting at ``t0``."""
    sel = (notes[:, 1] >= t0 - lookback) & (notes[:, 1] < t0 + window)
    n = notes[sel]
    perf = {
        "pitch": torch.as_tensor(n[:, 0], dtype=torch.long),
        "onset": torch.as_tensor(n[:, 1] - t0, dtype=torch.float32),
        "offset": torch.as_tensor(n[:, 2] - t0, dtype=torch.float32),
        "velocity": torch.as_tensor(n[:, 3], dtype=torch.float32),
    }
    for name in PEDAL_CCS:
        perf[name] = torch.as_tensor(
            pedal_frames(pedals[name + "_t"], pedals[name + "_v"], t0, n_frames, cfg.hop, cfg.sample_rate),
            dtype=torch.float32)
    return perf


class MaestroSegments(Dataset):
    """Random windows of prepared MAESTRO pieces.

    Each example covers ``warmup + segment`` seconds of audio. The model renders
    the whole window but the loss only sees the last ``segment`` seconds, so
    reverb, sympathetic resonance and notes struck before the window have time
    to build up. Notes up to ``lookback`` seconds before the window are included
    (with negative onsets); the closed-form string model renders them exactly.
    """

    def __init__(self, root, split, cfg: PianoConfig, segment=2.0, warmup=1.0, lookback=4.0, length=10000,
                 deterministic=False):
        with open(os.path.join(root, "index.json")) as f:
            self.pieces = [p for p in json.load(f) if p["split"] == split]
        if not self.pieces:
            raise ValueError(f"no pieces for split {split!r} in {root}")
        self.root, self.cfg = root, cfg
        self.segment, self.warmup, self.lookback = segment, warmup, lookback
        self.length, self.deterministic = length, deterministic
        dur = np.array([p["duration"] for p in self.pieces])
        self.weights = dur / dur.sum()
        self._midi = {}

    def __len__(self):
        return self.length

    def _notes(self, piece):
        if piece["id"] not in self._midi:
            with np.load(os.path.join(self.root, piece["midi"])) as z:
                self._midi[piece["id"]] = {k: z[k] for k in z.files}
        return self._midi[piece["id"]]

    def __getitem__(self, i):
        import soundfile as sf

        cfg, sr = self.cfg, self.cfg.sample_rate
        rng = np.random.default_rng(i if self.deterministic else None)
        piece = self.pieces[rng.choice(len(self.pieces), p=self.weights)]
        n_samples = int(round((self.warmup + self.segment) * sr))
        path = os.path.join(self.root, piece["audio"])
        total = sf.info(path).frames
        start = int(rng.integers(0, max(1, total - n_samples)))
        audio, _ = sf.read(path, start=start, frames=n_samples, dtype="float32")
        audio = np.pad(audio, (0, n_samples - len(audio)))
        z = self._notes(piece)
        t0 = start / sr
        perf = _perf_from_notes(z["notes"], z, t0, n_samples / sr, self.lookback, n_samples // cfg.hop + 2, cfg)
        perf["condition"] = torch.tensor(year_to_condition(piece["year"]))
        perf["audio"] = torch.from_numpy(audio)
        perf["loss_start"] = torch.tensor(int(self.warmup * sr))
        return perf


class SyntheticPerformances(Dataset):
    """Random note/pedal performances (no audio), e.g. for teacher-student sanity checks."""

    def __init__(self, cfg: PianoConfig, seconds=1.0, max_notes=8, length=1000, seed=0):
        self.cfg, self.seconds, self.max_notes, self.length, self.seed = cfg, seconds, max_notes, length, seed

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        cfg = self.cfg
        rng = np.random.default_rng(self.seed * 1_000_003 + i)
        n = int(rng.integers(1, self.max_notes + 1))
        on = np.sort(rng.uniform(-0.3, self.seconds * 0.8, n))
        notes = np.stack([rng.integers(21, 109, n), on, on + rng.uniform(0.05, 1.0, n), rng.integers(20, 120, n)], 1)
        pedals = {}
        for name in PEDAL_CCS:
            k = int(rng.integers(0, 3)) if name == "sustain" else 0
            pedals[name + "_t"] = np.sort(rng.uniform(0, self.seconds, k))
            pedals[name + "_v"] = rng.choice([0.0, 127.0], k)
        n_samples = int(self.seconds * cfg.sample_rate)
        perf = _perf_from_notes(notes, pedals, 0.0, self.seconds, 1.0, n_samples // cfg.hop + 2, cfg)
        perf["condition"] = torch.tensor(int(rng.integers(0, cfg.n_conditions)))
        return perf


def collate(batch):
    """Pad per-example note lists to a common length and stack everything else."""
    n_max = max(1, max(len(b["pitch"]) for b in batch))
    out = {}
    for key in ("pitch", "onset", "offset", "velocity"):
        fill = 21 if key == "pitch" else 0
        out[key] = torch.stack([torch.cat([b[key], b[key].new_full((n_max - len(b[key]),), fill)]) for b in batch])
    out["mask"] = torch.stack([torch.arange(n_max) < len(b["pitch"]) for b in batch])
    for key in batch[0]:
        if key not in out:
            out[key] = torch.stack([b[key] for b in batch])
    return out
