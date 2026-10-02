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

    pm = pretty_midi.PrettyMIDI(path if hasattr(path, "read") else str(path))
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


def _open_member(source, rel):
    """A MAESTRO file as a readable binary object: from the extracted directory or straight from the zip."""
    import io
    import zipfile

    if os.path.isdir(source):
        return open(os.path.join(source, rel), "rb")
    with zipfile.ZipFile(source) as z:
        name = next(n for n in z.namelist() if n.endswith("/" + rel) or n == rel)
        return io.BytesIO(z.read(name))


def prepare_piece(source, row, out_dir, sr, channels=2):
    """Resample one MAESTRO recording to ``sr`` FLAC and cache its MIDI as npz.

    ``source`` is the extracted MAESTRO directory or the zip itself. Both channels are kept
    (``channels=2``): summing a spaced pair comb-filters the spectrum in a key-dependent way,
    which per-key tables would otherwise learn as if it were the piano (review 3, F2).
    """
    import soundfile as sf
    from scipy.signal import resample_poly

    stem = os.path.splitext(os.path.basename(row["midi_filename"]))[0]
    audio_rel, midi_rel = f"audio/{stem}.flac", f"midi/{stem}.npz"
    audio_out, midi_out = os.path.join(out_dir, audio_rel), os.path.join(out_dir, midi_rel)
    if not os.path.exists(audio_out):
        with _open_member(source, row["audio_filename"]) as f:
            x, in_sr = sf.read(f, dtype="float32", always_2d=True)
        x = x.mean(1, keepdims=True) if channels == 1 else np.repeat(x, 2, 1)[:, :2] if x.shape[1] == 1 else x[:, :2]
        g = gcd(int(in_sr), sr)
        x = resample_poly(x, sr // g, int(in_sr) // g, axis=0).astype(np.float32)
        sf.write(audio_out + ".part.flac", np.clip(x, -1, 1), sr, subtype="PCM_16")
        os.replace(audio_out + ".part.flac", audio_out)  # a killed run never leaves a truncated file behind
    if not os.path.exists(midi_out):
        with _open_member(source, row["midi_filename"]) as f:
            notes, pedals = load_midi(f)
        np.savez(midi_out, notes=notes, **pedals)
    info = sf.info(audio_out)
    return {"id": stem, "split": row["split"], "year": int(row["year"]), "duration": info.frames / info.samplerate,
            "channels": info.channels, "audio": audio_rel, "midi": midi_rel}


def _perf_from_notes(notes, pedals, t0, window, lookback, n_frames, cfg, hist_frames=0):
    """Build the model's per-example performance dict for a window starting at ``t0``.

    Notes struck up to ``lookback`` s before the window are included (negative onsets). The
    control curves start ``hist_frames`` frames before the window, so the dampers, sostenuto
    latches and re-strikes of those notes are exact at the window start (review 3, F8);
    ``n_frames`` is the number of frames of the window itself.
    """
    sel = (notes[:, 1] >= t0 - lookback) & (notes[:, 1] < t0 + window)
    n = notes[sel]
    perf = {
        "pitch": torch.as_tensor(n[:, 0], dtype=torch.long),
        "onset": torch.as_tensor(n[:, 1] - t0, dtype=torch.float32),
        "offset": torch.as_tensor(n[:, 2] - t0, dtype=torch.float32),
        "velocity": torch.as_tensor(n[:, 3], dtype=torch.float32),
        "hist_frames": torch.tensor(hist_frames),
    }
    t_start = t0 - hist_frames * cfg.hop / cfg.sample_rate
    for name in PEDAL_CCS:
        perf[name] = torch.as_tensor(
            pedal_frames(pedals[name + "_t"], pedals[name + "_v"], t_start, hist_frames + n_frames, cfg.hop, cfg.sample_rate),
            dtype=torch.float32)
    return perf


def history_frames(lookback, cfg):
    return int(round(lookback * cfg.sample_rate / cfg.hop))


class MaestroSegments(Dataset):
    """Random windows of prepared MAESTRO pieces.

    Each example covers ``warmup + segment`` seconds of audio. The model renders
    the whole window but the loss only sees the last ``segment`` seconds, so
    reverb, sympathetic resonance and notes struck before the window have time
    to build up. Notes up to ``lookback`` seconds before the window are included
    (with negative onsets); the closed-form string model renders them exactly, and the
    control curves cover the lookback too. 12 s lets the loss see bass aftersound.

    ``audio`` is ``[channels, samples]``.
    """

    def __init__(self, root, split, cfg: PianoConfig, segment=2.0, warmup=1.0, lookback=12.0, length=10000,
                 deterministic=False, years=None, seed=0):
        with open(os.path.join(root, "index.json")) as f:
            self.pieces = [p for p in json.load(f) if p["split"] == split and (not years or p["year"] in years)]
        if not self.pieces:
            raise ValueError(f"no pieces for split {split!r} in {root}")
        self.root, self.cfg = root, cfg
        self.segment, self.warmup, self.lookback = segment, warmup, lookback
        self.length, self.deterministic, self.seed = length, deterministic, seed
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

    def silence_clips(self, year=None, min_seconds=0.3, margin=0.1, max_seconds=2.0):
        """The recorded silence before each piece's first note (MAESTRO has about a second of it): ``[ch, n]``
        float tensors, from ``margin`` s after the start to ``margin`` s before the first onset."""
        import soundfile as sf

        sr, clips = self.cfg.sample_rate, []
        for piece in self.pieces:
            if year is not None and piece["year"] != year:
                continue
            first = float(self._notes(piece)["notes"][:, 1].min())
            a, b = margin, min(first - margin, margin + max_seconds)
            if b - a < min_seconds:
                continue
            x, _ = sf.read(os.path.join(self.root, piece["audio"]), start=int(a * sr), frames=int((b - a) * sr),
                           dtype="float32", always_2d=True)
            x = x.T if x.shape[1] == self.cfg.channels else x.mean(1, keepdims=True).T.repeat(self.cfg.channels, 0)
            clips.append(torch.from_numpy(np.ascontiguousarray(x)))
        return clips

    def __getitem__(self, i):
        import soundfile as sf

        sr = self.cfg.sample_rate
        rng = np.random.default_rng(self.seed * 1_000_003 + i if self.deterministic else None)
        piece = self.pieces[rng.choice(len(self.pieces), p=self.weights)]
        n_samples = int(round((self.warmup + self.segment) * sr))
        total = sf.info(os.path.join(self.root, piece["audio"])).frames
        return self._item(piece, int(rng.integers(0, max(1, total - n_samples))))

    def item_at(self, piece_id, start_s):
        """The example whose window (warm-up included) starts at ``start_s`` seconds into piece ``piece_id``."""
        piece = next(p for p in self.pieces if p["id"] == piece_id)
        return self._item(piece, int(round(start_s * self.cfg.sample_rate)))

    def _item(self, piece, start):
        import soundfile as sf

        cfg, sr = self.cfg, self.cfg.sample_rate
        n_samples = int(round((self.warmup + self.segment) * sr))
        path = os.path.join(self.root, piece["audio"])
        audio, _ = sf.read(path, start=start, frames=n_samples, dtype="float32", always_2d=True)
        audio = np.pad(audio, ((0, n_samples - len(audio)), (0, 0))).T
        if audio.shape[0] != cfg.channels:
            audio = audio.mean(0, keepdims=True).repeat(cfg.channels, 0)
        z = self._notes(piece)
        t0 = start / sr
        perf = _perf_from_notes(z["notes"], z, t0, n_samples / sr, self.lookback, n_samples // cfg.hop + 2, cfg,
                                history_frames(self.lookback, cfg))
        perf["condition"] = torch.tensor(year_to_condition(piece["year"]))
        perf["audio"] = torch.from_numpy(np.ascontiguousarray(audio))
        perf["piece"] = torch.tensor(self.pieces.index(piece))
        perf["start"] = torch.tensor(start)
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
