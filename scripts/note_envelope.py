"""N13: the whole note's envelope, partial by partial, recording against models (docs/tone_measures.md 17).

    python scripts/note_envelope.py data/maestro24k --out runs/phase6/envelope --validate \\
        --model phase5=runs/phase5/train_run/train/last.pt:physics

Events (``measures.envelope_events``): notes of MIDI 36-88 in any texture, every piece of ``--year`` (any split: this
measures the sound, not a fit), each read while it sounds freely (key or pedal holding its damper off, no re-strike;
at most 3 s), at most ``--cap`` per (register, velocity, length < 1 s; 1.5 and 3 times that for 1-2 and 2+ s). Each is rendered by every model
in its context. ``measures.note_envelope`` reads each partial (up to 40, below 6 kHz) at 0.1-2.5 s after the onset,
beside its local floor, where no other sounding note has a partial near it (2 s of earlier notes).

The fade of a partial at time t: its level at t minus at 0.1 s, each level held at or above its local floor (a
partial lost in the floor reads as the floor, so a side whose partial dies early is not dropped). Cells: model fade
minus recording fade, median over the partials of the events where both read clear, the partial stands 6 dB over
its floor at 0.1 s on both sides and at t on at least one; in brackets the partials counted. ``sounding``: the share
of those partials still 6 dB over their floor at t (recording / model). ``--validate``: the first model again with
the prompt decay of partials 13 and up doubled, and with partial 2's aftersound 6 dB louder: what the cells read.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from pianonn import measures as M  # noqa: E402
from pianonn.config import year_to_condition  # noqa: E402
from pianonn.envfit import LENGTHS, REGS, T_MAX, mine  # noqa: E402,F401
from pianonn.render import load_variant  # noqa: E402

PRE, POST = 0.3, 3.1
GROUPS = (("1", 0, 1), ("2", 1, 2), ("3-4", 2, 4), ("5-8", 4, 8), ("9-12", 8, 12), ("13-20", 12, 20), ("21-40", 20, 40))
BANDS = (("<500 Hz", 0, 500), ("0.5-1 kHz", 500, 1000), ("1-2 kHz", 1000, 2000), ("2-4 kHz", 2000, 4000),
         ("4-6 kHz", 4000, 6000))
SNR = 6.0
SHOW = (0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5)


def validation_variants(spec, dev):
    """The first model with known changes: (label, model, residual)."""
    out = []
    for label, change in (("x2 decay 13+", "decay"), ("p2 after +6 dB", "after")):
        _, model, residual = load_variant(spec, device=dev)
        orig = model.physics.modes

        def modes(ki, *a, _orig=orig, _change=change, **k):
            r = _orig(ki, *a, **k)
            if _change == "decay":
                al = r["alpha"].clone()
                al[..., 12:, 0] = al[..., 12:, 0] * 2.0
                r["alpha"] = al
            else:
                amp = r["amp"].clone()
                amp[..., 1, 1:] = amp[..., 1, 1:] * 2.0
                r["amp"] = amp
            return r

        model.physics.modes = modes
        out.append((label, model, residual))
    return out


def med(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return (float(np.median(v)), len(v)) if len(v) else (float("nan"), 0)


def cell(v):
    m, n = med(v)
    return "" if not n else f"{m:+.1f} ({n})"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data")
    ap.add_argument("--model", action="append", required=True,
                    help="label=checkpoint:physics[:options] (pianonn.render.load_variant); the first gives the partial table")
    ap.add_argument("--year", type=int, default=2018)
    ap.add_argument("--cap", type=int, default=20, help="per cell for notes under 1 s; x1.5 for 1-2 s, x3 for 2+ s")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.6)

    events, avail = mine(args.data, args.year, args.cap, args.seed)
    models = [load_variant(m, device=dev) for m in args.model]
    if args.validate:
        models += validation_variants(args.model[0], dev)
    labels = [m[0] for m in models]
    cfg, sr = models[0][1].cfg, models[0][1].cfg.sample_rate
    table = M.partial_table(models[0][1], year_to_condition(args.year), dev)
    tabs = {p: table[p - 21][(table[p - 21] > 0) & (table[p - 21] < 6500)] for p in range(21, 109)}
    print(f"{len(events)} notes; available per (register, velocity, length): {avail}", flush=True)
    srcs = ["recording"] + labels
    T, K = len(M.ENV_TIMES), 40
    R = {s: {k: np.full((len(events), T, K), np.nan) for k in ("L", "ctrl")} for s in srcs}
    for s in srcs:
        R[s]["clear"] = np.zeros((len(events), T, K), bool)
        R[s]["freqs"] = np.full((len(events), K), np.nan)
    for s0 in range(0, len(events), args.chunk):
        part = events[s0: s0 + args.chunk]
        clips = M.render_clips(args.data, part, models, cfg, dev, pre=PRE, post=POST, batch=8, at="onset")
        for i, ev in enumerate(part):
            e = s0 + i
            for s in srcs:
                r = M.note_envelope(clips[s][i], sr, PRE, table[ev["pitch"] - 21], ev["t_end"], ev["others"], tabs)
                k = len(r["freqs"])
                for key in ("L", "ctrl", "clear"):
                    R[s][key][e, :, :k] = r[key]
                R[s]["freqs"][e, :k] = r["freqs"]
        print(f"{min(s0 + args.chunk, len(events))}/{len(events)}", flush=True)
    np.savez_compressed(os.path.join(args.out, "envelope.npz"),
                        **{f"{s}|{k}": R[s][k] for s in srcs for k in ("L", "ctrl", "clear", "freqs")})
    with open(os.path.join(args.out, "events.json"), "w") as fh:
        json.dump([{k: v for k, v in ev.items() if k != "others"} for ev in events], fh, indent=1)
    report(args, events, avail, srcs, labels, R)


def fades(R, s):
    """Fade re 0.1 s with each level held at or above its floor, and whether it stands SNR dB over it."""
    Lc = np.fmax(R[s]["L"], R[s]["ctrl"])
    up = R[s]["L"] >= R[s]["ctrl"] + SNR
    return Lc - Lc[:, :1], up


def report(args, events, avail, srcs, labels, R):
    reg = np.array([ev["register"] for ev in events])
    vel = np.array([ev["vel_bin"] for ev in events])
    F = {s: fades(R, s) for s in srcs}
    rec_f, rec_up = F["recording"]
    times = list(M.ENV_TIMES)
    cols = [times.index(t) for t in SHOW]
    kk = np.arange(40)
    freq = R["recording"]["freqs"]  # [E, K]

    def base(s):
        """[E, T, K]: both sides clear, both 6 dB up at 0.1 s, at least one up at t."""
        m_f, m_up = F[s]
        clear = R["recording"]["clear"] & R[s]["clear"] & R["recording"]["clear"][:, :1] & R[s]["clear"][:, :1]
        return clear & rec_up[:, :1] & m_up[:, :1]

    def table_rows(s, ev_sel, key_sel, label_of):
        m_f, m_up = F[s]
        b = base(s) & ev_sel[:, None, None]
        rows = []
        for name, sel in key_sel:
            cells_ = []
            for i in cols:
                ok = b[:, i, :] & sel & (rec_up[:, i, :] | m_up[:, i, :])
                cells_.append(cell((m_f[:, i, :] - rec_f[:, i, :])[ok]))
            rows.append(f"| {label_of(name)} | " + " | ".join(cells_) + " |")
        return rows

    groups = [(g, np.broadcast_to((kk >= a) & (kk < b), freq.shape)) for g, a, b in GROUPS]
    bands = [(g, (freq >= a) & (freq < b)) for g, a, b in BANDS]
    hdr = "| | " + " | ".join(f"{t:g} s" for t in SHOW) + " |"
    sep = "|---|" + "---|" * len(SHOW)
    n_reg = {r: int((reg == r).sum()) for r, _, _ in REGS}
    L = [f"# N13: the whole note's envelope, partial by partial ({args.year})", "",
         f"Notes of MIDI 36-88 in any texture, read while they sound freely (at most {T_MAX:g} s): {len(events)} (at "
         f"most {args.cap} / {int(1.5 * args.cap)} / {3 * args.cap} per register × velocity for lengths < 1 / 1-2 / 2+ s, of {sum(avail.values())}); per register "
         + ", ".join(f"{r} {n}" for r, n in n_reg.items()) + ". Models: " + "; ".join(f"`{m}`" for m in args.model)
         + (" and the validation variants of the first" if args.validate else "") + ".", "",
         "Cells: model fade − recording fade (dB; the fade of a partial at t is its level re 0.1 s, each level held at "
         "or above its local floor), median over the partials of the notes where both sides read clear and stand 6 dB "
         "over their floor at 0.1 s, and at least one at t; in brackets the partials counted. Positive: the model's "
         "partial has faded less, i.e. rings too long.", ""]
    for s in labels:
        L += [f"## {s}: by partial number, all registers", "", hdr.replace("| |", "| partials |"), sep]
        L += table_rows(s, np.ones(len(events), bool), groups, str)
        L += ["", f"### {s}: by frequency, all registers", "", hdr.replace("| |", "| partial frequency |"), sep]
        L += table_rows(s, np.ones(len(events), bool), bands, str)
        L.append("")
    s = labels[0]
    L += [f"## {s} by register", ""]
    for r, _, _ in REGS:
        L += [f"### {r} ({n_reg[r]} notes)", "", hdr.replace("| |", "| partials |"), sep]
        L += table_rows(s, reg == r, groups, str)
        L.append("")
    L += [f"## {s} by velocity, all registers", ""]
    for v, _, _ in M.VELOCITY:
        L += [f"### {v} ({int((vel == v).sum())} notes)", "", hdr.replace("| |", "| partials |"), sep]
        L += table_rows(s, vel == v, groups, str)
        L.append("")
    # how many partials still sound
    m_f, m_up = F[s]
    b = base(s)
    show = [times.index(t) for t in (0.5, 1.0, 2.0)]
    L += [f"## Still sounding (6 dB over the floor): recording / {s}, % of the partials read", "",
          "| register | partials | " + " | ".join(f"{times[i]:g} s" for i in show) + " |", "|---|---|" + "---|" * len(show)]
    for r, _, _ in REGS:
        for g, sel in groups:
            cells_ = []
            for i in show:
                ok = b[:, i, :] & sel & (reg == r)[:, None]
                n = int(ok.sum())
                cells_.append("" if n < 10 else f"{100 * rec_up[:, i, :][ok].mean():.0f} / {100 * m_up[:, i, :][ok].mean():.0f} ({n})")
            L.append(f"| {r} | {g} | " + " | ".join(cells_) + " |")
    L.append("")
    text = "\n".join(L) + "\n"
    with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
