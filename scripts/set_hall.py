"""Set the hall's T60 and level per octave band from free decays (P4) of one year's recordings (phase 3, step 1).

The hall's T60 was fitted by the music loss and came out 10-30 % short in every band (docs/tone_measures.md, 10.3),
while its level left the model's direct and early sound 1.4-2.7 dB lower over its reverberant field than the
recordings'. Here both are set from measurement instead:
- P4 (``pianonn.measures.room_measures``) on the calibration group's free decays, for the recordings and for the
  model's renders of the same MIDI;
- per band, the hall's T60 is multiplied by the median ratio recording / model, and its band gain moved by the median
  early-level difference (model - recording) / 0.67 dB (the measure reads +4.0 dB for the hall 6 dB down, 10.2);
- repeated ``--iterations`` times, since the model's reading is not the parameter (the body's tail, the undamped
  strings and the floor enter it, and the T60 moves the early level);
- then checked on the evaluation group.
Writes a checkpoint with the new hall (and whatever options the model was loaded with), to be frozen in training.

    python scripts/set_hall.py data/maestro24k --out runs/phase3/step1/hall \\
        --model base=runs/round2/b_control/best.pt:physics:bridge_end_comb=1,body_q_max=50,mined=runs/measurements/mined_2018.json
"""

import argparse
import json
import math
import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.6,max_split_size_mb:256")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from note_bench import ROOM_BANDS, floor_bands, measure_decays  # noqa: E402
from pianonn.config import year_to_condition  # noqa: E402
from pianonn.dsp import bounded  # noqa: E402
from pianonn.render import load_variant  # noqa: E402
from pianonn.room import HALL_BANDS  # noqa: E402

EARLY_SLOPE = 0.67  # dB of P4 early per dB of hall level (10.2: hall -6 dB reads +4.0)


def paired(rows, label, key, log=False):
    """Median over decays of model - recording (of logs with ``log``) where both have ``key``; and n."""
    v = [(math.log(r[label][key] / r["recording"][key]) if log else r[label][key] - r["recording"][key])
         for r in rows if key in r[label] and key in r["recording"]]
    return (float(np.median(v)), len(v)) if len(v) >= 5 else (float("nan"), len(v))


def summary(rows, label):
    out = {}
    for c in HALL_BANDS:
        rec = [r["recording"][f"P4 T60 {c}"] for r in rows if f"P4 T60 {c}" in r["recording"]]
        mod = [r[label][f"P4 T60 {c}"] for r in rows if f"P4 T60 {c}" in r[label]]
        dt, n = paired(rows, label, f"P4 T60 {c}", log=True)
        de, ne = paired(rows, label, f"P4 early {c}")
        out[c] = {"rec_t60": float(np.median(rec)) if rec else float("nan"), "mod_t60": float(np.median(mod)) if mod else float("nan"),
                  "log_ratio": dt, "n": n, "early_diff": de, "n_early": ne}
    return out


def fmt(s):
    return " | ".join(f"{c}: T60 {v['rec_t60']:.2f}/{v['mod_t60']:.2f} s ({100 * (math.exp(v['log_ratio']) - 1):+.0f} %, n {v['n']}), "
                      f"early {v['early_diff']:+.1f} dB" for c, v in s.items())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data")
    ap.add_argument("--years", type=int, nargs="*", default=[2018])
    ap.add_argument("--bench")
    ap.add_argument("--model", required=True, help="label=checkpoint:physics[:options] (pianonn.render.load_variant)")
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.6)
    os.makedirs(args.out, exist_ok=True)
    with open(args.bench or f"runs/measurements/bench_{'_'.join(map(str, args.years))}.json") as f:
        bench = json.load(f)
    label, model, residual = load_variant(args.model, device=dev)
    cfg, cond = model.cfg, year_to_condition(args.years[0])
    room = model.room
    groups = {g: [d for d in bench["decays"] if d["group"] == g] for g in ("calib", "eval")}
    lines = [f"# Hall set from free decays ({args.years[0]})", "", f"Model: `{args.model}`. "
             f"{len(groups['calib'])} calibration and {len(groups['eval'])} evaluation free decays. Per band: T60 "
             "recording/model (median), the paired median change model − recording, and the early level difference "
             "(model − recording, dB; negative: the model's direct sound sits lower over its reverberant field).", ""]

    def run(group):
        floors = {"recording": floor_bands(model, cond, cfg, dev), label: floor_bands(model, cond, cfg, dev)}
        return measure_decays(args.data, groups[group], [(label, model, residual)], cfg, dev, floors)

    def show(tag, s):
        lines.append(f"- **{tag}**: " + fmt(s))
        print(lines[-1], flush=True)

    show("evaluation, before", summary(run("eval"), label))
    with torch.no_grad():
        t60_0 = torch.exp(room.prior_log_t60 + bounded(room.raw_log_t60[cond], 0.7)).clone()
        gain_0 = room.band_log_gain[cond].clone()
    for it in range(args.iterations):
        s = summary(run("calib"), label)
        show(f"calibration, iteration {it}", s)
        with torch.no_grad():
            for b, c in enumerate(HALL_BANDS):
                if math.isfinite(s[c]["log_ratio"]):
                    log_t60 = room.prior_log_t60[b] + bounded(room.raw_log_t60[cond, b], 0.7) - s[c]["log_ratio"]
                    x = ((log_t60 - room.prior_log_t60[b]) / 0.7).clamp(-0.99, 0.99)
                    room.raw_log_t60[cond, b] = 0.7 * torch.atanh(x)
                if math.isfinite(s[c]["early_diff"]):
                    room.band_log_gain[cond, b] += s[c]["early_diff"] / EARLY_SLOPE / (20 / math.log(10))
    s = summary(run("calib"), label)
    show("calibration, after", s)
    show("evaluation, after", summary(run("eval"), label))
    with torch.no_grad():
        t60 = torch.exp(room.prior_log_t60 + bounded(room.raw_log_t60[cond], 0.7))
        gain = room.band_log_gain[cond]
    lines += ["", "| band | T60 before (s) | T60 set (s) | hall band level change (dB) |", "|---|---|---|---|"]
    for b, c in enumerate(HALL_BANDS):
        lines.append(f"| {c} | {t60_0[b]:.2f} | {t60[b]:.2f} | {(gain[b] - gain_0[b]).item() * 20 / math.log(10):+.1f} |")
    text = "\n".join(lines) + "\n"
    with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8") as f:
        f.write(text)
    torch.save({"cfg": cfg.to_dict(), "model": model.state_dict(), "source": args.model, "step": 0}, os.path.join(args.out, "model.pt"))
    print(text)


if __name__ == "__main__":
    main()
