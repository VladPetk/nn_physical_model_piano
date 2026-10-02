"""Evaluate a trained model on held-out MAESTRO excerpts and read out what it learned.

    python scripts/evaluate.py runs/trial/best.pt data/maestro24k --years 2018 --split test --out runs/trial/eval

Every distance table has a scale (review 4, section 2; docs/plan_round2.md, A5):
* the models: the training start (prior + mined frequencies + data init), the fitted physics, and the fitted
  physics with the residual (if the run trained one);
* references: the same model with another noise seed, the recording plus the model's floor, a per-excerpt gain
  oracle, the model driven by another excerpt's MIDI, and stationary noise with the recording's spectrum.
Distances: the round-2 loss and its terms, the trial's MR-STFT loss and log-mel L1 (dB), both channels.

Then: the level per octave band and against dynamics and pedal, the level bias of every loss term, the fitted
floor against the test pieces' silence, what the residual does, the fitted recording condition (with the level
chain), per-key tables, and with ``--mined`` the fitted vs tracked inharmonicity and stretch (review 3, 4.6).
Renders two long excerpts: the recording, the training start, the physics, and the physics with the residual.
"""

import argparse
import copy
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

from pianonn.config import PianoConfig, year_to_condition  # noqa: E402
from pianonn.data import MaestroSegments  # noqa: E402
from pianonn.dsp import bounded  # noqa: E402
from pianonn.fit_init import apply_mined_priors, floor_from_silence, initialise_from_data  # noqa: E402
from pianonn.metrics import (CENTERS, band_levels, bias_report, level_bias, level_regression, make_terms,  # noqa: E402
                             references, render)
from pianonn.physics import LOWEST_MIDI, N_KEYS  # noqa: E402
from pianonn.synth import NeuralPhysicalPiano  # noqa: E402
from pianonn.train import fixed_batches  # noqa: E402

COLUMNS = ("new: total", "new: band", "new: fine", "new: attack", "old: MR-STFT", "log-mel (dB)")


@torch.no_grad()
def distances(model, batches, residual):
    terms = make_terms(model.cfg.sample_rate, batches[0]["audio"].device)
    acc = {}
    for b in batches:
        s = int(b["loss_start"][0])
        for k, v in terms(render(model, b, residual), b["audio"][..., s:], b["onset"] - s / model.cfg.sample_rate, b["mask"]).items():
            acc.setdefault(k, []).append(v.cpu())
    return {k: float(torch.cat(v).mean()) for k, v in acc.items()}


def table(rows):
    lines = ["| | " + " | ".join(COLUMNS) + " |", "|---|" + "---|" * len(COLUMNS)]
    for name, d in rows.items():
        lines.append(f"| {name} | " + " | ".join(f"{d[c]:.3f}" if c != "log-mel (dB)" else f"{d[c]:.2f}" for c in COLUMNS) + " |")
    return lines


@torch.no_grad()
def read_residual(model, batches):
    """What the residual does on held-out audio (review 3, 9.2, safeguard 5): per-note corrections by register
    (mean and rms, in their own units), the band gains and noise per band, and the residual noise's share of the
    output's band energy at the microphones."""
    from pianonn.losses import band_energies
    from pianonn.synth import ContextNet

    regs = ((21, 48), (48, 72), (72, 109))
    note = {name: {r: [] for r in regs} for name in ContextNet.NOTE}
    gains, noise, share = [], [], []
    M = model.noise.band_masks(512)
    for b in batches:
        n = b["audio"].shape[-1]
        out = model(b, n, residual=True, generator=torch.Generator(device=b["audio"].device).manual_seed(0),
                    extras=("residual_out",))
        mask = b["mask"] & (b["onset"] >= 0)  # notes struck inside the rendered window
        for name, v in out["ctx"].items():
            v = v.mean(-1) if v.dim() == 3 else v
            for lo, hi in regs:
                sel = mask & (b["pitch"] >= lo) & (b["pitch"] < hi)
                note[name][(lo, hi)].append(v[sel].cpu())
        gains.append(out["frame_ctx"]["band_gain"].mean((0, 2)).cpu())
        noise.append(out["frame_ctx"]["noise_level"].mean((0, 2)).cpu())
        if "noise_res_out" in out:
            e_res = band_energies(out["noise_res_out"], M).mean((0, 1, 3))
            e_out = band_energies(out["audio"], M).mean((0, 1, 3))
            share.append((e_res / (e_out + 1e-20)).cpu())
    lines = ["| per-note output (bound) | " + " | ".join(f"MIDI {lo}-{hi - 1} mean / rms" for lo, hi in regs) + " |",
             "|---|" + "---|" * len(regs)]
    for name, (k, s) in ContextNet.NOTE.items():
        cells = []
        for r in regs:
            v = torch.cat(note[name][r]) if note[name][r] else torch.zeros(0)
            cells.append(f"{float(v.mean()):+.3f} / {float(v.pow(2).mean().sqrt()):.3f}" if v.numel() else "-")
        lines.append(f"| {name} (+-{s}) | " + " | ".join(cells) + " |")
    g, nz = torch.stack(gains).mean(0), torch.stack(noise).mean(0)
    lines += ["", "R3 band gains, mean over time (dB, 16 bands 40 Hz - 12 kHz): "
              + " ".join(f"{20 * float(x) / math.log(10):+.1f}" for x in g),
              "", "R3 noise level re its base (dB): " + " ".join(f"{20 * float(x) / math.log(10):+.0f}" for x in nz)]
    if share:
        sh = torch.stack(share).mean(0)
        lines += ["", "Residual noise share of the output's band energy (32 bands 40 Hz - 12 kHz, %): "
                  + " ".join(f"{100 * float(x):.1f}" for x in sh)]
    return lines


@torch.no_grad()
def fitted_table(model, cond):
    ph = model.physics
    c = torch.tensor([cond], device=ph.gain_db.device)
    ki = torch.arange(N_KEYS, device=c.device)[None]
    u = torch.full(ki.shape, 64 / 127, device=c.device)
    m = ph.modes(ki, u, torch.zeros_like(u), c)
    f1 = m["freq"][0, :, 0, 0]
    B = torch.exp(ph.prior_log_B + bounded(ph.raw_log_B, 1.5))
    cents_et = 1200 * torch.log2(f1 / (440 * 2 ** ((ki[0] + LOWEST_MIDI - 69) / 12)))
    room = model.room
    out = {
        "tuning_cents(cond)": float(bounded(ph.cond_cents[cond], 30.0)),
        "damper_delay_ms": 1000 * float(ph.damper_delay(c)),
        "level chain (dB): mic gain L/R": [round(float(v), 2) for v in room.mic_gain_db[cond]],
        "level chain (dB): condition gain": float(bounded(ph.cond_gain_db[cond], 12.0)),
        "level chain (dB): mean per-key gain re prior": float((ph.gain_db - ph.prior_gain_db).mean()),
        "level chain (dB): velocity slope re prior (cond)": float(bounded(ph.cond_vel_slope[cond], 10.0)),
        "level chain (dB): velocity curve, u=0..1": [round(float(v), 2) for v in bounded(ph.cond_vel_curve[cond], 12.0)],
        "hall_t60_s": [round(float(v), 2) for v in torch.exp(room.prior_log_t60 + bounded(room.raw_log_t60[cond], 0.7))],
        "hall_log_gain": [round(float(v), 2) for v in room.log_gain[cond]],
        "pedal_theta": float(0.42 + bounded(ph.raw_pedal_theta, 0.3)),
        "pedal_power": float(2.5 * torch.exp(bounded(ph.raw_pedal_power, 0.47))),
        "decay_exponent": float(1.092 * torch.exp(bounded(ph.raw_decay_p, 0.25))),
    }
    per_key = {}
    for k in range(0, N_KEYS, 3):
        per_key[k + LOWEST_MIDI] = {
            "B": float(B[k]), "cents_et": float(cents_et[k]),
            "restrike_nats": float(m["restrike"][0, k]),
            "phantom_db_re_prior": float(bounded(ph.raw_phantom_db[k], 20.0)),
            "impulse_db_re_prior": float(bounded(ph.raw_impulse_db[k], 20.0)),
            "b1_x": float(torch.exp(bounded(ph.raw_log_b1[k], 1.5))), "tc_x": float(torch.exp(bounded(ph.raw_log_tc[k], 1.0))),
            "prompt_x": float(torch.exp(bounded(ph.raw_prompt[k], 1.5))), "gain_db": float(ph.gain_db[k] - ph.prior_gain_db[k]),
        }
    return out, per_key, B.cpu().numpy(), cents_et.cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt")
    ap.add_argument("data")
    ap.add_argument("--years", type=int, nargs="*", default=[2018])
    ap.add_argument("--split", default="test")
    ap.add_argument("--examples", type=int, default=96)
    ap.add_argument("--mined", help="scripts/mine_notes.py output for the identifiability check")
    ap.add_argument("--render-seconds", type=float, default=20.0)
    ap.add_argument("--init-examples", type=int, default=32)
    ap.add_argument("--start-without-mined", action="store_true",
                    help="the run started from the prior + data init only (no --mined): model its start that way")
    ap.add_argument("--no-renders", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.6)
    os.makedirs(args.out, exist_ok=True)
    state = torch.load(args.ckpt, map_location=dev)
    cfg = PianoConfig.from_dict(state["cfg"])
    from pianonn.render import load_weights

    load_log = []
    trained = load_weights(NeuralPhysicalPiano(cfg), state["model"], log=load_log.append).to(dev).eval()
    year = args.years[0]
    cond = year_to_condition(year)
    run_args = state.get("args", {})
    has_residual = state.get("stage", 1) >= 2 and not run_args.get("no_residual", False)

    ds = MaestroSegments(args.data, args.split, cfg, 2.0, 1.0, 12.0, length=args.examples, deterministic=True,
                         years=args.years, seed=11)
    batches = fixed_batches(ds, args.examples, 8, dev)

    torch.manual_seed(0)
    init = NeuralPhysicalPiano(cfg).to(dev).eval()  # the trained model's starting point, as in pianonn.train
    mined = None
    if args.mined:
        with open(args.mined) as f:
            mined = json.load(f)
        if not args.start_without_mined:
            apply_mined_priors(init, mined, log=lambda *a: None)
    init_set = MaestroSegments(args.data, "train", cfg, 2.0, 1.0, 12.0, length=args.init_examples, deterministic=True,
                               years=args.years, seed=3)
    initialise_from_data(init, fixed_batches(init_set, args.init_examples, 8, dev), log=lambda *a: None,
                         tuning=mined is None or args.start_without_mined, silence=init_set.silence_clips(year))
    rows = {"training start (prior + measurements)": distances(init, batches, False),
            "fitted physics (residual off)": distances(trained, batches, False)}
    if has_residual:
        rows["fitted physics + residual"] = distances(trained, batches, True)
    refs, oracle = references(trained, batches, has_residual)
    lines = [f"# Evaluation of `{args.ckpt}` on {args.examples} {args.split} excerpts ({', '.join(map(str, args.years))})", "",
             f"Step {state.get('step')}, stage {state.get('stage')}"
             + (", stage 2 without the residual (control run)" if run_args.get("no_residual") else "")
             + (", GAN in stage 2" if run_args.get("adv_with_stage2") else "")
             + ". 2 s excerpts after 1 s warm-up and 12 s lookback, both channels.", ""]
    if load_log:
        lines += ["Checkpoint loading: " + "; ".join(load_log), ""]
    lines += ["## Distances", "", "Lower is better. `new` is the round-2 training loss (log10 power units, 1 = 10 dB).", ""]
    lines += table(rows)
    lines += ["", f"References for scale (the {'residual' if has_residual else 'physics'} model); the gain oracle's "
              f"median gain is {oracle:+.1f} dB:", ""]
    lines += table({k: v for k, v in refs.items() if k != "model"})

    per_band, lvl = band_levels(trained, batches, has_residual)
    coef, r = level_regression(lvl)
    edges = np.percentile(r[:, 0], [100 / 3, 200 / 3])
    terc = [np.median(r[np.digitize(r[:, 0], edges) == k, 3]) for k in range(3)]
    lines += ["", "## Level", "",
              f"Broadband level, model − recording: median {np.median(lvl[:, 3]):+.2f} dB "
              f"(IQR {np.percentile(lvl[:, 3], 25):+.2f} … {np.percentile(lvl[:, 3], 75):+.2f}), {len(lvl)} excerpts.", "",
              "| octave band | " + " | ".join(f"{c:.0f} Hz" for c in CENTERS) + " |", "|---|" + "---|" * len(CENTERS),
              "| model − recording, median (dB) | " + " | ".join(f"{np.median(per_band[:, k]):+.1f}" for k in range(len(CENTERS))) + " |",
              "", f"By the excerpt's mean velocity (terciles): soft (< {edges[0]:.0f}) {terc[0]:+.2f} dB, middle {terc[1]:+.2f}, "
              f"loud (≥ {edges[1]:.0f}) {terc[2]:+.2f}; loud − soft {terc[2] - terc[0]:+.2f} dB.",
              "", f"Regression over {len(r)} excerpts: error = {coef[0]:+.1f} {coef[1]:+.1f}·velocity/127 "
              f"{coef[2]:+.1f}·half-pedal {coef[3]:+.1f}·full-pedal (dB; fractions of the window; the slope is over the "
              "full velocity range, which the excerpts' means span only partly). On these excerpts the trial's "
              "220-minute model gives −1.7 − 0.8·velocity/127 (docs/round2_results.md)."]

    bias, err = level_bias(trained, batches, has_residual)
    lines += ["", "## Level bias of the loss terms", "",
              "The term's optimum gain in each band minus the energy-matching gain (dB, median over excerpts). "
              "Gate: ±0.5 dB overall, ±1 dB per band.", ""] + bias_report(bias, err)

    test_silence = ds.silence_clips(year)
    if test_silence:
        probe = copy.deepcopy(trained)
        floor_from_silence(probe, cond, test_silence, log=lambda *a: None)
        fit = trained.room.floor_db(torch.tensor([cond], device=dev))[0].mean(0)
        meas = probe.room.floor_ref_db[cond].mean(0)
        c = trained.room.floor_log2_centers.exp2()
        lines += ["", f"## Noise floor: fitted vs the {args.split} pieces' leading silence ({len(test_silence)} clips)", "",
                  "| band | " + " | ".join(f"{float(c[i]):.0f}" for i in range(0, len(c), 2)) + " |",
                  "|---|" + "---|" * len(range(0, len(c), 2)),
                  "| fitted (dBFS, L/R mean) | " + " | ".join(f"{float(fit[i]):.1f}" for i in range(0, len(c), 2)) + " |",
                  "| silence | " + " | ".join(f"{float(meas[i]):.1f}" for i in range(0, len(c), 2)) + " |",
                  "", "Hum (RMS dBFS, 60/120/180 Hz, L/R mean): fitted "
                  + " ".join(f"{float(v):.1f}" for v in trained.room.hum_db(torch.tensor([cond], device=dev))[0].mean(0))
                  + ", silence " + " ".join(f"{float(v):.1f}" for v in probe.room.hum_ref_db[cond].mean(0))]

    if has_residual:
        lines += ["", "## What the residual does (held-out)", ""] + read_residual(trained, batches)
    fitted, per_key, B, cents = fitted_table(trained, cond)
    _, _, Bi, centsi = fitted_table(init, cond)
    _, _, B0, cents0 = fitted_table(NeuralPhysicalPiano(cfg).to(dev).eval(), cond)
    lines += ["", "## Fitted recording condition", "",
              "The level is split over the mic gain, the condition gain, the per-key gains and the velocity law; their "
              "means are pinned (the level gauge), so the mic gain carries the level.", "", "```", json.dumps(fitted, indent=1), "```"]
    lines += ["", "## Per key (every third key)", "",
              "| MIDI | B | cents re ET | re-strike (nats) | phantom dB re prior | impulse dB re prior | b1 x | T_c x | R x | gain dB re prior |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    lines += [f"| {p} | {v['B']:.2e} | {v['cents_et']:+.1f} | {v['restrike_nats']:.2f} | {v['phantom_db_re_prior']:+.1f} | "
              f"{v['impulse_db_re_prior']:+.1f} | {v['b1_x']:.2f} | {v['tc_x']:.2f} | {v['prompt_x']:.2f} | {v['gain_db']:+.1f} |"
              for p, v in per_key.items()]
    if args.mined:
        with open(args.mined) as f:
            mk = {int(k): v for k, v in json.load(f)["per_key"].items()}
        a4 = mk.get(69, {}).get("cents")
        lines += ["", "## Identifiability: fitted vs tracked on isolated notes of the same recordings", "",
                  "Tracked with `pianonn.calibration` on notes with a clear first second (scripts/mine_notes.py). "
                  "Registers pool keys; B uses reliable fits only. 'start' is the trained model's starting point "
                  "(prior + mined + data init).", "",
                  "| register | notes | B tracked | B prior | B start | B fitted | cents tracked | cents prior | cents start | cents fitted |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for lo, hi in ((21, 36), (36, 48), (48, 60), (60, 72), (72, 84), (84, 96), (96, 109)):
            ks = [p for p in range(lo, hi) if p in mk]
            if not ks:
                continue
            bt = [mk[p]["B"] for p in ks if mk[p]["B"] is not None]
            ct = [mk[p]["cents"] for p in ks]
            n = sum(mk[p]["n"] for p in ks)
            idx = [p - LOWEST_MIDI for p in ks]
            med = lambda v: float(np.median(v)) if len(v) else float("nan")
            lines.append(f"| {lo}-{hi - 1} | {n} | {med(bt):.2e} | {med(B0[idx]):.2e} | {med(Bi[idx]):.2e} | {med(B[idx]):.2e} | "
                         f"{med(ct):+.1f} | {med(cents0[idx]):+.1f} | {med(centsi[idx]):+.1f} | {med(cents[idx]):+.1f} |")
        if a4 is not None:
            lines.append(f"\nA4 tracked {a4:+.1f} cents re 440 Hz.")

    if not args.no_renders:  # long renders of held-out excerpts
        rs = MaestroSegments(args.data, args.split, cfg, args.render_seconds - 1.0, 1.0, 12.0, length=2, deterministic=True,
                             years=args.years, seed=12)
        for i, b in enumerate(fixed_batches(rs, 2, 1, dev)):
            n = b["audio"].shape[-1]
            clips = [b["audio"][0].T.cpu().numpy()]
            sf.write(os.path.join(args.out, f"long{i}_target.wav"), clips[0], cfg.sample_rate)
            for tag, model, res in (("start", init, False), ("physics", trained, False)) + ((("residual", trained, True),) if has_residual else ()):
                with torch.no_grad():
                    y = model(b, n, residual=res, block_seconds=4.0, generator=torch.Generator(device=dev).manual_seed(0))["audio"][0]
                clips.append(y.T.clamp(-1, 1).cpu().numpy())
                sf.write(os.path.join(args.out, f"long{i}_{tag}.wav"), clips[-1], cfg.sample_rate)
            gap = np.zeros((cfg.sample_rate // 2, clips[0].shape[1]), dtype=np.float32)
            sf.write(os.path.join(args.out, f"long{i}_ab.wav"), np.concatenate([x for c in clips for x in (c, gap)]), cfg.sample_rate)
    text = "\n".join(lines) + "\n"
    with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
