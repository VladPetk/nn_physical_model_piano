"""Evaluate a trained model on held-out MAESTRO excerpts and read out what it learned.

    python scripts/evaluate.py runs/trial/best.pt data/maestro24k --years 2018 --split test --out runs/trial/eval

Distances (lower is better) for the untrained prior, the prior after the data-driven init, the fitted
physics alone and the fitted physics with the residual: the multi-resolution STFT loss used in training
and a log-mel L1 distance (dB), both averaged over both channels. The fitted per-condition values
(tuning, damper delay, mic gain, hall T60, re-strike, phantoms, ...) are printed, and with ``--mined``
the fitted inharmonicity and stretch are compared with the values tracked on isolated notes of the
same recordings (the identifiability check of review 3, section 4.6). Renders a few long excerpts.
"""

import argparse
import copy
import json
import math
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.6,max_split_size_mb:256")

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

from pianonn.config import PianoConfig, year_to_condition  # noqa: E402
from pianonn.data import MaestroSegments  # noqa: E402
from pianonn.dsp import bounded  # noqa: E402
from pianonn.fit_init import initialise_from_data  # noqa: E402
from pianonn.losses import MultiResolutionSTFTLoss, _mag  # noqa: E402
from pianonn.physics import LOWEST_MIDI, N_KEYS  # noqa: E402
from pianonn.synth import NeuralPhysicalPiano  # noqa: E402
from pianonn.train import fixed_batches  # noqa: E402


def mel_matrix(sr, n_fft, n_mels=80, f_lo=30.0, f_hi=None, device="cpu"):
    f_hi = f_hi or sr / 2
    mel = lambda f: 2595 * np.log10(1 + f / 700)
    imel = lambda m: 700 * (10 ** (m / 2595) - 1)
    pts = imel(np.linspace(mel(f_lo), mel(f_hi), n_mels + 2))
    f = np.fft.rfftfreq(n_fft, 1 / sr)
    M = np.zeros((n_mels, len(f)))
    for i in range(n_mels):
        lo, c, hi = pts[i], pts[i + 1], pts[i + 2]
        M[i] = np.clip(np.minimum((f - lo) / (c - lo), (hi - f) / (hi - c)), 0, None)
    return torch.tensor(M, dtype=torch.float32, device=device)


def log_mel_db(x, M, n_fft=2048):
    P = _mag(x.reshape(-1, x.shape[-1]), n_fft, n_fft // 4) ** 2
    return 10 * torch.log10(torch.einsum("mf,nft->nmt", M, P) + 1e-8)  # floor ~ -80 dB re full scale per mel band


@torch.no_grad()
def distances(model, batches, residual, M):
    recon = MultiResolutionSTFTLoss()
    mr, lm = [], []
    for b in batches:
        n, s = b["audio"].shape[-1], int(b["loss_start"][0])
        g = torch.Generator(device=b["audio"].device).manual_seed(0)
        y = model(b, n, residual=residual, generator=g)["audio"][..., s:]
        t = b["audio"][..., s:]
        mr.append(float(recon(y, t)))
        lm.append(float((log_mel_db(y, M) - log_mel_db(t, M)).abs().mean()))
    return float(np.mean(mr)), float(np.mean(lm))


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
        "mic_gain_db": [round(float(v), 2) for v in room.mic_gain_db[cond]],
        "hall_t60_s": [round(float(v), 2) for v in torch.exp(room.prior_log_t60 + bounded(room.raw_log_t60[cond], 0.7))],
        "hall_log_gain": [round(float(v), 2) for v in room.log_gain[cond]],
        "vel_slope_db(cond)": float(bounded(ph.cond_vel_slope[cond], 10.0)),
        "gain_db(cond)": float(bounded(ph.cond_gain_db[cond], 12.0)),
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
            "prompt_x": float(torch.exp(bounded(ph.raw_prompt[k], 1.5))), "gain_db": float(ph.gain_db[k]),
        }
    return out, per_key, B.cpu().numpy(), cents_et.cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt")
    ap.add_argument("data")
    ap.add_argument("--years", type=int, nargs="*", default=[2018])
    ap.add_argument("--split", default="test")
    ap.add_argument("--examples", type=int, default=48)
    ap.add_argument("--mined", help="scripts/mine_notes.py output for the identifiability check")
    ap.add_argument("--render-seconds", type=float, default=20.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.6)
    os.makedirs(args.out, exist_ok=True)
    state = torch.load(args.ckpt, map_location=dev)
    cfg = PianoConfig.from_dict(state["cfg"])
    trained = NeuralPhysicalPiano(cfg).to(dev)
    trained.load_state_dict(state["model"])
    trained.eval()
    cond = year_to_condition(args.years[0])

    ds = MaestroSegments(args.data, args.split, cfg, 2.0, 1.0, 12.0, length=args.examples, deterministic=True,
                         years=args.years, seed=11)
    batches = fixed_batches(ds, args.examples, 8, dev)
    M = mel_matrix(cfg.sample_rate, 2048, device=dev)

    torch.manual_seed(0)
    raw = NeuralPhysicalPiano(cfg).to(dev).eval()
    init = copy.deepcopy(raw)
    init_set = MaestroSegments(args.data, "train", cfg, 2.0, 1.0, 12.0, length=32, deterministic=True,
                               years=args.years, seed=3)
    initialise_from_data(init, fixed_batches(init_set, 32, 8, dev), log=lambda *a: None)
    rows = {
        "untrained prior": distances(raw, batches, False, M),
        "prior + data init": distances(init, batches, False, M),
        "fitted physics (residual off)": distances(trained, batches, False, M),
        "fitted physics + residual": distances(trained, batches, True, M),
    }
    lines = [f"# Evaluation of `{args.ckpt}` on {args.examples} {args.split} excerpts ({', '.join(map(str, args.years))})", "",
             f"Step {state.get('step')}, stage {state.get('stage')}. 2 s excerpts after 1 s warm-up and 12 s lookback, both channels.", "",
             "| model | MR-STFT loss | log-mel L1 (dB) |", "|---|---|---|"]
    lines += [f"| {k} | {a:.4f} | {b:.2f} |" for k, (a, b) in rows.items()]
    fitted, per_key, B, cents = fitted_table(trained, cond)
    prior_fitted, _, B0, cents0 = fitted_table(raw, cond)
    lines += ["", "## Fitted recording condition", "", "```", json.dumps(fitted, indent=1), "```"]
    lines += ["", "## Per key (every third key)", "",
              "| MIDI | B | cents re ET | re-strike (nats) | phantom dB re prior | impulse dB re prior | b1 x | T_c x | R x | gain dB |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    lines += [f"| {p} | {v['B']:.2e} | {v['cents_et']:+.1f} | {v['restrike_nats']:.2f} | {v['phantom_db_re_prior']:+.1f} | "
              f"{v['impulse_db_re_prior']:+.1f} | {v['b1_x']:.2f} | {v['tc_x']:.2f} | {v['prompt_x']:.2f} | {v['gain_db']:.1f} |"
              for p, v in per_key.items()]
    if args.mined:
        with open(args.mined) as f:
            mined = {int(k): v for k, v in json.load(f)["per_key"].items()}
        a4 = mined.get(69, {}).get("cents")
        lines += ["", "## Identifiability: fitted vs tracked on isolated notes of the same recordings", "",
                  "Tracked with `pianonn.calibration` on notes with nothing else sounding (scripts/mine_notes.py). "
                  "Registers pool keys; B uses reliable fits only.", "",
                  "| register | notes | B tracked | B prior | B fitted | cents tracked | cents prior | cents fitted |", "|---|---|---|---|---|---|---|---|"]
        for lo, hi in ((21, 36), (36, 48), (48, 60), (60, 72), (72, 84), (84, 96), (96, 109)):
            ks = [p for p in range(lo, hi) if p in mined]
            if not ks:
                continue
            bt = [mined[p]["B"] for p in ks if mined[p]["B"] is not None]
            ct = [mined[p]["cents"] for p in ks]
            n = sum(mined[p]["n"] for p in ks)
            idx = [p - LOWEST_MIDI for p in ks]
            med = lambda v: float(np.median(v)) if len(v) else float("nan")
            lines.append(f"| {lo}-{hi - 1} | {n} | {med(bt):.2e} | {med(B0[idx]):.2e} | {med(B[idx]):.2e} | "
                         f"{med(ct):+.1f} | {med(cents0[idx]):+.1f} | {med(cents[idx]):+.1f} |")
        if a4 is not None:
            lines.append(f"\nA4 tracked {a4:+.1f} cents re 440 Hz.")

    # long renders of held-out excerpts
    rs = MaestroSegments(args.data, args.split, cfg, args.render_seconds - 1.0, 1.0, 12.0, length=2, deterministic=True,
                         years=args.years, seed=12)
    for i, b in enumerate(fixed_batches(rs, 2, 1, dev)):
        n = b["audio"].shape[-1]
        sf.write(os.path.join(args.out, f"long{i}_target.wav"), b["audio"][0].T.cpu().numpy(), cfg.sample_rate)
        for tag, model, res in (("prior_init", init, False), ("physics", trained, False), ("residual", trained, True)):
            with torch.no_grad():
                y = model(b, n, residual=res, block_seconds=4.0, generator=torch.Generator(device=dev).manual_seed(0))["audio"][0]
            sf.write(os.path.join(args.out, f"long{i}_{tag}.wav"), y.T.clamp(-1, 1).cpu().numpy(), cfg.sample_rate)
    text = "\n".join(lines) + "\n"
    with open(os.path.join(args.out, "report.md"), "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
