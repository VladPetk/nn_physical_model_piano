# From the reviews to a first MAESTRO trial run

This is the plan built from reviews 1–3 ([`reviews/`](reviews/)), what was implemented, and how the
first trial on real audio was run. The results are in [`trial_2018.md`](trial_2018.md).

The plan follows review 3, section 7: fix what would make the first real fit meaningless (phase 0),
enrich the physics where review 3 found the model incomplete for loud, pedalled music, add the
black-box residual of section 9.2 with the safeguards that keep the physics in charge, then run the
go/no-go on one MAESTRO year (phase 1) within about 2 h of GPU time.

## 1. Open findings and what was done

Reviews 1 and 2 were already fixed in spec v3/v4, except for the items below. Review 3's findings are
F1–F15.

| finding | status | where |
|---|---|---|
| F1 log-magnitude loss dominated by the recordings' noise floor | **done**: the loss floor is white noise at −80 dBFS (scaled per window), the model has a learned stationary floor per condition and channel (initialised from the recordings), and every noise decay time is bounded. Test: same piano + floor scores < 0.5× a different piano | `losses.py`, `room.py` (`floor_noise`), `synth.py` (`NoiseBank.taus`) |
| F2 mono downmix bakes mic combing into per-key tables | **done**: stereo end to end. MAESTRO 2018's channels correlate at only 0.2–0.4. Two body FIRs, decorrelated hall tails, a gain per channel, a per-key channel balance; the loss runs on both channels | `scripts/prepare_maestro.py`, `data.py`, `room.py` |
| F3 `partial_gain` and the context net can absorb the physics | **done**: `partial_gain` bounded ±8 dB (was ±26) and smooth across keys; staged training freezes it, the colouration and the whole residual in stage 1 | `physics.py`, `train.py` (`set_stage`) |
| F4 re-strike accumulates energy | **done**: every later strike of the same key takes a learnable number of nats (per key; bass 0.35, treble 1.0) out of the ringing vibration. Measured: 8-strike tremolos under the pedal built up +1 to +5 dB without it (not the ~10 dB the review guessed), ≤ +1.3 dB with it | `oscbank.py`, `synth.py` (`next_strikes`) |
| F5 no longitudinal modes / phantom partials | **done** as a parametrised set: components at 2f_j and f_j + f_(j+1) of the prompt and first aftersound modes, level ~ a_j a_k (grows with velocity squared), emphasised around the longitudinal resonance near 15 f1; per-key level, learnable | `physics.py` (`_phantoms`) |
| F6 bridge conductance too coarse and global | **done**: per condition, 1/6-octave knots from A0 to 14 kHz, smoothness-regularised | `physics.py` (`bridge_conductance`) |
| item 6: per-key soundboard colouration | **done**: a smooth (key, log f) table per condition (23 × 36 knots, ±8 dB), kept zero-mean over keys so it only holds what the body FIR cannot; stage 2 | `physics.py` (`coloration`) |
| F7 no learned attack residual | **done**, see section 2 (R2) | `synth.py` |
| F8 lookback too short; damper/sostenuto history ignored (also review 1 m4) | **done**: 12 s lookback, and the control curves (pedals, key rolls, damper integral, sostenuto latch) start 12 s before the window. Test: a note released 2.5 s before the window is > 60 dB under a held one | `data.py`, `synth.py` |
| F9 no validation, dumps, resume | **done**: fixed validation excerpts with the residual on and off, audio dumps, best/last checkpoints with the optimiser, resume, time limit, JSONL log, per-module gradient norms | `train.py` |
| F10 24 kHz, 96 partials, mono | stereo done; 48 kHz and more bass partials deferred to the product stage (phase 5) | |
| F11 float32 absolute time in the oscillator bank | **done**: the cycle count is formed in float64 and only its fractional part kept. Test: a note 10 min in renders identically to one at t = 0 (< −100 dB) | `oscbank.py` |
| F12 `_ring_end` ignores dampers | **done**: exact activity test per chunk from the envelope bound with dampers and pedals, *per oscillator* (a ringing bass note under the pedal only pays for its partials that are still audible) | `synth.py` (`render_strings`) |
| F13 damper delay fixed | **done**: per condition, 15 ms ± 50 ms, differentiable (the release edge is fractional in frames) | `physics.py`, `synth.py` (`key_rolls`) |
| F14 stop Iowa calibration | done: nothing further was tuned on Iowa | |
| F15 README framing | done | `README.md` |
| review 1 m1 loudness prior over total energy | done: equal energy over the first 0.3 s | `physics.py` |
| review 1 nits: release noise ignored the sostenuto latch; `pedal_log_power` unbounded | done | `synth.py`, `physics.py` |
| review 3 section 5: tests that the loss and the data can drive the parameters | done: gradient points back to a perturbed teacher (B, b1, T_c) through a room and a noise floor; the floor test; stage freezing; plus the go/no-go ladder on a real excerpt (`scripts/overfit_excerpt.py`) | `tests/test_training.py` |
| review 3 section 4.6: isolated-note mining | done: `scripts/mine_notes.py` (421 notes of MAESTRO 2018 with a clear first second) and the fitted-vs-tracked table in `scripts/evaluate.py` | |
| review 3 section 9.1: felt model at note-on | deferred (phase 3): the spectral hammer table stays | |
| sympathetic bank | kept off for the trial (2× per step); works with stereo | |
| GAN | not used: it belongs after the residual (review 3, 9.2) | |

Also new in the physics: a deterministic **knock impulse** (the hammer's contact pulse sent straight into
the body FIR, so the knock has the soundboard's modal character rather than only band noise), and the
data-driven initialisation of the recording chain (below).

## 2. The residual (review 3, section 9.2)

The context net (a causal GRU over the piano roll and pedals, now including the 12 s history) emits:

| level | what | bound |
|---|---|---|
| R1, per note | gain, brightness (via contact time), decay rate and its tilt over frequency, a smooth spectral correction (6 log-f bumps) | ±6 dB, ±0.3, ±0.3, ±0.15/oct, ±4 dB |
| R2, per note | knock level and spectrum (8 bumps), plus a slower learned attack-noise component with its own level, spectrum and decay time (15–110 ms) | ±1 nat, ±8.7 dB; attack starts 40 dB under the knock |
| R3, per frame (200 Hz) | 16 band gains on the dry signal, and a filtered-noise path | ±6 dB; noise starts below the loss floor |

R4 (an additive neural post-filter) is not built: it is the escalation if listening says R1–R3 are not
enough.

The five safeguards of section 9.2:
1. **Staging**: stage 1 fits the physics and the recording chain alone; stage 2 switches the residual on
   and lowers the physics learning rate ×0.3 (not frozen).
2. **Budget**: penalties on the per-note corrections and band gains (normalised by their bounds, weight
   0.05 each) and on the residual noise's share of the dry signal's band energy (weight 0.1).
3. **Correction, not replacement**: every output is bounded and zero-initialised; the residual cannot
   create a note.
4. **Ablation as a metric**: validation reports the loss with the residual on and off.
5. **Read the residual**: its outputs are returned by the forward pass (`out["ctx"]`, `out["frame_ctx"]`,
   `out["noise_res"]`) for analysis.

## 3. Initialisation from the recordings

Gradients on frequencies and onset times are only informative close to the answer, so these are
estimated directly before training (`pianonn/fit_init.py`), by comparing renders of the prior with the
recordings of the same MIDI:
- **latency** from onset-strength cross-correlation;
- **tuning** from log-frequency spectral cross-correlation;
- **level and long-term spectrum** per channel → mic gain + a third-octave EQ folded into the body FIR;
- **noise floor** per channel and band from the recordings' quietest frames;
- **inharmonicity and stretch per register** from the isolated notes (`--mined`). For MAESTRO 2018 the
  bass B is about half the prior (5.5e-5 vs 1.2e-4 around MIDI 31–42), which puts high bass partials
  tens of Hz away: out of reach of spectral gradients.

## 4. Speed

The first real-data step took 3.5–15 s (batch 4). Two fixes brought it to about 1.4 s at batch 8:
- `oscbank.py`: the oscillator bank is one autograd function with an analytic backward (checked with
  `gradcheck`): it recomputes `E sin φ` and `E cos φ` once and reduces them with batched matrix products,
  over flattened (example, note) pairs, so each example only pays for its own sounding notes.
- The CUDA caching allocator's reserve grew past the card's 24 GB, and Windows (WDDM) silently spills
  into system RAM (10–40 s steps). The process memory fraction is capped at 0.8 with a GC threshold.

## 5. Trial protocol (phase 1)

- Data: MAESTRO v3, 2018 only (22.9 h train, 3.1 h validation, 1.4 h test), stereo, 24 kHz.
- Examples: 1 s warm-up + 2 s loss window, 12 s lookback, batch 8.
- Go/no-go ladder on one 10 s validation excerpt: (A) recording chain and per-condition scalars,
  (B) + all physics, (C) + residual, 200 steps each from the same initialised prior.
- Main run: 120 min of training compute; stage 2 after 60 %; validation (32 fixed excerpts) every 250
  steps, audio dumps every 1000.
- Evaluation on the test split: MR-STFT loss and log-mel L1 for the untrained prior, the initialised
  prior, the fitted physics, and the fitted physics with the residual; fitted vs tracked inharmonicity
  and stretch; resyntheses.
