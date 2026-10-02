# nn_physical_model_piano

A physically parametrised piano whose parameters are **fitted to MAESTRO by gradient descent** instead of
by a person tuning them by ear. It is a *parametric modal synthesizer*: every note is a sum of closed-form
damped sinusoids whose frequencies, decays and amplitudes come from per-key tables with physically
motivated shapes (stiff strings, coupled unisons, hammer, dampers, pedals, soundboard, hall). Nothing
interacts at run time, so every effect that would emerge from interaction in a simulation like
Pianoteq's (re-strike, phantom partials, sympathetic resonance) is added by hand as a parametrised
feature (review 3, section 2). MAESTRO gives about 200 h of Disklavier recordings with note- and
pedal-level MIDI aligned to about 3 ms, which is exactly the supervision this needs.

The core is about 1.9k interpretable per-key and global physical parameters, 888 more per MAESTRO year
(bridge conductance, per-key colouration, tuning, level and velocity law, contact time, damper delay), an
8.4k table of per-partial corrections held to +-8 dB, and a recording chain per year (two body FIRs, a
parametric hall, mic gains, a noise floor: 14.4k). A small causal context network is the learned
residual: bounded, zero-initialised corrections per note and per frame that absorb what the physics
omits, switched on only after the physics has been fitted. Synthesis is closed-form or a linear
recurrence, and nothing is autoregressive.

## Honest assessment

**Why it should work.** Most of what makes a piano sound like a piano is linear physics
that we can write down: stiff-string inharmonicity, unison mistuning (beats), two-stage
decay, damper timing, sympathetic resonance through the bridge, and a linear
soundboard/room. Spectral and diffusion models have to learn all of this from data and
still tend to get phase-coherent beating and long decays wrong. A physical model gets it
right by construction, runs in real time on a CPU, and stays editable after training
(you can re-voice, re-tune or change the hall).

**What's not new.** Two earlier projects cover part of this ground:
- *DDSP-Piano* (Renault, Roebel & Mignot, DAFx 2022 and JAES 2023) already trained on
  MAESTRO with learned inharmonicity, per-key detuning, a polyphonic context network and
  a learned reverb. It is more physically informed than "a synth as the base".
- *Wave2Midi2Wave* (Hawthorne et al., ICLR 2019, the MAESTRO paper) did MIDI-to-audio
  with a WaveNet.

What is new here is how far the physics goes:
- coupled-string modes (prompt sound and aftersound),
- time-varying damper physics with continuous half-pedalling,
- a real driven sympathetic-resonance bank,
- una corda,
- hammer knock and damper noise,
- adversarial training to recover the attack.

**Where it will be hard (in rough order of pain):**
1. **Frequencies can't be learned from scratch with spectral losses.** Gradients with
   respect to sinusoid frequency are only informative within about one FFT bin (this is
   a known DDSP failure mode). Inharmonicity B, stretch tuning and unison detune
   therefore start from literature priors, with bounded learned offsets. The robust
   version estimates them first from MAESTRO by partial tracking at known MIDI pitches,
   then fine-tunes them.
2. **MAESTRO is about 10 pianos, not one.** Each competition year means a different
   Disklavier, hall and set of microphones. Everything instrument-specific is
   conditioned on year: tuning, gain, velocity curve, brightness and the full IR. You
   choose the piano at render time.
3. **Identifiability.** Strike position, hammer spectrum, soundboard EQ and the room IR
   can trade off against each other. The sound will fit the data, but the parameters
   will only mean what their names say if priors and smoothness regularisation hold
   them in place.
4. **The last 10% is the attack.** Hammer–string contact, soundboard knock and
   longitudinal/phantom partials are what listeners notice, and they are where
   physical models are weakest. The plan is physics for the tonal part, learned noise
   and context corrections for the rest, and a GAN loss, because an STFT loss alone
   averages transients away.
5. **Evaluation.** Use held-out MAESTRO resynthesis (spectral distance, FAD), round-trip
   transcription F1, and listening tests against the real recording and Pianoteq.

## Signal flow

```
MIDI notes + pedals (+ year), with 12 s of history before the rendered window
   │
   ├─ ContextNet (causal GRU over piano roll/pedals) ─► the residual, only in stage 2:
   │      R1 per-note corrections (gain, brightness, decay + tilt, spectral shape)
   │      R2 attack noise (knock spectrum, a slower learned attack component)
   │      R3 per-frame band gains on the dry signal + a filtered-noise path
   ▼
PianoPhysics: per-key params = prior (literature, Iowa, isolated MAESTRO notes) + bounded learned offset
   │   f_n = n f0 √(1+Bn²) · stretch · unison detune          (inharmonic, beating)
   │   α_n = b1 + b3 f_n^p (+ bridge loss g_year(f) for the prompt mode, 1/6-octave detail)
   │   a_n = gain(v) · hammer(f; T_c(v)) · sin(nπx0) · colouration_year(key, f)
   │   phantom partials at 2f_j, f_j + f_j+1 (~a_j a_k, longitudinal emphasis near 15 f1)
   │   damper decay after note-off + delay_year · (1-lift(sustain))^p · (not sostenuto-latched)
   │   re-strike: a new blow takes part of the ringing vibration out
   ▼
Strings: closed-form damped-sinusoid bank (float64 phase, analytic backward), per-oscillator activity
   │ bridge force, panned per key into two channels
   ├─► knock impulse (the contact pulse into the body)
   ├─► SympatheticBank (off by default): 88 keys × 4 partials as resonators with time-varying poles
   ├─► NoiseBank: hammer knock, key-bottom thump, damper noise, pedal noise (bounded decay times)
   ▼
per channel: mic gain · body FIR ⊛ (direct + parametric octave-band hall), per year
   ▼
+ stationary noise floor per year and channel ─► stereo audio
```

| component | params | learned from data |
|---|---|---|
| `physics` | 1.9k + 888 per year + 8.4k `partial_gain` | inharmonicity, tuning, unison detune, loss curves, prompt/aftersound, strike point, hammer cutoff/rolloff/velocity response, damper strength and delay, pedal curve, una corda, re-strike, phantoms, knock impulse; per year: bridge conductance, colouration, scalars; `partial_gain` (±8 dB, stage 2) |
| `context` | 160k | the residual R1–R3 (zero-initialised, stage 2) |
| `symp` / `noise` | ~9k | coupling gains; knock, attack and release spectra and envelopes |
| `room` | 14.4k per year: 2 × 7.2k body taps + hall, gains, per-key pan, 2 × 32 floor bands | soundboard body per mic; hall T60 and level per octave band; noise floor |

## Usage

Training is GPU-first. Install a CUDA build of PyTorch (the default `pip install torch`
gives the CPU wheel), then the package. On a GPU the string bank and the sympathetic resonators run as fused CUDA
kernels, compiled on first use into `pianonn/csrc/build/` (about a minute); that needs the CUDA toolkit's `nvcc` and,
on Windows, the Visual Studio Build Tools (C++). Without them the model renders with the PyTorch code, three to four
times slower ([`docs/speed.md`](docs/speed.md)); `PIANONN_FUSED=0` forces that.

```bash
python -m venv .venv && .venv/Scripts/activate      # or: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124  # CUDA build first
pip install -e .[dev]
pytest

# 1. prepare MAESTRO v3 straight from the zip (stereo FLAC at 24 kHz plus cached MIDI); one year takes a minute,
#    all ten (omit --years) ~26 GB. Scripts without a --years default (pianonn.train, mine_notes, measure_phantoms,
#    overfit_excerpt) use every year present in data/maestro24k, so pass --years to stay on one
python scripts/prepare_maestro.py data/maestro-v3.0.0.zip data/maestro24k --years 2018

# 2. measure inharmonicity and stretch on isolated notes of that year (CPU)
python scripts/mine_notes.py data/maestro24k --years 2018 --out runs/measurements/mined_2018.json

# 3. go/no-go: overfit one excerpt with growing parameter sets
python scripts/overfit_excerpt.py data/maestro24k --years 2018 --out runs/exp1/overfit

# 4. train (stage 1 physics + recording chain, stage 2 + residual after 60 %; --minutes caps the time)
python -m pianonn.train --data data/maestro24k --years 2018 --mined runs/measurements/mined_2018.json --out runs/exp1/train --minutes 120 --batch 8
# or continue from a fitted checkpoint's weights and config (fresh optimiser; phase 3, step 4: runs/phase3/step4/chain.sh)
python -m pianonn.train --data data/maestro24k --years 2018 --init-from runs/phase3/step3/spread/model.pt --out runs/exp1/music --minutes 450 --batch 8 --lr 5e-4 --lr-warmup 200 --stage2-at 0.5 --lr-decay-at 0.5 --lr-final 0.05 --onset-weight 0.5 --onset-pool 0.97 --freeze physics.raw_log_B physics.raw_cents physics.cond_cents room.raw_log_t60 room.band_log_gain room.log_gain

# 5. evaluate on the test split, with the identifiability table
python scripts/evaluate.py runs/exp1/train/best.pt data/maestro24k --years 2018 --mined runs/measurements/mined_2018.json --out runs/exp1/train/eval

# the note bench: isolated notes of the recordings vs each model, by register, velocity and pedal (docs/tone_measures.md)
python scripts/note_bench.py data/maestro24k --years 2018 --out runs/exp1/bench --validate --music 24 --model a=runs/exp1/train/best.pt:physics
# pitch glide (N3) and extra narrow peaks (N12) on the bench; N12 on notes after silence, recordings of every year
python scripts/measure_glide_extra.py data/maestro24k --model runs/exp1/train/best.pt --out runs/exp1/glide_extra
python scripts/survey_after_silence.py data/maestro24k --model runs/exp1/train/best.pt --out runs/exp1/after_silence
# a model's options go after its mode: config overrides and re-applied per-key B, e.g.
#   --model b=runs/exp1/train/best.pt:physics:bridge_end_comb=1,mined=runs/measurements/mined_2018.json
# when the strings' first pulse arrives after the strike, bass and tenor (waveforms, not magnitudes)
python scripts/attack_waveforms.py data/maestro24k --out runs/exp1/attack --model a=runs/exp1/train/best.pt:physics

# compare checkpoints excerpt by excerpt over several noise seeds, and render them in turn for listening
python scripts/compare_runs.py data/maestro24k --out runs/exp1/compare.md --model a=runs/exp1/a/last.pt:residual --model b=runs/exp1/b/last.pt:physics
python scripts/ab_render.py data/maestro24k --out runs/exp1/listen --model a=runs/exp1/a/last.pt:residual --model b=runs/exp1/b/last.pt:physics
python scripts/listen_page.py runs/exp1/listen   # index.html: every excerpt's versions side by side, for a browser

# render (stereo; without --ckpt you hear the untrained prior; --physics-only switches the residual off)
python -m pianonn.render some.mid out.wav --ckpt runs/exp1/train/best.pt --year 2018

# sanity check without data: fit a randomly perturbed copy of the model
python -m pianonn.train --synthetic --out runs/scratch/synthetic

# check a model against the literature targets in docs/physical_parameters.md
python -m pianonn.diagnostics [--ckpt runs/exp1/train/best.pt]
```

Outputs go under `runs/` (not in git except its index, [`runs/README.md`](runs/README.md)): one folder per
experiment, one subfolder per run, each holding its checkpoints, `log.jsonl` (every log line), `train.log` (the
console, if redirected) and `eval/`. Measurements taken from the recordings go in `runs/measurements/`, throwaway
runs in `runs/scratch/`.

The physics prior is specified in [`docs/physical_parameters.md`](docs/physical_parameters.md), with
a source for every value. The literature reviews are in [`docs/literature/`](docs/literature/), the
recording calibration is in [`docs/calibration_iowa.md`](docs/calibration_iowa.md) (reproduce it with
`python -m pianonn.calibration data/iowa`), and the current acceptance report is
[`docs/diagnostics_prior.md`](docs/diagnostics_prior.md).
`samples/demo.mid` is the demo score. Renders are no longer kept in the repository: make
them with `python -m pianonn.render samples/demo.mid out.wav --ckpt ...`, or take the listening sets in
`runs/round2/listen/` and `runs/round2/ab/`.

## Status

The documents, in order:
- [`docs/plan_phase0_1.md`](docs/plan_phase0_1.md): the plan that took the project from the reviews to its
  first fit on real audio;
- [`docs/trial_2018.md`](docs/trial_2018.md): the first trial on MAESTRO 2018, corrected after review 4;
- [`docs/plan_round2.md`](docs/plan_round2.md): the response to review 4, with a new loss, floor, budget and
  evaluation scale;
- [`docs/round2_results.md`](docs/round2_results.md): the re-fit with a control branch and a GAN branch.
- [`docs/tone_measures.md`](docs/tone_measures.md): what a piano note is made of, the measures that follow,
  and what each loss term can see (the design for round 3); sections 9–11 report the note bench built from it,
  its validation and what it found, and 10.4 / 11.4 the physics it points to.

In short, on one year (RTX 3090):
- **Distance.** On 96 held-out test excerpts, the round-2 loss (log band energies, fine and attack terms, no
  spectral convergence) matches in 50 minutes the trial's 220-minute log-mel distance: 3.27 vs 3.25 dB, from
  5.88 dB at the measured starting point.
- **Level.** The level is right to within 0.5 dB broadband and 1 dB per octave band, except 4 kHz (−1.8 dB). The
  trial's model was 2.4 dB low, because of its loss.
- **The residual.** It beats a physics-only control by more than the noise-seed spread, but the physics leans
  on it (it takes over turning the knock down). Its noise path stays silent: the per-frame loss shuts it.
- **Frequencies.** Inharmonicity and tuning are measured on isolated notes and frozen: no loss term sees them
  well enough to fit them.
- **The note bench** (isolated notes of the recordings against the models' renders, every measure validated
  on synthetic notes and on known changes pushed through the renderer) found what listening heard:
  - the attack has the wrong spectrum in every register: too much at 8 kHz everywhere (the fitted knock noise),
    too little 250 Hz thump in the bass (−4…−5 dB) and 1 kHz knock in the middle (−10 dB), a low thud too strong
    in the treble (+13…+18 dB);
  - the hall decays 10–30 % too fast;
  - at equal key and velocity the model's notes vary 30–70 % as much as the piano's (step 3 closes part of it);
  - phantom partials are too weak from the bass up to C♯6;
  - the model's soundboard filter has narrow modes that ring after every note.

  Re-strikes, dampers, the touch precursor and the level are close; the pitch glide is real but small; duplex
  strings cannot be told apart from other faint components.
- **Phase 3, steps 0–1** (no training): the hall's T60 and level set from free decays, the soundboard filter's
  narrow modes capped (12.4; the notes come out 1.6–1.7 dB loud in R3–R5 and their onset rise too slow until step 2);
  per-key B fixes a 15–20 % error from MIDI 43 to 63 (the recordings' partials 10–30
  sat ~5 cents sharp of the model's); in the bass the recordings' tone starts about (1 − x0) T/2 after the strike, as
  the bridge end's force predicts, where the round-2 models start it at x0 T/2.
- **Phase 3, step 2 in part** (fits on isolated notes, 12.5): a per-year velocity map with per-key level and
  brightness brings the level within 1.2 dB in R2–R5 and cuts the brightness excess in R2 from +209 to +66 cents; a refit of the existing
  attack parts removes a 10–20 dB low knock excess in R6–R7, but isolated notes cannot set the knock above ~2.5 kHz
  (it barely rises over the background there), and one knock decay per key cannot be right in both low and high bands.
  On music, every attack-aware loss term puts the knock 4.5–7 dB lower than the note fit left it.
- **An onset-aligned attack term for training** (12.6, 12.8; `losses.OnsetLoss`, `train.py --onset-weight
  --onset-pool`, off by default): windows at each note's expected sound onset see the knock impulse below 400 Hz and
  pin the contact time better than the training loss. Pooled over one batch of 2 its bias gate fails (up to 3.2 dB on
  512 segments); pooled across steps (a running pool of ~32 batches) it passes (0.1–0.7 dB). A relative form (the
  attack re the same onset's 30–100 ms) passes too but moves nothing a run would fit. No band-level term at onsets sets
  the knock's level in music.
- **Phase 3, step 3** (12.7): at equal key and velocity the piano's notes differ strike to strike (not key to key) in
  level (R2–R3), brightness, attack, early decay and onset timing, nearly independently. The model now draws a
  per-note offset for each (config `strike_*`; brightness and decay keep the note's level), with spreads set from
  the bench by a variance match, not fitted: onset timing and level now vary as the piano's, brightness and early
  decay part of the way; the attack at 1–2 kHz still varies less (no model part moves it alone; the knock barely
  reaches it). Medians stay; the training loss against the recordings rises 2–6 % (a random model loses to its median
  under L1). Checkpoint `runs/phase3/step3/spread/model.pt`.
- **Phase 3, step 4** (12.9): one 450-minute run on music (2018) from the step-3 checkpoint, 17,840 steps. On 96 held-out
  test excerpts it is the nearest model to the recordings so far: total 0.691 (physics alone; 0.686 with the residual),
  log-mel 3.19 dB, against the round-2 control's 0.719 and 3.35 dB, better on every term beyond two standard errors;
  steps 1–3 alone had cost 0.053 on music. On held-out isolated notes the brightness excess in R2–R3 is gone (+204 /
  +160 → +7 / −72 cents), the level is within 1 dB in R2–R5 at every velocity, the 8 kHz attack excess roughly halves
  (+9…+14 → +5…+10 dB), the bass thump and the phantoms in R4–R5 come close; the level in music is within 0.7 dB in
  every octave band (energy). Left: the quiet stretches of music 0.5–2 dB quieter than the piano's, the 8 kHz attack
  still too strong, the onset rise too fast in R4–R5, phantoms weak in R2. The residual adds 0.005; rendered with the
  per-strike variation on, the distances read 0.032 higher (a random model under median-seeking terms). Checkpoint
  `runs/phase3/step4/train/last.pt`; listening in `runs/phase3/step4/listen/` (the per-strike variation on).
- **The residual's ceiling** (`docs/tone_measures.md` 13): the context net sees the MIDI history but not what the physics
  renders, fixes each note's corrections at its onset, and trained late under a budget; it adds 0.006 on test. Free
  outputs in its own language, fitted per test excerpt on the frozen step-4 model and scored on unseen noise seeds,
  take off 0.136 per note, 0.219 per frame, 0.268 both (24 excerpts): the output language is not the limit. The gain is
  per note (a constant correction gives none); how much of it the context can predict is open.
- **A residual that sees the physics** (13.2, `pianonn/residual.py`, `residual_kind=aware`): each note's expected energy per
  octave group of partials, attention across the sounding notes, a gain curve per group over each note's life inside
  the oscillator bank. 30 min on the frozen step-4 physics: −0.010 ± 0.004 on 96 test excerpts (the GRU residual
  −0.005); the reachable fraction is unknown, since the ceiling includes strike-to-strike variation no context
  predicts (review 5). Fitted to 8 excerpts alone it reaches 86 % of the full ceiling there (13.3): capacity only;
  whether the corrections carry over to new pieces is open.
- **Phase 4** (`docs/tone_measures.md` 14): the attack in three parts (knock noise capped at 2.5 kHz, a thump to 2 kHz,
  a string-borne precursor; smooth rises and decays per band and register; config `attack_model=parts`), fitted on
  isolated notes (held out 3.064 → 2.997 dB) and frozen; a gain per training piece and the whole-excerpt level term
  (`--piece-gain --level-weight`); a cheap sympathetic bank (MIDI 21–59, partials below 2.5 kHz, at 6 kHz). One 50-min
  run: −0.003 ± 0.003 on 96 test excerpts against step 4; the 8 kHz attack excess on isolated notes drops in R2, R5
  and R6 (R6 N4 +9.5 → +3.7 dB), but in music the attack is as abrupt as before (E4), and the bank sits 32–45 dB under
  the strings, so the pedal halo gap is unchanged. Checkpoint `runs/phase4/train_run/train/last.pt`; listening in
  [`samples/phase4/`](samples/phase4/).
- **Isolated tenor notes, partial by partial** (`docs/tone_measures.md` 15, `scripts/note_profile.py`): on 56 isolated
  R3 notes the average model note is close (partial levels within ±2 dB at 50–900 ms, beating as strong, fast and
  irregular), but the notes are too much alike (the spread of the partials' late decays across notes 3–8 dB/s against
  6–24 in the recordings), the fundamental fades too fast early (−15 vs −6 dB/s) and the middle partials peak 15–35 ms
  too soon. A/B files in [`samples/r3_notes/`](samples/r3_notes/).
- **Phase 5** (`docs/tone_measures.md` 16): an onset measure found the model's notes switch on where the piano's build up
  over 10-35 ms, the same at every velocity: the soundboard's ring-up, now in the body (config `body_ring_ms`). After
  one run the attack in music is no longer too abrupt in the bass and tenor (E4) and the onsets are closer in most
  registers; the distances on test excerpts read +0.027 against phase 4, which per-bin terms charge for any fine
  structure not at the recording's own frequencies. Also: the onset jitter re-measured (~6 ms), per-partial strike
  variation, the decays fitted on isolated notes with a per-partial term; the fundamental's early fade is the body's
  low-band ringing (fixed on notes, back after training); the pedal-halo gap is an excess with the pedal up. Checkpoint
  `runs/phase5/train_run/train/last.pt`; listening in [`samples/phase5/`](samples/phase5/).
- **Phase 6** (`docs/tone_measures.md` 17): a note that rang too long led to a measure of the whole note's envelope in
  music (N13: each partial's fade to 2.5 s) and a map of what every measure sees. The upper partials rang 4-7 dB too
  long after 1 s; their decays were fitted on N13 and the term kept in the training loss, the decays free. After a
  310-min run the excess is about halved, the music loss is slightly better (−0.0084 on test excerpts) and does not
  pull against the envelope term. Checkpoint `runs/phase6/train_run/train/last.pt`; listening in
  [`samples/phase6/`](samples/phase6/).
- **Review 6** ([`docs/reviews/review_6_training.md`](docs/reviews/review_6_training.md), 2026-10-02): the training loss
  calibrated on 96 validation excerpts (13 pieces). The model against the recording reads 0.86 with the per-strike
  variation off and 0.91 with it on; against another draw of itself 0.59 and 0.65: a gap of about 0.26, of which 0.05
  is each piece's recording level. The floor is a stand-in (it depends on the variation set by hand). Phases 4–6 read
  0.846, 0.873, 0.860. 75 % of the loss's weight is on notes younger than 0.5 s, 12 % after 1 s; the unison mistuning
  is still its random draw (correlation 0.98). A critique corrected several of the review's readings (its section 7):
  the loss is not shown to be *the* cause, and what makes the renders sound synthetic is not known. The proposal for
  moving forward is its section 8 (not built): listen to excerpts fitted with the residual's current and extended
  outputs, then one long run of the residual from step 0 at the full rate without the budget, and the note
  objectives (attack, early level, fades) moved from fits before training into the training batch with every
  parameter free; the mined frequencies, the initialisation from data and the bench stay.
- **The composite score** (review 6, section 11, 2026-10-02; uncommitted): one training loss with read-by-read and
  pooled comparisons (`pianonn/partial_view.py`, `pianonn/composite.py`), checked on known changes
  (`scripts/score_check.py`). The main bias found was the render, not the score: trained without the per-strike
  variation, every term pulls the aftersound ~4 dB low against varied takes; as an energy score with the variation on
  the pulls shrink. Training on it (`--score composite --energy`) with a wider residual (a curve per partial, noise
  per note, random inputs): the first smoke run diverged (pools across steps lagged the residual); the second, 30
  min, lowers the composite on held-out and training pieces level matched per piece (0.726 → 0.690 held out), through
  the pooled, level and onset terms; the band and partial terms do not move, and unmatched the held-out score rises
  (0.770 → 0.806: the model got louder). The setup is written up in
  [`docs/composite_score.md`](docs/composite_score.md); listening in [`samples/composite_smoke2/`](samples/composite_smoke2/).
- **Training speed** ([`docs/speed.md`](docs/speed.md), 2026-10-02): the composite training step from 4.5 s to 1.0 s on the
  same batches: the CUDA allocator back at its default (its garbage-collection and split settings cost 1.2 s), the
  string bank and the sympathetic resonators as fused CUDA kernels with analytic backwards (`pianonn/csrc`; the PyTorch
  code stays the reference and the fallback), the score reading the render once for both of the energy score's
  comparisons. Audio and gradients agree with the reference to float precision (`tests/test_cuda.py`). Also fixed:
  after running out of memory once, `train.py` kept the failed step's graph and skipped every later batch.

The prior is calibrated against the literature (the KTH *Five Lectures on the Acoustics of the Piano*,
arXiv and Zenodo papers) and against 260 recorded notes of a Steinway B, analysed with the same code as the
model's renders ([`docs/calibration_iowa.md`](docs/calibration_iowa.md)); per-year inharmonicity and
stretch now come from isolated MAESTRO notes. Reviews are in [`docs/reviews/`](docs/reviews/); every open
finding of reviews 1–3 is either fixed or explicitly deferred in the plan.

Phase 3, in this order ([`docs/tone_measures.md`](docs/tone_measures.md), 12.1; the physics from 10.4 and 11.4).
Each step is checked on the note bench with re-renders; one training run comes after step 3:
- [x] 0. Fixes that move the baseline: per-key B (the round-2 models' B was 15–20 % low from MIDI 43 to 63; now on by
      default in `apply_mined_priors`); the strike comb's sign (the bridge end's force; the recordings' bass notes
      side with it, 12.3; config `bridge_end_comb`, off for older checkpoints, proposed on for the next run).
- [x] 1. Set from measurement: hall T60 and level from free decays (P4; T60 was 12–20 % short, now within −7…+3 % on
      held-out decays); the soundboard filter's Q capped at 50 (config `body_q_max`; its recurring narrow peaks go from
      37 to 16 of the model's extra peaks). Side effects left for step 2: notes 1.6–1.7 dB loud in R3–R5, the onset
      rise too slow (R4 median 82 vs 49 ms). Checkpoint `runs/phase3/step1/hall/model.pt` (12.4).
- [ ] 2. Fit on isolated notes, per-note terms aligned to each note's onset (`scripts/fit_notes.py`): done, the
      velocity map and a refit of the existing attack parts (checkpoint `runs/phase3/step2/knock_2k5/model.pt`, 12.5).
      Open: the attack in three parts (a string-borne precursor up to ~5 kHz, a structure thump limited to ~2 kHz with
      the treble's long low ringing, no knock noise above that; the refit shows one knock decay per key is too few)
      with a per-register velocity law; phantoms from (j−1, j+1) pairs.
- [x] 3. Per-strike variation, its spread set from the bench by a variance match (`scripts/strike_spread.py`; config
      `strike_*`; checkpoint `runs/phase3/step3/spread/model.pt`, 12.7). Left: the attack at 1–2 kHz varies ~70 % as
      much as the piano's.
- [ ] Alongside: measures T1–T3 (two-stage decay, beating, pedal and decay) from partial fits with phase; E2, E5.
- [x] 4. One training run on music (12.9): 450 min from the step-3 checkpoint with warm-up and a cosine decay (a constant
      rate lets the level parameters wander with each batch's pieces), the onset-aligned attack term with a running
      pool (`--onset-weight 0.5 --onset-pool 0.97`, 12.8), the hall frozen at its measurement, the per-strike variation
      off in training and on in rendering (12.7); checkpoint `runs/phase3/step4/train/last.pt`. Not in it, still open:
      per-key unison mistuning with per-key long-window terms, the sympathetic bank for the bass and tenor, a
      whole-excerpt band-level term (in energy the octave bands are now within 0.7 dB; the quiet stretches read 0.5–2
      dB low), texture.
- [ ] 5. All years: mine isolated notes per year (B, stretch, velocity curve, damper delay, T60); the per-key stretch
      offset is still shared across conditions.
Phase 5 (`docs/tone_measures.md` 16), one round with one training run at the end:
- [x] 1. The first 20 ms (`measures.onset_profile`, `scripts/onset_profile.py`, 16.1): the model's tone switches on
      where the piano's builds up over 8-35 ms, the same at every velocity: the board's ring-up, missing from the fitted
      body. Built: config `body_ring_ms` (a decaying noise kernel per octave band of the body, its fine structure divided
      out at the partials). The onset jitter matches at 5.7-6.1 ms per strike in R2-R5, not 3.3.
- [x] 2. The medians on isolated notes (16.2): `--fit onset` and `--fit decay` (a per-partial note term); small held-out
      gains (3.734 → 3.695 dB). The fundamental's fast early fade is the body's low-band ringing, not the strings:
      config `body_q_max` per band, Q 20 at 125-250 Hz.
- [x] 3. Per strike and partial (16.3): config `strike_partial_decay`, `strike_after`; the late decay still varies about
      half as much as the piano's.
- [x] 4. The sympathetic bank (16.4): its level does not move the pedal halo (E1), and the E1 gap is an excess with the
      pedal up, not a missing halo; nothing changed.
- [x] 5. One training run (16.6): +0.027 ± 0.005 on 96 test excerpts against phase 4 (the ring-up's fine structure and
      the Q change under the per-bin terms, 16.5), but the attack in music is no longer too abrupt in the bass and tenor
      (E4 8 kHz R2-R4 1.2/1.1/2.1 → 0.2/0.9/1.2 dB, recordings 0.3/0.5/0.5) and closer in the treble; the fundamental's
      early fade came back in training. Listening in [`samples/phase5/`](samples/phase5/).
Phase 6 (`docs/tone_measures.md` 17), the whole note:
- [x] 1. N13, the whole note's envelope in music (`scripts/note_envelope.py`, 17.3): partials to 40 / 6 kHz at 0.1-2.5 s.
      On phase 5 the upper partials ring too long (9+ fade 4-6 dB too little by 1 s, 6-7 by 1.5 s; R3 about +10);
      the fundamental is right; the per-strike spread is not too wide. The coverage map of every measure (17.2).
- [x] 2. The envelope fit (`scripts/fit_envelope.py`, 17.4): the strings' decay and the aftersound on N13's fades.
- [x] 3. One long run (310 min) with the envelope term in the training loss and the decays free (`--env-weight`,
      17.5-17.6), the aware residual in stage 2. The music loss does not pull against the envelope term (gradient
      cosine about −0.01): it barely sees the tails. The upper partials' excess ring is about halved (N13 9-12 at 1.5 s
      +6.4 → +2.0 dB; R3 partial 2's late decay now right), and on 96 test excerpts −0.0084 ± 0.0036 against phase 5.
      Checkpoint `runs/phase6/train_run/train/last.pt`; listening in [`samples/phase6/`](samples/phase6/).
Then: all years (5 above), the residual's read-out and budget, texture.

- [ ] Deferred on measurement: pitch glide (~1.5–3 cents at *ff*), duplex strings. Later: felt model at note-on,
      Weinreich eigenmodes, evaluation suite (FAD, transcription F1, listening tests), 48 kHz stage, real-time
      C++/JUCE engine.
