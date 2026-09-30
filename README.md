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
gives the CPU wheel), then the package:

```bash
python -m venv .venv && .venv/Scripts/activate      # or: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124  # CUDA build first
pip install -e .[dev]
pytest

# 1. prepare MAESTRO v3 straight from the zip (stereo FLAC at 24 kHz plus cached MIDI); one year takes a minute
python scripts/prepare_maestro.py data/maestro-v3.0.0.zip data/maestro24k --years 2018

# 2. measure inharmonicity and stretch on isolated notes of that year (CPU)
python scripts/mine_notes.py data/maestro24k --years 2018 --out runs/mined_2018.json

# 3. go/no-go: overfit one excerpt with growing parameter sets
python scripts/overfit_excerpt.py data/maestro24k --years 2018 --out runs/overfit

# 4. train (stage 1 physics + recording chain, stage 2 + residual after 60 %; --minutes caps the time)
python -m pianonn.train --data data/maestro24k --years 2018 --mined runs/mined_2018.json --out runs/trial --minutes 120 --batch 8

# 5. evaluate on the test split, with the identifiability table
python scripts/evaluate.py runs/trial/best.pt data/maestro24k --years 2018 --mined runs/mined_2018.json --out runs/trial/eval

# render (stereo; without --ckpt you hear the untrained prior; --physics-only switches the residual off)
python -m pianonn.render some.mid out.wav --ckpt runs/trial/best.pt --year 2018

# sanity check without data: fit a randomly perturbed copy of the model
python -m pianonn.train --synthetic --out runs/synthetic

# check a model against the literature targets in docs/physical_parameters.md
python -m pianonn.diagnostics [--ckpt runs/trial/best.pt]
```

The physics prior is specified in [`docs/physical_parameters.md`](docs/physical_parameters.md), with
a source for every value. The literature reviews are in [`docs/literature/`](docs/literature/), the
recording calibration is in [`docs/calibration_iowa.md`](docs/calibration_iowa.md) (reproduce it with
`python -m pianonn.calibration data/iowa`), and the current acceptance report is
[`docs/diagnostics_prior.md`](docs/diagnostics_prior.md).
`samples/` has renders of `samples/demo.mid`: `physics_prior_v1.wav` is the first guess and
`physics_prior_v2.wav` is the first literature pass, `v3` has the review fixes, `v4` is the v2
calibration, and `v5` is the corrected v3 calibration (fast three-string prompt decay). `v6` is the v4
per-partial calibration (steeper hammer top, high partials that sustain, frequency-dependent bridge loss), and
`ab_bass_notes_v5_v6.wav` plays A1, C2 and C3 at mf three times each: the Iowa Steinway recording, v5, then v6
(each through the soundboard body only, no hall, loudness-matched).

## Status

The plan that took the project from the reviews to its first fit on real audio is in
[`docs/plan_phase0_1.md`](docs/plan_phase0_1.md), and the first trial on MAESTRO 2018 is reported in
[`docs/trial_2018.md`](docs/trial_2018.md).

The prior is calibrated against the literature (the KTH *Five Lectures on the Acoustics of the Piano*,
arXiv and Zenodo papers) and against 260 recorded notes of a Steinway B, analysed with the same code as the
model's renders ([`docs/calibration_iowa.md`](docs/calibration_iowa.md)); per-year inharmonicity and
stretch now come from isolated MAESTRO notes. Reviews are in [`docs/reviews/`](docs/reviews/); every open
finding of reviews 1–3 is either fixed or explicitly deferred in the plan.

Next steps:
- [ ] Longer training on one year; then all years (per-condition tables are in place; the per-key stretch
      offset is still shared across conditions).
- [ ] Mine isolated notes for every year: per-year B, stretch, velocity curve, damper delay.
- [ ] Switch the sympathetic bank on in a later stage; then the GAN, after listening.
- [ ] Felt model at note-on instead of the hammer spectrum table (review 3, 9.1); Weinreich eigenmodes if
      the fitted aftersound tables look unphysical; pitch glide at *ff*.
- [ ] Evaluation suite (FAD, transcription F1, listening tests).
- [ ] 48 kHz stage and more bass partials; real-time C++/JUCE engine (recursive oscillators, online damper
      integral, partitioned convolution, the context GRU at 200 Hz).
