# nn_physical_model_piano

A piano built like Pianoteq (strings, hammer, dampers, pedal, sympathetic resonance,
soundboard) whose parameters are **fitted to MAESTRO by gradient descent** instead of by
a person tuning them by ear. MAESTRO gives about 200 h of Disklavier recordings with
note- and pedal-level MIDI aligned to about 3 ms, which is exactly the supervision this
needs.

The core is about 7k interpretable physical parameters. A small causal context network
(about 145k params) predicts bounded corrections, and a soundboard body plus a parametric
hall is kept for each recording condition. Synthesis is closed-form or a linear
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
MIDI notes + pedals (+ year)
   │
   ├─ ContextNet (causal GRU over piano roll/pedals) ─► bounded per-note corrections
   │                                                    (gain, brightness, decay, knock)
   ▼
PianoPhysics: per-key params = literature prior + bounded learned offset (docs/physical_parameters.md)
   │   f_n = n f0 √(1+Bn²) · stretch · unison detune          (inharmonic, beating)
   │   α_n = b1 + b3 f_n²  (+ bridge loss for the prompt mode)  (two-stage decay)
   │   a_n = gain(v) · half-sine pulse(f; T_c(v)) · sin(nπx0)   (bridge force; velocity, strike point)
   │   damper decay α_d,n × (key up) · (1-lift(sustain))^2.5 · (not sostenuto-latched)
   ▼
Strings: closed-form damped-sinusoid bank, attack ramp over the contact time,
   │     chunked + checkpointed (exact for pedalling; notes can start before the window)
   │ bridge force
   ├─► SympatheticBank (off by default, see roadmap): 88 keys × S partials as resonators with time-varying poles
   │                    (dampers), driven by the bridge minus the key's own strings
   ├─► NoiseBank: hammer knock, key-bottom thump, damper noise, pedal noise
   ▼
Soundboard body FIR (modal, 60-70 Hz high-pass) ⊛ (direct + parametric octave-band hall), per year
   ▼
audio
```

| component | params | learned from data |
|---|---|---|
| `physics` | ~7k | inharmonicity, tuning, unison detune, loss curves, prompt/aftersound, strike point, hammer cutoff/rolloff/velocity response, damper strength, pedal curve, una corda, per-partial residual |
| `context` | ~145k | per-note corrections (zero-initialised, so training starts from pure physics) |
| `symp` / `noise` | ~6k | coupling gains; knock and release spectra and envelopes |
| `room` | 7.2k body taps + 15 hall params per condition | soundboard body; hall T60 and level per octave band, for each MAESTRO year |

## Usage

```bash
pip install -e .[dev]
pytest

# 1. prepare MAESTRO v3 (about 100 GB download; writes mono FLAC at 24 kHz plus cached MIDI)
python scripts/prepare_maestro.py /path/to/maestro-v3.0.0 data/maestro24k

# 2. train (add --adv-start N to switch on the GAN loss after N steps)
python -m pianonn.train --data data/maestro24k --out runs/v0

# sanity check without data: fit a randomly perturbed copy of the model
python -m pianonn.train --synthetic --out runs/synthetic

# 3. render (without --ckpt you hear the untrained physics prior)
python -m pianonn.render some.mid out.wav --ckpt runs/v0/last.pt --year 2018

# check a model against the literature targets in docs/physical_parameters.md
python -m pianonn.diagnostics [--ckpt runs/v0/last.pt]
```

The physics prior is specified in [`docs/physical_parameters.md`](docs/physical_parameters.md), with
a source for every value. The literature reviews are in [`docs/literature/`](docs/literature/), the
recording calibration is in [`docs/calibration_iowa.md`](docs/calibration_iowa.md) (reproduce it with
`python -m pianonn.calibration data/iowa`), and the current acceptance report is
[`docs/diagnostics_prior.md`](docs/diagnostics_prior.md).
`samples/` has renders of `samples/demo.mid`: `physics_prior_v1.wav` is the first guess and
`physics_prior_v2.wav` is the first literature pass, `v3` has the review fixes, and `v4` is calibrated
against the sourced literature and the Steinway recordings.

## Status

The package runs end to end and has tests. The prior is calibrated against the literature (the KTH
*Five Lectures on the Acoustics of the Piano*, arXiv and Zenodo papers) and against 260 recorded
notes of a Steinway B, which are analysed with the same code as the model's renders
([`docs/calibration_iowa.md`](docs/calibration_iowa.md)). It passes 62 of 65 acceptance checks;
the three failures are a known limitation of the bass knee metric. Reviews are in
[`docs/reviews/`](docs/reviews/). Gradients reach every
physical parameter, block-wise rendering matches single-pass rendering, dampers, sustain,
una corda and sympathetic resonance all behave as expected, and a student fitted to a
perturbed teacher moves towards it. **It has not been trained on MAESTRO yet.**

Next steps:
- [ ] Estimate B, tuning and decays from MAESTRO directly (partial tracking at known pitches) and use them as the prior.
- [ ] First real training run on a single year, then all years.
- [ ] Weinreich coupled-string eigenmodes instead of the mode-space shortcut.
- [ ] Initial pitch glide at *ff* (tension modulation), longitudinal modes and phantom partials.
- [ ] Re-strike interaction on a string that is still vibrating.
- [ ] Calibrate the damper delay, the damper boundary key and hall T60s per year from MAESTRO.
- [ ] Stereo output.
- [ ] Speed up the sympathetic bank's scan (about 90% of a CPU training step; profile on GPU first), then re-enable it (`use_sympathetic=True`).
- [ ] Real-time C++/JUCE engine: recursive two-pole resonators replace the training-time closed form.
- [ ] Evaluation suite (FAD, transcription F1, listening tests).
