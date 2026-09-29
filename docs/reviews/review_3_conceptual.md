# Review 3: approach, physics completeness, training idea, tests, and a plan (commit 356b461)

Scope of this review: not "is the spec copied into the code" (reviews 1 and 2 did that) but
whether the project is pointed at the goal. The goal is a MIDI-to-audio piano that is
convincing "in all its glory", later a real-time VST, with physics as the bedrock and a
network plus MAESTRO closing the gap.

Everything measured here was run on this machine (RTX 3090, torch 2.6 cu124). Scripts:
`docs/reviews/review_3_scripts/`. `pytest`: 22/22 pass, which is table stakes and not
otherwise discussed.

## 1. Verdict

**The approach is sound and the choice of model family is the right one for the goal.** A
differentiable, physics-parametrised synthesizer fitted to recordings is the only family that
gives real-time synthesis, editability (re-voice, re-tune, change the hall) and data-driven
timbre at once. Sampling gives fidelity but no continuity; pure hand-tuned physical
modelling (Pianoteq) tops out at "very good synth"; neural audio (WaveNet, diffusion, codec
language models such as MIDI-VALLE 2025) gives the most recording-like timbre but is not
real-time, not controllable and not editable. This project sits where DDSP-Piano sits,
with more physics. That is the correct place to sit.

**But the project has spent its effort on the part that was never in doubt.** Twelve commits
and three spec versions calibrate the *linear sustain* of a Steinway B from isolated notes.
The linear sustain is exactly what the README itself says physics gets right by construction.
The things that decide whether the result sounds like a piano or like a very good synth (the
attack, the fit to real performance recordings, the loss, identifiability, stereo, the
missing nonlinear effects) are untested. Nothing has been fitted to a single second of a real
performance yet, and the first attempt will hit a loss pathology that I measured (section 4.1).

**What should happen next** is a go/no-go experiment on real MAESTRO audio (section 7,
phase 1), with a small number of fixes first (phase 0). Not more Iowa calibration.

## 2. What the model actually is, and what that implies

The README says "a piano built like Pianoteq". It is not, and the difference matters for
what "sufficiently complete physics" means.

Pianoteq simulates: a nonlinear hammer-felt contact interacting with a string that is
already moving, waves propagating on a stiff string, strings coupled through a bridge with a
frequency-dependent admittance, a soundboard. Two-stage decay, beating, velocity-dependent
brightness, re-strike behaviour and phantom partials *emerge* from the simulation.

This model is a **parametric modal synthesizer**: every note is a sum of closed-form damped
sinusoids whose frequency, amplitude and decay are read from per-key tables with physically
motivated shapes (`physics.py`). Nothing interacts with anything at run time. Two-stage decay
is a table (`prior_prompt_ratio`), the hammer is a spectral envelope (`hammer_spectrum`), the
soundboard is a FIR on the summed output, sympathetic resonance is a separate resonator bank.

This is the right design for gradient-based fitting (closed-form, chunkable, exact under
pedalling, cheap) and for a real-time engine (recursive oscillators). It has one consequence
that the plan must own: **any effect that would emerge from interaction has to be added by
hand as a parametrised feature, and anything not added is simply absent, forever.** So
"complete enough" is a checklist, and section 3 is that checklist.

The second consequence: because the parameters are *bounded* offsets on priors, the model
cannot fit what its forms cannot express. Where a form is wrong, the fit will push the
error into whatever is flexible enough to absorb it (the 88x96 `partial_gain` table at
+-26 dB, the 7200-tap body FIR, the context net). Section 4.3 is about that.

## 3. Are the physics correct and sufficiently complete?

Checked against what listeners and the acoustics literature say a grand piano is. Status:
**yes** = modelled in a physically meaningful form; **param** = present as a parametrised
shape that fitting can tune but that does not follow from physics; **no** = absent.

| # | what a piano does | status | assessment |
|---|---|---|---|
| 1 | inharmonic stiff-string partials, stretch tuning | yes | correct form; the sounding-fundamental convention is right; bounded offsets (x/4.5 on B, +-30 cents) are wide enough |
| 2 | unison mistuning, beats, two-stage decay | param | mode-space shortcut: prompt mode + 2 aftersound modes, per-key random mistuning. Adequate for the sound. Real Weinreich coupling also makes the *phase* between modes and the knee depth depend on mistuning and bridge admittance; here the knee (`prior_log_after`) is a free table, so the fit can still get it |
| 3 | frequency-dependent decay (internal, air, bridge) | yes / param | b1 + b3 f^p plus a bridge-conductance curve g(f). Correct structure. **g(f) has 9 knots over 8 octaves and is global.** Real soundboards impose resonance-scale (1/12-octave) variation of decay across neighbouring partials, the very thing the Iowa comparison flagged (C2-B2, and the within-note spread 3.4 vs 5.3 dB). This is the bridge admittance and it is per instrument. A finer, per-condition g(f) is cheap and physically meaningful (section 7) |
| 4 | hammer: velocity-dependent brightness, contact time | param | two-corner envelope with velocity-dependent order, fitted. Fine as a spectral model. The felt nonlinearity's other signatures (multiple contacts in the bass, pulse skew) are not represented; low priority |
| 5 | strike position comb | yes | signed sin(n pi x0). Finite hammer width (extra roll-off) is absorbed by the hammer envelope; fine |
| 6 | per-key soundboard colouration (bridge position differs per key) | param | the global body FIR cannot do this; `partial_gain` (per key, per partial) *is* effectively this. It should be regularised as a smooth function of *frequency* across neighbouring keys, not just kept small |
| 7 | sympathetic resonance | yes, off | 88 x 4 resonators driven by the bridge minus own key; the dominant effect (undamped treble and lifted strings ringing at coincident partials) is covered. Costs 2x per step (4.2 vs 2.0 s). Keep off for the first fits, on later |
| 8 | dampers: delay, partial-dependent rate, half pedal, sostenuto, undamped top | yes | good, and exact under time-varying pedal in closed form. Damper delay fixed at 15 ms (should become per-condition, as the spec says) |
| 9 | una corda | param | gain, aftersound, contact time; reasonable |
| 10 | **re-strike of a ringing string** | **no** | the second strike is added to the first. Under pedal, repeated notes and tremolos accumulate energy and get louder than any real piano. MAESTRO is full of this. Needs a fix before training on dense music: treat a re-strike as a damper-like event on the earlier instance of the same key (an extra term in the closed-form damping integral, cheap) |
| 11 | **longitudinal modes and phantom partials** | **no** | the bass "growl" at f and ff and the metallic edge of loud low notes. Amplitude grows as velocity squared. Without it, loud bass will sound like a soft bass turned up. Parametric addition is feasible (a second partial set at sum/difference frequencies of transverse partials, gain ~ v_h^2, per-key level) |
| 12 | pitch glide at ff (tension modulation) | no | audible only in loud bass, small; defer |
| 13 | hammer knock, key-bottom thump, damper and pedal noise | param | time-domain band noise with exponential envelopes, per-key spectra. Good enough to start. The knock's *modal* character (soundboard low modes rung by the impulse) comes through the body FIR. What is missing is per-note variability and any *learned* residual noise (section 4.4) |
| 14 | **stereo** | **no** | the recordings are stereo; the instrument is wide; the room is stereo. Also a training-target problem, see 4.2 |
| 15 | bandwidth | partial | 24 kHz, 96 partials: A0 tops out near 4.8 kHz; a real ff bass note has energy to 8-10 kHz (the calibration itself measured partials to 10 kHz). Right for a first fit; the end product needs 48 kHz and ~150 partials in the bass |
| 16 | per-note randomness (same key, same velocity, not identical) | no | minor; the noise seed varies, the tone does not |
| 17 | soundboard and room: body FIR + parametric hall per year | param | sensible and identifiable. The hall is a fixed noise IR with 7 T60s and gains; real halls have early reflections and a stereo image. Adequate for mono training |

Summary: **the linear sustain is complete and correct in form.** The gaps are 10, 11, 14 and
the resolution of 3 and 6. Items 10 and 11 are not polish; they are what separates "piano"
from "electric piano with long decay" on loud, pedalled music, which is most of MAESTRO.

The Iowa calibration validated the *forms* (two hammer corners, p about 1.1 for the aftersound
loss, additive bridge loss). That was worth doing once. The per-partial *values* are
Steinway-B, close-miked, and will be overwritten by the MAESTRO fit; further tuning of them is
wasted. Review 2's "transfer" list still holds: keep the forms and the frequency priors,
loosen the rest.

## 4. Does the training idea hold up?

### 4.1 The loss will be dominated by the recordings' noise floor (measured, high priority)

The model renders digital silence between partials and after notes. Recordings have a noise
floor (hall, audience, preamps). `MultiResolutionSTFTLoss` uses log-magnitude L1 with
eps = 1e-5, which is about -100 dBFS per bin, far below any recording's floor. Every bin in
which the recording has only floor and the model has nothing contributes a constant, large
penalty, and the number of such bins is most of the spectrogram.

Measured (`review_3_scripts/floor.py`): a 3-s four-note phrase from the prior, peak
-25.5 dBFS, compared with itself plus a stationary white floor:

| target | loss(model = the same piano) | loss(model = every note 10 velocity units softer) |
|---|---|---|
| no floor | 0.000 | 0.873 |
| floor -80 dBFS | 2.580 | 3.287 |
| floor -70 dBFS | 3.488 | 4.178 |
| floor -60 dBFS | 4.516 | 5.150 |

A perfectly matching piano scores 4.5 against a -60 dBFS floor, five times worse than a
wrong piano against a clean target. The floor term is constant with respect to the physics
parameters, so it does not bias the gradient direction, but it will (a) push whatever can make
broadband noise (knock and pedal taus are unbounded, band levels are free) to fill the
silence with noise, and (b) make the logged loss uninformative. Fix before the first run:

- add a learned stationary noise floor per condition (32 log-band levels, a few hundred
  parameters; physically it is the hall and the chain), added to `dry` or after the room;
- and/or raise eps to about the recording floor, or weight the log term by the target's
  local level (a perceptual weighting);
- bound `knock_log_tau`, `release_log_tau`, `pedal_log_tau` (review 1 m5, still open).

Add a test: loss(same piano + floor) must be well below loss(different piano + floor).

### 4.2 The mono target will teach the model the microphones, not the piano

`prepare_maestro.py` sums L and R. Review 2 found that summing the Iowa pair comb-filters
the spectrum badly (a C2 partial's decay read 73 s from the sum, 6-8 s on either channel).
MAESTRO's pairs are spaced too. The notches of a spaced pair depend on the source position,
which differs per key along the bridge, so the comb is key-dependent. The body FIR is per
year, not per key, and cannot fit it; `partial_gain` (per key, per partial, +-26 dB) can, and
will, so the "piano" tables will absorb the microphone geometry.

Fix: keep both channels. Give the room two body FIRs per condition (the cost is one more
convolution) and compute the loss on both channels. This is also step one of the stereo
output that the end goal needs anyway. If mono has to stay for now, train on one channel.

### 4.3 Identifiability and the order of unfreezing

Several components can explain the same spectrum:

- `partial_gain`: 88 x 96 free values, bounded +-3 nats (+-26 dB), regularised only by
  1e-2 x mean square. This can replace the hammer model and the strike comb outright. Bound it
  to about +-8 dB and regularise smoothness over neighbouring keys at equal frequency (it is a
  per-key body response, see section 3 item 6).
- `ContextNet` outputs `log_decay` +-0.5 (x1.65 per note) and `log_fc` +-0.5 per note. A
  per-note decay multiplier of 1.65 lets the network fake the decay physics from step one, and
  Adam will use it immediately even though it is zero-initialised. It should stay frozen until
  the physics has converged, and then be regularised towards zero.
- body FIR: 7200 free taps per condition (115k parameters, more than the physics). Fine
  once the physics is fixed; hazardous while both move.

Recommended staging (details in section 7): physics + room + per-condition scalars first,
with `partial_gain` narrowed; then `partial_gain`; then the context net; then the GAN.
Between stages, check that the fitted B, stretch and decay rates agree with values tracked
directly from the same recordings (section 4.6). That is the identifiability test.

### 4.4 The residual model is too small for "the last 10 %"

The context net predicts four scalars per note (gain, brightness, decay, knock level). The
README correctly says the attack is where physical models are weakest, and answers with a
GAN. A GAN cannot create what the generator has no parameters for. What the generator can
vary at the attack is a knock level and an exponential envelope per key. Two options, in
order of how much interpretability they cost:

1. A **learned per-note noise residual**: the context net emits a short frame-rate envelope
   (say 25 frames at 200 Hz, 125 ms) times a band spectrum, as in DDSP's filtered noise, in
   place of the fixed exponential knock. Cheap, per-note, still a "noise" component.
2. A **learned per-note per-partial correction** (amplitude and decay per partial, bounded
   small): what DDSP-Piano does. More capacity, more risk of eating the physics.
3. A **neural post-filter** on the dry signal (a small causal TCN conditioned on the roll).
   Closes the gap fastest and breaks interpretability least visibly, since the physics stays
   underneath. Keep as the fallback if 1 and 2 are not enough, and make it switchable.

Recommendation: 1 now, 2 after the physics fit is trusted, 3 only if listening says so.

### 4.5 Windows, decays and throughput

- 2-s loss windows with a 4-s lookback: a note is at most 7 s old when the loss sees it. The
  bass aftersound at 8-16 s, which the calibration spent effort on, is never observed. Raise
  the lookback to about 12 s (it only adds notes, and inactive notes are skipped per chunk).
- Damping history before the window start is ignored (`c_onset` reads the cumulative damper
  time at t = 0): a note released 2.5 s before the window rings at -73 dB in the first 200 ms,
  10 dB under a legitimately held note, and is gone by 300 ms (`review_3_scripts/history.py`).
  Harmless with the 1-s warmup, wrong in principle. The sostenuto latch at the window start
  (review 1 m4) is the same root cause and is still open. Fix both by building the pedal
  curves and key rolls from t0 - lookback.
- Throughput on the 3090 (`review_3_scripts/bench.py`, 56 notes per 3-s example):

  | config | s/step | peak memory |
  |---|---|---|
  | batch 4, sympathetic off | 2.0 | 1.5 GB |
  | batch 16, sympathetic off | 7.0 | 8.1 GB |
  | batch 4, sympathetic on | 4.2 | 9.2 GB |

  About 6x real time either way (the oscillator bank is compute-bound, so a larger batch
  does not help). One pass over MAESTRO's training split (about 160 h) is roughly 40 h of
  wall clock; the default 200k steps at batch 4 is about 4.6 days. Feasible, but each
  full-data experiment is a multi-day commitment, which is why the go/no-go must run on
  one year and on excerpts first.
- Long renders: the oscillator bank evaluates `freq * (t - onset)` in float32 with absolute
  time, so phase quantises as the piece goes on. The inter-partial floor rises from -126 dB
  re peak at t = 0 to -90 dB at a 10-minute onset (`review_3_scripts/precision.py`). Not
  audible yet; it will be in the C++ engine if copied as is. Fix by computing the cycle count
  in float64 and passing only the fractional part.

### 4.6 What the Iowa work should turn into

The most valuable piece of code in the repo is `pianonn/calibration.py`: a partial tracker
and decay analyser that works on recordings and on the model identically. Pointed at
MAESTRO, it becomes three things at once: (1) per-year priors for B, stretch and decays,
(2) the identifiability check after fitting, (3) evidence for the "one piano per year"
assumption (cluster tracked B per piece within a year; the literature check could not
confirm which Disklavier was used when). MAESTRO has thousands of notes that are effectively
isolated for a second or two (sparse passages, no pedal, no overlapping key). Mine them.

## 5. Do the tests and acceptance checks test the right things?

The 22 tests are mostly plumbing and invariants: shapes, finite gradients, block rendering
equals single pass, per-key sums, the sostenuto latch. Those are worth having. Four tests
carry meaning and deserve a comment each:

- `test_student_moves_towards_teacher` is read in the README as "training works". It shows
  that Adam can reduce a spectral loss between two instances of the *same model class* on
  clean synthetic audio. It says nothing about MAESTRO, where the target is a different
  generative process with a noise floor and a room. The pathology in 4.1 passes this test.
- `test_decay_profile_matches_measured_piano` is analytic on the parameter tables that were
  fitted to those same targets. It is a regression test that the tables were not edited, not
  evidence about the sound.
- `test_damper_and_sustain_pedal`, `test_soft_pedal_darkens`, `test_sympathetic_resonance_needs_pedal`
  are the right kind of test: a behaviour a pianist would name, checked on rendered audio.
- The 50 diagnostics checks are the real acceptance suite. As reviews 1 and 2 already said,
  many are regression checks on the prior. 45/50 is fine and not the point.

What is missing is any test that the *loss and the data* can drive the *parameters*:

1. **Gradient sanity per parameter**: perturb one physical parameter (B by +10 %, b1 by x1.5,
   a T_c by x0.8), render, and assert that the loss gradient on that parameter points back.
   Do it on a target that has a noise floor and a room. This would have caught 4.1.
2. **Explain-one-excerpt**: with a 10-s MAESTRO excerpt as target, fitting only the
   per-condition scalars and the room must reduce the loss substantially below the untrained
   prior, and fitting everything must reach a small residual. This is the go/no-go and also a
   permanent regression test once a small excerpt is committed (MAESTRO is CC-BY-NC-SA 4.0,
   which allows a short excerpt in the repo for testing).
3. **Identifiability**: fit to synthetic audio rendered by a *perturbed* model plus a floor
   and a room, then assert the recovered B, stretch and b1 are within tolerance of the
   teacher's. The current student-teacher test only checks that the loss fell.

## 6. Findings, prioritised

Severity reflects impact on the goal, not code size.

**Blocking for a meaningful first training run**
- F1. Log-magnitude loss dominated by the recording's noise floor (4.1). Add a learned floor
  per condition and bound the noise taus.
- F2. Mono downmix bakes microphone combing into per-key tables (4.2). Train on both channels
  with two body FIRs per condition.
- F3. `partial_gain` and the context net can absorb the physics from step one (4.3). Narrow
  the bound, stage the unfreezing, regularise.
- F4. Re-strike accumulates energy (section 3, item 10). Add a re-strike damping term.

**Important for the result sounding like a piano**
- F5. No longitudinal modes / phantom partials (item 11).
- F6. Bridge conductance too coarse and global (item 3); make it per condition and finer.
- F7. No learned attack residual beyond a knock level (4.4).
- F8. Lookback too short for the bass aftersound; damper and sostenuto history at the window
  start ignored (4.5).
- F9. No validation loop, no held-out metric, no audio dumps, no resume in `train.py`.
- F10. 24 kHz, 96 partials, mono: fine for the fit, not for the product (items 14, 15).

**Correct but worth fixing**
- F11. Float32 absolute time in the oscillator bank (4.5).
- F12. `_ring_end` ignores dampers, so released notes stay "active" for up to 60 s in
  full-piece renders (cost only).
- F13. Damper delay fixed instead of per condition (spec says per year).

**On the process**
- F14. Further calibration against the Iowa Steinway B has negative return now. The forms are
  validated; the values will be refitted. Stop.
- F15. The README's framing ("built like Pianoteq") should say "parametric modal synthesizer
  with physically parametrised tables", because that is what decides what has to be added by
  hand (section 2).

## 7. Plan for what is missing

Ordered so that the biggest unknown (does the model explain real performance audio?) is
resolved first and every later phase is gated on it.

### Phase 0: make the first real fit possible (about 2 days)
1. Learned per-condition stationary noise floor; bounded noise taus; loss eps or weighting
   (F1). Test: loss(same + floor) << loss(different + floor).
2. Stereo target: prepare MAESTRO keeping both channels; two body FIRs per condition; loss on
   both channels (F2). Keep the per-year hall mono for now.
3. Pedal curves and key rolls built from t0 - lookback; lookback 12 s (F8).
4. Re-strike damping term in the closed form (F4). Test: a repeated note under pedal does not
   exceed a single note's level by more than a few dB.
5. `partial_gain` bound to about +-8 dB and smoothness across keys at equal frequency; context
   net frozen by flag (F3).
6. `train.py`: validation split loss every N steps, audio dumps of a fixed validation excerpt,
   resume from checkpoint, per-parameter-group logging (F9).
7. Gradient-sanity test per physical parameter on a target with floor and room (section 5).

### Phase 1: go/no-go on real audio (about 3 days, one MAESTRO year, 2018 or 2017)
1. Prepare one year only (`--limit` or a year filter in `prepare_maestro.py`).
2. **Overfit one 10-s excerpt** with per-condition scalars + room only, then with physics,
   then with everything. Record the loss ladder and listen. If the physics stage cannot get
   close, the forms are wrong somewhere and that is the finding to chase; do not proceed to
   full training.
3. Train physics + room on the whole year (context frozen, GAN off, sympathetic off), about
   20k steps (about 11 h). Measure held-out MR-STFT and log-mel distance against the
   untrained prior; listen to resyntheses of held-out pieces next to the recordings.
4. Mine isolated notes from that year with `calibration.py`; compare tracked B, stretch and
   decay profiles with the fitted parameters (identifiability, 4.6). Disagreement means the
   fit is explaining the data with the wrong knobs.

Success criterion for phase 1: a clear reduction of the held-out distance, fitted frequencies
that agree with tracked ones, and resyntheses that a listener places nearer the recording
than the prior. If this fails, the remaining phases are moot; if it passes, the rest is
engineering.

### Phase 2: per-year priors and all years (about 1 week)
1. Isolated-note mining for every year: per-year B, stretch, velocity-to-level curve, damper
   delay (F13), decay profiles. Use them as initial values for the per-condition parameters
   and as a validation table.
2. Train all years, physics + room; then unfreeze `partial_gain`, then the context net with an
   L2 penalty on its outputs; then the sympathetic bank (F7's option 1, the learned noise
   residual, can start here).
3. GAN last, and only after listening says the STFT-fitted model is close; log the
   spectrogram distance while the GAN runs to see whether it trades fidelity for texture.

### Phase 3: physics completeness (about 2 weeks, in parallel with phase 2 training)
1. Longitudinal modes and phantom partials as a parametrised second partial set with a
   velocity-squared gain (F5). Validate on ff bass notes mined from MAESTRO.
2. Finer per-condition bridge conductance g(f) (F6), regularised in log frequency.
3. Weinreich coupled-string eigenmodes replacing the mode-space shortcut, if the fitted
   aftersound tables look unphysical; otherwise skip.
4. Float64 cycle count in the oscillator bank (F11); damper-aware `_ring_end` (F12).

### Phase 4: evaluation suite (about 1 week; start the metrics in phase 1)
- Held-out MAESTRO test split: MR-STFT and log-mel distance per year; FAD with a stated
  embedding (VGGish for comparability, plus a piano-specific one such as Piano-Encodec if
  available) against the real recordings.
- Transcription round-trip with a public transcription model: note F1 and velocity error.
- Physical diagnostics on MAESTRO (the tracked-vs-fitted table from 4.6) as a per-year report.
- A small MUSHRA-style listening test: recording, this model, the untrained prior, DDSP-Piano
  (public checkpoints exist), Pianoteq if available.

### Phase 5: the product (after phase 2 passes)
1. 48 kHz stage: retrain body FIRs and noise bands at 48 kHz, raise the partial count in the
   bass; physics parameters transfer unchanged because they are in physical units.
2. Stereo hall (two IRs per condition) and per-key panning.
3. Real-time engine: recursive complex oscillators per partial, the closed-form damper
   integral accumulated online, partitioned convolution for body and hall, the context GRU
   run causally at 200 Hz. Export a parameter file per condition. The CPU budget is small
   (tens of thousands of oscillators per second per note are cheap with SIMD).

## 8. Answers to the three questions asked

- **Is the approach sensible at all?** Yes. It is the only family that meets all of the
  goal's constraints, and the literature (DDSP-Piano 2022/23, Simionato and Fasciani 2024,
  Berendes et al. 2023) has no system that combines this much physics with fitting to full
  performances. The bet is reasonable; the ceiling is set by what is parametrised, so the
  missing parametrisations (re-strike, phantom partials, stereo, attack residual) are the
  work.
- **Are the physics correct and sufficiently complete?** Correct in form for the linear
  sustain, and calibrated more carefully than necessary. Not complete for loud, pedalled
  music: re-strike and longitudinal/phantom partials are absent, and the bridge model is too
  smooth to be a specific piano.
- **Do the tests make sense?** As invariants, yes. As evidence that the model can be fitted
  to recordings, no: nothing tests the loss against realistic targets or the recoverability of
  parameters, and the one test that looks like a training test does not cover the failure
  mode that will occur first.

## 9. Two additions after discussion

### 9.1 A felt model at note-on instead of a hammer spectrum table

Pianoteq's parameters are causes (felt hardness, string length, soundboard impedance) and
its spectra are results; this repo's spectra are the parameters. The pragmatic middle
ground keeps the closed-form modal bank for everything after the contact and replaces only
`hammer_spectrum` with a small simulation of the contact itself:

- a lumped hammer of mass m with a nonlinear felt F = K x^p (p about 2 to 3.5, K and p per
  key with a smooth prior), striking the string's modal model at the strike point, at the
  hammer velocity that `hammer_velocity` already gives;
- integrated for a few milliseconds at the audio rate (a few hundred steps, differentiable,
  batched over notes); the string's response during contact is the modal sum of the same
  partials the bank already has, so no new state;
- the force history's projection on each mode gives the modal amplitudes and phases that
  the bank then rings with, and the contact time and roll-off come out instead of going in.

Benefits: hardness, contact time, brightness versus velocity and the number of contacts
become causes with a handful of parameters per key; **re-strike is emergent** if the string's
current modal state is used as the initial condition; the felt exponent is a directly
interpretable voicing parameter. Cost: a few hundred small steps per note-on, negligible
next to the bank; more engineering than the table. Keep the table as the fallback and the
initialisation (fit the felt model to reproduce the current envelopes first). This belongs
in phase 3.

### 9.2 A black-box residual that absorbs what the physics omits

The physics list in section 3 will stay incomplete no matter how far it is extended. The
question is whether a learned component can absorb the difference during fitting, and it
can: it is just another differentiable module in the same forward pass, trained by the same
loss. What matters is making sure the physics keeps first claim on the data, so that the
residual explains only what the physics cannot, and so that its content stays readable as a
map of what is still missing.

**Where the residual can live**, from most to least structured:

| level | form | absorbs | risk to the physics |
|---|---|---|---|
| R1 note, modal language | per-note bounded corrections to partial amplitudes and decays, plus a few extra free sinusoids (phantom-partial slots) | fine spectral shape, missing partials | low if bounded |
| R2 note, attack grain | a small decoder from (key, velocity, context embedding) to a 50-200 ms waveform added at the strike | the attack: felt, knock, multiple contacts, early soundboard | low: it cannot sustain |
| R3 signal, bounded time-varying filter | the context net emits frame-rate band gains (bounded, e.g. +-6 dB) applied to the dry signal, plus a filtered-noise path with a learned envelope | time-varying colouration, mic and room detail, noise texture | low: it cannot create notes, only reshape |
| R4 signal, additive neural correction | a small causal TCN on [dry physics, control features] producing an additive correction, zero-initialised | anything | high: this is the one that can replace the physics |
| R5 neural vocoder on top | mel from physics, waveform from a network | everything | defeats the purpose; not proposed |

Recommendation: build R1 + R2 + R3 as the standard residual stack, and keep R4 as the
escalation if listening says the stack is not enough. All are causal and cheap enough for
real time (a TCN of the size used in neural amp modelling runs on one CPU core).

**How to keep the physics honest.** Five mechanisms, all needed:

1. **Staging.** Fit physics and room alone until the held-out loss plateaus. Only then
   switch the residual on, with the physics learning rate reduced (not frozen: the residual
   taking over part of the explanation should let the physics settle into a cleaner fit).
2. **A residual budget.** Penalise residual energy relative to the dry signal (per band and
   per time), so the optimiser prefers a physics explanation whenever one exists at equal
   loss. For R4 also bound the correction's gain with a tanh so it can never exceed a set
   level below the dry signal.
3. **Correction, not replacement.** Give the signal-level residual the physics output as its
   main input and only coarse control features (pedal state, a note-density signal, the
   context embedding), not the raw onset roll. Then it can reshape what the physics produced
   but cannot re-synthesise notes on its own.
4. **Ablation as a metric.** Always report held-out distance with the residual on and off.
   The gap is the amount of sound that is still unexplained by physics. If the gap grows
   while the physics-only distance stops improving, the physics has stopped learning and the
   residual has taken over.
5. **Read the residual.** Its spectrogram, averaged per key and velocity, is a map of what
   the model lacks. A residual concentrated in the first 30 ms says attack; one at
   sum-and-difference frequencies of the partials says phantom partials; one that is
   stationary says the chain. This is the mechanism by which the residual tells you what to
   model next, so the physics list in section 3 grows from evidence rather than guesswork.

**Interaction with the GAN.** A spectrogram discriminator needs generator capacity to
sculpt; with only knock levels and exponential envelopes to move, it has none. The residual
stack is what gives the adversarial loss something to work with, so the GAN belongs after
the residual is in, not before.

**Effect on the plan.** R3 and R1 join phase 2 step 2 (they are what the context net should
emit); R2 joins phase 3 alongside the felt model (the two overlap: if 9.1 works, R2 shrinks);
R4 is a gated escalation after the phase 4 listening test.
