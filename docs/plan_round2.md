# Round 2: response to review 4, and the plan

Review 4 ([`reviews/review_4_trial_2018.md`](reviews/review_4_trial_2018.md)) checked the first MAESTRO trial
([`trial_2018.md`](trial_2018.md)). This page answers every point. It adds five measurements that test the
review's central claims, and sets the plan for the next round. The scripts are in
[`reviews/review_4_response_scripts/`](reviews/review_4_response_scripts/); run them from the repository root.
Where comparable, they use the review's test excerpts (seed 11) and the 220-minute checkpoint.

## 1. In short

I agree with the verdict and with most of the review. My checks confirm its most important finding, refute
one claim and sharpen four others.

- **Confirmed: the loss sets the level too low, and the spectral-convergence (SC) term is what does it.** I
  scanned a gain per band and compared each term's optimum with the energy match in the same band, so the
  model's spectral-shape errors cancel out:
  - SC is about 1.8 dB low in every band and at every resolution;
  - the per-bin log term is nearly unbiased below 1.3 kHz, but 1.5–2.5 dB low above 2 kHz;
  - log band energies are unbiased within ±0.9 dB in every band, even with 20 ms frames.

  The model is 1–3 dB under the recordings in every band from 80 Hz to 5 kHz, close to where SC wants it.
- **Refuted: "the floor is doing physics' work".** I measured the recordings' floor on the second of silence
  before each piece's first note. It agrees with the *fitted* floor within −3…+5 dB in 30 of 32 bands. The
  other two sit next to mains-hum lines at 60 and 120 Hz.
  - At 40 Hz: silence −25.4, fitted −28.2 dBFS.
  - The initialisation was what was wrong: 15 dB too low at 40 Hz, and 6–16 dB too high between 300 Hz and
    1.6 kHz. The review measured the fit against that initialisation.
  - Below 50 Hz the music is only 1–2 dB above the room's own rumble, so that band really is mostly noise.
- **New: infrasound.** 0–20 Hz holds about 5 % of the recordings' energy, more than all of 1–4 kHz, and it
  is pure noise. Both signals should be high-passed at 20 Hz before the loss.
- **Sharpened: the phantom tests.**
  - The decay test is weak in this model. A phantom decays 0.7–2.6× as fast as partial 2j at the start and
    only 1.1–1.5× in the aftersound, not about 2×.
  - The velocity test stands. A frequency test is stronger: the peak must sit within about a cent of
    exactly 2f_j.
  - One more alternative has to be ruled out: the undamped strings above about F6 ring sympathetically near
    harmonic frequencies.
- **Sharpened: the 60–125 Hz "drift" is not special.** On 24 test excerpts the 56–113 Hz bands are 1–2 dB
  under, like every other band. The report's −5.9 dB came from three validation excerpts. The review already
  says to use the test set; this shows why.
- **Ruled out: missing stereo placement.** On the 420 isolated notes the inter-channel delay is flat across
  the keyboard (register medians within 0.12 ms) and the level difference is within ±0.7 dB. So the owner's
  "one-dimensional" is not a missing per-key position.
- **Sharpened: the texture penalty.** The per-bin loss prefers no texture to the right texture.
  - Under an L1 of log magnitudes, a bin whose content is random is scored best by a constant at its
    median. That constant is 1.6 dB under the mean and has no texture at all.
  - A random signal with the right statistics scores √2 worse. This is a second reason the residual's
    noise path went down, besides the budget flaw the review found.
  - Band energies shrink this penalty about tenfold, but only a distribution-level criterion removes it.
    Pooling k bins cuts both the penalty and the level bias (−0.38 dB at 4 bins, −0.02 dB at 64), but the
    √2 ratio stays (`texture_penalty.py`).

The order of work changes accordingly:
1. the loss (SC out, log band energies as the backbone, a 20 Hz high-pass);
2. the budget and the evaluation scale;
3. a re-fit with a proper control, plus a GAN branch (added after discussion with the owner).

The floor needs a better initialisation and a bound, not a rescue.

## 2. The measurements

### 2.1 Level bias of each loss term, per band (`bias_per_band.py`)

For each of the 24 test excerpts and each half-octave band, I compare two gains:
- the gain that minimises the term restricted to that band;
- the gain that matches the model's mean energy to the recording's in the same band.

The table shows their difference, the median over excerpts, in dB. An unbiased term reads 0, whatever the
model's shape errors. Columns are the bands' lower edges; the last column is the median over every excerpt
and band.

| term | 40 Hz | 80 | 160 | 320 | 640 | 1280 | 2560 | 5120 | all |
|---|---|---|---|---|---|---|---|---|---|
| *model − recording, energy* | *+1.3* | *−1.2* | *−2.2* | *−2.2* | *−2.0* | *−2.1* | *−2.4* | *−2.7* | |
| SC, 4096 points | −3.0 | −1.7 | −0.9 | −0.9 | −1.3 | −2.0 | −2.2 | −4.1 | **−1.80** |
| SC, 1024 | −2.2 | −1.8 | −1.0 | −1.0 | −1.3 | −1.9 | −1.9 | −3.4 | −1.74 |
| SC, 256 | | −2.0 | −1.7 | −1.4 | −1.5 | −1.9 | −1.9 | −2.8 | −1.79 |
| log-magnitude L1, 4096 | +1.6 | 0.0 | −0.4 | +1.4 | +0.6 | −0.5 | −1.6 | −2.1 | −0.23 |
| log-magnitude L1, 1024 | −0.2 | −0.3 | −0.5 | +0.8 | +0.5 | −0.4 | −1.5 | −2.1 | −0.40 |
| log-magnitude L1, 256 | | −0.3 | −0.2 | −0.1 | +0.3 | −0.4 | −0.8 | −2.0 | −0.36 |
| log band energy, 1/6 octave, 20 ms | +0.4 | −0.3 | −0.5 | +0.7 | +0.3 | −0.5 | −0.6 | −0.7 | **−0.14** |
| same, 100 ms | +0.3 | −0.3 | −0.4 | +0.5 | +0.1 | −0.5 | −0.7 | −0.9 | −0.09 |
| same, 500 ms | −0.5 | −0.2 | −0.5 | +0.1 | +0.2 | −0.5 | −0.4 | −0.4 | −0.18 |

- **SC is biased at every resolution.** The bias is the least-squares form, not fine resolution. For
  magnitudes whose fine structure cannot be predicted, the L2 optimum gain is π/4 (−2.1 dB) when both
  signals are random. It is √π/2 (−1.0 dB) when a smooth model faces a random target. That is the review's
  regression to the mean, and the table's −1.8 dB lies between the two.
- **The per-bin log term is nearly unbiased where the piano's energy is**, below about 1.3 kHz. It is
  −1.5 to −2.5 dB above 2 kHz, where hammer noise, the hall and the floor make most bins random.
- **Log band energies are unbiased within ±0.9 dB in every band, with frames from 20 to 500 ms.** Frame
  length hardly matters, so the backbone can keep 20 ms frames for attacks and damping (the review suggested
  50–100 ms).
- **SC holds the level down.** Adding the two rows gives each term's optimum re the current model:
  - SC's optimum is about 0.3 dB above the current level;
  - the log terms' optimum is about 1.8 dB above it.

  The model sits at SC's optimum, which agrees with the review's broadband scan.

### 2.2 The noise floor, measured on real silence (`silence_floor.py`, `silence_low.py`, `hum_lines.py`)

- **How much silence.** 84 of the 93 pieces of 2018 have at least 0.5 s before the first note (median
  0.97 s).
- **It is steady.** The low-frequency energy there is flat over time, within 1.5 dB, with no fade-in or
  start transient.

Floor levels in white-equivalent dBFS, left channel (the right is within 3 dB), median over 86 pieces:

| band | 40 Hz | 84 | 174 | 364 | 760 | 1586 | 3310 | 6910 |
|---|---|---|---|---|---|---|---|---|
| leading silence | −25.4 | −47.8 | −49.9 | −58.6 | −67.0 | −73.5 | −77.8 | −81.8 |
| trial init (2nd percentile of 0.2 s windows) | −40.3 | −49.2 | −47.2 | −43.0 | −52.8 | −67.0 | −81.2 | −85.8 |
| fitted, 220 min | −28.2 | −47.7 | −47.8 | −55.6 | −66.6 | −75.5 | −80.2 | −83.2 |

Power spectral density of the music (mid-piece) relative to the silence:

| 0–20 Hz | 20–50 | 50–70 | 70–100 | 100–150 | 150–250 | 250–500 | 0.5–1 kHz | 1–4 kHz |
|---|---|---|---|---|---|---|---|---|
| +0.3…+1.0 | +1.2…+2.2 | +4.6 | +11.5 | +12.9 | +29.1 | +40.8 | +42.4 | +35.5 dB |

- **The fit moved the floor to the silence measurement.** In 30 of the 32 bands the fitted floor is within
  −3.3…+4.8 dB of it, correcting a bad initial measurement. The review's "−4.4 dB at 40 Hz" is a 2.8 dB
  shortfall there, plus infrasound that the floor's lowest band cannot shape.
- **The two exceptions are mains hum.** The fitted floor is −7.7 dB at the 48 Hz band and −5.8 dB at
  145 Hz, the neighbours of the 60 Hz and 120 Hz lines.
  - The recordings hum at 60 Hz (+9 dB above the surrounding floor) and 120 Hz (+26 dB), in the silence and
    under the music.
  - The band measurement includes the lines. A smooth floor cannot hold a line, so the fit settled on the
    floor between them.
- **The initialisation was wrong in both directions.**
  - At 40 Hz the band holds 1–2 bins and its energy is mostly slow infrasound. With 2 degrees of freedom,
    the 2nd percentile sits 17 dB under the mean.
  - In the middle bands the quietest 0.2 s of continuous music is quiet piano, not floor (the review's
    point).
- **At 20–50 Hz the recordings contain almost no piano.** The floor's 41 % share of the model there is
  right, if anything low.
- **Infrasound.**
  - The silence's power density at 0–10 Hz is about 16 dB above its level at 30–40 Hz, and the music adds
    nothing there.
  - Integrated, 0–20 Hz is about 5 % of a recording's energy. That is more than all of 1–4 kHz, and it all
    goes into the loss's lowest bins.
  - The model's floor is flat below 40 Hz, so it cannot follow it.

### 2.3 Can decay identify a phantom? (`phantom_decay_ratio.py`)

The fitted model at velocity 50: a phantom's decay rate at 2f_j (α_j + α_j) divided by the rate of the
transverse partial 2j next to it.

| | j=3 | j=4 | j=5 | j=6 | j=8 |
|---|---|---|---|---|---|
| MIDI 60, prompt | 2.42 | 1.83 | 1.45 | 1.15 | 0.79 |
| MIDI 60, aftersound | 1.46 | 1.38 | 1.32 | 1.27 | 1.20 |
| MIDI 72, prompt | 1.16 | 0.79 | 0.70 | 0.76 | 1.31 |
| MIDI 72, aftersound | 1.36 | 1.28 | 1.22 | 1.17 | 1.09 |

Losses grow with frequency, so partial 2j already decays faster than partial j. Once the aftersound
dominates, a phantom decays only 1.1–1.5× as fast as its neighbour, and in the prompt stage the ratio
crosses 1. Short windows on soft notes cannot separate the hypotheses this way. Three tests can:

1. **Velocity.** A phantom's level relative to partial 2j should rise about 1 dB per dB of partial level,
   because it goes as the product of two transverse amplitudes.
2. **Frequency.** The peak should sit at exactly twice the tracked f_j (about 1 cent, using the note's own
   B and f0). Partials of a still-ringing string or a sympathetically ringing undamped string do not sit
   there, except by accident.
3. **Sympathetic check.** Flag detections within 10 cents of an undamped string (above about F6) or of a
   string held open by the pedal.

### 2.4 Stereo placement per key (`itd_per_key.py`)

I measured the inter-channel delay (GCC-PHAT, first 60 ms, 150 Hz–6 kHz) and the level difference on the
420 isolated notes:
- The register medians of the delay are between 0.00 and +0.12 ms, with interquartile ranges of about
  ±0.2 ms, and show no trend along the keyboard.
- The level difference is −0.4 to +0.7 dB.
- Inter-channel coherence is low even in the direct sound (peak ≈ 0.07).

There is no per-key placement for the model to learn.

## 3. Point by point

| review 4 | assessment | action |
|---|---|---|
| §3 the loss wants a quiet piano | **Agree.** The cause is SC, in every band and at every resolution (2.1) | A1 |
| drop SC, or weight it ≤ 0.2 | **Drop it.** The model already sits at SC's optimum, so any weight hands it the level | A1 |
| gain-bias diagnostic in `evaluate.py` | **Agree, measured per band** against the energy match in the same band (2.1), not against the broadband level, which mixes bias with shape error. Gate: ±0.5 dB overall, ±1 dB per band | A5 |
| log band energies as the backbone (1/3–1/6 octave, 50–100 ms) | **Agree**, at 1/6 octave; 20 ms frames are enough (2.1) | A1 |
| a frequency term on mined notes | **Agree in substance.** For 2018, freeze B and stretch at the measurements: the loss cannot see them. For all years, a term on the model's analytic partial frequencies against the tracked ones (no rendering needed), with the review's B × 2 regression test | A4, C4 |
| an attack-window term | **Agree**: band energies at a ~3 ms hop, 0–40 ms after each onset | A1 |
| 8192–16384 points is the wrong direction | **Agree with the conclusion**: frequencies come from measurement. **Not quite the reason**: the per-bin log term is no more biased at 4096 than at 1024 points in the bass (2.1). The real problem is that the loss cannot see B in polyphonic audio (trial, section 10) | — |
| re-run the level regression after the loss change | **Agree** | B3 |
| §2 noise ceilings, gain oracle, wrong-model reference; drop the untrained row | **Agree to all.** One caution: under a per-bin L1, a constant at the median beats a random signal with the right statistics by √2 (section 1). So the seed-to-seed "ceiling" is not a lower bound for the per-bin loss. For band energies it nearly is | A5 |
| §4 the floor does physics' work at 40 Hz | **Disagree on the evidence.** The fit moved the floor to the silence measurement; the initialisation was wrong. 20–50 Hz is mostly noise in the recordings too (2.2) | A2 |
| the middle-band initialisation was 9–14 dB too high | **Agree**, confirmed (15.6 dB at 364 Hz) | A2 |
| the floor is unbounded, at 20× the learning rate | **Agree** | A2 |
| §5 the budget's share term is degenerate | **Agree** (`residual_budget`: `e_res / (e_res + e_dry)`, with the dry signal physics only). The per-bin loss's texture penalty (section 1) also pushed the noise path down | A3 |
| the residual's gain is confounded | **Agree.** The control is built into the next run | B2; wording fixed now (A9) |
| the read-out works | **Agree** | — |
| §6 the bass B is a genuine measurement | **Agree**; the re-tracking at Rigaud/32 settles it | — |
| nothing is known about the treble | **Agree.** Mine it with criteria that suit it (few, widely spaced partials) | C4 |
| phantoms: plausible, not established | **Agree.** The decay test is weak (2.3); use the velocity, frequency and sympathetic tests. The script goes into the repo. Also check whether the fitted levels drop under the new loss: at 1024 points and below, a phantom and partial 2j often share a bin (j ≤ 4 in the middle register), so the slots can carry the partial's level | A8, B3 |
| +14 dB at MIDI 72–84, "where phantoms vanish above C6" | **A detail**: MIDI 72–84 is below C6 (MIDI 84). The stronger point is that many of those slots lie above 5 kHz, where the per-bin loss sees mostly floor | — |
| the hall's T60 is non-monotonic | **Agree**; watch it when the sympathetic bank is on | C5 |
| level is split over five parameter sets | **Agree.** Fix the gauge: zero-mean per-key gains and velocity curve, so the mic gain carries the level. Present one "level chain" in the fitted table | A6 |
| the re-strike did not move: add a synthetic check | **Agree.** Synthetic identifiability under the new loss for re-strike (teacher at 2×), damper delay and velocity curve | A7 |
| §7 test material is small; the band tables use 3 excerpts | **Agree.** Use 96 test excerpts, and take every band table from the test set (2.1 shows why) | A5 |
| gradient clipping is always active | **Agree.** The trial's logs show room ≈ 2.1 (median), physics 0.27, context 0.21, and every logged step clipped. Clip per module | A6 |
| tolerant loading was silent in the evaluation | **A correction**: `scripts/evaluate.py` prints missing parameters. The 220-minute evaluation predates the velocity curve, so nothing was missing then; the review's own scripts pass a silent logger. Still, put it in every report | A6 |
| no test of the data initialisation | **Agree**: latency sign, EQ folding and the silence floor on a synthetic teacher | A7 |
| listening: "one-dimensional", "a synth", "much less powerful" | **Agree with the readings; two hypotheses tested.** Per-key stereo placement is ruled out (2.4). The 1–3 dB deficit in every band is SC's, so fix that and listen again before touching the body, lid or hall. Regression to the mean and the texture penalty remain the lead suspects for "one-dimensional" and "a synth" | B4 |
| §8 the order | **Agree**, with the floor demoted (it needs a better initialisation, not a rescue) and the 20 Hz high-pass added to the loss | below |

## 4. Plan

### Phase A: code, tests, diagnostics (no long training)

- **A1 Loss.**
  - Remove SC.
  - High-pass both signals at 20 Hz before any term.
  - Backbone: L1 of log band energies (1/6-octave raised-cosine bands up to 11 kHz, about 20 ms frames).
    The analysis window grows towards the bass, so each low band has enough bins. The floor is white noise
    at −80 dBFS, as now.
  - Fine structure: the per-bin log-magnitude term, only below 2 kHz where it is unbiased, at weight 0.25.
  - Attack: band energies at 128/256-point windows, 0–40 ms after each onset.
  - The log-mel term becomes an evaluation metric only.
  - **Gate** on the 220-minute checkpoint, before any training: the combined loss's bias is within ±0.5 dB
    overall and ±1 dB per band. If the fine-structure term breaks the gate, it goes.
- **A2 Floor.**
  - Initialise from the leading silence of the training pieces: the median per piece, then the median over
    pieces.
  - Bound the learned offset to ±3 dB, drop it from the ×20 learning-rate group, and report fitted vs
    silence.
  - Add mains-hum lines (60, 120 and 180 Hz, a level per condition and channel, measured on the same
    silence), since a smooth floor cannot hold them.
- **A3 Budget.** Measure the share term against the output's band energy (dry physics + floor + residual),
  so cells where the physics is silent no longer count as 100 % residual.
- **A4 Frequencies.** For 2018, freeze B, the per-key stretch and the condition's tuning after the mined
  initialisation. The measurement term waits for all years (C4).
- **A5 Evaluation.**
  - Every table gets the two ceiling rows, the gain oracle, a wrong-MIDI reference and a stationary-noise
    reference (the recording's long-term spectrum).
  - Add the per-band bias table and the fitted vs silence floor.
  - Use 96 test excerpts and take every band table from the test set.
  - Keep the level regression against velocity and pedal.
  - Drop the untrained row.
  - Report the new loss, the old MR-STFT (for continuity) and log-mel.
- **A6 Hygiene.**
  - Clip gradients per module.
  - Fix the level gauge.
  - Put missing checkpoint parameters in the evaluation report.
- **A7 Tests.**
  - The data initialisation on a synthetic teacher: latency sign, EQ folding, silence floor.
  - Synthetic identifiability under the new loss: re-strike at 2×, damper delay, velocity curve.
  - A documented check that the backbone is blind to B × 2, which is why the frequencies come from
    measurement.
- **A8 Phantoms.** `scripts/measure_phantoms.py`: the trial's measurement recovered, plus the velocity,
  frequency and sympathetic tests of 2.3. Mine louder mid-register notes for the velocity test.
- **A9 Report corrections** in [`trial_2018.md`](trial_2018.md):
  - the headline starts from the initialised prior;
  - "the residual became a real gain" becomes "costs nothing and reads as a map" until the control exists;
  - "never added noise" becomes "the budget forbade it";
  - the phantoms become "plausible, not established";
  - the 60–125 Hz drift is a three-excerpt artefact;
  - the floor matches the silence;
  - the "40 % of the reduction from the initialisation" was mostly level.
- **A10 GAN safeguard.** The per-bin loss penalises realistic texture (section 1), and only a
  distribution-level criterion rewards it, so the GAN moves forward from phase C into phase B as a third
  branch.
  - The discriminator sees the same audio as the loss, high-passed at 20 Hz so infrasound is no giveaway.
  - Its gradients reach only the noise bank and the residual (its attack noise and noise path). In the
    critic's view of the output, the noise is rendered a second time from the same random draw with every
    physical input detached (damper timing, pedal, levels), and the strings, the knock impulse, the room and
    the floor are detached. So it cannot bend frequencies, decays, damping or the recording chain to fool
    the critic.

### Phase B: the re-fit, with the control and the GAN (about 1.5 h of GPU time)

- **B1** Stage 1 for ~30 min (to the validation at step 1250) with the new loss. The data, batch and mined
  priors are the trial's. It was shortened from 72 min at the owner's request: the trial's loss fell mostly in
  the first half hour, and these runs compare variants rather than produce a final model.
- **B2** Three stage-2 branches of 20 min each from B1's end, on the same schedule (stage 1 to stage 2 is
  60/40, as in the trial):
  - (a) with the residual;
  - (b) without it: the control;
  - (c) with the residual and the GAN (the A10 safeguard): compared with (a), it isolates what the
    discriminator adds.

- **B3** Evaluate all three on 96 test excerpts:
  - the new scale and the per-band level;
  - the level regression (did the soft deficit shrink?);
  - the floor;
  - the phantom levels (did they drop?);
  - B and stretch;
  - the residual's read-out, including the noise path now that the budget is fixed.
- **B4** Listening: the recording, the trial's model, the new physics, the new physics with the residual,
  and the GAN branch. The GAN branch can score worse on the spectral distances and still sound better (it
  adds texture the per-bin terms penalise), so for (c) listening and the texture statistics decide.

**Moving on requires:**
- the level within ±0.5 dB broadband and ±1 dB per band on the test set;
- the soft-vs-loud slope (+5.3 dB over the velocity range) at least halved;
- the residual branch beating the control by more than the metric's seed-to-seed spread, or the report
  saying it costs nothing;
- the owner hearing it as nearer the recording than the trial model;
- for the GAN: kept if the owner hears (c) as nearer the recording than (a).

**Results** are in [`round2_results.md`](round2_results.md):
- the level passes broadband but fails at 4 kHz, where the loss is biased on the fitted models;
- the soft-deficit criterion does not apply: on the test set the trial has no soft deficit;
- the residual beats the control, but the physics co-adapts;
- the GAN had no measurable effect;
- listening is pending.

### Phase C: afterwards, in order

The trial report's section 12, re-ordered by this review:

1. **Pedal decay and aftersound** on mined pedal-down notes, if the soft deficit survives the loss change.
   The Iowa C2/C3 decay failures stand on their own either way.
2. **Attack**: the knock spectrum per register, then the felt model at note-on.
3. **Longer runs** with a cosine learning-rate decay.
4. **All years**: per-year mining (including the treble), per-year stretch, the frequency measurement term,
   then training on everything.
5. **Sympathetic bank** on in a later stage; watch the hall's 1–4 kHz T60.
6. **Texture**: longer GAN training if branch (c) helped; otherwise matching band-energy variance and
   modulation instead. R4 only if R1–R3 still fall short.

## 5. Phase A, as built

- **A1 Loss** (`pianonn/losses.py`, `PianoLoss`; gate: `scripts/loss_bias.py`). On the trial's 220-minute
  checkpoint and 48 test excerpts, the combined loss passes before any training: its optimum is −0.11 dB from
  the energy match overall and within ±0.7 dB in every octave band. The trial's loss sits at −0.89 dB
  (spectral convergence alone −1.33). Every term's distance to a model with the bass B × 2, divided by its
  distance between two noise seeds:

  | term | ratio |
  |---|---|
  | round-2 total | 0.65 |
  | band | 0.35 |
  | fine | 1.19 |
  | trial's loss | 0.52 |

  None sees the doubled inharmonicity much beyond the noise, so B, the per-key stretch and the tuning are
  frozen at the measurements (A4, `--freeze`).
- **A2 Floor** (`fit_init.floor_from_silence`, `Room.floor_db` / `hum_db`).
  - It is measured on 63 training pieces' leading silence, high-passed at 20 Hz.
  - The 60/120/180 Hz hum is measured at −70/−69/−90 dBFS RMS.
  - Old checkpoints load: their learned floor becomes the reference.
- **A3 Budget.** The share term measures the residual noise at the microphones against the whole output.
- **A5 Evaluation** (`pianonn/metrics.py`, `scripts/evaluate.py`): all of the list above, on 96 test
  excerpts by default. `scripts/ab_render.py` writes the listening material.
- **A6 Hygiene.**
  - Clipping is per module, at 4× each module's running norm.
  - The level gauge is in `PianoPhysics.regularizer`.
  - Missing checkpoint parameters are printed in the report.
- **A7 Tests**: 41, including the loss's level unbiasedness on random signals, the silence floor and hum, the
  latency sign and level of the initialisation, the non-degenerate budget, the critic's isolation from the
  physics, and gradients pointing back to a teacher for the internal loss, contact time, damper delay and
  velocity curve.
- **A8 Phantoms** (`scripts/measure_phantoms.py`; 1,398 training notes of MIDI 51–87 with a clear 0.65 s).
  - **For j = 3 and 5, a real component at exactly 2f_j.** It is detected at 2f_j in 26 % and 31 % of
    clean notes, against 6 % and 20 % at a control position, and it sits within 0.8–1.3 cents of 2f_j.
    Chance peaks sit 4–5 cents off.
  - **For j ≥ 6, nothing above chance.**
  - **It does not behave like a phantom.** Regressed on partial j's level, its level rises with slope
    0.2–0.8, below partial 2j's 0.5–1.6 and far from a phantom's ~2. That is partly the detector's
    selection bias. It also decays no faster than partial 2j (ratio 0.3–1.0).

  So: a line at 2f_j exists in a minority of notes, but its mechanism is not established. The phantom prior
  stays; the fitted +10–14 dB is not evidence.
- **A9** The corrections are marked in [`trial_2018.md`](trial_2018.md).
- **A10 GAN safeguard, as built.** The critic's gradients reach only the noise bank and the residual. The
  knock impulse is left out, because its level carries the physical velocity law.
  - The isolation is structural: the critic sees the output with the strings, impulse, room and floor
    detached, and with the noise rendered a second time from the same random draw with every physical input
    detached.
  - A test checks that no other parameter receives a gradient.
  - The noise renders are checkpointed, which brings the peak memory to 10.5 GB.
