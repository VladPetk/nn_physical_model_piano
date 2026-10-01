# Round 2, phase B: results

Phase B of [`plan_round2.md`](plan_round2.md): the re-fit with the round-2 loss, a control and a GAN branch.
- **Training:** stage 1 ran for 29.5 min (1,250 steps). Three stage-2 branches then ran for 20 min each from
  the same checkpoint:
  - (a) with the residual;
  - (b) without it, the control;
  - (c) with the residual and the GAN.
- **Setup:** data, batch and mined priors are the trial's ([`trial_2018.md`](trial_2018.md)). B, the per-key
  stretch and the tuning are frozen at the measurements.
- **Evaluation:** 96 test excerpts from `scripts/evaluate.py` (`runs/round2/*/eval/report.md`); a paired comparison
  over three render-noise seeds (`scripts/compare_runs.py`, `runs/round2/compare.md`); and a level diagnostic
  (`docs/reviews/review_4_response_scripts/round2_level_diag.py`, `runs/round2/level_diag.md`).
- **Trial model:** the trial's 220-minute model was re-evaluated on the same excerpts.

The branches had equal wall-clock time, not equal steps. The residual costs time, and the critic costs more:

| branch | stage-2 steps | seconds per step |
|---|---|---|
| (a) residual | 726 | 1.6 |
| (b) control | 822 | 1.4 |
| (c) GAN | 506 | 2.3 |

## In short

- **Level: fixed, except at 4 kHz.** The trial model is 2.45 dB under the recordings; round 2 is 0.44–0.55 dB
  under. Every octave band is within ±1 dB except 4 kHz (−1.7 to −1.9 dB). The loss's level bias, which passed
  on the trial checkpoint, comes back at 4–8 kHz on the fitted models (section 2).
- **There is no soft-playing deficit on the test set, for either model.** The trial model is 1.2 dB under on
  soft excerpts and 2.6 dB under on loud ones. Round 2 is within 0.7 dB in every velocity tercile. The trial
  report's "+5.3 dB slope" came from other excerpts, so the criterion "halve it" has nothing to halve.
- **The residual beats the control beyond the measurable spread,** by 0.017 ± 0.003 on the new loss. Changing
  the render-noise seed moves the mean by at most 0.002. But the physics under it gives up about as much: alone,
  it is 0.016 worse than the control's physics. The residual turns the knock down, and in the control the
  physics does that itself (section 3).
- **The residual's noise path is still silent (0.0 % of the output)** with the budget fixed. The loss pushes it
  down on its own: the texture penalty predicted in the plan, section 1.
- **The GAN had no measurable effect in 506 steps.** The critic trained stably, but the noise bank moved no more
  than in (a). The branch sits between (a) and (b) on every distance, which fits its fewer steps.
- **Short training, same place.** 50 minutes of round 2 reach the trial's log-mel distance after 220 minutes
  (3.27 vs 3.25 dB).
- **Other readings:**
  - the fitted floor is within ~1 dB of the test pieces' silence, and the hum matches;
  - the middle register's phantom partials rose at most 4 dB above the prior (the trial: up to 15 dB, in four
    times the training);
  - B and the stretch stayed at the measurements, by construction.

## 1. Distances

The paired comparison uses 96 test excerpts, averaged over 3 render-noise seeds. Across those seeds, a model's
mean moves by at most 0.002 on the new loss.

| | new: total | new: band | new: fine | new: attack | old: MR-STFT | log-mel (dB) |
|---|---|---|---|---|---|---|
| training start (prior + measurements) | 1.172 | 0.620 | 0.903 | 0.654 | 1.454 | 5.88 |
| trial, 220 min, with its residual | 0.735 | 0.401 | 0.614 | 0.362 | 0.994 | 3.25 |
| (b) control, physics | 0.719 | 0.377 | 0.619 | 0.375 | 1.105 | 3.35 |
| (a) residual branch, physics alone | 0.735 | 0.385 | 0.627 | 0.388 | 1.088 | 3.44 |
| (a) residual branch, with the residual | **0.702** | **0.370** | 0.615 | **0.358** | 1.086 | 3.27 |
| (c) GAN branch, with the residual | 0.711 | 0.374 | 0.622 | 0.363 | 1.091 | 3.32 |

For scale, on the same excerpts:

| reference | new: total |
|---|---|
| the same model under another noise seed | 0.15–0.17 |
| the recording plus the model's floor | 0.14 |
| a per-excerpt broadband gain oracle applied to the model | 0.66–0.68 |
| stationary noise with the recording's long-term spectrum | 1.03 |
| the model driven by another excerpt's MIDI | 2.29 |

The oracle's median gain is +0.0 to +0.2 dB for round 2 and +1.5 dB for the trial. What the oracle still gains
for round 2 is per-excerpt level scatter (IQR about ±1.5 dB), not a global offset.

Paired against the control (model − control per excerpt; mean ± 2 standard errors over excerpts; all three
seeds agree in sign in every cell):

| | new: total | new: band | new: fine | new: attack | old: MR-STFT | log-mel (dB) |
|---|---|---|---|---|---|---|
| (a) physics alone | +0.016 ± 0.007 | +0.008 ± 0.004 | +0.008 ± 0.004 | +0.012 ± 0.005 | −0.016 ± 0.011 | +0.09 ± 0.04 |
| (a) with the residual | **−0.017 ± 0.003** | −0.007 ± 0.002 | −0.004 ± 0.002 | −0.017 ± 0.004 | −0.018 ± 0.008 | −0.08 ± 0.02 |
| (c) with the residual | −0.008 ± 0.003 | −0.003 ± 0.001 | +0.004 ± 0.002 | −0.013 ± 0.003 | −0.013 ± 0.006 | −0.04 ± 0.02 |
| trial | +0.016 ± 0.016 | +0.024 ± 0.009 | −0.005 ± 0.010 | −0.013 ± 0.011 | −0.111 ± 0.026 | −0.10 ± 0.10 |

- **The trial model does better on the old MR-STFT loss** (−0.11). That is the loss it was trained on, and its
  spectral-convergence half rewards being quiet (plan, section 2.1). On the new loss's band term, which is mostly
  level, it is worse.
- **On log-mel, the metric neither run trained on,** the trial and round 2 are level.

## 2. Level

Model − recording, high-passed at 20 Hz, median over 96 excerpts (dB):

| | broadband | 32 Hz | 63 Hz | 126 Hz | 252 Hz | 504 Hz | 1 kHz | 2 kHz | 4 kHz | 8 kHz |
|---|---|---|---|---|---|---|---|---|---|---|
| trial | −2.45 | +2.9 | −1.4 | −1.9 | −2.3 | −2.3 | −2.2 | −1.8 | −2.6 | −2.0 |
| (b) control | −0.48 | −1.2 | −0.1 | −0.6 | −0.8 | −0.3 | −0.0 | −0.8 | **−1.9** | −0.4 |
| (a) residual | −0.44 | −0.8 | +0.1 | −0.4 | −0.5 | −0.1 | −0.6 | −0.6 | **−1.7** | −0.3 |
| (c) GAN | −0.55 | −0.9 | −0.3 | −0.4 | −0.7 | −0.3 | −0.3 | −0.9 | **−1.8** | −0.4 |

The trial's +2.9 dB at 32 Hz is mostly its noise floor, 6 dB high at 40 Hz (section 5).

**Against dynamics.** Broadband error by the excerpt's mean velocity, in terciles (median dB):

| | soft (< 48) | middle | loud (≥ 66) | loud − soft |
|---|---|---|---|---|
| trial | −1.18 | −2.67 | −2.60 | −1.42 |
| (b) control | −0.70 | −0.97 | +0.36 | +1.06 |
| (a) residual | −0.44 | −0.85 | −0.19 | +0.24 |
| (c) GAN | −0.61 | −1.10 | −0.18 | +0.43 |

The linear regressions (error = a + b·velocity/127 + pedal terms) are:

| | slope b |
|---|---|
| trial | −0.8 |
| (b) control | +4.0 |
| (a) residual | +3.1 |
| (c) GAN | +2.3 |

The slope is quoted over the full velocity range, which the excerpts' means span only partly (about 30–90). The
terciles are the robust reading, and the regression overstates the spread.

The trial report's "−3.8 + 5.3·velocity/127" and "soft playing ~2 dB quiet" came from 92 validation and test
excerpts, including two long soft ones. On these 96 test excerpts the trial model is, if anything, quieter when
loud. Round 2's fitted velocity curve is compressive (+1.0 dB at the softest, −1.9 to −2.2 dB at the loudest),
and the soft excerpts are within 0.7 dB. The pedal-decay item (phase C1) was conditional on the soft deficit
surviving. On the test set there is none to survive, so it drops down the list.

**4 kHz.** The shortfall is broad, a little worse in the attacks:

| 4 kHz octave (dB) | attack frames (−5 … +40 ms around onsets) | the rest |
|---|---|---|
| (b) control | −2.2 | −1.6 |
| (a) residual | −2.2 | −1.6 |
| (c) GAN | −2.4 | −1.5 |

The loss holds this band down on the fitted models. Its optimum sits below the energy match at 4 and 8 kHz:

| optimum − energy match (dB) | 4 kHz | 8 kHz |
|---|---|---|
| (a) | −1.2 | −0.9 |
| (b) | −1.7 | −1.2 |
| (c) | −1.3 | −1.1 |
| the trial checkpoint (the gate of phase A) | −0.6 | −0.5 |

So the gate passed on the checkpoint it was run on, but not on the models the loss produces. The residual
follows the same bias: its band gains cut 4–8 kHz by 1–2 dB.

A pooled log band energy per frame is median-seeking over frames. If the recording's 4 kHz energy is more
concentrated in time than the model's, the loss's optimum falls below the energy match. Release, pedal and key
noise are the candidates, besides the attacks. This is not established.

The robust fix is a term compared over the whole excerpt: the log of each band's total energy per excerpt. Its
optimum is the energy match by construction, whatever the time structure. `scripts/evaluate.py` now runs the
gate on every fitted model.

## 3. The residual against the control

- **Beyond the measurable spread.** (a) with its residual beats the control by 0.017 ± 0.003 on the new loss, and
  by 0.08 ± 0.02 dB on log-mel.
  - Changing the render-noise seed moves a model's mean by at most 0.002.
  - The spread between training seeds is not measured, with one run per branch. Within a run, the 32-excerpt
    validation moved by ±0.005 between checks.
  - The control also had 13 % more steps.
- **Where the gain is.** Half of it is the attack term (−0.017 on that term, weight 0.5); the band term gives
  most of the rest.
- **What it does, per note (held-out; units of each output, bound in brackets):**
  - knock level −0.19 to −0.37 (±1) and knock spectrum −0.17 to −0.34 (±1), largest in the bass;
  - gain −0.06 to −0.18 dB (±6);
  - everything else near zero;
  - band gains within ±0.5 dB up to 3 kHz, then −1 to −2 dB at 4–8 kHz (the loss's bias, section 2).
- **Co-adaptation.** Alone, (a)'s physics is 0.016 worse than the control's. The control moved the physical
  knock (level, velocity law, decay) about three times as far as (a) or (c) did. The residual took over a
  correction the physics makes by itself when it has to.
  - The residual is worth its cost as a model.
  - But it makes the physics a worse physical description. For reading the physics, use the control.
  - For the final model, train the residual after the physics has converged.
- **The noise path.** Its share of the output is 0.0 % in every band, and its level fell 3–15 dB below its start
  (most at the top) in 20 minutes of stage 2. The budget is no longer the cause, so the loss is. This is the
  texture penalty of the plan's section 1: under per-frame terms, a noise the model cannot align costs more than
  its absence.
- **Knock spectrum.** In stage 1, the physics raised the knock impulse by +3 to +7.7 dB over MIDI 36–60 and
  lowered it by 5 dB in the treble. The residual then turns it down again in every register, while the attack
  frames at 4 kHz stay 2.2 dB low. The knock's spectrum is wrong, not just its level: too much in the low-middle
  bands, too little at 4 kHz.

## 4. The GAN branch

- **It trained stably.** The critic's least-squares loss, summed over its resolutions, held near 1.5. The
  generator's adversarial term fell from 1.18 to 0.69, and the feature matching stayed near 0.3. Both enter
  the loss at weight 0.1 (adversarial + 2 × feature matching).
- **The isolation held.** Only `noise.` and `context.` can receive its gradients: that holds by construction,
  and a test checks it.
- **It moved nothing measurable.** Every noise-bank parameter moved from the shared start by no more than in (a).
  The residual's noise path ended 12 dB down at the top instead of 15, and its share of the output is still
  0.0 %.
- **On the distances** it sits between the control and (a), 0.009 behind (a). That fits 30 % fewer steps.
- **Its residual jumped at the end.** Its validation with the residual went from 0.723 to 0.734 in the last six
  steps, while the physics did not move.

In 506 steps, at weight 0.1, the per-frame terms shut the noise path faster than the critic can open it. Listening decides
whether (c) is kept, but no audible difference from (a) is expected.

## 5. Floor, phantoms, frequencies

- **Floor.** The fitted floor is within about 1 dB of the silence of the 9 test pieces in every band, except
  58 Hz: +3.4 dB, at the ±3 dB bound, because the training and test silences differ there.
  - The hum is fitted at −70.3 / −68.1 / −90.0 dBFS (60/120/180 Hz) against −69.2 / −68.6 / −91.7 measured.
  - The trial's floor had absorbed the hum into its 58 and 121 Hz bands (+5 and +10 dB) and was 6 dB high at
    40 Hz.
- **Phantoms**, in dB re the prior, MIDI 51–75:

  | | phantom partials |
  |---|---|
  | round 2 | −1 to +4 |
  | the trial | +6 to +15 |

  Round 2 had a quarter of the training, so this does not show that the new loss stops the rise. It does show
  that 50 minutes that fix the level do not need it.
- **Frequencies.** B, the per-key stretch and the tuning are at the measurements, as intended (the
  identifiability table in each report).

## 6. Against the plan's criteria for moving on

| criterion | result |
|---|---|
| level within ±0.5 dB broadband | **pass** for (a) −0.44 and (b) −0.48; (c) −0.55 |
| within ±1 dB per band | **fail at 4 kHz** (−1.7 to −1.9) in every branch, and at 32 Hz in the control (−1.2); the loss is biased there on the fitted models |
| soft-vs-loud slope at least halved | **not applicable**: on these excerpts the trial has no soft deficit; round 2's soft excerpts are within 0.7 dB |
| the residual beats the control beyond the seed-to-seed spread | **yes** against the render-noise spread (0.017 vs ≤ 0.002); the training-seed spread is unmeasured; the physics co-adapts |
| the owner hears it nearer the recording than the trial | **pending** |
| the GAN kept if (c) sounds nearer than (a) | **pending**; no measurable effect |

## 7. Listening

`runs/round2/listen/0_ab.wav` and `1_ab.wav` play two 20 s test excerpts, with half a second between:
1. the recording;
2. the trial (220 min, with its residual);
3. the control's physics;
4. (a) with the residual;
5. (c), the GAN branch.

`demo_<label>.wav` renders `samples/demo.mid` with each model. Nothing is normalised: level is part of what is
being judged.

## 8. What this suggests next

*Superseded by [`tone_measures.md`](tone_measures.md): the note bench measured these gaps; its sections 10.4 and
11.4 give the order now.*

For discussion before any code:
1. **A whole-excerpt level term.** Add the log of each band's total energy per excerpt, compared with L1. It is
   unbiased for level by construction, and it closes the 4 kHz gap without touching the per-frame terms.
   Separately, find what makes the recording's 4 kHz energy more concentrated in time than the model's.
2. **Attack, phase C2, moves to the front.** Three findings point there:
   - the knock spectrum is wrong: too much low-middle, too little at 4 kHz;
   - the residual's main job is turning the knock down;
   - the 4 kHz attack frames are 2.2 dB low.
3. **Texture.** The GAN needs the noise path open to have anything to shape. Options:
   - keep the per-frame spectral terms from shutting the residual noise path, so that only the critic and the
     budget act on it;
   - or match band-energy variance and modulation, as phase C6 suggested.
4. **The residual's schedule.** For the final model, train the residual only after the physics has converged, so
   it cannot absorb corrections the physics can make.
5. **Longer runs after that.** 50 minutes already match 220 minutes of the trial on log-mel.
