# The composite score and how we train on it (2026-10-02)

The training loss and the training run as they stand in the code now (uncommitted, branch `maestro`), for the owner
to read and validate. History and evidence: `docs/reviews/review_6_training.md`, sections 7-8 (why the loss was
rethought) and 11 (the build, the check, the two smoke runs). Labels as elsewhere: **read** (from a log or report),
**mine** (my design choice or reasoning, not tested unless it says so).

Code: `pianonn/partial_view.py` (the partial and between views), `pianonn/composite.py` (the score),
`pianonn/losses.py` (`PianoLoss`'s band term, `OnsetLoss`), `pianonn/train.py` (`--score composite --energy`),
`pianonn/residual.py` (the residual), `scripts/score_check.py` (the check), `scripts/composite_eval.py` (scoring
checkpoints). The run: `runs/composite_train/smoke2/run.sh`.

## 1. One step in brief

1. Draw a batch: 8 excerpts of the training pieces, each 12 s of MIDI history, 1 s rendered but not scored
   (warm-up), 2 s scored.
2. Render the batch twice with the same model: `X` with gradient, `X'` without, every random draw independent
   (per-strike variation, noises, the residual's random inputs). Both get the piece's gain.
3. Compare `X` with the recording `Y`, and `X` with `X'`, in seven terms (section 2).
4. Each term `d` enters as `d(X, Y) - d(X, X') / 2`, the energy score (section 4); the total is their weighted
   sum (section 5) plus the physics' smoothness regulariser.
5. Backward, clip per module, Adam, update the weight average.

Units everywhere: log10 of power, so 1 = 10 dB and 0.1 = 1 dB; every distance is a mean |log10 difference|. Both
signals are high-passed at 20 Hz first.

## 2. The seven terms

| term | reads | resolution | pooled over | weight |
|---|---|---|---|---|
| `band` | energy in 1/6-octave bands centred 30 Hz-9.7 kHz (51 bands) | per band, 10 ms frame | nothing (cell by cell) | 0.4 |
| `partials` | the mix at every sounding note's partial frequencies | per partial, 10 ms frame | nothing | 0.4 |
| `between` | the mix between the partials, 1/3-octave bands | per band, 10 ms frame | nothing | 0.7 |
| `pooled_exposed` | the `partials` readings, weighted by exposure | partial group × note age | the batch | 1.0 |
| `between_pooled` | the `between` readings | band × time since the last onset | the batch | 0.9 |
| `level` | the `band` energies | per band | the excerpt, then the batch | 0.8 |
| `onset` | band powers in a window at each onset | 1/3 octave from 100 Hz | the batch's onsets | 0.9 |

The first three are **read by read**: one log reading per cell, compared cell by cell. They keep where and when.
The last four are **pooled**: energy is summed over many readings before the log, so a consistent error stands out
of the take's own scatter (section 3).

### 2.1 `band` (`PianoLoss`, band term only)

- 51 raised-cosine bands at 1/6 octave, centred 30 Hz-9.7 kHz (centres up to 0.45 × 24 kHz; the top band takes
  everything above its centre), 10 ms hop. The analysis window grows
  towards the bass so each band spans a few bins: 8192 points below 200 Hz, 2048 to 1.6 kHz, 512 above.
- Floor in each log: white noise at −80 dBFS.
- Value: the mean over channels, bands and frames of |log10 band energy (render) − log10 band energy (recording)|.

### 2.2 `partials` (`PartialView`)

- **Which partials:** the model's own note list from its forward pass: every note sounding in the window (struck in
  it or ringing in from before), its partial frequencies as rendered. A partial is read if its expected energy
  (amplitude² and decay, dampers ignored) somewhere in the window is within 70 dB of the example's loudest, and it
  lies at 30 Hz-10.8 kHz.
- **Reading:** the mix's power at that frequency every 10 ms, render and recording alike. The analysis window
  follows the note: the power of two holding 4 periods of its fundamental (256 to 8192 points), so its own partials
  are at least 4 bins apart. The reading sums 3 bins around the frequency with a triangle 1.5 bins wide (a partial
  up to about a bin off still counts). Where two notes share a frequency, the reading holds their sum.
- **When:** from 10 ms before the note's sound onset to the end of the window.
- **Gate (a stand-in for masking, mine):** a reading's floor is −80 dB plus 50 dB below the loudest partial reading
  in the recording at that moment, anywhere in the spectrum. Below that, both sides count as silent.
- Value: the mean |log10 difference| over the readings.
- The frequencies are the model's: a reading at fixed frequencies cannot steer them, so this term does not train the
  tuning or B (they are frozen anyway, section 6.2).

### 2.3 `between` (`PartialView._between`)

- 23 bands at 1/3 octave centred 60 Hz-9.7 kHz (8192 points below 200 Hz, 2048 above), per 10 ms frame.
- A bin counts as "between" if it is more than 1.5 bins from every sounding partial (any note, from its onset on);
  a band counts in a frame if at least 3 such bins remain. The reading is the mean power per free bin: the noise,
  the soundboard, the hall that fill the gaps.
- Floor: white noise at −80 dB per bin. Value: the mean |log10 difference| over the counted cells.

### 2.4 `pooled_exposed`

- The `partials` readings, summed per cell before the log. Cells: 11 partial-number groups (1, 2, 3, 4, 5-6, 7-8,
  9-12, 13-16, 17-24, 25-32, 33+) × 5 note-age bands (0-0.1, 0.1-0.3, 0.3-1, 1-3, 3+ s). A cell counts if it holds at
  least 20 readings (weighted).
- **Exposure weighting:** in dense music most readings of an old note's partial hold younger notes' partials at the
  same frequency, which would dilute what that note's own fade does to the pool. So each reading is weighted by its
  note's expected share of it, squared: the note's expected energy over the expected energy of every partial
  within two bins of it. Expected energies come from the model's amplitudes, decay rates, dampers and re-strikes
  (the room and the residual's curves ignored). Both sides get the same weights.
- What it buys (`runs/score_check/sweeps/`, read; 48 validation excerpts, pooled per excerpt): every term sees a
  two-way sweep of the decay of partials 1-8 (rise at the ends z 7-8 for `band`, `pooled`, `pooled_exposed` and the
  old total alike; the pooled ones by a larger share of their floor, 0.12-0.18 against 0.04). For partials 9 and up
  the exposure weighting triples the pooled term's share: 0.022 of its floor against 0.007 without it (z 4.8 against
  4.1; `band` 0.005).

### 2.5 `between_pooled`

- The `between` readings summed per (1/3-octave band, time since the latest onset: 0-50, 50-150, 150-500, 500+ ms)
  before the log. At least 20 free-bin readings per cell. Separates the attack's noise from what rings on between
  the notes.

### 2.6 `level`

- The `band` term's band energies summed over each excerpt's 2 s, per band, then pooled over the batch. Its optimum
  is the energy match per band whatever the time structure.

### 2.7 `onset` (`OnsetLoss`)

- A window at each note's expected sound onset: MIDI onset plus a delay that depends on pitch and velocity
  (7.17 − 0.273 (pitch − 60) − 0.073 (velocity − 64) ms, at least 0), from 3 ms before to 33 ms after, flat-topped.
  Onsets within 10 ms share one window (a chord is one attack).
- 20 bands at 1/3 octave from 100 Hz (to ~8 kHz, the last open above). A band of a window counts where either side
  stands 6 dB over its own background (330 to 30 ms before the onset).
- The counted windows' powers are summed per band over the batch before the log (raw powers: unlike the other
  pooled terms, loud excerpts weigh more). Value: the mean |log10 difference| over the counted bands.

## 3. Pooling

- **Why pool:** against one recorded take, a cell-by-cell L1 of logs hides a consistent error `d` that is smaller
  than the take's own scatter `s`: it raises the mean |difference| by only about `d²/s`. Summing energy before the
  log averages the scatter out (`runs/score_check/pooled/`, read: partials 5-8 +3 dB reads 5.3 times more of the
  floor pooled than read by read).
- **Why across examples, not within one:** pooled within one excerpt a cell is still median-seeking across excerpts
  (it prefers the typical take, not the average). Pooled over many excerpts, the optimum is the energy match.
- **Weights per example:** 1 / the render's mean power per reading, detached, so a soft passage counts as much as a
  loud one. The render's, not the recording's: the recording's own random level would leak into its weight and bias
  the pool (`runs/score_check/across/`, a slip of mine, caught).
- **Over the batch only, not across steps.** Pooling across steps (running sums decaying by 0.97, ~33 batches) was
  tried and diverged: the sums hold renders of an older model, so the term's sign lags the model and a fast-learning
  residual overshoots (smoke run 1, section 8). `--pool-decay 0` is the default now. So each pooled term pools 8
  excerpts per step; its optimum per step is the energy match over those 8.

## 4. The energy score

- **Why:** a plain distance between a random model and one random take is smallest when the model is less random
  than the takes. Rendered without its per-strike variation (phases 3-6), the model's unison modes start in phase,
  so a render is louder early than a typical varied strike, and every term, old or new, then prefers the aftersound
  about 4 dB too quiet (`runs/score_check/varoff_seed3/`, read: z 6-13, two takes).
- **What:** each term becomes `d(X, Y) - d(X, X')/2`, with `X'` an independent second render. Its expected value is
  minimised when the model's distribution matches the takes', spreads included (a proper score). With the variation
  on and this form the pulls shrink (`energy_seed3/`, read: aftersound −2.4 ± 0.9 dB, knock −1.9 ± 1.8 dB, the decays
  within 2 errors; one pair of reference draws).
- **Gradient:** both renders depend on the model, symmetrically, so the gradient is that of `d(X, Y) - d(X, sg X')`
  (`sg`: no gradient through `X'`). The code returns that for the backward pass and the energy-score value for the
  log. On every term, pooled ones included: pooled over one batch of 8, their self distances are half to all of
  their distances to the recording (read, smoke 2's log, means over thirds of the run: `pooled_exposed` 0.14-0.17
  against itself 0.11, `between_pooled` 0.18-0.21 against 0.10-0.11, `onset` 0.12-0.15 against 0.10-0.12).
- **What is random between `X` and `X'`:** the per-strike variation (config, fixed, not learned: the note's level
  1.12 dB sd up to about MIDI 53, none from MIDI 65 up; brightness 0.16, all decay rates 0.25 and their tilt 0.26 in
  log units; the onset 3.3-6.1 ms; each partial's decay 0.3, each aftersound mode a random part 4 times its key's value; draws
  clipped at 2.5 sd), the white noises of the noise bank and the residual's noise paths, the room's noise floor, and
  the residual's 8 random inputs per note.
- **What is not proper (mine):** the score is proper only per feature's marginal (an L1 per cell), not for the joint
  distribution. Several pieces depend on one side: the `partials` gate uses the recording's loudest reading; the
  `onset` cells count where either side stands over its background; the readings sit at the render's frequencies and
  onsets; the exposure and the example weights come from the render. Each is the same for `X'` and `Y` within a
  step, but they are not symmetric in `X` and `Y`.
- **At the optimum the value is not 0:** it is half the takes' own spread, `E d(Y, Y')/2`, which cannot be measured
  without a second take. So a term's value tells how far it is from its floor only by assumption (section 9).

## 5. Weights

`WEIGHTS = {band 0.4, partials 0.4, between 0.7, pooled_exposed 1.0, between_pooled 0.9, level 0.8, onset 0.9}`.

- Derived (mine) as `floor(pooled_exposed) / floor(term)`, with the floors of `runs/score_check/exposed_p2/` (the
  model, variation off, against a varied draw of itself; 48 validation excerpts; pooled terms then pooled per
  excerpt): band 0.281, partials 0.301, between 0.154, `pooled_exposed` 0.114, `between_pooled` 0.124, level 0.136,
  onset 0.125. So each term counts in units of its own floor, the pooled partials at 1.
- Those floors predate pooling across the batch and the energy form; the pooled terms' floors are different now. The
  weights have not been re-derived.
- What each term contributes at the end of smoke 2 (held out, level matched, weight × value): band 0.10, partials
  0.10, between 0.13, `pooled_exposed` 0.06, `between_pooled` 0.14, level 0.06, onset 0.09; total 0.69.

## 6. The training run (smoke run 2)

### 6.1 Data

- MAESTRO 2018, training split: 70 pieces, 22.9 h. Validation split 13 pieces (3.2 h), test 10 (1.4 h).
- Excerpts drawn at random, pieces weighted by duration (uniform over time); batch 8.
- **Piece gains:** a free gain in dB per training piece, applied to both renders before the score, averaging zero
  over the pieces, 50 times the base learning rate; loaded from phase 6. A nuisance (each piece's recording level
  differs by ~1.5 dB sd); not used in validation or evaluation.

### 6.2 What trains

- **Start:** phase 6 (`runs/phase6/train_run/train/last.pt`), its physics and room; the residual fresh.
- **Physics, room, noise bank, sympathetic bank:** everything trains except the frequencies (`physics.raw_log_B`,
  `raw_cents`, `cond_cents`: a magnitude loss cannot steer them, review 6, 8.4). Phase 6 had also frozen the hall's
  decay, the room's band gains and level, the attack's parts and the knock impulse's level; those train now.
- **The residual** (`residual_kind=aware`, 476k parameters):
  - *Sees*, per note and 20 ms control frame, without gradient into the physics: its expected energy in 8 octave
    groups of partials (from the physics' modal parameters: amplitudes, decay rates, dampers, re-strikes; the
    per-strike variation not included), the mix's, their difference, the note's age, key held, damper, the pedals,
    re-strikes, velocity, key, the example's level; a key and a condition embedding; a GRU's summary of the MIDI
    history; and 8 random numbers per note, drawn afresh at every render.
  - *Network:* a token per note and frame; 2 layers of attention across the sounding notes at each frame, an MLP,
    and a GRU along each note's own frames; width 96, 4 heads.
  - *Outputs* (every output layer starts at zero, so it starts from the physics; every output bounded):

| output | per | range |
|---|---|---|
| gain curve per partial (1-31, 32+ shared), applied inside the oscillator bank | note × 20 ms | ±12 dB |
| onset-time corrections: level, brightness, decay and its tilt, 6 spectral bumps, knock level and spectrum, a slower attack noise (level, spectrum, decay) | note | as `ContextNet.NOTE` (e.g. level ±6 dB) |
| band gains on the dry signal, 16 bands | frame | ±6 dB |
| a noise path, 32 bands | frame | −104 dB at zero, ±35 dB |
| a noise path per note, 16 bands, re the note's own expected energy (follows its decay and dampers; silent before its onset, ramps in over the next 20-40 ms) | note × 20 ms | −60 dB at zero, ±48 dB |

### 6.3 Optimisation

| | |
|---|---|
| optimiser | Adam (default betas), one group per parameter |
| base rate | 5e-4; the residual, the tables that can absorb form errors (`partial_gain`, the colour table) and the residual's attack-noise spectrum (`noise.att`) × 1.0; everything else in the physics, room and noise bank × 0.3; on top, a per-unit scale (dB parameters × 20, the body FIR × 0.03, the piece gains × 50) |
| warm-up | linear over the first 100 steps, then constant (no decay in a smoke run) |
| clipping | per module (physics, room, noise, residual): at 4 × its running typical gradient norm |
| regulariser | the physics' smoothness across keys and level gauges (weight 1); the residual budget off |
| weight average | EMA, decay 0.98 (~50 steps), used by validation, the audio dumps and the checkpoints |
| per-strike variation | on in training and validation (both renders) |

- **Cost (read, smoke 2):** 5.4 s per step (second render 1.25 s, render 1.45 s, score 0.37 s, backward 2.34 s on a
  dense batch), peak 15 GB. 30 min = 332 steps = 2,656 excerpts = 1.5 h of scored audio, 6.5 % of one pass over
  the 22.9 h. (I wrote 12 % earlier; that was wrong.)
- **Cost since 2026-10-02** ([`docs/speed.md`](speed.md)): 1.0 s per step on the same setup (the allocator at its
  default, the string bank and the sympathetic resonators as fused CUDA kernels, the score reading the render once;
  the same audio and gradients to float precision).

### 6.4 Validation and evaluation

- **During training** (every 100 steps; the weight average): 32 fixed excerpts of the validation pieces and 32 of
  the training pieces, without and with the residual, the two renders at seeds 0 and 1; the pooled terms pooled over
  all 32. Also the old score (`PianoLoss` with its level term at 0.5) on the first render. Smoke 2 ran this without
  level matching; it is level matched by default now.
- **Level matching:** each render scaled to its piece's recorded level, measured on the piece's other excerpts in the
  set (leave one out, as `scripts/compare_runs.py`); an excerpt alone in its piece keeps 0 dB.
- **After a run:** `scripts/composite_eval.py`: 64 excerpts of each set, the same scoring, level matched or not.

## 7. Left out, and why

- `PianoLoss`'s `fine` term (per bin below 2 kHz): the partial and between views read the same bins, sorted by what
  they hold.
- Its frame-by-frame `attack` term: against a take with timing scatter it prefers the knock 3.4 dB too quiet
  (`runs/score_check/run1/`, read); the onset term integrates over each attack's window.
- The N13 envelope term (each partial's fade on isolated notes, phase 6): off in the smoke runs, to read the composite
  alone. `pooled_exposed` now sees the first partials' decay in music.
- A critic for texture. Round 2's critic (`--critic patch`) was checked for the first time on 2026-10-02
  (`scripts/gan_check.py`, `runs/gan_check/check1/`, read) and failed all three checks:
  - between recordings, with a 3 dB shelf above 1 kHz as the only thing the generator can change, the shelf does
    not return to 0: from +3 dB it ends at −2.2 dB, from −3 dB at −1.0 dB, the critic's loss at chance throughout;
  - trained 600 steps against renders of `env_fit2`, it tells them from recordings at a held-out AUC of 0.53
    (64 + 64 excerpts), and its push per octave band does not follow the measured band error (correlation +0.05;
    the paired feature matching's +0.64: one more spectral distance);
  - at weight 0.1 its gradient on the residual is 0.1 % of the composite's.

  Mine: each of its outputs sees ~11 frequency bins, so it cannot compare registers. Its texture view also reached
  only the noise bank and the residual's noise paths, not the residual's band gains or per-partial curves. Built
  since, not yet checked on the GPU: `--critic wide` (`losses.WideSpecDiscriminator`: strides down the frequency
  axis, one output per frame over the whole spectrum, bfloat16 convolutions, the recordings' spectra computed once
  per step), `--critic-reach residual` (the critic judges the output itself; its term goes back into the noise bank
  and the residual alone by a second, restricted backward), `--fm-weight`. Cost of the round-2 GAN per step (read,
  same batches): 0.72 s without, 1.21 s with it.

## 8. What the two smoke runs showed

**Smoke 1** (pools across steps, decay 0.97): diverged. At step 100 the residual made every term worse on both sets
(validation: 0.919 → 1.248 with the residual; pooled level 0.12 → 0.22, onset 0.18 → 0.28) and tilted the spectrum
(+1.5 to +4 dB below 1 kHz, −2 to −8 dB at 1-8 kHz). The lagging pools read the pooled terms as nearly matched.
Trained on one fixed batch with fixed seeds and no pool memory, the same residual lowered the composite steadily
(0.892 → 0.573 in 40 steps), so the gradient is right. Stopped at ~170 steps.

**Smoke 2** (pools over the batch): `runs/composite_train/smoke2/eval.md` and `eval_unmatched.md`, 64 excerpts per
set (held out: 13 pieces; training: 42 pieces, 26 with a single excerpt), one seed pair, no error bars. Read:

| | held out, matched | held out, unmatched | training, matched | training, unmatched |
|---|---|---|---|---|
| phase 6, physics | 0.726 | 0.770 | 0.808 | 0.843 |
| smoke 2, physics | 0.703 | 0.771 | 0.773 | 0.796 |
| smoke 2, with the residual | **0.690** | **0.806** | **0.752** | **0.763** |

- **Level matched, the score falls on both sets.** Unmatched, it falls on the training pieces but **not held out**:
  there it is flat for the physics and rises with the residual (0.770 → 0.806). The training made the model louder
  (read, the listening manifest: the physics 0.2-0.8 dB and with the residual 0.7-1.5 dB louder than phase 6 on
  the 8 test excerpts). My reading: that suits the training pieces' recording levels, which the score with the
  piece gains learns, and not the held-out pieces'. Level matching removes exactly that per piece. **So the
  held-out gain holds only for the tone, not for the absolute level.** (I told you earlier that the unmatched gap
  "came from the pieces' levels, not from the model"; that was wrong: the model got louder.)
- **The gain comes from the pooled, level and onset terms.** Held out, matched, phase 6 → smoke 2 with the residual:
  `pooled_exposed` 0.076 → 0.064, `between_pooled` 0.170 → 0.159, onset 0.113 → 0.103, `between` 0.183 → 0.179;
  `band` 0.249 → 0.250 and `partials` 0.261 → 0.260 do not move. Unmatched they get slightly worse (0.267 → 0.278,
  0.273 → 0.282).
- The training log (one batch every 10 steps, 33 values) is too noisy to show a trend: the score per batch ranges
  0.54-1.01 (sd 0.09) with the music.
- The residual after 332 steps: mostly +0.4 to +0.7 dB below 1 kHz; its noise paths stay quiet (the strongest 26 dB
  under the render, in the top octave). The model is still 1.9-2.7 dB too bright above 1 kHz on the validation
  excerpts (before level matching). Gradient agreement between successive steps (means over 50 steps): the physics
  −0.04 to +0.20, the residual +0.06 to +0.25; the old score's at phase 6 was +0.01 to +0.04 and +0.08 (review 6,
  2.3, measured differently). Not clearly better.
- Listening: `samples/composite_smoke2/` (the 8 test excerpts of `samples/phase6/listen/`).

## 9. Open: why the band and partial terms do not move

**What is known (read):**

- Their distance to the recording is not far above the model's distance to itself. Smoke 2's training log (means
  over thirds of the run): `band` 0.38-0.39 against itself 0.30-0.31, `partials` 0.40-0.41 against 0.31-0.32, a
  ratio of 1.25-1.30. If the takes vary as much as the model, the difference is what the model can still lower:
  0.08 (`band`) and 0.09 (`partials`), as much as or more than the pooled terms' (`pooled_exposed` 0.03-0.07,
  ratio 1.3-1.6; `between_pooled` 0.08-0.10, ratio 1.7-2.0; `level` 0.05, ratio 1.5-1.7). So in absolute terms
  there is room left; per cell it sits inside a scatter of about 0.3.
- **Even on one fixed batch they barely move.** In the memorisation test (the residual alone, one batch of 4,
  fixed seeds, 40 steps; the self term then on the read-by-read terms only) `band` went 0.405 → 0.390 and `partials` 0.453 → 0.436, while the total fell 0.892 → 0.573 through
  the pooled and onset terms. So it is not only a matter of generalising to new music.
- Their weights are the lowest (0.4 each), but their weighted share of the total is about the others' (0.10 each of
  0.69).
- 332 steps is 6.5 % of one pass.

**Possible reasons (mine, none tested), and what would tell them apart:**

1. **The gradient is mostly noise.** An L1 of logs per cell gives each cell a gradient of ±1 by the sign of its
   difference; where the take's scatter dominates a cell, the sign is nearly random, and the expected gradient is
   about `d/s` of its size (the `d²/s` mechanism of section 3, seen from the gradient's side). The self term doubles
   the noise. *Check:* each term's gradient on the residual and the physics, separately, over ~16 batches: its norm,
   its share of the total, and its agreement between batches.
2. **What remains is not in the residual's reach.** The read-by-read terms see timing (the take's onset scatter
   against the model's), beating and exact partial frequencies (fixed; a partial more than a bin off reads the gap
   next to it), and the hall's detail. The residual's outputs are smooth (20 ms, bounded) and predicted from context.
   *Check:* free outputs fitted to one excerpt with the band term alone (`scripts/fit_passages.py` with this score):
   how far can anything the residual can output take `band` down?
3. **What remains is the take's own randomness.** If the recordings vary from strike to strike more than the model
   (their `d(Y, Y')` larger than the model's 0.31), the band term is near its floor already and nothing will move
   it. *Check:* a lower bound on the takes' spread from repeated notes in the data (the same key, velocity and
   context struck more than once), compared with the model's at the same notes.
4. **Time.** 6.5 % of a pass. *Check:* the same terms after a longer run; cheapest to read on the training pieces,
   where generalisation is not in the way.

Checks 1 and 2 cost minutes; 3 is a small measurement; 4 is the long run itself.

## 10. Choices of mine to question

- The term set and the weights (section 5), not re-derived since the pooling changed.
- The `partials` gate: 50 dB under the loudest partial anywhere, the recording's; and the 70 dB selection.
- The exposure power 2, the age bands, the partial groups, the 20-reading minimum.
- Pooling the `onset` term on raw power (loud excerpts weigh more) while the others weigh excerpts equally.
- The energy form on the pooled terms, which doubles their gradient noise.
- Level-matched evaluation as the gate: it hides any change in absolute level (section 8).
- The residual's new outputs: the per-note noise at −60 dB and its range, 8 random inputs, the 32 per-partial curves.
- The rates: the residual at the full base rate, the physics at 0.3, warm-up 100.

## 11. The loss comparison (2026-10-03)

`runs/loss_compare/` (`chain.sh`, `post.sh`): the old loss (A) against the composite (B) and the composite with the
wide critic (C, weight 0.3, reaching the residual's every output), each from `phase6/env_fit2`, stage 1 then the
residual, every leg stopped early at its own held-out plateau; the envelope term off. One run per arm, no error bars
across training runs. Read:

- **Training:** A's stage 1 17,000 steps (stopped by hand; plateaued by ~6,000), B's 9,250 (79 min); with the residual
  A 7,750 steps, B 4,000, C 2,750 (all early stops).
- **Each loss wins on its own score.** Held out, level matched (64 validation excerpts): composite A 0.689, B **0.619**,
  C 0.706 (phase 6 0.731); old score A **0.829**, B 0.843, C 0.857 (phase 6 0.863). Not level matched, B's held-out
  gain on the composite disappears (A 0.734, B 0.740) and its old score is no better than the start (0.917 against
  0.920): its level suits the training pieces, as in smoke 2. On the training pieces B's composite falls far more
  (0.644 against A 0.800).
- **The old distances on 96 test excerpts** (variation off, which is how A trained and not how B and C did): A
  −0.046 ± 0.009 against phase 6, B +0.013 ± 0.008, C +0.038 ± 0.014.
- **Decays (N13; no envelope term in any arm):** only the composite pulled the upper partials' excess ring in: fade
  error at 1 s, partials 9-12 / 13-20, A +4.1 / +3.0 dB, B +2.2 / +0.5, C +2.0 / +2.4 (start +4.9 / +3.2). B pulls
  partial 2 low late (−4.8 dB at 1.5 s, few notes).
- **The GAN:** the critic won outright (end of C: critic loss 0.48, chance 1.5; generator term 2.9), the physics alone
  got worse through C's training (held out 0.63 → 0.79 on the composite: the noise bank, which the critic reaches, and
  the physics following the residual) and C ends worse than B on every score.
- The note bench (onsets, the pedal halo, attacks and texture in music) shows no clear difference between the arms.
- Listening: `samples/loss_compare_blind/` (phase 6, A, B, C with their residuals, shuffled per excerpt; the key in
  `runs/loss_compare/listen/blind_key.*`), the named page `runs/loss_compare/listen/index.html`. Not heard yet.

## 12. The GAN rebuilt (2026-10-03), stopped at the residual

Read, unless marked mine:

- **What tells renders from recordings** (`runs/gan_check/views1/`, a fresh critic per restricted view, held-out AUC):
  every band from 500 Hz up (0.98-1.00), the loudness-equalised signal (1.00), the mono mix (1.00) and the side signal
  (1.00, the easiest: critic loss 0.29); only < 500 Hz is close (0.72). No single artifact.
- **The critic:** wide (`--critic wide`), on the mono mix (`--critic-mono`), R1 10 (`--r1`): on per-octave gains between
  recordings (the known answer, `gan_check.py`, part `octaves`) the RMS of the second half's means 0.53 dB (0.67 without
  R1), held-out AUC against B's renders 0.78 (1.00 without). It keeps a bias: it pushes 4-8 kHz down by 0.5-0.9 dB even
  between recordings. Not the 24 kHz files' top: the recordings do not roll off near 12 kHz, and the renders are
  already 1-2.5 dB quieter there (re 2-4 kHz).
- **Scored only on the full output, the physics learns to cancel what the residual does.** C2 (`runs/gan2`, the critic
  reaching the residual alone): the physics alone 0.640 → 0.851 held out in 750 steps while the full output held; B
  drifted the same way, slowly (0.623 → 0.642). So C's degradation (section 11) was this, more than the noise bank.
  `--split-grad`: the physics learns from its own render alone, the residual from the full output. In C3
  (`runs/gan3/C3`) the physics held (0.624-0.633 over 1,250 steps).
- **Memory:** C skipped 205 of its 2,750 batches (the densest), C2 32 of 1,280, B 6 of 4,000: the GAN arms trained on
  lighter music. With the split, both pushes are balanced on the output audio and one backward goes into the residual;
  at a memory fraction of 0.9, C3 skipped 4 of 1,375.
- **Stopped by the owner:** with the residual adding nothing on held-out pieces (B: 0.629 → flat, then 0.650; C3: 0.619
  → 0.634, the physics alone 0.624-0.633), a GAN that acts only through the residual cannot be judged. Next: make the
  residual learn something that holds on new pieces; then the GAN again, or a GAN into the physics (with the split).
  The fresh-critic test of C3's renders (pass mark 1) was not run.

## 13. Three fixes to what the score reads (2026-10-03, branch `physics_revamp`)

Found by a review of the loss code (read):
- **Mirrored edges.** Every spectrum of the scored 2 s slice was a centred STFT with reflect padding, so the frames
  near either edge read a mirror of the slice's own audio (17 of 201 frames at each end at 8192 points: an attack
  20 ms before the end reads ~1.6 times its energy, `tests/test_partial_view.py`), and the FFT high-pass wrapped the
  slice's end into its start. Now (`losses.extend_scored`, `crop_frames`; `CompositeLoss.scored`) the spectra read
  8,400 samples of the rendered warm-up before the slice and silence after it, and keep the frames centred on the
  scored samples (the same count as before).
- **Onsets.** The partial view read each note from its MIDI onset (minus 10 ms) and counted its age from there; the
  sound starts ~7 ms later at middle C, ~15 ms in the bass (`ONSET_DELAY_MS`, the law the onset term already used).
  Now `composite.sound_onsets` / `scored_notes` add it, for training and for `validate_composite`.
- **A0, A♯0.** The partial view's floor was 30 Hz, above both fundamentals; now 25 Hz.

What they move (`scratch/loss_fixes/compare.py`, phase-composite B's `best.pt`, physics alone, 48 validation excerpts,
d(render, recording) / d(render, second draw), not level matched):

| | band | partials | between | pooled_exposed | between_pooled | level |
|---|---|---|---|---|---|---|
| before | 0.422 / 0.315 | 0.442 / 0.330 | 0.263 / 0.176 | 0.130 / 0.067 | 0.210 / 0.101 | 0.125 / 0.035 |
| after | 0.420 / 0.311 | 0.441 / 0.329 | 0.264 / 0.174 | 0.129 / 0.069 | 0.185 / 0.107 | 0.124 / 0.040 |

Five terms move by under 1 %; `between_pooled` (energy between the partials by time since the latest onset) falls
from 0.210 to 0.185 (its energy-score form 0.160 → 0.132): part of its mismatch was mirrored attacks and early
readings landing in the wrong age cells. The A0 floor changes nothing on these excerpts. Scores logged before this
change are not comparable on `between_pooled`. Whether the gradients changed is read by the coherence probe, not here.
