# Review 5: rounds 2 and 3 (the measurement suite, phase 3, the residual)

Scope: commits d954fb9..819fc65, i.e. everything since review 4: the response to review 4 and the round-2 re-fit
(`docs/plan_round2.md`, `docs/round2_results.md`), the measurement suite and phase 3 (`docs/tone_measures.md`,
sections 0–12), and the residual's diagnosis and redesign (section 13). The question asked was not whether the code
runs but whether the direction, the losses, the physics and the residual make sense, and what comes next. I read the
documents, the run index, the loss and residual code and the training script; I ran nothing. Where I give a number
that is not in the documents it is marked *mine, estimate*.

## 1. Verdict

**The direction is right, and the method has matured.** Round 1 fitted a model to a biased loss and then guessed
at what was wrong by listening. Rounds 2 and 3 replaced that with a chain that holds up: name the constituents of a
note, build a validated measure for each, audit every loss term against them, set from measurement what measurement
sees better than the loss (B, tuning, hall T60, the body's Q), fit on isolated notes what notes identify (velocity
map, knock level), and fit on music only what remains. The results follow the method:

- the level bias of the loss is gone (spectral convergence out, log band energies in; the gate is in the evaluation);
- the per-key inharmonicity error, the strike comb's sign, the hall's T60 and the body's ringing modes were each found
  by a measure, fixed, and confirmed on held-out notes;
- step 4 is the nearest model to the recordings on every term of the test distance beyond two standard errors, and the
  bench shows the brightness, level and early-decay gaps closed where the physics was set from measurement;
- the per-strike variation is the first part of the model that addresses "one-dimensional" at the source, and its
  spreads were set by a variance match, not fitted, which is the only way a median-seeking loss can be kept from
  shrinking them.

Four things need correcting or deciding before the next round:

1. **The attack physics is overdue.** The attack's wrong spectrum has been the largest measured gap since the second
   pass of the bench (10.3): too much at 8 kHz in every register, too little thump in the bass, too little 1 kHz
   knock in the middle. Three rounds of documents list "the attack in three parts" as the first physics to build
   (10.4, 11.4, 12.1 step 2) and it is still open. Step 2 refitted the *existing* knock parts instead and found that
   the structure is wrong (one decay per key cannot serve the low and the high bands; the knock noise reaches 8 kHz
   where the piano's does not; the envelope starts with a step). Fitting the level of a wrong structure, in notes or
   in music, is what produced the unexplained 4.5–7 dB disagreement between the note fits and every music term (12.6).
   Section 3 says what to build.
2. **The step-4 gain over round 2 is not attributed.** Step 4 had 450 minutes, a warm-up and cosine schedule, and the
   onset term; the round-2 control had 50 minutes at a constant rate. "Steps 1–3 cost 0.053 and the run recovered it
   and went 0.028 past" compares a long run against a short one. The bench evidence for each physics change stands on
   its own (per constituent, held out), so the physics is not in doubt; the music-distance attribution is. The same
   confound was flagged for the residual in review 4 and fixed with a control. The equivalent here is cheap relative to
   the run: the round-2 control continued with the step-4 schedule for the same number of steps.
3. **The residual's "ceiling" overstates what a predictor can reach, and the new residual's output language rewrites
   what the physics is for.** The free per-note outputs were fitted to each test excerpt's realised recording, so the
   0.136 includes the strike-to-strike variation that no context predicts. The documents' own numbers say how much:
   the fitted per-note level spreads 3.1 dB (sd) where the piano's own strike-to-strike spread is 1.5–2.3 dB, so a
   quarter to a half of the per-note level variance is unpredictable by construction, and the context features
   correlate with the fitted corrections at |ρ| ≤ 0.14. The aware residual's −0.010 may be a third of the reachable
   gain rather than 7 % of it. Separately, a ±12 dB gain curve per octave group over each note's life, inside the
   oscillator bank, is a learned re-shaping of the decay structure that the physics exists to model. Section 4.
4. **Validation cannot rank checkpoints, and listening has not caught up.** The 64-segment validation moves by ±0.005
   between checks and by 0.05–0.10 under the level wander in stage 1; effects of interest are 0.005–0.03. `best.pt` was
   picked from noise in both the step-4 and the aware-residual runs, and the documents say so. The owner has not yet
   heard step 4. "Synth" and "one-dimensional" are the two findings no measure has yet been shown to track, so the
   listening is the check that decides whether step 3 and step 4 moved them.

## 2. The losses

### 2.1 What makes sense

- **`PianoLoss` as the backbone.** L1 of log 1/6-octave band energies with the window growing towards the bass, the
  per-bin term kept only below 2 kHz where it is unbiased, the attack term in 1/3-octave bands at a 2.7 ms hop. The
  response to review 4 measured each term's level bias per band against the energy match and chose by the result,
  which is the right procedure; the 20 Hz high-pass was found by measurement (5 % of the energy is infrasonic
  rumble). The gate runs on every fitted model in the evaluation. All good.
- **Freezing what the loss cannot see.** B, the per-key stretch and the tuning stay at the measurements because the
  regression test (bass B × 2 against the seed-to-seed spread) showed no term sees them. The hall's T60 and band
  levels are set from free decays and frozen. This is the principle the whole project rests on, applied
  consistently: the loss is for what only music shows.
- **The onset term.** Its purpose is right: the training loss had no band below 400 Hz at onsets and smeared the bass
  over 341 ms, so the thump and the knock impulse were invisible to it. Its construction is careful: expected onsets
  from a fitted delay law, cells counted where either side stands over its own background (the symmetric rule that
  fixed step 2's selection bias), the running pool so that the optimum is the energy match whatever the batch size.
  One remark on the pool (read in `OnsetLoss.forward`): the per-band value comes from the running sums and the
  gradient from the batch's own log ratio with the pool's sign. That is not the gradient of any loss, so nothing is
  being minimised in the usual sense; its fixed point is the pooled energy match, which is what was wanted, and the
  bias gate passes on 512 segments. Fine as built, but it should be said in the docstring that the gradient's size is
  that of a batch-sized pool, so the effective weight is roughly 1/(1 − decay) times what the value suggests, and the
  weight 0.5 was chosen with that in it.
- **Training with the variation off and rendering with it on.** Under median-seeking terms a random model loses to its
  median (the measured 0.032), so this is the only consistent choice while the loss has no distribution term. It
  does mean nothing in training ever sees the variation; the bench's spread ratio is the only check on it, and it must
  stay in every report.

### 2.2 What does not yet

- **No term matches a distribution.** Class V and S constituents (section 4.2 of the design) still have no loss. The
  GAN branch is counted as "no measurable effect" after 506 steps at weight 0.1, which is not a test of the idea; the
  diagnosis that the per-frame terms shut the noise path faster than a critic can open it is right, and the fix
  suggested in round 2 (section 8, item 3) is the one to take: detach the residual's noise path from the spectral terms
  so that only a distribution-level criterion and the budget act on it. Without that, every texture experiment will
  read "the noise path is silent". Whether that criterion is a critic or P3's statistics (band-envelope variance,
  modulation power, correlations) is secondary; the statistics are cheaper to validate and to read.
- **The whole-excerpt band-energy term** has been proposed since round 2 and is still not built. It is unbiased by
  construction and costs nothing; the per-frame bias at 4–8 kHz came back on the fitted models in round 2 and the
  step-4 gate "fails by 0.1 dB" at 4 kHz. Build it.
- **The quiet stretches read 0.5–2 dB low** (P3's compressed envelope, 12.9) while the energy is right. The documents
  say "not looked into". Two candidates, one of which is the loss: (a) a per-frame L1 of log band energy is
  median-seeking over frames, and in a reverberant tail a band's energy per frame fluctuates; with k effective
  degrees of freedom the median sits below the mean by about 1.6 dB (k = 2), 0.6 (5), 0.3 (10), 0.15 (20) (*mine,
  chi-square medians*). With 2–10 bins per band in the bass and middle this gives 0.3–0.6 dB, so the loss explains a
  part but not the 2 dB at 1 kHz. (b) Missing energy between and after notes: the pedal halo (the bench's E1 found a
  3–5 dB gap in R2–R3 with the pedal down), the sympathetic bank (off), release and pedal noise. A split of P3's
  deficit by pedal state and by "tail vs between notes" would separate them, and it is a cheap measurement.
- **The knock's level in music is not identifiable by any band-level term at onsets** (12.6). This was found, and the
  right conclusion follows the project's own rule: a quantity the music loss cannot see is set on isolated notes and
  frozen, as B is. Step 4 instead let the knock impulse move −1…−6 dB under a loss that cannot see it. Once the attack
  is rebuilt, the knock's level should come from the note fit and be frozen, or the note fit and the music terms must
  first be shown to agree.

## 3. The physics

### 3.1 Steps 0–3 are sound

- **Per-key B and the comb's sign (12.2, 12.3).** The B error was a smoothing artefact of the initialisation, found by
  a per-partial cent measurement on held-out notes; the comb's sign was settled by two readings with different
  confounds. Both are the kind of finding the suite was built for.
- **Hall and body (12.4).** Setting T60 and the band levels from free decays and capping the body's Q are right. The
  Q cap removed the recurring narrow peaks the recordings never have. The side effects (level +1.6–1.7 dB in R3–R5,
  the slower rise) were measured and handed to step 2, which is how it should go.
- **The note fits (12.5).** The machinery is good: windows at each side's own onset, a free level per piece (the
  sessions differ by ~2 dB at equal key and velocity), piecewise-linear per-key corrections instead of free tables.
  The symmetric cell selection was an important fix and the biased fits were kept and labelled. The velocity map's
  result is modest (0.06 dB of shape) and the documents say so.
- **Per-strike variation (12.7).** The right response to "one-dimensional": the piano's notes differ strike to strike,
  not key to key, along nearly independent dimensions, so each is drawn on its own; brightness and decay keep the
  note's energy, so the level is a dimension of its own; spreads are solved from the probes' sensitivities by a
  variance match. Two cautions:
  - the onset jitter (3.29 ms sd) is set from a measure that cannot tell strike timing from detection noise, and the
    model's own detection noise was 0.6–2.8 ms. After step 4 the model's onset spread ratio fell to 0.64–0.74 with the
    jitter on, which the document attributes to the sharper attack tightening N0. That reading implies the recordings'
    4.5–5 ms includes a detection component, and the jitter to add is the root of the difference of the squares, not
    the full spread. Cheap check: N0's precision on step-4 renders with every dimension on except the onset;
  - the knock got zero spread because no bench target needed it, but the piano's attack at 1–2 kHz varies more than the
    model's and no dimension moves it. That is the attack structure again (3.2): a per-strike dimension on a
    string-borne precursor would have something to move.

### 3.2 The attack: what to build

Everything the bench and the fits found about the attack points one way, and the documents already say it (10.4 item
1, 12.5, 12.6). Stated as a build list:

1. **No knock noise above ~2.5 kHz.** The fitted noise's 8 kHz content is the +5…+10 dB percussive excess on the
   bench and the too-abrupt high attack in music (E4). Askenfelt's precursor and Bank's structure-borne thump do not
   put broadband noise there.
2. **A rise and a decay per band, not per key.** Step 2 showed one decay per key cannot serve 250 Hz and 4 kHz at
   once, and the envelope's step onset (`NoiseBank._env`, applied after the band split) spreads every band's onset
   over all frequencies. This lead was written down and not checked; it is the most likely single cause of the
   music-vs-notes disagreement on the knock's level, because every music term "fixes" the broadband click by turning
   the whole knock down.
3. **A structure thump limited to ~2 kHz** with the treble's long low ringing (0.2–0.5 s), replacing the treble's short
   low thud.
4. **A string-borne precursor**, 1–2 ms, up to ~5 kHz, ~−10 dB re the first transverse wave, scaling with the blow,
   through the body like the strings. It is the part that grows with velocity the way the piano's 1 kHz knock does
   and the model's does not.

Fit it on isolated notes with the step-2 machinery (the attack re the early window, and between the partials), then
check that the music terms and the note fit agree on its level before any music run. Only then does a per-strike attack
dimension make sense.

### 3.3 Other physics items

- **The sympathetic bank** is still off in every training run. The bench measured the pedal halo gap (3–5 dB in
  R2–R3) and the literature puts the pedal's extra energy below 2.5 kHz. Memory is the stated obstacle (2 clips per
  batch). A bank limited to R2–R3's keys and partials below 2.5 kHz is a fraction of the full 88 × 4; build the cheap
  version and turn it on. It is also a candidate for the quiet-stretch deficit and for the owner's "less powerful".
- **Phantoms.** Step 4 raised the phantom table by +6.5…+10.9 dB in R4–R6, as the trial did, and the mechanism
  question from review 4 is still open (A8: a line at exactly 2f_j in a minority of notes, not behaving like a
  phantom). The bench says R4–R5 came close, so this is not urgent, but the frozen-or-free rule applies: if the music
  loss raises something whose mechanism is unestablished, it may be filling in for something else.
- **The level wander in training** (12.9, `lr_checks/notes.md`): at a constant rate the level parameters follow each
  batch's pieces, because the sessions differ by ~2 dB at equal key and velocity. The schedule tames it; the cause is
  a missing nuisance parameter. The note fit already has a free level per piece. Give the music training the same
  (one gain per piece, zero-mean prior, discarded at evaluation). It would cut the validation noise, stop the mic
  gain from averaging over sessions, and sharpen every other parameter's gradient.

## 4. The residual

### 4.1 The diagnosis was done well

Section 13.1 is the right kind of experiment: replace the net by free outputs in its own language, score on noise
seeds the fit never saw, control with constant outputs, correlate the fitted corrections with context features. It
settled that the output language was not the limit of the GRU residual. The move to a residual that sees the physics'
own state (per-note expected energies per octave group, the mix, age, damper, pedals) and attends across sounding notes
is a reasonable design answer, and training it on frozen physics keeps the physics readable, which is the lesson of
round 2's co-adaptation.

### 4.2 Three corrections

1. **The ceiling is an upper bound on a different question.** It asks how much a per-note correction fitted to *this*
   recording can remove; the residual is asked how much a function of the context can remove. The gap is the
   strike-to-strike variation (class V), which the project's own bench says is 1.5–2.3 dB of level at equal key and
   velocity, plus brightness, attack and decay spreads. The fitted per-note outputs spread 3.1 dB in level, so at
   least a quarter to a half of that output's variance is noise to any predictor; the feature correlations (|ρ| ≤
   0.14) say the predictable part is small. A reachable ceiling needs a fit that cannot see the realisation: fit free
   per-note outputs on training pieces, then predict them from context on held-out pieces with a simple regressor and
   report the explained variance; or train the aware residual on all years and watch train and test diverge. Until
   then "7 % of the ceiling" should not be the headline; "−0.010 ± 0.004, twice the GRU, reachable fraction unknown"
   is.
2. **The memorisation test shows capacity only.** A 470k-parameter net fitting 8 excerpts of 2 s to 86 % of a
   per-excerpt free fit is what one expects of any such net; it does not bear on what carries over. The document says
   this in its last paragraph; the section's title and lead should say it too.
3. **The output language should be budgeted and read before it is scaled.** Per-note gain curves of ±12 dB per octave
   group over the note's life, inside the oscillator bank, can re-draw the two-stage decay, the damper's action and
   the pedal's effect per note. With the physics frozen this does not corrupt the physics' parameters, but it does
   make the rendered decay whatever the net says, and the project's contract is "physics for the tonal part". Two
   measures keep it honest: a budget on the curves' size and their time derivative (smooth, small), and the read-out
   that 13.2 lists as "not yet looked at". If the curves are systematic (every note's upper groups falling faster
   after 0.3 s, say), that is a physics finding (decay tilt, bridge conductance, the aftersound) and belongs in the
   physics; the residual's best use so far has been as a detector of missing physics (it turned the knock down in
   round 2), and that is worth keeping as its stated role.

### 4.3 What the numbers say about priorities

The residual adds 0.005 (GRU) or 0.010 (aware) on test. Steps 0–4 of the physics moved the test distance by 0.028 over
round 2 and closed bench gaps of 100–200 cents, 1–2 dB and 10–20 % T60. The attack excess still on the bench is
5–10 dB. The residual is not where the next 0.03 is; the attack is.

## 5. Method and protocol

- **Validation.** Fixed segments and a fixed noise seed are in place (read in `train.validate`), so the ±0.005
  between checks is the model moving, mostly its level parameters. Report validation as the paired difference to the
  start checkpoint on the same segments with its standard error, and use enough segments (the test comparison uses 96
  × 2 seeds and resolves 0.005) that a checkpoint choice means something; otherwise take `last.pt` by rule, as the
  evaluation already does.
- **Controls.** Every claim of the form "X improved the distance" needs the same schedule without X. Step 4 has none;
  the aware residual has one (the GRU on the same physics). Add the continued round-2 control (section 1, item 2).
- **Variation on and off.** The documents are careful to say which numbers were taken with the variation off (distances)
  and on (bench, listening). Keep that, and add the spread ratio to the headline table so that the variation's purpose
  is reported alongside its cost.
- **One year, one piano.** Everything measured is 2018's Disklavier in 2018's hall. The shared physics (hammer law,
  knock, decays) is confounded with that instrument until a second year is in. The bench is also thin in R1, R6 and R7
  on one year. All years are converted and counted (7× the isolated notes); the per-year mining (B, stretch, floor,
  hum, T60, velocity map) is the next data step and the one that makes the residual's generalisation testable.
- **Documentation.** `tone_measures.md` is the project's status, its design, its lab notebook and its results, at
  1,180 lines. It is readable and honest, but a one-page decision log (what is set from measurement, what is fitted on
  notes, what on music, what is frozen, as of which checkpoint) would stop the next reviewer, or the owner, from
  reconstructing it from twelve sections. The README's status list is nearly that already; move it to its own file and
  keep the README short.
- **Listening.** Step 4's listening sets exist with player pages and have not been heard. Two A/Bs would answer the
  two open perceptual findings directly: step 4 with the variation on against off ("one-dimensional"), and step 4
  against the round-2 control ("synth"). Nothing in the suite yet predicts either, so these are the measurements that
  are missing.

## 6. Next steps, in order

1. **Listen** to step 4: variation on vs off, and vs the round-2 control and the recording. Half an hour; it decides
   whether steps 3–4 moved the two perceptual findings.
2. **The attack in three parts** (3.2): per-band knock envelope and decay, no noise above ~2.5 kHz, thump ≤ 2 kHz with
   the treble's long ringing, the precursor. Fit on isolated notes; confirm the note fit and the music terms agree on
   its level; then a per-strike attack dimension.
3. **Training hygiene** (3.3, 5): a per-piece gain nuisance, the whole-excerpt band-energy term, validation as paired
   differences; the continued round-2 control for the step-4 attribution.
4. **The sympathetic bank** for R2–R3 below 2.5 kHz, cheap enough to train with; the P3 quiet-stretch deficit split by
   pedal state before and after.
5. **All years**: per-year mining and bench, the shared physics trained across instruments.
6. **The residual**: the read-out of the aware residual's curves; a curve budget; a reachable ceiling; training on all
   years tracking train and test. Move anything systematic into the physics.
7. **Texture**: detach the noise path from the spectral terms; then P3's statistics or a critic with real weight, on
   the noise path only; listen.

The order follows the sizes of the measured gaps and the project's own rule of measuring before building. Items 1 and
3 are short; item 2 is the round.
