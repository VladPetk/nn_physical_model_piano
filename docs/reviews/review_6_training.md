# Review 6: the training approach (phases 3–6)

Scope: the owner's question of 2026-10-02: is what we measure, what we compare and how we train right for reproducing
a piano note from start to finish, in context; why phases 4 → 5 → 6 sound alike; what the envelope term is worth; why
the residual learns nothing; whether fitting notes and freezing is the way; 24 against 48 kHz.

What I did: read the losses, the training loop, the residual, the envelope term and the physics (`pianonn/losses.py`,
`train.py`, `residual.py`, `envfit.py`, `physics.py`, `synth.py`), the logs of the step-4, phase-4, phase-5, phase-6 and
aware-residual runs (`log.jsonl`), and sections 4 and 13–17 of `docs/tone_measures.md`. I did not run the test suite.
I took four measurements that the question needed and nobody had taken (scripts in `review_6_scripts/`, all on the
phase-6 checkpoint unless said, validation pieces of 2018). Claims are marked *read* (code, logs, documents) or *mine*
(measured or reasoned here). Proposals are in section 5 and are proposals.

**Corrected 2026-10-02 after a critique (section 7).** Withdrawn or changed: the size of the gap and the fine term's
worth (2.1), "at this loss's optimum" (2.3), the third bullet of 3.1, "the same cause in every case" (3.4), the
scoreboard of 5.1a, and the claims made for 5.2 and 5.4. The measurements stand; several readings of them leaned
further towards "the loss is the culprit" than the numbers carry. Read section 7 before sections 1 and 5.
Section 8 is the proposal for moving forward that came out of it; it replaces the order of section 5.

## 1. Verdict

The model (modal strings, noises, body, hall) is a reasonable thing to fit. **What is wrong is the learning signal and
the scoreboard, and they are the same object.** Every run since round 2 has been trained and judged by a cell-by-cell
distance between one render and one recording. That distance is saturated: most of its value is not model error, the
model's parameters sit at its optimum, and each phase since step 4 has moved it by 2–17 % of what is left, inside
its own noise. Nothing audible should be expected from such moves, and the owner hears none.

The work of phases 3–6 went around the loss instead of through it: measure a feature on a few dozen isolated notes,
build physics for it, fit it on notes, then freeze it because the music loss pulls it back. `tone_measures.md` 4.2
already says what the loss needs per class of constituent (per-key long-window terms, distribution terms, a critic or
texture statistics), and 7 item 5 records it as *not started*. It still is not. That, not any one missing piece of
physics, is the main thing to change.

## 2. Four measurements

### 2.1 What the loss reads (`loss_floor.py`, `level_match.py`)

96 validation excerpts (13 pieces), 2 s windows, the training loss as the runs use it (band + 0.25 fine + 0.5 attack +
0.5 level; 1 = 10 dB). Mine.

| pair compared | total | band | fine | attack | level |
|---|---|---|---|---|---|
| a render against itself, another noise seed (variation off) | 0.18 | 0.10 | 0.16 | 0.04 | 0.02 |
| the recording against itself + 1 dB | 0.21 | 0.09 | 0.10 | 0.08 | 0.09 |
| the recording against itself 5 ms late | 0.24 | 0.06 | 0.18 | 0.20 | 0.05 |
| white noise against white noise | 0.56 | 0.26 | 0.60 | 0.26 | 0.05 |
| the recording, left microphone against right | 0.64 | 0.29 | 0.54 | 0.28 | 0.16 |
| **the model against another draw of itself** (per-strike variation on) | **0.65** | 0.31 | 0.49 | 0.31 | 0.14 |
| the model against the recording, level matched per piece | 0.81 | 0.38 | 0.63 | 0.34 | 0.22 |
| **the model against the recording** (as every report scores it) | **0.86** | 0.40 | 0.64 | 0.35 | 0.25 |
| another excerpt of piano music, level-matched, against the recording | 1.91 | 0.88 | 1.26 | 0.76 | 0.67 |

- **A model that was the piano would score about 0.65, not 0.** Two draws of the same model, with the strike-to-strike
  variation the bench measured on the piano (and which is, if anything, smaller than the piano's), are 0.65 apart. The
  recording is one take; the loss cannot tell "wrong" from "another take". (The piano's true take-to-take distance
  cannot be measured: there is no second take. This is the model's stand-in for it.)
- **0.05 of the rest is the recording level of each piece.** The validation pieces sit −3.8 … +3.1 dB from the model
  (sd 2.0 dB over 13 pieces, 1.0 dB within a piece), mostly the same in every octave band: a gain per session that no
  MIDI predicts. Training has a gain per piece; evaluation does not.
- **What is left is about 0.16.** Phases 4, 5 and 6 on these excerpts: 0.846, 0.873, 0.860. Phase 5 − 4 +0.027 ±
  0.008, phase 6 − 5 −0.013 ± 0.006, phase 6 − 4 **+0.014 ± 0.009**: by its own loss phase 6 is slightly *worse* than
  phase 4. −0.013 is what a 0.06 dB gain change does to this number.
- **The fine term carries almost no information.** Model against recording 0.64; two unrelated noises 0.60; the
  recording's other microphone 0.54. Per bin, the model is no nearer the recording than noise of the right level is.
  The attack term is 0.35 against a floor of 0.31, and 5 ms of timing alone costs 0.20.

### 2.2 Which part of a note the loss weighs (`grad_coherence.py`, `loss_floor.py`)

Per cell of the loss window (octave group × 20 ms, 48 training excerpts), the share of the physics' expected energy by
the age of the note it comes from. Mine.

| note age | 0–50 ms | 50–100 ms | 0.1–0.2 s | 0.2–0.5 s | 0.5–1 s | 1–2 s | 2–4 s | 4 s + |
|---|---|---|---|---|---|---|---|---|
| share of the loss's cells | 14 % | 13 % | 20 % | 27 % | 13 % | 7 % | 3 % | 3 % |

75 % of what the loss looks at is the first half second of notes; everything after 1 s gets 12 %, after 2 s 6 %. By
frames (96 validation excerpts): 90 % of the band term comes from frames within 0.5 s of the latest onset, 4 % from
frames more than 1 s after it. **The owner's complaint ("nothing past half a second") still holds in phase 6**; the 2 s window is not the
cause, the music is dense and every cell counts the same. Only the envelope term (3.1) looks later, and it is weak.

### 2.3 Whether training still has anywhere to go (`grad_coherence.py`)

Cosine between the loss gradients of independent batches of 4 excerpts (16 batches, 120 pairs), the phase-6 model with
its piece gains. 0 = the batches agree on nothing; the value is roughly the share of a batch's gradient that is signal.
Mine.

| parameters | mean cosine | sd |
|---|---|---|
| physics, all | +0.01 | 0.15 |
| strings' decay | +0.03 | 0.14 |
| per-key level | +0.02 | 0.16 |
| `partial_gain`, colour tables | +0.03, +0.04 | 0.10, 0.17 |
| room (body FIR) | +0.02 | 0.15 |
| the residual | +0.08 | **0.47** |
| noise bank (frozen in phases 4–6) | +0.26 | 0.32 |

- **The physics and the room are at this loss's optimum.** 1–4 % of each step's gradient is direction, the rest is the
  batch. More minutes on this loss will move nothing; Adam at these rates random-walks around the optimum, which is
  the ±0.03–0.08 bounce of the validation loss between checks in every stage 1 (read, logs).
- **The residual is not at an optimum, and its gradient is dominated by one direction that flips sign from batch to
  batch**: pairs of batches agree or oppose at about ±0.5; that one direction holds half the gradient's power, and its
  sign follows the batch's level error only loosely (r = −0.42, 16 batches; I did not identify it further). With it removed
  the rest agrees at +0.19.
- **The frozen attack noises have the most consistent gradient in the model** (+0.26): the music loss disagrees with
  the note fit there, and freezing hides it (4.3).

### 2.4 The beats were never fitted (`unison_check.py`)

The unison mistuning that makes each key's partials beat is two numbers per key, drawn from a random generator at the
first commit (`physics.py`, seed 1234). After every run to phase 6 it correlates 0.98 with that draw and has moved by
0.14 cents on a mean of 0.81. The aftersound's level is one number per key and mode for all partials; a partial's decay
is a smooth form in frequency. So how each partial of each key lives after the first half second (its own beat, its own
knee) is a prior plus a random draw, not this piano. Read and mine.

## 3. The owner's points

### 3.1 The envelope term

What it is (read, `train.py`, `envfit.py`): besides the 8 music excerpts, each step renders 2 notes drawn from a list
of 804 (0.2 % of 2018's 362,000 notes: of a capped draw of 3,678, those with some partial clear of every other
sounding note; see 8.4 for the size of the pool), reads on
average 18.5 values per step (one partial's level at 0.2–2.5 s re its level at 0.1 s) and takes their L1 distance to
the recording's. Its gradient on the decay parameters is about 3 times the whole music loss's.

What the run shows (read, phase-6 log; mine where said):
- **It did not train.** On the training notes it reads 3.78, 3.82, 3.57, 3.66, 3.74, 3.73, 3.64, 3.60, 3.63, 3.64 dB
  over the ten tenths of the 310 minutes; held out 4.01 → 3.81. Across the notes of a register the piano's own fades
  spread by 4–8 dB (IQR, partials 13–20, 17.3), so a median-seeking term on single notes cannot go much below this:
  it is likely at its own floor (mine, not measured).
- **The "halving" of the excess ring is mostly the fit before the run** (`env_fit2`), scored on notes of which 1,800
  of 2,200 come from the pieces it was fitted on. Held out: partials 9 and up improve on 5–17 readings per cell,
  partials 5–8 get worse (17.6).
- **"The music loss does not pull against it" is not what the log says.** The cosine between the two gradients is
  negative in all ten tenths (−0.0005 … −0.017). With gradients this noisy (2.3: 1–3 % signal on these parameters), a
  full opposition would read about −0.01 to −0.05. And at a point where the sum of two terms has settled, their mean
  gradients are opposed by definition. The likelier reading: the fit moved the decays, the music loss pulls back, and
  the run sat at the balance (mine; the summed-gradient check 17.6 lists as open would settle it).

So the owner's doubt is right: 18 readings a step from a hand-picked 0.2 % tell the decays little, loudly. The idea
inside it is right, though: read every note's partials through its life, in context. It is applied at a hundredth of
the scale it needs, on a side render, on a selected sample.

### 3.2 Phase 5 against phase 6 by ear

Nothing audible is expected: the two differ by 0.013 on 0.86 (2.1), and the fades that changed are in the part of a
note that both the loss (12 % after 1 s) and dense music expose least.

### 3.3 The residual

The owner's reading is right: it is not useless, it is set up so that it cannot learn. Four reasons, in order of
weight:

1. **The loss leaves it almost nothing.** A residual that is a function of the context can only move the render
   towards the *typical* take; the per-key tables have already taken what is typical per key. Everything specific to
   the take reads as error whatever it does (2.1), and anything it adds that is not exactly the take costs: the noise
   path was shut by the loss in round 2, the ring-up's fine structure cost +0.05, the per-strike variation +0.05 (0.86
   → 0.91). The free per-excerpt fit of 13.1 gained 0.14–0.27 because it was fitted to the take.
2. **It may only turn the physics' own knobs**: gains per octave group of the existing partials, the onset-time knobs,
   16 band gains. Nothing the physics lacks (other partials, other noise, other time structure inside a group) is in
   its reach. "The physics cannot model everything, so the residual learns the rest" is not what was built.
3. **It has never been trained.** Read, logs: its one dedicated run was 1,000 steps (30 min); validation with it went
   −0.004, −0.014, −0.012, −0.021, −0.015 and was still moving; it gained −0.010 ± 0.004 on test. Per minute that is
   the best return of anything since step 4, and it was dropped. Phase 6 started a fresh one at 55 % of the run: about
   600 steps at the full rate (200 of them warm-up), then the cosine decay, 1,840 steps in all, under the budget. Every
   residual so far has trained only in the tail of a run, into a falling rate.
4. **Its gradient is mostly one sign-flipping direction** (2.3), so at batch 8 it is pushed up and down by something
   it cannot predict: from check to check its effect on validation (with − without) swings between −0.010 and +0.022
   in phase 6 and between −0.011 and +0.038 in step 4 (read, logs).

### 3.4 Fitting notes, priors, freezing

Read across 12.5–17: the knock fitted on notes, music wants it 4.5–7 dB lower, frozen. The fundamental's early fade
fixed on notes, "back after training". The decays fitted on notes, frozen in phase 5. The Q change and the ring-up
"cost" +0.03 and +0.05, "trained anyway". Each time the note measure and the music loss disagree, and each time the
parameter was protected. In every case the disagreement has the same cause: the cell-by-cell loss puts anything
noise-like, timing-scattered or finely structured below its true level or smooths it away (4.1 of `tone_measures.md`
says so itself). Those are bug reports about the loss. Two more costs of the note route:

- **It starves on data.** 56 isolated tenor notes; "37 R3 notes with 2 s clear"; cells of 5–17 readings. The finding
  that the piano's variety is "strike to strike, not per key" (section 15) rests on 42 notes of 13 keys, about three
  per key, with other notes ringing under them. Physically one expects the opposite for beats and the knee (they
  belong to the key's unison and its place on the bridge), and the model then got random per-strike draws where
  per-key structure may be what is missing (2.4). 2018 has about 4,000 strikes per key; only music has that data.
- **Freezing has two different meanings here.** B and the tuning are frozen because a spectral gradient cannot move a
  frequency further than a bin: that is sound. Knock, decays and hall are frozen to protect them from a biased loss:
  that is the pattern to stop.

A fit on notes is a fine way to *start* a parameter. It should then train, and where training drags it away, the loss
gets fixed.

### 3.5 24 kHz

Fine for now, not for the end. At 24 kHz the top octave has 2–5 partials below the ceiling and the attack's content
above 11 kHz is gone; the model also stops at 96 partials (A0 to ~5 kHz). None of that is why the tenor sounds
synthetic. Going to 48 kHz roughly doubles to triples the cost of a step (samples, partials) and needs the body, hall,
noise bands and the loss's windows (now given in points, not ms) redone, while the per-key physics carries over. Do it
as the last stage, after the objective is right; meanwhile write new code in seconds and Hz, not points.

## 4. What else is off

1. **The scoreboard is the training loss.** So every report shares its blind spots, and a change that is right by
   measurement (the ring-up) reads as worse. It also resolves nothing at this point (2.1), and `best.pt` has been
   chosen from noise.
2. **Run length and schedule.** Phases 4 and 5 trained 1,320–1,350 steps: a quarter of one pass over 2018's 22.9 h.
   Their training terms are flat after the first few hundred steps because the physics is at the optimum, not because
   of the schedule; but a network cannot be trained in such runs at all (3.3).
3. **Level parameters at 20–50 times the base rate under Adam** wander by up to 1.5 dB (read, 12.9, 17.6) and take the
   validation loss with them. An average of the weights over the last steps would remove most of it for free.
4. **One strike variation, set by hand, off in training.** It exists because the loss would shrink it; so the part of
   the model meant to make notes differ is the one part no training ever sees.

## 5. What I would change (proposals)

In this order; each is a few hours to a day of building and a short run.

1. **A scoreboard that can see.** (a) Report the excess over the model's own second draw, with the level matched per
   piece, as paired differences on a fixed set. (b) Train a small critic to tell held-out recordings from renders of
   a frozen model and report its accuracy, overall and by register, note age and density. It is a measure in the
   project's sense (50 % = cannot be told apart) and it says where the model is given away, instead of the ear
   finding one thing per phase and a hand-built measure following it. I cannot hear the samples; this is the
   instrument that replaces my guess at what "synth" is.
2. **Explain the take, then compare.** Give each note a few latent numbers for its strike (level, brightness, decay,
   onset time, the same dimensions the per-strike variation has) and infer them from the recording with a small
   encoder, under a prior that keeps them small. The loss then compares the recording with the model's account of
   *this* take. 13.1 is the proof it works: free per-note values took 0.14–0.27 off in 150 steps. What it buys: a
   clean gradient for the physics; the spread of the inferred values *is* the per-strike variation, learned in training
   on every note instead of matched by hand on a bench; and the residual becomes a regression of those values on the
   context, which also answers how much of them the context can predict at all (review 5's open question).
3. **One note-attributed term in the main window instead of the 2-note side term.** The model knows every sounding
   note's partial frequencies and can render each note alone, so it can read each partial of each note in the training
   window wherever that note dominates, weigh by note and age rather than by cell, and do it for the hundreds of notes
   a batch already holds. A share of batches with 6–8 s windows, so that slow beats are seen within one strike.
4. **Then let the per-key structure go free**: mistuning, aftersound and decay per key and partial, with today's forms
   as priors rather than as the parametrisation. With 2 and 3 in place the data can hold them; today it cannot.
5. **A critic as a loss term on the whole output**, with the spectral terms as anchor, and a signal path for the
   residual that the cell-by-cell terms do not govern. The round-2 GAN (506 steps, weight 0.1, noise path only) was
   not a test of this.
6. **One joint run**: everything from notes as a start, nothing frozen but the frequencies; constant rate, averaged
   weights, the residual from the beginning. Then all years, then 48 kHz.

Item 1 is the one I would do first whatever else is decided: it is cheap and it tells whether 2–5 are aimed right.

## 6. What the earlier reviews missed

Review 5 called the direction right, endorsed "freezing what the loss cannot see" as the principle the project rests
on, and put a distribution-level term last on its list. It audited each loss term's bias and never asked the two
questions that take ten minutes: what does the loss read for a perfect model, and which part of a note does it weigh.
Both answers were available then and both change the plan. It also accepted the conclusions drawn from a few dozen
isolated notes at the sample sizes given.

## 7. Corrections (2026-10-02, after a critique of this review)

The owner passed on a critique of this review. I checked each point against this review's own outputs, the code and
`tone_measures.md`. *Critique* marks a number I took from it and did not re-measure.

1. **The gap in 2.1 compared unlike things.** 0.86 is a render with the per-strike variation off; 0.65 is two renders
   with it on (`level_match.py`: `P = r(0, False)` against `Q0`, `Q1`). Like for like from my own run: variation on
   against the recording 0.9075, against its own other draw 0.653, gap 0.25. The critique's version (variation off
   against one varied draw 0.59, gap 0.27 ± 0.05 at 2 se grouped by piece; critique) agrees. After the per-piece
   level about 0.21, not 0.16. The floor is also not a measured floor: it is whatever variation was set by hand, which
   17.3 found smaller than the piano's. "About 70 % of the number is floor-like, and the phases moved it by 0.03 or
   less" stands; "0.16 is left" does not.
2. **"The fine term carries almost no information" is wrong**, and the table in 2.1 says so: before weighting, the
   excess over the model's own second draw is fine 0.15, level 0.11, band 0.09, attack 0.04. White noise against white
   noise was the wrong reference; noise with the recording's spectral envelope reads 0.95 on the fine term (critique).
   After weighting the fine term's excess is 0.04 of the total, which is a statement about its weight, not its worth.
3. **"At this loss's optimum" (2.3) does not follow from the cosines.** At an optimum the expected cosine is 0; a
   cosine of +0.03, if real, means the signal outweighs the noise after some tens to hundreds of steps. With 16
   batches the means are good to about ±0.04, so +0.01 … +0.04 can be told neither from 0 nor from each other. The
   evidence for "little left to gain on this loss" is the flat training terms over phase 6's 4,800 steps (read), and
   it is weaker than "will move nothing".
4. **3.1's third bullet is withdrawn.** 17.6 hedged its claim ("a weak conflict cannot be excluded") and gave a second
   piece of evidence I passed over: the envelope fit moved the upper partials' fades by several dB and the music
   validation did not change (0.9445 against 0.948). That supports "the music loss barely sees the tails", which is
   also what 2.2 of this review measures. My "the music loss pulls back" contradicted my own 2.2. The cosine's sign is
   consistent (negative in ten of ten tenths) but its size (−0.0005 … −0.017) is near orthogonal; it supports neither
   reading strongly. The first two bullets of 3.1 stand.
5. **"The same cause in every case" (3.4) was asserted.** For the knock a timing-scatter bias is documented (4.1);
   for the ring-up and the Q change the texture penalty is a fair account; for the decays the evidence is a loss that
   is indifferent, not biased; and 17.5's second reading (long decays standing in for energy between the partials
   that the model lacks) is open. That reading is about the model, not the loss, and section 1's "the model is a
   reasonable thing to fit" rests on nothing I measured.
6. **5.1a can be gamed.** d(model, recording) − d(model, model′) falls without bound as the model's variation grows
   (for a Gaussian under L1: 0.80 σ − 1.13 σ). With half the second term it is the energy score, which is proper
   (Gneiting & Raftery 2007; cited). While the variation is fixed by hand the two differ by a constant; once it is
   learned (5.2) only the second is safe. Simplest for now: the variation-off score against the like-for-like floor,
   errors grouped by piece.
7. **5.1b (the critic as a measure) skipped this project's own rule for a measure.** It needs: two seeds of one model
   read 50 %; known changes (knock +6 dB, upper decays doubled) are detected; and a guard against the easiest cue
   being the recording chain (noise floor, audience, hall), which would pin it near 100 % and say nothing about notes.
8. **5.2 claimed more than 13.1 shows.** 13.1 shows free per-note values lower the loss, not that the physics gets a
   cleaner gradient; its per-note level spread was 3.1 dB against the piano's 1.5–2.3 dB, so free values soak up
   model error and their spread is not the per-strike variation. If built: zero mean per key and velocity bin, a
   prior that holds the spread, and scoring with values drawn from the prior, never inferred from the scored take.
9. **5.4 on the mistuning was aimed wrong.** `raw_unison` is already a free parameter (it moved 0.14 cents). It stays
   at its draw because a beat rate behaves like a frequency under a magnitude loss, the reason 3.4 accepts for
   freezing the tuning. It needs a measured start per key (and a check that a key's beat rate survives the retunings
   within 2018). I do not fully share the critique's "the band term averages beats away": below about partial 8 a
   1/6-octave band holds one partial and a 10 ms frame follows a slow beat (mine, not measured); the conclusion is
   the same.
10. **Run length is the owner's decision.** 4.2 and 5.6 implied longer runs against the standing short-run rule
    without asking. It is a question, not a finding.
11. Smaller: the ± of 2.1 treat 96 excerpts as independent (13 pieces: they are too narrow, and phase 6 − 5 may not
    be distinguishable from 0); "0.06 dB" uses the steepest slope, so read "at least"; "90 % within 0.5 s of the
    latest onset" mostly describes how dense the music is (the note-age table is the better statement, and it too
    partly describes the music); the term of 5.3 should weigh readings by whether the note is unmasked, as N13 does,
    not by age.

**What this changes.** The errors lean one way: each made the loss look more saturated or more guilty than the
numbers show. The owner's brief carried that thesis and I read towards it. Standing: most of the 0.86 is not model
error; the phases moved it within its noise; the loss's weight is on young notes; the residual's set-up (3.3); the
envelope term's scale and flat curve; the mistuning is still its random draw. Not standing: that the loss is *the*
cause, and the order of section 5. What makes the renders sound synthetic is not known; neither this review nor its
critique measures it, so no proposal here can claim to aim at it.

## 8. How to move forward (proposal, 2026-10-02; replaces the order of section 5)

The owner's proposal: take the phase-6 model and unlock the residual. Let it learn more than the physics' own knobs,
and let it train properly, without a throttled learning rate. Longer runs are accepted as a requirement. This section
is that proposal with one listening test in front of it, and (8.4) the note objectives moved into training.
Nothing here is built or agreed in detail.

### 8.1 What the logs and documents already say about the residual

- **It has barely trained** (read, 13.2, `runs/residual/aware/`). Its one dedicated run was 1,000 steps of 8 excerpts
  of 2 s: 4.4 h of audio, 19 % of one pass over 2018's 22.9 h. On 96 excerpts of the training pieces it gained
  0.016 ± 0.004, on test 0.010 ± 0.004: it had not fitted even the pieces it saw.
- **It is throttled in phase 6** (read, `train.py`, `chain.sh`): half the base learning rate (`lr_scale` 0.5 for
  `context.`), switched on at 55 % of the run, the cosine decay from 70 % to 0.05 of the rate, and a budget
  (`--budget 0.05 0.05 0.1`) that penalises its outputs.
- **By the loss number its outputs are not the limit** (read, 13.3). Fitted to 8 excerpts alone it took the loss from
  0.727 to 0.489, 86 % of the way to free outputs in its own language (0.452).
- **Nobody has listened to a fitted excerpt.** `runs/residual/ceiling/` and `memorise/` hold scores and no audio. So
  it is not known whether 0.49 sounds like the piano.

Three things are unknown at once: what is audibly wrong, whether the residual's outputs can express it, and whether
the loss rewards it. Section 13 answered the second by the loss number only, which assumes the third.

### 8.2 Step 1: listen to fitted excerpts

Fit a handful of validation excerpts directly (as `scripts/residual_ceiling.py` does; about 10 s each so that they
can be judged by ear) and render four versions of each as WAV in `samples/`:

1. the physics alone (phase 6);
2. the residual's current outputs, fitted to that excerpt;
3. extended outputs, fitted to that excerpt;
4. the recording.

Extended outputs (mine, a first guess; the listening decides): a gain curve in time per partial rather than per
octave group; a noise path attached to each note, for energy between its partials that follows the note; finer
global noise than 16 bands. Not a frequency or beat-rate output: a magnitude loss cannot steer those (section 7,
item 9).

| the owner hears | it means | then |
|---|---|---|
| 2 sounds like the piano | the outputs are enough; the limit is training | the long run with the current residual |
| 2 synthetic, 3 like the piano | the outputs are the limit, as the owner suspects | the long run with the extended residual |
| both synthetic at a low loss | the loss does not hear what matters | change the loss before any long run |

This is the localisation by ear the owner raised, done in music rather than on isolated notes. To build: audio export
from the fits, and the extended outputs. A fit to one take can also fit what no context predicts, so a version that
sounds right bounds what the trained residual can reach; it does not promise it.

### 8.3 Step 2: the residual run

- **Start:** the phase-6 physics; the residual fresh, from step 0, not from 55 %.
- **Rate:** the full base rate for the residual, a short warm-up, constant; no decay until the very end.
- **Budget:** off.
- **Physics:** trainable at a low rate rather than frozen. Every check reports the physics alone and with the
  residual, so what each contributes stays readable.
- **Length:** at least one pass of 2018 (about 5,000 steps of 8 × 2 s; phase 6 ran 4,800 steps in 310 min), probably
  several. This sets aside the short-run rule for this run, as the owner accepted.
- **Tracking:** excerpts of the training pieces and of the validation pieces scored side by side at each check
  (13.3's open suggestion). Training improves and validation does not: the corrections do not carry over (more data,
  or they cannot be predicted from context). Both improve: keep going.
- **Listening:** the same six excerpts rendered at each checkpoint. The ear is the gate, not the loss.

Caveat (mine): under this loss a residual that is a function of the context can learn what is predictable, not what
differs from take to take. That may still be most of what is heard; it is not known.

### 8.4 Step 3: the note objectives inside training, the real priors kept

The owner agreed to this in principle on 2026-10-02 (section 9 has the numbers behind it). Not built.

**Rationale.**

- The fits before training are a second training loop with another loss, on a few hundred notes, one parameter group
  at a time with the rest held (9.3). From phase 5 on they move their own held-out objective by 0.01–0.16 dB on
  3.4–4.9 dB and the music loss by under 0.01 (9.2), at 1.4–6 hours of work per phase (9.1).
- Fitting one group with the others held lets it take up the others' error; four cases are in the record (9.3).
- The music loss alone does not find what the fits target: phases 3 and 4 trained the decays on music and the upper
  partials still rang too long (17.5), and 12 % of its weight is on notes older than 1 s (2.2). So the note
  objectives are needed; what is wrong is where they sit.
- Run apart, the two objectives never meet: the main training either cannot see a fitted parameter or pulls it
  elsewhere, and the parameter gets frozen (3.4). In one loop, with the parameter free, a disagreement between them
  shows in the log instead of being hidden by a freeze.
- Phase 6 did put one note term into training, at 2 notes per step from a capped list of 804; it stayed at
  3.6–3.8 dB (3.1). The idea was right and the scale was not.

**The proposal.**

- **One loop.** Each batch holds music excerpts scored with the music loss, and windows chosen for their exposed
  notes (isolated strikes for the attack and early level; notes that ring free for the fades) scored with the note
  terms as well. The terms exist and are differentiable: `notefit.NoteTerm` (band levels in windows at the note's
  own onset: early, sustain, the attack re the early window, between the partials) and `envfit.EnvelopeTerm` (each
  partial's fade from 0.1 to 2.5 s).
- **Everything free:** every physics group, the room and the residual train on the sum. Only the frequencies stay
  fixed (below). No group is frozen to protect a fit.
- **Drawn from all the notes, not a fixed list.** N13 treats every one of 2018's 362,000 notes as an event and keeps
  the readings that are clear. Of 3,678 drawn for the phase-6 run 804 (22 %) had a reading that counts, and of
  1,383 drawn for the first envelope fit 307 (22 %) (read, `train.log`, `env_fit/log.txt`; stratified draws, so not
  a population rate). The usable pool is tens of thousands of notes, against the 804 used.
- **Start from the phase-6 values.** What the fits found is kept as the starting point; nothing is thrown away.
- **In the log:** each term per step, and the cosine between the note terms' gradient and the music loss's on the
  parameters they share, over enough steps to read it (section 7, item 4).

**How many notes per step (a correction).** I told the owner "hundreds of notes per step". That is not what this
gives. A batch of 8 × 2 s of music holds about 70 notes (362,000 notes in 22.9 h), and each note-term clip is its own
render (1.8 s for the attack terms, 4.4 s for the fades). At a render budget like phase 6's, tens of notes per step
is realistic: for example half the batch as longer windows around exposed notes, every note in them that has clear
readings scored. That is still ten or more times phase 6's 2, from a pool a hundred times larger (mine, not
measured; the count of clear notes per window is the first thing to measure when building it).

**What stays as a prior, set once from measurement:**

- **The frequencies:** B and the tuning per key, mined from isolated notes. A magnitude loss cannot move a frequency
  by more than about a bin, so training cannot find them and they stay fixed. The unison's beat rates are the same
  kind of thing and have no measured start yet (section 7, item 9).
- **The initialisation from data** (`fit_init.py`), and the phase-6 values as the start.
- **The bench as a scoreboard.** The measures keep running on each checkpoint; what stops is building a fit for each
  measure.
- **Fitting as a test, not a stage:** "can the model express this at all", on a few excerpts, with audio (8.2).

**Open.**

- Whether the note terms and the music loss still disagree once they train together (the knock is the known case,
  12.6). If they do, that is the thing to look into, with the parameter free.
- The weights of the note terms: phase 6's 0.5 gave the envelope term about 3 times the music loss's gradient on the
  decays with 2 notes; with tens of notes it needs setting again.
- The hall (T60, band levels) is frozen today at its measured values. Under the same principle it would start there
  and train, with free-decay windows among the exposed ones. The owner's call.
- Whether this goes into the same run as the residual (8.3) or after it. One run after both are built fits the
  owner's rule of training once after a batch of changes; it gives up telling the two apart.

### 8.5 Step 4: the rest, in this order

1. **The scoreboard.** The variation-off score against its like-for-like floor, errors grouped by piece (section 7,
   items 1 and 6).
2. **A critic as a loss.** Only if the trained residual sounds blurred or averaged, and with the checks of section 7,
   item 7.
3. **All years, then 48 kHz.**

Set aside for now: the per-note latents (5.2) and freeing the per-key structure (5.4).

## 9. The fits before training (owner's question, 2026-10-02)

The question: phases 5 and 6 spent more hours fitting parameters on notes before training than on training. Does
that get anywhere, or should training do it? Read from the run folders' file times, the fits' `report.md` and
`runs/phase5/attribution/compare.md`; the readings are mine.

### 9.1 Where the hours went

| | before training | of which fits and measures computing | training |
|---|---|---|---|
| phase 5 | 13:52 → 15:19, 1.4 h (2.9 h with the isolated-note study from 12:24) | onset fit 17 min, decay fits 16 + 18 min, scans about 25 min | 50 min |
| phase 6 | 17:48 → 23:51, 6.0 h | N13 on 2,200 notes 26 min, envelope fits 25 + 54 min | 310 min |

The fits themselves are short (5–54 min each). The hours are the cycle around each feature: build a measure, check
it, build a fit for it, fit, re-run the bench, write it up. The fit code is of the size of the training code
(`notefit.py`, `envfit.py` and the fit scripts about 1,470 lines; `train.py` 690).

### 9.2 What the fits bought

Each fit's own objective on held-out notes (mean |model − recording| over its cells, dB, before → after):

| fit | held out | music loss (32 test excerpts, re the step before) |
|---|---|---|
| velocity map (phase 3) | 3.90 → 3.65 | |
| knock (phase 3) | 3.37 → 3.11 | music wants it 4.5–7 dB lower (12.6) |
| attack parts (phase 4) | 3.17 → 3.00 | |
| onset fit (phase 5) | 3.430 → 3.424 | −0.008 |
| decay fit, decay fit 2 (phase 5) | 3.734 → 3.712 → 3.695 | +0.001, +0.003 |
| envelope fit, fit 2 (phase 6) | 4.91 → 4.82 → 4.75 | validation 0.948 → 0.9445 |

- From phase 5 on the fits move their own held-out objective by 0.01–0.16 dB on 3.4–4.9 dB, and the music loss by
  under 0.01. They do shift medians in the cells they target (the upper partials' fade at 1 s by several dB, 17.6),
  which a mean over all cells hides. The owner hears no difference between phases 5 and 6.
- "Training does nothing because the priors are already good" is not what the numbers show. The music loss does not
  register the fits at all; it was flat before them and after them. The two objectives see different things.

### 9.3 What a fit is here

Not a prior. It is a second training loop: another loss (a note measure), other data (819 calibration notes, or 307
envelope notes; 2018 has 362,000 notes), one parameter group at a time with the rest held, run before the main
training. The main training then either cannot see what was fitted (decays) or pulls it elsewhere (knock), and the
parameter is frozen.

The reason the fits exist is sound: the music loss does not see these features. Phases 3 and 4 trained the decays on
music and the upper partials still rang too long (17.5), so "let the training do it" does not work with today's loss
as it is. What does not follow is the place the fix was put. The interactions the owner expects are in the record:

- a band-level term let the knock noise stand in for high partials the tone lacks (12.5);
- the knock fitted on notes sits 4.5–7 dB above where music puts it (12.6);
- the envelope fit improved partials 9 and up and made partials 5–8 worse on held-out notes (17.6);
- phase 5's training, or one band's Q, undid a fitted change at partials 6–8 (16.6).

Each is one group fitted with the others held, taking up the others' error.

### 9.4 Proposal

- **Stop the fit-then-train pipeline.** No new measure-plus-fit per feature before a run.
- **Move the note objectives into training**, every parameter and the residual free, and **keep the real priors**
  (frequencies, the initialisation from data, the bench as a scoreboard). The owner agreed to both in principle;
  the proposal with its rationale, and a correction of "hundreds of notes per step", is 8.4.
- **Keep fitting as a test, not a stage:** "can the model express this at all" on a few excerpts (8.2), with audio.

## 10. Why a step takes seconds (owner's question, 2026-10-02)

`review_6_scripts/step_profile.py`: the phase-6 model, the residual on, one batch as the run trained (8 excerpts of
1 s warm-up + 2 s = 24 s of audio), mean of 3 batches after a warm-up batch, the GPU synchronised around each part.
Mine.

| | s per step | share |
|---|---|---|
| forward | 1.23 | 41 % |
| loss | 0.02 | 1 % |
| backward | 1.77 | 59 % |
| **step** (without the optimiser, the data and phase 6's 2 envelope notes) | **3.02** | |

| part | forward (s) | backward (s) |
|---|---|---|
| the oscillator bank (the strings' partials) | 0.49 | 0.77 |
| the noises | 0.27 | not split |
| the sympathetic bank | 0.23 | not split |
| the residual network | 0.08 | not split |
| tables to modes, impulses | 0.02 | not split |

- **The cost has nothing to do with the parameter count** (0.74 million). It follows the audio: per step 24 s at
  24 kHz, 1,044 notes in the batch (most ring in from the 12 s before the window), and 2.0 billion
  oscillator-samples (notes × oscillators × samples, after the activity test), each one a sine and an exponential at
  audio rate with the phase formed in float64. Peak memory 13.0 GB.
- **No single bottleneck:** the oscillator bank is 42 % of the step (1.26 of 3.02 s); the noises, the sympathetic
  bank, the room and their backward passes share the rest. The data costs 4–29 ms per excerpt, the loss 0.02 s.
- Phase 6 logged 3.9 s per step (4,800 steps in 310 min): this plus the envelope term's 2 notes of 4.4 s, the
  optimiser, validation and dumps.
- **Against a 170-million-parameter transformer at 1 s per step** (mine, an estimate, not measured): such a step is
  of the order of 1e13 arithmetic operations, matrix products in mixed precision, which is what the GPU is built
  for. This step is of the order of 1e11: element-by-element sines and exponentials over tensors of tens of
  millions of elements, each operation a separate pass through memory, in float32 and float64. So the GPU does
  roughly a hundredth of the arithmetic per second here. Part of that is the nature of the work; part is that nobody
  has optimised it.

Not tried, things to look into before runs of many hours: the bank as one fused kernel instead of separate tensor
passes; mixed precision (`--amp` exists and is off); the activity threshold (a note is rendered until every partial
is 90 dB below its peak); rendering the partials straight into the loss's spectral frames instead of as audio.

## 11. The composite score and the first run on it (2026-10-02)

The owner's request: keep both the pooled and the read-by-read comparisons in one loss, add a critic for variety
and texture later, check that the loss falls and that the change is audible, and make sure the residual can do more
than turn the physics' own knobs. Built in this order: the score, its check, then training on it. Uncommitted.

### 11.1 The score (`pianonn/partial_view.py`, `pianonn/composite.py`)

- **Read by read**, per example: `band` (`PianoLoss`'s 1/6-octave bands), `partials` (the mix at every sounding
  note's partial frequencies), `between` (the energy between the partials, 1/3 octave, per frame).
- **Pooled across examples and steps** (`RunningPool`, as the onset term's pool, decay 0.97): `pooled_exposed` (the
  partials' readings summed per partial group and note age, each weighted by its note's expected share of the
  reading from the model's own envelopes, dampers and re-strikes), `between_pooled` (per band and time since the
  latest onset), `level` (`PianoLoss`'s bands over the excerpt); plus the onset term. Each excerpt is weighted by 1 /
  the render's power (not the take's: its random level would leak into the weight).
- **Weights** (`WEIGHTS`): each term in units of its own floor, the pooled partials at 1. Heuristic, mine.

### 11.2 The check (`scripts/score_check.py`, `runs/score_check/`)

Phase 6, 48 validation excerpts (grouped by piece for the errors; jackknife, leave one piece out). The reference is
the model, the "take" a varied draw of it (a piano that is the model), so every right setting is 0. Mine.

| | old total | composite | source |
|---|---|---|---|
| gap to the recording / floor | 0.43 | 0.60 | `across/` |
| decay of partials 1-8, two-way sweep: rise at the ends | z 7.6 | z 6.0-9.1 | `varoff_seed3/`, `across/` |
| known changes (7): multiple of the error | 0.1-13 | 0.3-12 | `across/`, `varoff_seed3/` |
| aftersound, best setting (reference variation off) | −3.76 ± 0.20 dB | −3.60 ± 0.54 dB | `varoff_seed3/` |
| knock, best setting (reference variation off) | −3.4 ± 0.8 dB | −2.6 ± 0.6 dB | `varoff_seed3/` |
| aftersound, best setting (energy score) | −3.54 ± 0.55 dB | −2.44 ± 0.90 dB | `energy_seed3/` |
| knock, decay 1-8, decay 9+ (energy score) | | −1.9 ± 1.8 dB, +0.10 ± 0.12, +0.27 ± 0.16 | `energy_seed3/` |

- **Pooling before the log** reads a consistent error the read-by-read terms hide inside the take's scatter (5.3
  times the sensitivity to partials 5-8 +3 dB, `pooled/`); every term sees a sweep of the first partials' decay (z 7-8),
  and the exposure weighting triples the pooled term's response to the decay of partials 9 and up (0.022 of its floor
  against 0.007, `sweeps/`). The composite does not detect the known changes better than the old total: its pooled terms vary a
  lot from piece to piece.
- **The main bias is the render, not the score.** Rendered without the per-strike variation, as phases 3-6 trained,
  the unison's aftersound modes start in phase, so the render is louder early than a typical varied strike; against
  varied takes every term, old or new, then prefers the aftersound about 4 dB quieter (z 6-13, two takes). Phase 6's
  aftersound may therefore be fitted low for the varied renders that are listened to: a hypothesis, not measured on
  recordings.
- **Energy score** (variation on, each term `d(X, Y) - d(X, X')/2` with `X'` a second draw): proper, its optimum the
  takes' own distribution. It shrinks the pulls; the aftersound's −2.4 ± 0.9 dB remains (z 2.1). One pair of
  reference draws (seeds 0 and 2) served both takes.
- Three slips of mine caught on the way: weights from the take's power, weights that moved with the swept setting,
  a reading-order bug in the between view (the first `composite/` run carries it).

### 11.3 Training on it (`pianonn/train.py`, `pianonn/residual.py`)

- `--score composite --energy`: each step renders the batch a second time without gradient (every random draw
  afresh) and each read-by-read term becomes `d(X, Y) - d(X, X')/2`; the gradient is that of `d(X, Y) - d(X, sg X')`.
  The pooled terms and the onset term are left as they are: they pool over ~33 batches, so their optimum is the
  energy match whatever the spread, and their self distances are small (pooled over 48 excerpts their floors are
  0.03-0.10 against 0.17-0.32 read by read). Use with `--strike-train`.
- Validation reports the composite (pooled terms pooled over all the validation excerpts; the second draw at seed 1),
  the old score (`PianoLoss`, level 0.5) on the first draw, and with `--val-train-examples` the same on excerpts of
  the training pieces. The log's `gcos_<module>` is the cosine between successive steps' gradients, the
  regulariser's own gradient excluded.
- **The residual's wider outputs** (config, off by default): `res_curve_partials` (a gain curve per partial),
  `res_noise_bands` (the frame-wide noise in the noise bank's 32 bands), `res_note_noise` (a noise path per note, its
  level re the note's own expected energy, so it decays and is damped with the note, never before its onset; −60 dB
  at zero output), `res_latent` (random inputs per note, drawn afresh at every render from their own stream, so a
  seed's other draws stay as without them: the residual can vary from strike to strike, which the energy score can
  train).
- **Memory:** the gain curves are now interpolated inside the oscillator bank from their 20 ms control values,
  forward and backward; with a curve per partial they had cost ~5 GB of an 18 GB step. A dense batch (8 excerpts,
  1,474 notes) now peaks at 12.0 GB with everything on (11.4 GB with the phase-6 residual), and takes 5.4 s: second
  draw 1.25, forward 1.45, loss 0.37, backward 2.34 (mine, `step_profile`-style, one batch).

### 11.4 Smoke run 1: the pools across steps diverge (`runs/composite_train/smoke/`)

From phase 6, the wider residual fresh at the full base rate from step 0, no budget, the physics at 0.3 of the rate,
only the frequencies frozen; the composite as an energy score (self term on the read-by-read terms only), the pooled
terms and the onset term pooled across steps (decay 0.97), variation on; batch 8, validation on 32 excerpts of the
validation pieces and 32 of the training pieces. Read from the log, mine.

| | validation pieces | training pieces |
|---|---|---|
| step 0, physics / with the residual | 0.920 / 0.916 | 1.030 / 1.025 |
| step 100, physics / with the residual | 0.919 / **1.248** | 0.998 / **1.243** |

- With the residual, every term was worse at step 100 on both sets: pooled level 0.12 → 0.22, onset 0.18 → 0.28,
  band 0.41 → 0.44 (validation). It tilted the spectrum: +1.5 to +4 dB below 1 kHz, −2.3 to −7.9 dB at 1-8 kHz
  against the physics, where the physics had been about right below 1 kHz and 2-2.5 dB too bright above.
- The running pools read the pooled terms as nearly matched (onset 0.04 in training) while validation, which pools
  without memory, read them far off. The pools hold ~33 batches rendered by an older model, so a term's sign lags the
  model: with the physics alone moving slowly (phase 6's onset pool) that did no harm; a fresh residual at the full
  rate overshoots, and the read-by-read terms in training climbed (band 0.41 → 0.68 by step 160). Stopped at ~170.
- The gradient itself is right: the same residual trained on one fixed batch with fixed seeds and no pool memory
  lowered the composite from 0.892 to 0.573 in 40 steps.
- The high gradient agreement it logged (`gcos` 0.5-0.9) came from the pools' lagging sign, not from signal: without
  them (smoke run 2) it is −0.04 to +0.25 (11.5).

Changed: the pooled terms pool over the batch only (`--pool-decay 0`, the default now), and the energy score's self
term applies to every term (pooled over one batch of 8 the self distances are as large as the distances to the
recording); the composite validation matches each render's level to its piece's recording (leave one out, as
`compare_runs.py`).

### 11.5 Smoke run 2 (`runs/composite_train/smoke2/`)

As smoke run 1 with those changes: 332 steps in 30 min (5.4 s per step, peak 15 GB, no batch skipped), 2,656
excerpts of 2 s, 1.5 h of scored audio: 6.5 % of one pass over 2018's 22.9 h of training pieces (I first wrote 12 %).
`eval.md` and `eval_unmatched.md` (`scripts/composite_eval.py`): 64 excerpts of the validation pieces (13 pieces) and
64 of the training pieces (42 pieces, 26 of them with one excerpt, which keeps 0 dB), level matched per piece or
not, the energy score as in training; phase 6 for reference. Read, one seed pair, no error bars. The full setup is
written up in `docs/composite_score.md`.

| | held out, matched | old score | held out, unmatched | training, matched | old score | training, unmatched |
|---|---|---|---|---|---|---|
| phase 6, physics | 0.7256 | 0.8661 | 0.7701 | 0.8078 | 0.8894 | 0.8426 |
| phase 6, with its residual | 0.7313 | 0.8631 | 0.7960 | 0.7905 | 0.8853 | 0.8261 |
| smoke 2, physics | 0.7028 | 0.8628 | 0.7713 | 0.7727 | 0.8836 | 0.7963 |
| smoke 2, with its residual | **0.6898** | 0.8590 | **0.8057** | **0.7521** | 0.8792 | **0.7628** |

- **Level matched, the loss falls on both sets**, the physics (−0.023 held out, −0.035 on the training pieces) and
  the residual on top of it (−0.013, −0.021). The old score falls a little too (−0.007, −0.010).
- **Unmatched, it does not fall held out:** flat for the physics (0.770 → 0.771), up with the residual (→ 0.806).
  The training made the model louder (the listening manifest, read: the physics 0.2-0.8 dB, with the residual
  0.7-1.5 dB louder than phase 6 on the 8 test excerpts). My reading: that suits the training pieces' recording
  levels, not the held-out ones'; level matching removes it per piece. The held-out gain is in the tone, not in the
  absolute level.
- **Where:** the pooled terms, the level and the onset term (held out, phase 6 → smoke 2 with the residual:
  pooled partials 0.076 → 0.064, between pooled 0.170 → 0.159, onset 0.113 → 0.103); the read-by-read terms do not
  move (band 0.249 → 0.250, partials 0.261 → 0.260).
- Training's own validation (32 excerpts each, unmatched) read the same: the residual worse on the validation
  pieces (0.721 → 0.760), better on the training pieces (0.861 → 0.793). Measured per band (16 excerpts each), the
  residual applies about the same correction to both (+0.4 to +0.7 dB below 1 kHz); the physics was 0.7-1.1 dB too
  quiet at 120-960 Hz on those training excerpts and 0.2-0.6 dB too loud on the validation ones.
- **What the residual does after 332 steps:** mostly level below 1 kHz; its noise paths stay quiet (the strongest
  25.7 dB under the render, in the top octave). The model is still 1.9-2.7 dB too bright above 1 kHz on the
  validation excerpts (before level matching). Gradient agreement between successive steps (means over 50 steps):
  physics −0.04 to +0.20, the residual +0.06 to +0.25; the old score's at phase 6 (2.3, fixed batches at one point,
  so measured differently) was +0.01 to +0.04 and +0.08. Not clearly better.
- Listening: `samples/composite_smoke2/` (the same 8 test excerpts as phase 6's). A 30-minute run is not expected to
  be audible; the files are there to hear whether the new outputs broke anything.
