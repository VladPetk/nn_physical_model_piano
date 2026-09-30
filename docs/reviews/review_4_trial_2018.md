# Review 4: the first MAESTRO trial (2018)

Scope: commits `69f12c4..d954fb9` (phase 0 fixes, the richer physics, the residual stack, the data
initialisation, the training and evaluation tools) and the report [`../trial_2018.md`](../trial_2018.md).
Method: I read the code and the run logs (`runs/*/log.jsonl`), and re-measured the claims I could on the
220-minute checkpoint (`runs/round1_trial/main/best.pt`, step 9000) with 16–24 held-out test excerpts. The
scripts are in [`review_4_scripts/`](review_4_scripts/); every number below that is not from the report
comes from them.

## 1. Verdict

**The trial is a go, and the report is a good piece of work.** The pipeline runs end to end on real audio,
the physics carries the fit, the staging and the residual read-out do what review 3 asked for, and the
report names its open errors instead of hiding them. Its main results hold:

- the fit reduces the held-out distance well beyond the data initialisation;
- the loss cannot find or even notice the bass inharmonicity, so frequencies must be measured per year
  (I also verified that the measured bass B is a genuine measurement, section 6);
- the residual stays small and reads as a map of what the physics gets wrong;
- on listening (section 7), the fitted model is clearly nearer the recording than the initialised prior,
  and the residual makes no audible difference.

Six things need correcting or completing before the next round, in order of importance:

1. **The loss wants a quiet piano** (section 3, new). The trained model is 2.8 dB under the recordings on
   the test excerpts. The spectral-convergence term's optimum is *0.5 dB quieter still*, the
   log-magnitude term's is 1.8 dB under the truth, log-mel's 1.3 dB under. This one mechanism explains
   the soft-playing deficit, why the velocity curve could not fix it, why the log-mel term helped, and
   probably the 60–125 Hz drift. It is a loss problem and it comes before any physics change.
2. **The loss numbers have no scale** (section 2). Two renders of the same model score 0.33 against each
   other and the recording scores 0.23 against itself plus its floor, so about a third of the reported 1.00
   is irreducible. The residual's whole gain (−0.02) is 3 % of the explainable range; a broadband gain of
   +1.5 dB is worth twice as much. The "untrained prior" row (15.1 dB) is a 16 dB level mismatch, not a
   model comparison.
3. **The noise-floor initialisation is wrong by 9–14 dB in several bands** (section 4). The fit corrected
   it: up 12 dB at 40 Hz, down in the middle bands. The corrected levels are right (the owner measured the
   recordings), so the floor is a measurement the fit repaired, not a hedge. An earlier draft of this
   review claimed the opposite without measuring the recordings; that claim is withdrawn.
4. **"The residual never added noise" is enforced, not found** (section 5). The additive budget term is
   degenerate wherever the physics is silent (13 % of band-frame cells, three quarters of the penalty), so
   it pushes the noise path down regardless of what the recordings contain.
5. **"The residual became a real gain" is confounded** with the physics learning rate: there is no
   physics-only control of the same length.
6. **The phantom-partial finding is plausible but not established**: the measurement that confirms it is
   not in the repository and lacks the two tests that would distinguish phantoms from other components.

Nothing here changes the direction. It changes what to do first: fix the loss, add a scale to the
evaluation, then re-fit, and only then judge the physics items of the report's section 12.

## 2. A scale for the numbers

On 24 test excerpts (2 s after a 1 s warm-up, both channels), MR-STFT loss as used in training:

| pair | loss | per resolution 4096 … 128 |
|---|---|---|
| fitted physics + residual vs recording | 1.012 | 0.997 0.989 0.990 1.007 1.029 1.059 |
| fitted physics alone vs recording | 1.036 | 1.023 1.014 1.014 1.030 1.051 1.081 |
| the same model, two noise seeds | **0.331** | 0.329 0.324 0.323 0.327 0.334 0.348 |
| the recording vs itself + the learned floor | **0.229** | 0.224 0.224 0.226 0.232 0.235 0.234 |
| model + best broadband gain per excerpt | 0.972 | |

What this says:

- **Roughly a third of the loss is irreducible.** Two renders that differ only in their noise realisation
  are 0.33 apart. The floor alone accounts for 0.23. The explainable range of the trained model is about
  0.7, not 1.0, and the trial's total progress (1.40 → 1.00) is 0.4 of that range.
- **The residual's gain is small on that scale**: −0.024 here, about 3 % of the explainable range. A
  single per-excerpt gain correction (median +1.5 dB) is worth −0.040, twice as much.
- **The attack claim survives the scale.** The noise ceiling is flat across resolutions (0.33 → 0.35), so
  the 128-point excess (1.059 vs 0.997) is model error, as the report says.
- **The headline row is a straw man.** The untrained prior renders 16 dB too quiet (the init found mic
  gains of +16.5 / +15.8 dB). Its 15.1 dB log-mel distance measures the level, not the prior. The honest
  starting point is the initialised prior, 6.7 dB, and the honest headline is 6.7 → 4.0 → 3.84.

Every evaluation table should carry the two ceiling rows and the gain oracle. A fourth reference is
missing entirely: a *wrong* model at the right level, e.g. the fitted model driven by another excerpt's
MIDI, or the recording's own long-term spectral envelope as stationary noise. Without it, "4.0 dB" has no
meaning to a reader; with it, the table says how much of the distance is *the notes* and how much is
*the colour*.

## 3. The loss wants a quiet piano

The trained model is quieter than the recordings on the test excerpts:

| | dB |
|---|---|
| broadband level, model minus recording, median over 24 excerpts (IQR) | **−2.84** (−3.33, +0.06) |
| gain that minimises the full loss, median | +1.5 |

So the loss's own optimum is 1.3 dB under the truth even on this fitted model. Splitting the loss by term
and scanning a broadband gain on the model's render (`level_bias_by_term.py`):

| term | gain at the term's optimum | i.e. re the recording |
|---|---|---|
| spectral convergence, 4096 / 2048 / 1024 | 0.0 dB | −2.8 dB |
| spectral convergence, 512 / 256 / 128 | −0.5 dB | −3.3 dB |
| log-magnitude L1, 4096 / 2048 | +0.5 dB | −2.3 dB |
| log-magnitude L1, 1024 … 128 | +1.0 dB | −1.8 dB |
| log-mel L1 | +1.5 dB | −1.3 dB |

**Mechanism.** Spectral convergence is a least-squares fit of magnitudes. Where the recording's fine
structure is unpredictable from the MIDI (three unison strings beating with unknown phases and rates,
the hall, sympathetic resonance, mechanical noise), the L2 optimum is the conditional mean, whose energy
is lower than the truth by the unexplained variance. That is ordinary regression to the mean, and it
shrinks the model in proportion to how much of the sound it cannot predict. The log-magnitude term is
median-seeking and shrinks less; band energies average the fine structure away and shrink least; a
coarser term shrinks less still. The ranking in the table is exactly that order.

**What it explains.** The shrinkage is largest where the unpredictable fraction is largest: soft,
pedalled, sustained sound, where beats, aftersound and hall dominate the prompt sound. That is the
report's regression (error ≈ −3.8 + 5.3·velocity/127 dB, −0.7 dB more under the pedal). It explains why
the per-year velocity curve learned a compressive shape *and the deficit did not move*: the loss does
not want it moved. It explains why the log-mel term improved the band balance everywhere. It explains
the residual's main job at 220 minutes, per-note gain in context: a bounded net can push the level
where the physics' velocity law, sitting at the loss's optimum, is held back. And it is the natural
suspect for the 60–125 Hz drift, where the bass unisons beat slowly and the 4096-point bins resolve
every beat.

**What I could not show.** Synthetic targets made from the same model with re-drawn unison mistuning or
a 1.1–1.3× bass inharmonicity (`bass_gain_synthetic.py`) do not reproduce a bass gain bias (optimum
0.0 / −0.5 dB). Those perturbations leave the prompt sound, which dominates, fully predictable; the real
recordings evidently do not. So the bass drift is *likely* the same mechanism but the proof is a real-data
one: re-fit with the spectral-convergence term off and see whether the 60–125 Hz band comes back.

**What to do.**

1. Drop the spectral-convergence term, or weight it at ≤ 0.2. It has the largest bias and no property the
   other terms lack.
2. Make the *level bias of the loss* a standard diagnostic in `scripts/evaluate.py`: scan a broadband gain
   on the fitted render and report where each term's optimum sits relative to the recording. Any loss
   candidate must land within about 0.5 dB of the truth on real excerpts. This is measurable in a minute,
   without training.
3. Main term: log band energies at coarse resolution (1/3–1/6-octave bands, 50–100 ms frames, L1 in dB).
   The mel term of section 10 is a first version of this; it should be the backbone, not an add-on.
4. Keep fine resolution only where it earns its keep: a frequency term on isolated notes (section 6), a
   short-window term restricted to the 30 ms after each onset for the attack.
5. The report's proposal of an 8192–16384-point resolution below 1 kHz points the wrong way: narrower bins
   resolve *more* of the beating, raise the unpredictable fraction and increase the shrinkage. Bass
   frequencies belong in a term on isolated notes, where the fine structure is predictable.
6. After the loss change, re-run the level-vs-velocity regression. Expect the soft deficit to shrink
   before any change to the pedal decay. The Iowa C2/C3 decay-profile failures are a separate, real
   physics question and should be pursued on mined pedal-down notes as the report proposes.

## 4. The noise floor: the fit corrected a wrong initialisation

The floor is meant to be a measurement: the recordings' quietest 0.2 s per band, learned only to
refine. After 220 minutes (`noise_ceiling.py`, white-equivalent dBFS, every fourth band):

| band centre | 40 Hz | 83 | 174 | 364 | 759 | 1584 | 3306 | 6900 |
|---|---|---|---|---|---|---|---|---|
| measured at init | −40.3 | −49.2 | −47.2 | −43.0 | −52.8 | −67.0 | −81.2 | −85.8 |
| fitted (L) | **−28.2** | −47.7 | −47.8 | **−55.6** | **−66.6** | −75.5 | −80.2 | −83.2 |

And its share of the model's output per band on 16 test excerpts (`floor_share_and_synthetic_bias.py`):

| band | 40 Hz | 58 | 84 | 121 | 174 | 252 | 364 … 4782 | 6910 | 9983 |
|---|---|---|---|---|---|---|---|---|---|
| model − recording, dB | −4.4 | −1.4 | −1.8 | −2.5 | −2.9 | −2.3 | −0.7 … −1.6 | −4.1 | −5.0 |
| same, floor removed | −6.8 | −1.4 | −1.9 | −2.5 | −2.9 | −2.3 | same | −4.2 | −6.6 |
| floor's share of the model | **41 %** | 0 | 1 | 0 | 0 | 0 | 0 | 2 | **31 %** |

What these tables show, and what they do not:

- **The fitted floor is the right one.** The owner measured the recordings' stationary noise directly
  and the fitted levels agree with it; the initialisation was 12 dB too low at 40 Hz and 9–14 dB too high
  in the middle bands. So the 41 % floor share of the model's 40 Hz band is what the recordings contain
  there, and the 30–60 Hz row of the report's band table converging to −0.3 dB is real. An earlier draft
  of this review read the 12 dB rise as the floor hedging for missing physics, on the strength of the
  parameter comparison alone, without measuring the recordings. That was the wrong order of operations
  and the claim is withdrawn.
- **The init method is the problem, not the parameter.** The 2nd percentile of 0.2 s windows in
  continuous piano music has too few bins per window in the lowest band (it reads low) and picks up
  quiet piano in the middle bands (it reads high). MAESTRO recordings have seconds of silence before the
  first note and after the last; measuring there would give the fit its starting point directly.
- The floor is unbounded and in the ×20 learning-rate group, which is what let it correct a 12 dB error
  in 5,000 steps. Leave that as it is. Do print it in the fitted table of `scripts/evaluate.py`, which
  currently omits it, so the next reader can compare it with a measurement instead of guessing.

Independent of the floor: the model is 1–3 dB under the recordings in every band from 58 Hz to 5 kHz
(second row of the second table, floor removed or not). That is section 3's shrinkage seen per band, not
a bass-specific effect; the bass is simply where it is largest.

## 5. The residual

**The budget's additive term is degenerate.** `residual_budget` measures the residual noise's share of
band energy as `e_res / (e_res + e_dry)`. Wherever the dry physics is silent (before the first note, high
bands in quiet passages) the share is 1 for *any* residual noise, however small. On the test excerpts 13 %
of band-frame cells are empty and they contribute three quarters of the penalty (share 0.039 overall,
0.010 in the non-empty cells). So the R3 noise path was pushed 25–32 dB down because the penalty forbids
noise wherever the physics is silent, which is exactly where a real piano's unmodelled sound (the
pedal-down halo, the hall) would live. The report's reading, "nothing in the recordings calls for
broadband noise", is not established. Fix: add the floor's (or the target's) band energy to the
denominator, or measure the residual against the recording's band energy. Then re-read the noise path.

**The 220-minute gain is confounded.** The claim that the residual "became a real gain" compares
physics + residual (trained jointly, physics at 0.3× lr) with the same physics with the residual switched
off. It does not compare with a physics-only run of the same length at the same schedule. The physics-only
validation kept improving through stage 2 (1.056 → 1.015), so part of the residual's −0.02 may be what
the physics would have done alone. The control is cheap: continue `stage1_best.pt` in stage 1 for the same
number of steps, or run stage 2 with the context net frozen. Until then the honest statement is "the
residual costs nothing and reads as a map".

**The read-out works and is the trial's best methodological result.** The per-note gain corrections
(0.6 dB rms in the bass) are section 3's loss bias, delegated to a net. The knock −0.5 to −1 dB and the
2.5–4 kHz −1 dB are real reads. The attack's texture is beyond R1–R3, as the report says; the report is
right not to escalate to R4 yet.

## 6. Physics findings, checked

**Bass inharmonicity: a genuine measurement, and the strongest result of the trial.** The mined
B ≈ 5.5e-5 for MIDI 28–41 sits within 10 % of the partial tracker's lower search bound (Rigaud/4) for
90 % of the notes, which looked like an artefact. It is not: re-tracking the same notes with the bound at
Rigaud/32 (`retrack_bass_B.py`) returns the same B to three digits for every note, from least-squares
fits over 14–44 partials with 0.4–2.5 cents rms. The 2018 piano's bass strings really are about half as
inharmonic as the prior (2.2× at MIDI 30–41, 1.7× at 24–29). Two consequences the report draws are right
and worth stating more strongly: (a) this is a *measurement* result, not a fit result, and it is what the
loss could never have found; (b) the ablation shows the loss cannot even *see* the difference: the run
without measured frequencies started at 1.4150 against 1.4206 with them, i.e. the correct frequencies
made the loss very slightly worse. Add a regression test for whatever frequency term replaces this: a
synthetic pair with B×2 in the bass must show a clear loss gap.

**Stretch and B in the registers with data agree with the tracked values**; MIDI 84–108 have 13 and 1
mined notes and no reliable B, so nothing is known about the treble yet. The mined notes are soft (median
velocity 31–48), which is what "isolated" selects for; that is fine for frequencies and unusable for
hammer or phantom levels.

**Phantom partials: plausible, not established.** The fit moved the middle-register phantoms +10 to
+14 dB above the prior and the report measured components at 2f_j in 25–35 % of mined notes. Three things
are missing. The measurement script is not in the repository, so the numbers are not reproducible. The
detection cannot yet tell a phantom from the other things that sit near 2f_j: a partial of a still-ringing
string, the aftersound modes' sidebands, or the hammer-spectrum error the phantom slots can absorb (the
slots are free sinusoids interleaved with partials 2j–2j+1, 20 dB of range per key, and above 5 kHz the
per-bin loss is mostly floor). And the fit put +14 dB at MIDI 72–84 where the literature has phantoms
vanishing above C6. The two distinguishing tests are cheap on the mined notes: **velocity dependence**
(a phantom's level re the partial product grows like v², so in dB it rises about twice as fast as the
partials) and **decay** (a phantom decays at α_j + α_k, about twice a partial's rate). Until they are run,
the phantom prior should not be changed.

**Recording chain.** The fitted hall T60s (1.38, 1.12, 1.07, 1.68, 1.58, 1.34, 1.00 s from 125 Hz to
8 kHz) are non-monotonic in a way real halls are not; the 1–4 kHz bump may be absorbing the sympathetic
and aftersound halo. Watch it when the sympathetic bank is switched on. Level is split across the mic
gain (moved from +16.5 to +9.7 dB during training), the condition gain, the per-key gains, the velocity
law and the body FIR; that is harmless but it means "what the model learned about the recording" and
"about the piano" are not separable for these, and the fitted table should not present them as if they
were.

**Decays.** b1 ×0.8 and R ×0.8 in the bass and middle (slower internal loss, slower prompt stage) and
the shorter damper delay (15 → 4 ms, trading against the later half-pedal point) are plausible for a
concert grand against the Steinway B prior. The re-strike table did not move, which means the loss does
not see it either; it is worth a synthetic check that it *can* (a teacher with 2× re-strike).

## 7. Method and protocol

- **Test material is small**: 48 × 2 s = 96 s. Acceptable for a first trial. The band-error tables,
  however, rest on three 7 s validation excerpts, too few for statements about octaves; use the test set.
- **Model selection** on fixed validation excerpts with the residual on, test never used: correct.
- **Gradient clipping is always active.** The room's gradient norm is 2–2.8 against a clip of 1.0 (physics
  0.2–0.4, noise 5e-4, context 0.2–0.5), so every step is scaled by the room. Adam absorbs a constant
  scale, so this is mostly harmless, but the clip is not doing what its name says. Clip per module or
  raise the threshold.
- **Tolerant checkpoint loading** silently used a default for `cond_vel_curve` in the 220-minute
  evaluation (the loader logs it, the evaluation script passes a silent logger). Harmless here; print it
  in the report.
- **Reproducibility**: the phantom measurement is missing; the identifiability ablation and the two
  follow-ups are reproducible from the logged commands. The scripts for this review are in
  `review_4_scripts/`.
- **The new tests are the right kind**: behavioural (re-strike accumulation, history damping, phase
  exactness at 10 min, analytic backward vs autograd, the floor-dominated loss, gradients pointing back
  to a perturbed teacher, staging). Missing: a test of the data initialisation (latency sign, EQ folding
  into the body), and a loss-bias check, which belongs in the evaluation (section 3) rather than in a
  unit test, because it only shows on real audio.
- **Listening (the project owner, on `runs/round1_trial/main/eval_cont/long{0,1}_ab.wav`, 220-minute model).**
  The recording is clearly much better than all three renders. The initialised prior is bad. The fitted
  physics and the fitted physics with the residual are very close to each other and "not bad". That meets
  the phase-1 listening criterion of review 3 (the fit is audibly nearer the recording than the prior) and
  agrees with the arithmetic: the residual's gain is 3 % of the explainable range and inaudible. On what
  is wrong with the fitted renders (the owner's words, not an engineer's, and not to be over-read):
  1. *The sound is one-dimensional.* In `long1` the notes of the first chord are easier to tell apart in
     the recording than in the renders. That is what regression to the mean sounds like (section 3): the
     loss rewards the average of what it cannot predict, so voices lose the individual beating, colour and
     placement that separate them. The per-key colouration and pan are also bounded to a few dB.
  2. *It still sounds like a synth.* Consistent with the attack being the worst resolution and with the
     static, purely exponential partials of a modal bank; the texture items (attack, phantoms, noise) are
     the ones the residual cannot reach.
  3. *Much less powerful*, which the owner attributes to the room, lid and soundboard emulation. The
     measured side of this is the 2.8 dB level deficit, the 1–3 dB per-band deficit, and the sympathetic
     bank being off. The lid and the board's directivity are not
     modelled at all; the body FIR is 0.3 s and learns at 0.03× the base rate. Section 3's loss fix should
     recover part of the power before any of these is changed, and is the way to find out how much is
     loss and how much is missing physics.

## 8. What to do next, in order

1. **Loss** (section 3): spectral convergence off; a coarse log-band-energy backbone; the gain-bias
   diagnostic in `evaluate.py`; a frequency term on the mined notes; an attack-window term. Re-fit for
   2 h. Re-check the level regression, the 60–125 Hz band and the bass B ablation.
2. **Floor** (section 4): initialise it from the leading and trailing silence of each piece, and print
   it in the evaluation report. No bound.
3. **Budget** (section 5): fix the share term, re-read the noise path.
4. **Control run** for the residual's gain; report "costs nothing" until it exists.
5. **Evaluation scale** (section 2): the two noise-ceiling rows, the gain oracle and a wrong-model
   reference in every table; drop or footnote the untrained-prior row.
6. **Phantoms** (section 6): put the measurement script in the repo; run the velocity and decay tests.
7. **Listened** (section 7): fit audibly better than the prior, residual inaudible. Next, the report's
   section 12 in its order: pedal decay on mined pedal-down notes, the knock spectrum, the felt model,
   longer runs with a decaying learning rate, all years, sympathetic bank, GAN.

The report's own list is sound; what this review changes is that items 1–5 are cheaper than any physics
change and will move the same three errors (soft deficit, bass drift, attack) that the physics items are
being asked to fix.
