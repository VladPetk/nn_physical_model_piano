# Piano tone: constituents, measures and losses

A design for round 3: derive what we measure, and what we train on, from what a piano note is made of. Draft
for discussion; nothing here is built yet except where marked *(exists)*.

## 0. Why

Rounds 1 and 2 ran a loop: train, listen, guess the cause, measure it, adjust, train again. Each turn found
something the metrics had not shown:
- in round 1, the level;
- in round 2, the attack and the "plucked", too-stringy tone.

There are three reasons:
1. **We measure music, and the ear judges notes.** Every metric is averaged over 2 s excerpts and over frequency
   bands, and an attack is ~30 ms per note. The one neural piano paper with a listening test found the same
   gap: its STFT and envelope losses looked good while listeners heard the attack fail (Simionato & Fasciani
   2024; [`literature/tone_constituents.md`](literature/tone_constituents.md), R2).
2. **The loss moves every unpredictable constituent towards its average.** Anything that varies from note to note
   in ways the MIDI does not say gets shrunk. Examples are the attack's exact shape, the knock, the noise and the
   beating pattern. We saw this for level and applied the lesson to nothing else.
3. **No parameter audit.** Only B and tuning were checked for whether the loss can see them and whether its
   optimum is right. Everything else was left to a loss that sees it weakly and with bias.

The fix is a chain that starts from the instrument:
1. what a note is made of (section 1);
2. where each constituent can be measured, given the data (section 2);
3. a suite of measures, one or more per constituent (section 3);
4. an audit of every loss term against those constituents (section 4);
5. training stages that use each kind of evidence where it is strong (section 5).

It will still miss things; listening stays. But listening should find what the suite cannot, not what we never
looked for.

### Principles

1. **Evaluate constituents, not the sum.** Every constituent that matters gets a measure, reported by register,
   velocity and pedal.
2. **Compare distributions over notes, not only medians.** The recording's note-to-note spread sets the
   tolerance. The spread is also a target in its own right: a model with less variation than the piano is the
   "one-dimensional", "synth" sound.
3. **Measure each constituent at its own time scale.** The scales run from 0.5 ms (hammer contact) to 20 s
   (bass aftersound). A 2 s window is too short for half of them.
4. **The constituent's class decides the loss.** Four classes:
   - **D**, set by the MIDI and fixed instrument properties: match it with aligned terms;
   - **K**, fixed per key but slow or unknown (the unison mistuning, for example): match it with per-key,
     long-window terms;
   - **V**, varying from note to note beyond the MIDI: match its distribution;
   - **S**, stochastic texture: match its statistics.
5. **Validate every measure before trusting it.** Each measure must recover a known change on synthetic notes: render the model, change one
   parameter by a known amount, and check that the measure reports it. It must also report its own failure
   rate on recordings. The onset detector of `scripts/measure_attack.py` failed silently on 273 of 482 notes
   (section 3.1); this principle exists because of it.

## 1. What a piano note is made of

Evidence: the project's spec ([`physical_parameters.md`](physical_parameters.md), which cites the KTH *Five
Lectures*, Weinreich, Conklin, Hall and others) and the literature report
[`literature/tone_constituents.md`](literature/tone_constituents.md) (R1–R16 there, labelled read/cited).

**Model column:**
- **yes**: modelled;
- **part**: modelled, but its shape is fixed or it has been off in training;
- **no**: not modelled.

**Class column:** the four classes of principle 4.

### 1.1 Along the sound-production chain

| # | constituent | physics, scale | register / velocity | audibility | model | class |
|---|---|---|---|---|---|---|
| A1 | touch precursor | finger–key impact noise 20–30 ms (up to 200 ms) before the tone | depends on touch, **not** on hammer velocity; varies by pianist | yes: touch identified above chance with it, at chance without (Goebl et al. 2014, R1) | no | V |
| A2 | key-bottom thump | key hits the keybed; +12 ms after the strike at p, −5 ms (before it) at ff | velocity sets its timing | discriminable in a test, cue uncertain (R1) | yes (noise event) | V |
| A3 | action clicks (let-off, backcheck), key release | mechanical | – | unknown | no | V |
| B1 | hammer–string contact | nonlinear felt; contact time 3.7 ms (A0) → 0.5 ms (C8) at mf, shorter when harder; sets the spectral envelope and its growth with velocity | strong in both | the main loudness–brightness coupling | yes: contact time, two-corner roll-off, velocity law | D |
| B2 | strike-point comb | notches at partials n ≈ L/x₀ (x₀/L 0.12 bass → 0.065 treble) | register | part of the timbre | yes | D |
| B3 | multiple hammer contacts (treble) | contact outlasts half the string period | treble, loud | unknown | no | D |
| B4 | knock | the hammer impact excites the frame and board directly: a short broadband transient | strongest re the tone in the treble; grows ~13 dB less than the tone from pp to ff | "the most important issue" for treble notes (Bank et al. 2003, R7) | part: a pulse of the contact time's shape + band noise; spectrum shape fixed | V (fine structure), D (level, spectrum) |
| C1 | transverse partials | f_n = n f₀ √(1+Bn²); stretch | B from 1e-4 (tenor) to 3e-2 (C8) | inharmonicity audible, most at low f₀ (Järveläinen et al., R5) | yes (measured, frozen) | D |
| C2 | partial amplitudes | hammer spectrum × comb × bridge colouration | register, velocity | timbre | yes | D |
| C3 | unison strings | 1 (A0–E1), 2 (F1–A♯2), 3 above; mistuning 0.2–2 cents | – | "unison quality" (tuners); no formal test found | yes: coupled modes, mistuning drawn at random per key | K |
| C4 | double decay | in-phase (prompt) mode drains through the bridge; the mid-range loses ~20 dB in the first second, then slow aftersound | strongest mid-range; treble nearly single-slope | the "prompt sound" vs "aftersound" | yes: R and bridge conductance g(f) | D (rates), K (knee, per key) |
| C5 | beating | amplitude modulation from mistuned unisons and polarisations; 0.1–2 Hz at low partials | 2–3-string keys | liveliness; untested formally | yes (via mistuning) | K |
| C6 | two polarisations | vertical couples to the board (fast), horizontal does not (slow) | all | inseparable from C4/C5 in audio | no (absorbed in C4) | K |
| C7 | longitudinal vibration, phantom partials | tension modulation: components at 2f_j and f_j+f_k; longitudinal modes near 15 f₁; precursor | bass and tenor; **audible up to C5** (Bank & Lehtonen 2010, R4); grows with velocity² | "metallic" bass | part: phantoms yes; longitudinal modes and precursor no | D |
| C8 | pitch glide | tension rises at large amplitude | bass, ff | unknown | no | D |
| C9 | internal and air losses | aftersound decay rising ~linearly with frequency | register | decay length | yes (b1, b3, p) | D |
| D1 | bridge/soundboard admittance | sets the prompt loss per partial and radiation; piano-specific resonances | register | long-term colour | yes: g(f) per year, body FIR, colouration | D |
| D2 | board build-up | attack rise 8–16 ms mid-range (Iowa) | register | attack softness | yes (body FIR) | D |
| D3 | weak radiation of bass fundamentals | board radiates poorly below its first modes | bass | "missing fundamental" | yes (body) | D |
| E1 | sympathetic resonance | undamped strings (pedal, treble above the dampers, held keys) driven by the bridge; halo; the pedal lengthens mid-range decays (Lehtonen et al. 2007, R9) | pedal, register | halo, sustain | part: bank exists, **off in all training** | K/S |
| E2 | duplex/aliquot scaling | string segments beyond the bridge resonate | treble | unknown | no | K |
| F1 | dampers | engage ~18 ms after key release; fast decay; none above ~F6 | register | release character | yes: rate, delay, boundary | D |
| F2 | damper and release noise | felt landing | – | possibly (R1) | yes (noise event) | V |
| F3 | pedal mechanics and noise | half pedal; thump when the rail moves | – | unknown | yes: lift curve, noise | V |
| F4 | re-strike | a new blow damps the ringing string | repeated notes | yes in tremolo | yes | D |
| F5 | una corda | fewer strings struck, softer felt; duller, no beating | CC67 | "softer and duller" (R8) | part: gains + contact time | D |
| G1 | room | reverberation per band, early reflections | – | space | yes: hall T60 per band, per year | S (tail), D (T60) |
| G2 | microphones | spaced pair: level and delay per register, decorrelated channels | register | image | yes: two bodies, pan per key | D |
| G3 | floor and hum | stationary noise, 60/120/180 Hz | – | background | yes (measured) | S |
| H1 | Disklavier velocity map | MIDI velocity → hammer speed, per instrument | – | dynamics | yes: velocity law + curve per year | D |
| H2 | onset timing | MIDI onset vs sound onset; scatter unknown (section 3.1) | velocity? | attack alignment | latency only | V |

### 1.2 The time course of a note

| phase | window | what dominates | measured by |
|---|---|---|---|
| pre-onset | −200 … 0 ms | touch precursor (A1), key-bottom at ff (A2) | N7 |
| contact | 0 … 0.5–4 ms | hammer pulse (B1) | spectrum via N2 |
| attack | 0 … ~30 ms | knock (B4), board build-up (D2), longitudinal precursor (C7), onset spectrum | N4, N5, N6, E4 |
| prompt decay | ~30 ms … ~1 s (mid-range) | in-phase mode loss (C4), fast decay of high partials | N8, T1 |
| aftersound | beyond ~0.5–1 s, up to 20 s in the bass | slow decay (C9), beating (C5), halo (E1) | T1, T2, E1 |
| release | from note-off + ~18 ms | dampers (F1), release noise (F2), halo and reverb tail | N10, P4 |

### 1.3 Register and velocity

The suite reports by the instrument's own boundaries, not by octaves:

| register | MIDI | why here |
|---|---|---|
| R1 | 21–28 (A0–E1) | one wound string |
| R2 | 29–46 (F1–A♯2) | two wound strings |
| R3 | 47–59 (B2–B3) | three strings; the wound-to-plain break is somewhere here (piano-specific: find it as a step in N2, N8) |
| R4 | 60–71 (C4–B4) | strongest double decay |
| R5 | 72–83 (C5–B5) | longitudinal audibility ends near C5 |
| R6 | 84–88 (C6–E6) | last damped keys (the boundary is a prior; F1 checks it) |
| R7 | 89–108 (F6–C8) | undamped; knock-dominated attacks |

**Velocity:**

| bin | MIDI velocity | hammer speed (the model's law) |
|---|---|---|
| p | < 40 | about 1.6 m/s at 40 |
| mp–mf | 40–63 | – |
| mf–f | 64–89 | 2.8 m/s at 64 |
| ff | ≥ 90 | 5.1 m/s at 90 |

**Pedal:** up, down, half.

## 2. Where each constituent can be measured

Counted from the MIDI of all MAESTRO years
([`runs/measurements/isolated_notes_all_years.md`](../runs/measurements/isolated_notes_all_years.md)): notes
with no other onset from 0.3 s before to *t* after.

| clear for | 2018 (27 h) | all years (199 h) | enough for |
|---|---|---|---|
| 0.65 s | 430–1,100 per register from MIDI 36 to 83 | 2,900–7,500 per register from MIDI 36 to 83 | attack, early decay, spectrum vs velocity, per year |
| 2 s | 37–120 per register | 200–950 per register | early decay, release; per year only for the middle |
| 4 s | 10–21 per register | 46–210 per register | aftersound and beats **pooled across years** only |

MIDI 84–108 are rare at every length: 155/771 (pedal up/down) at 0.65 s across all years, 12/154 above 96.

So the evidence comes from four places:
1. **The note bench (N):** isolated notes, per year. The fast constituents: A, B, C1–C2, C4 early, C7, D2, F1–F2.
2. **Tracks (T):** partials followed inside music where they collide with nothing, plus long final chords. The
   slow constituents (C4 late, C5, C9, E1) per year. Long isolated notes pooled across years check the
   mechanism.
3. **Events (E):** onsets, releases, pedal moves and repeats in music.
4. **Passages (P):** whole excerpts: level, long-term spectrum, texture statistics, room, stereo.

Two rules for the bench:
- The notes are split: training pieces for calibration, validation and test pieces for evaluation.
- The list is fixed per year (seeded), so every run is measured on the same notes.

## 3. The measures

Every measure is computed identically on the recording and on the model's render of the same MIDI in its
context: 12 s lookback, 1 s warm-up, both channels.

Reported per stratum (register × velocity × pedal):
- n;
- the recording's median and IQR;
- the model's median;
- the paired median difference, with a flag where |difference| > max(JND, ½ recording IQR);
- the **spread ratio**: model IQR over recording IQR, at equal key and velocity;
- for V and S constituents, a distance between the two distributions (Wasserstein, in units of the recording's
  IQR).

Each stratum also gets a listening file: recording and model notes alternating.

JNDs used where known:
- level 1 dB (R7: a < 1 dB change of resonator amplitudes is inaudible);
- T60 5 % (cited, ISO 3382);
- elsewhere the recording's own spread.

### 3.1 Note bench (N)

| id | measure | constituents | definition |
|---|---|---|---|
| N0 | onset | H2, all N | onset of **this note**: energy rise in the note's own partial bands (a comb at its known partials, ±1/4 of the partial spacing), 0.5 ms resolution, searched in ±50 ms of the MIDI onset; failures reported. Replaces the broadband detector that failed on 273/482 notes (it locked onto pedal ringing and earlier notes' tails) |
| N1 | level vs velocity | H1, B1 | energy of the note's partials, 0–300 ms, per key and velocity: the velocity map |
| N2 | spectrum vs velocity | B1, B2, C2 | level of partials 1–16 at 30–130 ms; the slope over partials (dB/oct), centroid, and residual vs the comb; the step at the wound/plain break |
| N3 | frequencies; glide | C1, C8 | f₁ and B per note *(exists: `scripts/mine_notes.py`)*. **Glide:** the pitch of the note's first 8 partials (below 4 kHz) early re late, in cents: each partial's frequency from the phase advance of its DFT between frames (Hann max(40 ms, 4/f₀), every 4 ms), over 0.1 s of frame centres from the first window starting 10 ms after N0, and over 0.1 s ending with the last window at 0.62 s; median over partials. A double decay of detuned unison strings moves the apparent pitch too, with each key's sign; tension modulation grows with velocity², so it is read from the velocity dependence |
| N4 | rise, sharpness | D2, B1, B4 | **rise:** 10 → 90 % rise of the note's power averaged over one period (partials beat at their spacing, so nothing finer is defined); only for f₀ ≥ 196 Hz. Measured at the microphones, so it includes the room (P4). **Percussive:** per band, the percussive part (median-filter harmonic–percussive separation, 21 ms frames every 2.7 ms) of −5…40 ms, background removed, re the note's 100–400 ms energy. It reads the knock and the partials' own onset alike: how abrupt the attack is |
| N5 | attack spectrum | B4, D2, C7 | 1/3-octave band energy 0–30 ms, 30–100 ms and 100–400 ms after N0, each re the note's own 100–400 ms energy *(first version: `scripts/measure_attack.py`)* |
| N6 | knock; halo | B4, E1 | **knock:** per band (125 Hz–8 kHz), the energy more than max(70 Hz, f₀/4) from every partial in −3…33 ms (a flat-topped window, so the first milliseconds count), **minus the same window's background** at −83…−47 ms, re the note's 100–400 ms energy. Needs f₀ ≳ 140 Hz (R3 up); below f₀ it is all knock. The first version (Hann window from −5 ms, background kept) was insensitive: +20 dB of knock read +3 dB. **Sustain:** the same at 100–400 ms without the background step (the pedal halo, E1) |
| N7 | pre-onset | A1, A2 | energy −200 … −5 ms before N0 re the local background, low band (< 1 kHz) and broadband; by velocity (touch is not velocity: report the spread) |
| N8 | early decay | C4, C9 | per partial 1–8, level drop from 30–100 ms to 300–600 ms |
| N9 | phantoms | C7 | the peak near 2f_j re partial 2j at 30–430 ms, median over the j whose position and a control position (half-way down to the nearest transverse partial) are both clear of every transverse partial; the control reads what a chance peak does. A standalone, more thorough version: `scripts/measure_phantoms.py` |
| N10 | release | F1, F2 | on note-offs with the pedal up **in any texture** (f₀ ≥ 130 Hz, no onset in −0.1…+0.3 s): level tracks (window max(30 ms, 6/f₀), every 2 ms) of the released note's first 8 partials that are clear of every other ringing note's partials, each fitted by two lines with a step between them. At the microphones the direct sound is only a few dB above the reverberant field, so a damper shows as a step and a steeper slope, not a deep drop. Delay (the breakpoint: contact plus part of the fall), step, slopes before and after. The first version (steepest descent on isolated note-offs) failed validation in context |
| N11 | spread | V classes | the note-to-note spread of N1, N2, N5 and N6 at equal key and velocity, model vs recording |
| N12 | extra peaks | E2, E1, C7 | narrow peaks at 50–650 ms (Hann, res 3.3 Hz) that are not the note's own partials (tracked up the series): new with the note (10 dB over the same stretch before it), 10 dB over the local floor and over the nearest partial's sidelobes, beyond 2 res + 0.15 % of every partial and not stronger than it. Labelled *phantom* (at an f_j + f_k), *near* (within 25 cents of a partial: unison splitting, undamped strings, duplex segments tuned near it) or *between*. Pedal up. Near peaks below 1397 Hz on damped keys cannot be undamped strings |

### 3.2 Tracks (T)

| id | measure | constituents | definition |
|---|---|---|---|
| T1 | two-stage decay | C4, C9 | per partial: prompt rate, aftersound rate, knee time and level, from partial tracks that stay ≥ 1/4 of a partial spacing clear of every other sounding partial for ≥ 2 s; long isolated notes pooled across years as a check |
| T2 | beating | C3, C5 | per partial 1–6: modulation rate and depth of the track's envelope (0.1–10 Hz), as a distribution per register (the Timbre Toolbox modulation descriptor, extended below 1 Hz) |
| T3 | pedal and decay | E1, F1 | T1's aftersound rate with the pedal down vs up, per register (R9: longer in the mid-range only); decay after note-off vs key (the undamped boundary) |

### 3.3 Events (E)

| id | measure | constituents | definition |
|---|---|---|---|
| E1 | halo | E1 | energy between the sounding notes' partials, and at the partials of undamped strings, with the pedal down vs up; first version in `scripts/measure_attack.py` |
| E2 | pedal moves | F3 | energy around pedal-down/up events with no onset or offset within 0.3 s |
| E3 (F4) | re-strikes | F4 | runs of one key struck again 0.08–0.6 s apart under the pedal: the level of the key's partials 20–100 ms after each strike; (model − recording at strike k) − (model − recording at the first strike). Positive: the model keeps too much of the ringing string |
| E4 | attacks in music | B4, D2 | positive spectral flux per octave band in −5 … +40 ms around onsets **minus the excerpt's flux away from onsets** (room and reverb fluctuate too), as a distribution over onsets, by the register of the struck note |
| E5 | una corda | F5 | spectrum and level with CC67 down vs up, same keys and velocities |

### 3.4 Passages (P)

| id | measure | constituents | definition |
|---|---|---|---|
| P1 | level and long-term spectrum | G, H1 | per octave band, whole excerpt *(exists: `scripts/evaluate.py`)* |
| P2 | level vs dynamics | H1 | by velocity tercile *(exists)* |
| P3 | texture statistics | V, S, "one-dimensional" | per band: mean, variance and skew of the envelope; modulation power in 0.5–64 Hz bands; correlations between band envelopes (McDermott & Simoncelli 2011, R10) |
| P4 | room and image | G1, G2 | **free decays** (every key up, pedal up, ≥ 0.7 s to the next onset): per octave band, T60 from a line fitted from 0.4 / 0.25 / 0.15 s after the MIDI's last damper (below 500 Hz / to 1 kHz / above) while 10 dB above the floor, its intercept corrected for the log-power bias; **early**: the 50 ms before the stop over that line, i.e. the direct sound and early reflections of what was sounding. **Image**, on the note bench: inter-channel cross-correlation (max over ±3 ms) per band at 0–30 ms and 150–400 ms, and the channels' level difference |
| P5 | floor and hum | G3 | *(exists)* |

## 4. The losses: what each term sees

### 4.1 The current terms

All are computed on 2 s windows after a 1 s warm-up.

| term | compares | time resolution | frequency resolution | optimum when the target is not predictable | sees | blind to |
|---|---|---|---|---|---|---|
| band | log 1/6-oct band energy per 10 ms frame, L1 | windows of 341 ms (< 200 Hz), 85 ms (to 1.6 kHz), 21 ms (above) | 1/6 octave | median over frames; pooled bins cut the level bias to ~0 | level, spectral envelope, slow envelopes | anything shorter than its window: the **1–1.6 kHz part of the attack burst is smeared over 85 ms**; partial detail; fine structure |
| fine | log magnitude per bin, L1, below 2 kHz | 171 and 43 ms | 5.9 and 23 Hz | median per bin; a smooth model beats a matched random one by √2 (the texture penalty) | low partials' frequencies and levels | above 2 kHz; beats slower than 0.5 Hz (less than one period in the window) |
| attack | log 1/3-oct band energy, −5 … +40 ms around onsets, L1, from 400 Hz | 10.7 ms window, 2.7 ms hop | 1/3 octave | median; **frame-by-frame, so onset timing errors of a few ms are penalised at the steepest point** | the attack's spectrum where timing is exact | the attack's variation from note to note (shrunk); anything under timing scatter |
| GAN (branch c) | multi-resolution critic on the noise-only view | – | – | the distribution | noise texture | nothing else, by construction; weight 0.1 was too weak to move anything |
| budget, reg | priors on the residual and parameters | – | – | – | – | – |

Seen together:
- **Nothing is longer than 2 s.** Beats at 0.1–2 Hz, the aftersound, the knee of the double decay, T60 and
  releases after 2 s are seen only mixed into windows holding the tails of older notes.
- **Nothing aligns per note except the attack term, and it has no timing tolerance.** Whether the onset scatter
  is large enough to matter is not known: N0 answers it.
- **Only the GAN matches a distribution.** Every V and S constituent is left to median-seeking terms, which
  shrink it.

### 4.2 The loss each class needs

| class | constituents | loss |
|---|---|---|
| D | spectrum vs velocity, frequencies, prompt rates, damper timing, level | aligned terms where alignment is reliable. On the note bench: per partial (level and decay of each partial of an isolated note). On music: band envelopes with windows matched to each constituent. Level: log band energy **summed over the whole excerpt** (its optimum is the energy match by construction) |
| K | double-decay knee, beating, per-key colouration, halo | per key, over long windows: statistics of T1/T2 tracks (rates, depths, knee), fitted to per-key parameters, including the unison mistuning, which is drawn at random today and never fitted |
| V | touch precursor, knock fine structure, attack variation, onset timing | distribution terms across notes at equal key and velocity (moments, or a critic conditioned on key and velocity). The model then needs a source of variation (a per-note random input), or it can only reproduce the median |
| S | noise, reverb tail, floor | texture statistics (P3) or a critic with enough weight |

Two more rules:
- **Timing.** Attack terms either align each note to its own detected onset (N0) before comparing, or use
  features that tolerate a few ms (onset-integrated energy, or scattering-type features; Vahidi et al. 2023,
  R12).
- **A bias gate for every term, on every fitted model.** For each measure in section 3, the term's optimum is
  checked against the energy-matched (or measured) value, as `scripts/loss_bias.py` does for level. Round 2
  showed that a gate passed on one checkpoint can fail on the model the loss produces (4–8 kHz).

## 5. Stages

1. **Measure.** Build the bench (per year, fixed, split by piece), the tracks, the per-year measurements, and
   validate every measure on synthetic notes (principle 5).
2. **Calibrate on notes.** Fit the per-key and per-year physics that isolated notes identify: spectrum vs
   velocity, early decays, knock, release, level. Use aligned per-note terms. Freeze what is measured more
   directly than fitted, as B and tuning are now.
3. **Fit on music.** Fit only what isolated notes cannot show: pedal and sympathetic resonance, the room,
   context, the residual. Use the terms of 4.2 by class.
4. **Evaluate.** The full suite runs after every stage, with listening files per stratum. Each run adds a row
   to the audit (section 6).

## 6. The audit

Built and run on the round-2 models and the trial model (sections 9 and 10; `runs/round2/bench/report.md`,
`runs/round2/bench_release/report.md`). The "round 2" column is the control (physics only) unless noted; the
other models differ from it by far less than from the recordings. It supersedes a first, flawed measurement
(`scripts/measure_attack.py`; section 9.3).

| constituent | model parameters | evidence | round 2, from the bench | decision |
|---|---|---|---|---|
| level, velocity map (H1) | gains, velocity law and curve | N1, P1, P2 | per note within ±0.6 dB in R2–R5 and in every velocity bin; broadband −0.5 dB; 4 kHz −1.8 dB in music (loss bias) | whole-excerpt level term |
| spectrum vs velocity (B1, C2) | contact time, roll-off order, second corner, colouration, `partial_gain` | N2, N5 | bass and tenor too bright: spectral centroid +212 cents (R2), +66 (R3), +22 (R4); sustain +3 dB at 4 kHz in R2. **Brightness grows too little with velocity**: soft notes +37 cents and +3.2 dB at 2 kHz in the attack, loud notes −21 cents and −1.2 dB | calibrate the hammer (contact time and roll-off vs velocity) on the bench |
| knock (B4) | knock impulse, knock noise (per-key spectrum), body | N6 knock, N4 percussive, E4 | **the attack has the wrong spectrum in every register** (section 10.3): too much 8 kHz everywhere (percussive +10…+20 dB; the fitted knock noise is what makes it: +20 dB of it reads +21 dB); bass and tenor lack a 250 Hz thump (percussive −4…−5 dB, E4 −1…−2 dB); the middle lacks a 1 kHz knock (R5 −10 dB, 182 notes), which grows with velocity in the piano and not in the model (mf–f −9 dB); the treble has a low thud 13–18 dB too strong at 125–500 Hz (121 notes), more so when loud | structure (section 10.4): a string-borne precursor up to ~5 kHz, a structure thump limited to ~2 kHz with the treble's long low ringing, no knock noise above that |
| board build-up (D2) | body FIR, hall | N4, P4 | rise R3 +16 ms (model slower), R4–R5 −8…−9 ms (faster). P4 rules out a drier model room: the model's direct and early sound sits 1.4–2.7 dB *lower* over its reverberant field than the recordings'. So the fast mid-register rise is the piano model's (strings, body, knock) | revisit after the knock change |
| touch precursor (A1) | none | N7 | within 1 dB except R2 (recordings +1.4 dB): small. N7 is validated only on synthetic notes | low priority |
| double decay (C4) | R, g(f), aftersound amplitudes | N8, T1 | N8 within ±0.9 dB in R2–R6. The fit's weaker R (−10…20 %) does not show at 0.4 s | T1 for the later stages |
| beating (C3, C5) | unison mistuning (random, fixed) | T2 | not built | measure; make the mistuning per-key learnable |
| phantoms, longitudinal (C7) | phantom levels (2f_j and f_j + f_(j+1) series) | N9 | **present in the recordings**: on 40 loud R2–R3 notes the component at 2f_j reads −10 dB re partial 2j, 10 dB over the control position. The model's reads −17 dB (7 dB weaker); +20 dB of them brings it to −8. In the middle register the room masks both | add the (j−1, j+1) pairs, which the literature finds dominant at 2f_j (follow-up F2, F4); keep N9 on loud bass and tenor notes |
| sympathetic (E1) | bank (off) | E1, T3 | pedal down − up, energy between the partials: recordings +1.9 (R2), 0.0 (R3), −2.1 (R4), −1.4 (R5) dB; model −2.9, −3.4, −1.8, −1.1. A 3–5 dB gap in the bass and tenor only | switch on for R2–R3; listen to `runs/round2/ab_symp` |
| dampers, release (F1, F2) | damper rates, delay, boundary, release noise | N10, T3 | rebuilt N10 (partial tracks, any texture; +20 ms reads +14 ms): the piano's dampers show as a −4.5…−4.7 dB step and a slope change from −9…−15 to −31 dB/s (the room). The model is within 7 ms and 0.7 dB in R3–R4 (105 note-offs) | none now |
| re-strike (F4) | re-strike damping | F4 | no build-up: from the second strike on the model stays within ±1 dB of the piano in R3–R7; in the bass its re-struck notes come out 1–5 dB *quieter*. The parameter moves little: no damping at all reads +0.6 dB | none now; the bass deficit is not the re-strike parameter |
| pedal noise (F3) | pedal noise | E2 | not built | measure |
| room, image (G1, G2) | hall T60, bodies, pan | P4 | **the model's hall decays too fast**: T60 0.2–0.4 s short in every band (recordings 2.3 s at 250 Hz, 1.75 at 1 kHz, 1.5 at 4 kHz; 52 free decays), well past the ~5 % a listener notices. Channel balance within ±1 dB; the model's late field is more correlated between the channels at 2 kHz (+0.1…+0.3) | measure T60 per year from free decays and set it (stage 1) instead of fitting it |
| floor (G3) | floor, hum | P5 | within 1 dB | done |
| variation (V) | per-key tables (smoothed), residual | N11 | **at equal pitch and velocity the model's notes vary 30–70 % as much as the piano's** in spectrum slope, attack at 2 kHz, early decay and onset timing; level 70–105 % (spread ratio, second pass) | per-key freedom and a source of per-note variation (section 4.2, class V) |
| texture (S) | residual noise, GAN | P3 | within ±1 dB mostly; 4–8 kHz has +1–2 dB more fast modulation (the clicks), −3.5 dB slow modulation at 8 kHz | follows the knock fix |

## 7. Build order

1. **N0** (a per-note onset) and the **bench** index; then validate N0 on synthetic notes and on the recordings (failure rate).
2. **N1–N11, E1, E4, P3** in a `pianonn/measures.py`, each with a synthetic-recovery test; the report added to
   `scripts/evaluate.py`.
3. **Baseline:** run the suite on the trial and round-2 models. That gives the audit's "round 2" column from
   the suite, not from guesses.
4. **T1–T3, P4:** tracks and free decays. Harder; after the baseline.
5. **Loss changes by class (section 4.2),** each gated for bias on the suite; then the stages of section 5.

## 8. Limits

- The suite will not capture everything: interactions in dense textures, perceived power, and anything no
  constituent names. Listening stays in every round, now organised by stratum.
- **Evidence behind the audibility column.** Several entries rest on abstracts or cited claims (see the
  literature report's labels). The report's open questions stand: the dB levels of aftersound, phantoms, knock
  and precursor re the tone; and JNDs for beat depth and decay time in piano tones.
- **Next reads:** Bank & Chabassier 2019 (IEEE SPM) and Lehtonen's 2010 thesis, for a numbers-rich list.

## 9. Built so far (2026-09-30)

### 9.1 What exists

- `pianonn/measures.py`: the bench (`build_bench`), context rendering (`render_clips`), N0–N8, N10, E4 and P3,
  and the paired summaries.
- `tests/test_measures.py`: 16 tests on synthetic notes with known answers.
- `scripts/note_bench.py`: runs the suite for any set of checkpoints. It also runs the validation: known
  parameter changes pushed through the renderer.
- `runs/measurements/bench_2018.json`: 1,462 notes (459 in the evaluation group; R1, R6 and R7 are sparse) and
  59 note-offs.

Method choices forced by validation:
- **Power is summed over the two channels.** Averaging the spaced pair comb-filters the spectrum.
- **N0 averages over one period and walks back from the steepest rise.** The first detector searched forward
  from the window's edge and locked onto earlier notes' ringing on 273 of 482 notes. The new one fails on 29 %
  in R2 (the hall's tail sits a few dB under a bass note), 20 % in R3, 13 % in R4 and 4 % above.
- **N4 is defined only from f₀ ≥ 196 Hz.** A sum of partials beats at their spacing, so rise times are
  resolved to about one period.
- **E4 is a contrast:** the flux at onsets minus the excerpt's flux away from them. The recordings' room and
  reverb fluctuate ~3.5 dB more everywhere at 250 Hz, so raw flux read as a missing thump.

### 9.2 Validation status

*Superseded by 10.2 for N6 and N10.*

| measure | status |
|---|---|
| N0 onset | +5 ms of latency reads +5.02 ms |
| N1 level | +6 dB of mic gain reads +6.00 dB |
| N2 spectrum | contact time ×0.7 reads +2.9 dB/oct and +196 cents |
| N5 attack spectrum | +20 dB of knock noise reads +10 dB at 4 kHz |
| N8 early decay | prompt ratio ×1.5 reads +1.0 dB (in the right direction, small at 0.4 s) |
| N6 between the partials | **insensitive**: +20 dB of knock reads +3 dB. A partial's own onset spreads energy between the partials. Needs a knock measure that separates the two: harmonic–percussive separation, or the first 0–5 ms before the partials build up |
| N10 release | **fails** in real passages: +20 ms of damper delay reads +0 over 21 releases (+12…+24 ms on isolated renders). The steepest descent after a note-off is usually something else in context. Needs per-partial tracks of the released note, clear of other notes' partials (as T1) |
| N4, N7, E1, E4, P3 | validated on synthetic notes only |

### 9.3 Corrections to the first measurement

`scripts/measure_attack.py` averaged the channels and used the broadband onset detector. Three of its findings do
not survive the bench, and are corrected in section 6 and in chat:
- The "1–2 kHz attack burst 1–3 dB weak" is within ±1 dB in R3–R5.
- "The 1 kHz partials lose 2–3 dB too little by 400 ms": N8 is within ±0.9 dB.
- "The pedal adds 4–9 dB between the partials in the recordings, nothing in the model": 3–5 dB in the bass
  and tenor only.

What the first measurement pointed at does hold, sharper: the attack is string-only, with a click and without
a thump; brightness does not follow velocity; and the notes are too uniform.

## 10. Second pass: knock, room, phantoms, re-strike, release (2026-09-30)

### 10.1 What was added

- **N6 knock** (replaces the insensitive N6 attack), **N4 percussive** (attack sharpness per band), **N9** on the
  bench (the note's own partials, found near the table's: the model's per-key B at E2–A2 is ~25 % low, which
  moves partial 25 by ~40 Hz), **N10** rebuilt on partial tracks in any texture, **P4** (free decays: T60 and
  early energy; image: inter-channel correlation and balance), **F4** (re-strike runs under the pedal).
- A fix to a shared primitive: `band_envelope` used a brick-wall band mask, whose time response falls as 1/t.
  A loud part of a clip leaked 30–50 dB down into quiet parts before and after it: it capped decay fits and
  could fake pre-onset energy (N7). Band edges are now raised-cosine, 1/6 octave on each side.
- The bench (`runs/measurements/bench_2018.json`) gained 202 free decays, 393 re-strike runs and 299 note-offs,
  drawn after the notes and releases so those are unchanged.
- 25 synthetic tests (`tests/test_measures.py`).

Choosing the knock measure: three candidates were pushed through the renderer with the knock changed and with
string changes that should not move it (`runs/round2/bench/report.md`, validation; the probe is in the session
notes). Extrapolating the string's decay back to the onset did not see the knock. Harmonic–percussive
separation sees the knock and the partials' own onset alike: kept, as a sharpness measure. The energy between
the partials sees the knock once its background is removed and its window is flat-topped.

### 10.2 Validation (known changes through the renderer)

| change | measure | reads | pass |
|---|---|---|---|
| latency +5 ms | onset | +5.01 ms | yes |
| mic gain +6 dB | N1 | +6.00 dB | yes |
| knock noise +20 dB | N6 knock 4 kHz / N4 percussive 8 kHz / N5 attack 4 kHz | +25.6 / +20.9 / +10.2 dB | yes |
| knock impulse +20 dB | N6 knock 500 Hz | +12.3 dB | yes |
| contact time ×0.7 | N2 slope / centroid / N6 knock 1 kHz | +2.9 dB/oct / +200 cents / +1.3 dB (not a knock) | yes |
| prompt ratio ×1.5 | N8 / N6 knock 1 kHz | +1.0 / 0.0 dB | yes |
| phantoms +20 dB | N9 | R3–R5 mixed velocity: +0.05 (masked by the room); loud R2–R3: +9 to +15 dB | mid: no; bass: yes |
| channel 0 +3 dB | P4 balance | +3.00 dB | yes |
| one body for both channels | P4 correlation early | +0.14 | yes |
| hall T60 ×1.3 | P4 T60 1 kHz / 4 kHz | +0.38 / +0.35 s | yes |
| hall −6 dB | P4 early 1 kHz | +4.0 dB | yes |
| no re-strike damping / ×3 | F4 | +0.58 / −0.18 dB | yes (small: the parameter barely matters) |
| damper delay +20 ms | N10 delay / step | +14 ms / +0.04 dB | yes |

### 10.3 What the second pass found

The attack, per register (control model; eval group unless noted; model − recording):

| register | N6 knock (energy between partials, −3…33 ms) | N4 percussive | E4 (music) |
|---|---|---|---|
| R2–R3 (bass, tenor) | not defined below f₀ ≈ 140 Hz; R3 within ±4 dB | 250 Hz −4…−5 dB; 4–8 kHz +7…+16 dB | 250 Hz −1…−2 dB; 8 kHz +4 dB |
| R4–R5 (middle) | 1 kHz −10 dB and 2 kHz −3…−7 dB in R5 (182 calib notes); 8 kHz +6…+12 dB | 8 kHz +12…+15 dB; 1 kHz −1…−4 dB | 1 kHz −1 dB; 8 kHz +6…+7 dB |
| R6–R7 (treble, 121 calib notes) | 125–500 Hz +13…+18 dB; 4–8 kHz +10…+16 dB | 250 Hz +18 dB; 8 kHz +18 dB | 8 kHz +4…+12 dB |

By velocity (R2–R6): the model's 1 kHz knock falls short most at mf–f (−9 dB); its 4 kHz sharpness is +14 dB at
p and −3.5 dB at mf–f: the model's attack does not grow with the blow the way the piano's does.

The literature follow-up (`docs/literature/tone_constituents_followup.md`, F1 Askenfelt 1993) explains the
pattern. The strong attack path is the string's longitudinal precursor: a 1–2 ms burst 9–14 dB under the first
transverse wave at the bridge, up to ~5 kHz, the "bite" above 1 kHz. The structure-borne thump (keybed, rim,
soundboard: 95–330 Hz) is ~40 dB under the strings at the bridge but louder in the radiated sound, and it
reaches only ~2 kHz; in the treble the board's low modes ring 0.2–0.5 s (Bank's thesis, F2). The model has no
precursor, a knock noise that reaches 8 kHz, and a treble impulse that puts a short low thud where the piano has
a weaker, longer one.

Also:
- **Brightness vs velocity is not the hammer's form.** The model's force-spectrum cutoff rises as v^0.46–0.60 in
  the bass and middle, as measured on Steinway D hammers (v^0.4–0.6, F9). The gap the bench sees (soft notes too
  bright, loud ones too dull) is the MIDI-velocity-to-hammer-speed map, which is fixed, and a fit that did not
  use its freedom there: a calibration on isolated notes (stage 1).
- **Room:** the hall's T60 is 10–30 % short in every band (10.2 checks the measure).
- **Phantoms** are in the recordings' loud bass and tenor, ~7 dB stronger than the model's (6).
- **Re-strike and release** are fine as far as the measures see (6).
- **Literature corrections:** the first report's "B up to 0.4 in the treble" is wrong (0.01–0.03); strike point
  1/8 in the bass to 1/12–1/17 at the top. The model's priors were not affected.

### 10.4 The physics this points to (proposed, not built)

In order of the measured gap:
1. **Attack in three parts** (B4): (a) a string-borne precursor, 1–2 ms, up to ~5 kHz, ~−10 dB re the first
   transverse wave, scaling with the blow, through the body like the strings; (b) a structure thump with modes
   near 95–330 Hz, limited to ~2 kHz, with a long (0.2–0.5 s) low ringing in the treble; (c) no knock noise
   above ~2–3 kHz. Measures: N6 knock, N4 percussive, E4 per register and velocity.
2. **Per-strike variation** (class V): at equal pitch and velocity the model's notes vary 30–70 % as much as the piano's in spectrum, attack, early decay and onset timing (level: 70–105 %).
3. **Hall T60** set from P4 per year.
4. **Phantoms** from (j−1, j+1) pairs as well; N9 on loud bass notes.
5. **Velocity map** calibrated on the bench (N2 by velocity).
6. **Sympathetic bank** on for R2–R3, limited to ~2.5 kHz (Lehtonen et al. 2007: the pedal's extra non-partial
   energy is below 2.5 kHz).
Measured and close enough: touch precursor (N7 within 1 dB), re-strike (F4), dampers (N10). Pitch glide and
duplex strings: measured in section 11, small or not separable (11.4). Multiple hammer contacts: not measured; read
from the velocity-map calibration (item 5): the map is one curve for all keys, contacts depend on register, so a
register-dependent, non-monotonic residual of the partial balance after the calibration would be their sign.

## 11. Glide and extra peaks (2026-09-30)

"Nothing shows a gap" in 10.4 first covered three constituents no measure could see. Two now have measures.

### 11.1 What was added

- **N3 glide** and **N12 extra peaks** (definitions in 3.1), with synthetic tests (`tests/test_measures.py`).
  `own_partials` gained `track=True`: each partial is searched near the last one's stretch, because the
  recordings' tenor partials drift from the model's table by 20 cents and more by partial 30.
- `scripts/measure_glide_extra.py`: N3 on 854 held R2–R4 notes of the bench, N12 on 217 pedal-up R3–R7 notes, the
  control model as rendered and with its sympathetic bank on (`runs/round2/glide_extra/`).
- `scripts/survey_after_silence.py`: N12 on single notes after 1 s of silence in the recordings of every year
  (`runs/survey/after_silence/`). For this the other nine MAESTRO years were converted from the local zip into
  `data/maestro24k` (1276 pieces, 199 h); scripts whose default is every year need `--years 2018` to repeat
  earlier runs.

### 11.2 Validation

| measure | check | reads | pass |
|---|---|---|---|
| N3 | synthetic 3-cent glide (τ 0.3 s), f₀ 55 / 110 / 330 Hz | 1.69 / 1.83 / 1.84 (definition 1.68 / 1.83 / 1.83); 0.00 without; a ringing earlier note moves one partial, not the median | yes |
| N3 | the same glide warped into 212 of the model's renders | +1.78 cents (definition +1.83) | yes |
| N3 | synthetic glide through an exponential-noise hall (T60 1.8 s), direct-to-reverberant +6 … −6 dB | 60–100 % of the dry reading: the room hides up to 40 % | caveat |
| N12 | synthetic tones at 1.02 × partial 2 and 0.985 × partial 4 | found, level within 0.1 dB, position within 3 cents; a steady tone before the note is not counted | yes |
| N12 | tones added to the bench's recordings (music) | re partial 3: −20 dB found 35–55 %, −30 dB 10–25 %, −40 dB 0–7 %. The surroundings hold earlier notes' partials, so most positions are not "new" (a shorter stretch before the note did worse) | **blind below ~−25 dB** |
| N12 | tones added to notes after silence whose floor allows it (18 notes) | half-way between partials: −40 dB found 89 %, −50 dB 17 %; 4 % from a partial: −30 dB 94 %, −40 dB 33 %; 2 %: −30 dB 50 % | yes, to ~−40 dB between and ~−30 dB beside the partials |

After silence the floor between partials is the recording's own noise (the stretch before the note reads the
same), 20–35 dB under partial 3 on soft notes and 45–60 dB on loud ones: a soft note can show only strong extras.

### 11.3 What they found

**Glide (N3, recording − model, cents, median [90 % bootstrap]).** The model reads +0.15…+0.6 at every velocity
(its detuned unison strings' double decay). The recordings' early-vs-late pitch grows with velocity in R3–R4:
R4 −0.11 (p), +0.06, +0.26, +0.97 (ff, 14 notes); R3 −0.03, +0.03, +0.17, +1.74 (ff, 5 notes). At ff the
difference is R4 +0.77 [+0.30, +0.88] and R3 +1.29 [+0.37, +1.42]; with the window's and the room's losses that is
about 1.5–3 cents at the onset, within the agent's 0.5–4-cent estimate (follow-up 5.5). R2 shows no growth with
velocity in these windows (its slower decay stretches the glide past 0.6 s). At p–mp the difference is −0.2…−0.7,
the same at every velocity: the recordings' pitch rises slightly as a note decays, the model's falls; that is the
unison and double-decay structure (T2), not glide.

**Extra peaks on the bench (N12, music, pedal up; blind below ~−25 dB).**
- Phantoms reach the treble at loud velocities. Phantom peaks per note, recording vs model: R3 p–mp–mf-f 0.26 /
  1.22 / 3.25 vs 0 / 0.04 / 0.62; R4 0 / 0.48 / 1.85 vs 0 / 0.04 / 1.07; R5 ff 1.40 vs 0; R6 ff 1.67 vs 0.33. On a
  loud C♯6 they sit at 2f₂, f₂+f₃ and 2f₃, 15–25 dB under the partials beside them, one B (0.0033) fitting all
  three. N9 found the bass and tenor; N12 extends it upwards.
- **The model's soundboard filter has narrow modes.** 34 of the model's 94 extra peaks sit at the same frequencies
  across keys (1158 Hz in 23 notes of 12 keys; 612 and 2740 Hz); the recordings' 208 have none. The body FIR's
  1158 Hz peak is 2.9 Hz wide, the resolution limit of its 300 ms: it rings through the whole filter after every
  knock. In R6 the model shows such a peak in 88 % of notes, ~29 dB under the partials. A real board's modes near
  1 kHz are far broader and overlap.

**Extra peaks after silence (N12, 45 notes of all years; 18 with a floor that allows −40 dB).** On the clean
tenor and middle notes (R3 5, R4 12): phantoms 2.2–2.3 per note, 14–16 dB under the partial beside them; *near*
peaks (within 25 cents) 0.3–0.4 per note at −11…−15 dB, none below 1397 Hz on damped keys, so most likely the
undamped strings (they answer only from F6's 1397 Hz up) and not unison splitting, which would show at every
partial; *between* peaks 1.4–2.0 per note at −32…−39 dB. Over all 45 notes the non-phantom peaks cluster 25–100
cents from a partial (23 peaks, −26 dB, on both sides) more than farther out (16 peaks, −40…−43 dB). None recurs
at a fixed frequency within a year.

### 11.4 What this means for the physics (10.4)

- **Glide:** real, at ff in the tenor and middle, about 1.5–3 cents at the onset. Pitch discrimination of complex
  tones is a few cents for steady tones; a 2-cent fall over 0.3 s is at or below it (not sourced). Deferred;
  cheap to add later (a frequency factor that follows the note's partial energy).
- **Duplex:** not separable. Clean notes carry narrow components the model lacks: 25–100 cents beside a partial at
  ~−26 dB (a beat of ±0.4 dB at 15–60 Hz near 1 kHz: a faint shimmer at most) and farther out at −32…−40 dB. Duplex
  segments, the undamped strings, longitudinal modes (plain strings put them at a few kHz) and unison mistuning all
  fit, and 45 notes do not separate them. The cheapest physics that covers the near components exists already: the
  sympathetic bank (undamped strings) and per-key unison mistuning (T2). No duplex element on this evidence.
- **Phantoms (item 4):** extend to R5–R6 at loud velocities.
- **New: the body filter's narrow modes.** Limit the soundboard FIR's Q (smooth its response, or make its tail
  decay faster), together with item 1, which changes what excites it.
- **Multiple contacts:** from item 5, as in 10.4.
