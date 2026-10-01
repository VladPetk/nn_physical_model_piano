# Piano tone: constituents, measures and losses

A design for round 3: derive what we measure, and what we train on, from what a piano note is made of. Sections
0–8 are the design; sections 9–11 report what has been built, how each measure was validated and what it found
(status 2026-09-30). The physics it points to is in 10.4, revised in 11.4; section 12 is phase 3: its order and its
steps so far (per-key B, the strike comb's sign; the body's Q and the hall); section 13 the learned residual and its ceiling; section 14 phase 4 (the attack in three parts, a gain per piece, the
whole-excerpt level term, a cheap sympathetic bank); section 15 isolated tenor notes partial by partial.

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
| B3 | multiple hammer contacts (treble) | contact outlasts half the string period | treble, loud | unknown; not measured (read from the velocity-map calibration, 10.4) | no | D |
| B4 | knock | the hammer impact excites the frame and board directly: a short broadband transient | strongest re the tone in the treble; grows ~13 dB less than the tone from pp to ff | "the most important issue" for treble notes (Bank et al. 2003, R7) | part: a pulse of the contact time's shape + band noise; spectrum shape fixed | V (fine structure), D (level, spectrum) |
| C1 | transverse partials | f_n = n f₀ √(1+Bn²); stretch | B from 1e-4 (tenor) to 3e-2 (C8) | inharmonicity audible, most at low f₀ (Järveläinen et al., R5) | yes (measured, frozen) | D |
| C2 | partial amplitudes | hammer spectrum × comb × bridge colouration | register, velocity | timbre | yes | D |
| C3 | unison strings | 1 (A0–E1), 2 (F1–A♯2), 3 above; mistuning 0.2–2 cents | – | "unison quality" (tuners); no formal test found | yes: coupled modes, mistuning drawn at random per key | K |
| C4 | double decay | in-phase (prompt) mode drains through the bridge; the mid-range loses ~20 dB in the first second, then slow aftersound | strongest mid-range; treble nearly single-slope | the "prompt sound" vs "aftersound" | yes: R and bridge conductance g(f) | D (rates), K (knee, per key) |
| C5 | beating | amplitude modulation from mistuned unisons and polarisations; 0.1–2 Hz at low partials | 2–3-string keys | liveliness; untested formally | yes (via mistuning) | K |
| C6 | two polarisations | vertical couples to the board (fast), horizontal does not (slow) | all | inseparable from C4/C5 in audio | no (absorbed in C4) | K |
| C7 | longitudinal vibration, phantom partials | tension modulation: components at 2f_j and f_j+f_k; longitudinal modes near 15 f₁; precursor | bass and tenor; **audible up to C5** (Bank & Lehtonen 2010, R4); grows with velocity² | "metallic" bass | part: phantoms yes; longitudinal modes and precursor no | D |
| C8 | pitch glide | tension rises at large amplitude | bass, ff | measured (N3, 11.3): ~1.5–3 cents at the onset at ff in the tenor and middle; at or below pitch discrimination | no | D |
| C9 | internal and air losses | aftersound decay rising ~linearly with frequency | register | decay length | yes (b1, b3, p) | D |
| D1 | bridge/soundboard admittance | sets the prompt loss per partial and radiation; piano-specific resonances | register | long-term colour | yes: g(f) per year, body FIR, colouration | D |
| D2 | board build-up | attack rise 8–16 ms mid-range (Iowa) | register | attack softness | yes (body FIR) | D |
| D3 | weak radiation of bass fundamentals | board radiates poorly below its first modes | bass | "missing fundamental" | yes (body) | D |
| E1 | sympathetic resonance | undamped strings (pedal, treble above the dampers, held keys) driven by the bridge; halo; the pedal lengthens mid-range decays (Lehtonen et al. 2007, R9) | pedal, register | halo, sustain | part: bank exists, **off in all training** | K/S |
| E2 | duplex/aliquot scaling | string segments beyond the bridge resonate | treble | not separable (N12, 11.3): clean notes carry narrow components the model lacks, which undamped strings, longitudinal modes and unison mistuning explain as well | no | K |
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

Built and run on the round-2 models and the trial model (sections 9–11; `runs/round2/bench/report.md`,
`runs/round2/bench_release/report.md`). The "round 2" column is the control (physics only) unless noted; the
other models differ from it by far less than from the recordings. It supersedes a first, flawed measurement
(`scripts/measure_attack.py`; section 9.3). Glide, extra peaks and the body's modes: `runs/round2/glide_extra/`
and `runs/survey/after_silence/` (section 11).

| constituent | model parameters | evidence | round 2, from the bench | decision |
|---|---|---|---|---|
| level, velocity map (H1) | gains, velocity law and curve | N1, P1, P2 | per note within ±0.6 dB in R2–R5 and in every velocity bin; broadband −0.5 dB; 4 kHz −1.8 dB in music (loss bias) | whole-excerpt level term |
| spectrum vs velocity (B1, C2) | contact time, roll-off order, second corner, colouration, `partial_gain` | N2, N5 | bass and tenor too bright: spectral centroid +212 cents (R2), +66 (R3), +22 (R4); sustain +3 dB at 4 kHz in R2. **Brightness grows too little with velocity**: soft notes +37 cents and +3.2 dB at 2 kHz in the attack, loud notes −21 cents and −1.2 dB | calibrate the hammer (contact time and roll-off vs velocity) on the bench |
| knock (B4) | knock impulse, knock noise (per-key spectrum), body | N6 knock, N4 percussive, E4 | **the attack has the wrong spectrum in every register** (section 10.3): too much 8 kHz everywhere (percussive +10…+20 dB; the fitted knock noise is what makes it: +20 dB of it reads +21 dB); bass and tenor lack a 250 Hz thump (percussive −4…−5 dB, E4 −1…−2 dB); the middle lacks a 1 kHz knock (R5 −10 dB, 182 notes), which grows with velocity in the piano and not in the model (mf–f −9 dB); the treble has a low thud 13–18 dB too strong at 125–500 Hz (121 notes), more so when loud | structure (section 10.4): a string-borne precursor up to ~5 kHz, a structure thump limited to ~2 kHz with the treble's long low ringing, no knock noise above that |
| board build-up (D2) | body FIR, hall | N4, P4 | rise R3 +16 ms (model slower), R4–R5 −8…−9 ms (faster). P4 rules out a drier model room: the model's direct and early sound sits 1.4–2.7 dB *lower* over its reverberant field than the recordings'. So the fast mid-register rise is the piano model's (strings, body, knock) | revisit after the knock change |
| touch precursor (A1) | none | N7 | within 1 dB except R2 (recordings +1.4 dB): small. N7 is validated only on synthetic notes | low priority |
| double decay (C4) | R, g(f), aftersound amplitudes | N8, T1 | N8 within ±0.9 dB in R2–R6. The fit's weaker R (−10…20 %) does not show at 0.4 s | T1 for the later stages |
| beating (C3, C5) | unison mistuning (random, fixed) | T2 | not built | measure; make the mistuning per-key learnable |
| phantoms, longitudinal (C7) | phantom levels (2f_j and f_j + f_(j+1) series) | N9 | **present in the recordings**: on 40 loud R2–R3 notes the component at 2f_j reads −10 dB re partial 2j, 10 dB over the control position. The model's reads −17 dB (7 dB weaker); +20 dB of them brings it to −8. In the middle register the room masks both. N12 (11.3): phantoms also in loud R5–R6, 1.4–1.7 phantom peaks per note vs the model's 0–0.3 | add the (j−1, j+1) pairs, which the literature finds dominant at 2f_j (follow-up F2, F4); keep N9 on loud bass and tenor notes |
| pitch glide (C8) | none | N3 | real at ff in R3–R4: early − late +1.0…+1.7 cents vs the model's +0.2…+0.5, i.e. ~1.5–3 cents at the onset; R2 shows no growth with velocity in these windows (11.3) | deferred: at or below pitch discrimination; cheap to add later |
| duplex, other narrow extras (E2) | none | N12 | blind below ~−25 dB in music. On 45 notes after silence: components 25–100 cents beside a partial at ~−26 dB and farther out at −32…−40 dB that the model lacks; not separable from undamped strings, longitudinal modes, unison mistuning (11.3) | no duplex element; the sympathetic bank and per-key unison mistuning (T2) cover the near ones |
| body modes (D) | body FIR | N12 | **the model's soundboard filter has narrow modes** (1158, 612, 2740 Hz; 2.9 Hz wide) that ring after every note: 34 of its 94 extra peaks recur across keys, the recordings' 208 never | limit the FIR's Q with the attack rebuild |
| multiple contacts (B3) | none (spectral hammer) | – | not measured | read from the velocity-map calibration (10.4, item 5) |
| sympathetic (E1) | bank (off) | E1, T3 | pedal down − up, energy between the partials: recordings +1.9 (R2), 0.0 (R3), −2.1 (R4), −1.4 (R5) dB; model −2.9, −3.4, −1.8, −1.1. A 3–5 dB gap in the bass and tenor only | switch on for R2–R3; listen to `runs/round2/ab_symp` |
| dampers, release (F1, F2) | damper rates, delay, boundary, release noise | N10, T3 | rebuilt N10 (partial tracks, any texture; +20 ms reads +14 ms): the piano's dampers show as a −4.5…−4.7 dB step and a slope change from −9…−15 to −31 dB/s (the room). The model is within 7 ms and 0.7 dB in R3–R4 (105 note-offs) | none now |
| re-strike (F4) | re-strike damping | F4 | no build-up: from the second strike on the model stays within ±1 dB of the piano in R3–R7; in the bass its re-struck notes come out 1–5 dB *quieter*. The parameter moves little: no damping at all reads +0.6 dB | none now; the bass deficit is not the re-strike parameter |
| pedal noise (F3) | pedal noise | E2 | not built | measure |
| room, image (G1, G2) | hall T60, bodies, pan | P4 | **the model's hall decays too fast**: T60 0.2–0.4 s short in every band (recordings 2.3 s at 250 Hz, 1.75 at 1 kHz, 1.5 at 4 kHz; 52 free decays), well past the ~5 % a listener notices. Channel balance within ±1 dB; the model's late field is more correlated between the channels at 2 kHz (+0.1…+0.3) | measure T60 per year from free decays and set it (stage 1) instead of fitting it |
| floor (G3) | floor, hum | P5 | within 1 dB | done |
| variation (V) | per-key tables (smoothed), residual | N11 | **at equal pitch and velocity the model's notes vary 30–70 % as much as the piano's** in spectrum slope, attack at 2 kHz, early decay and onset timing; level 70–105 % (spread ratio, second pass) | per-key freedom and a source of per-note variation (section 4.2, class V) |
| texture (S) | residual noise, GAN | P3 | within ±1 dB mostly; 4–8 kHz has +1–2 dB more fast modulation (the clicks), −3.5 dB slow modulation at 8 kHz | follows the knock fix |

## 7. Build order

1. **N0** (a per-note onset) and the **bench** index; then validate N0 on synthetic notes and on the recordings
   (failure rate). *Done (9).*
2. **N1–N11, E1, E4, P3** in a `pianonn/measures.py`, each with a synthetic-recovery test. *Done (9, 10), plus
   N3 glide and N12 (11); the report is `scripts/note_bench.py`, not `scripts/evaluate.py`.*
3. **Baseline:** run the suite on the trial and round-2 models. That gives the audit's "round 2" column from
   the suite, not from guesses. *Done (6).*
4. **T1–T3, P4:** tracks and free decays. Harder; after the baseline. *P4 done (10); T1–T3 not built.*
5. **Loss changes by class (section 4.2),** each gated for bias on the suite; then the stages of section 5.
   *Not started. Revised in 12.1: the note-fitting terms come with the first physics of 10.4, not after it.*

## 8. Limits

- The suite will not capture everything: interactions in dense textures, perceived power, and anything no
  constituent names. Listening stays in every round, now organised by stratum.
- **Evidence behind the audibility column.** Several entries rest on abstracts or cited claims (see the
  literature report's labels). The report's open questions stand: the dB levels of aftersound, phantoms, knock
  and precursor re the tone; and JNDs for beat depth and decay time in piano tones.
- **Next reads:** Bank & Chabassier 2019 (IEEE SPM) and Lehtonen's 2010 thesis, for a numbers-rich list.
- **Weak narrow components are out of reach in music.** Earlier notes' partials fill most frequencies, so N12
  sees nothing weaker than ~−25 dB re the partials; after silence it reaches ~−40 dB, but all ten years hold only
  45 such single notes (11.2).

## 9. Built so far (2026-09-30)

### 9.1 What exists

The first pass; 10.1 and 11.1 add to it (the test file now has 29 tests).

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
1. **Attack in three parts** (B4), with the soundboard filter's narrow modes damped (11.4): (a) a string-borne precursor, 1–2 ms, up to ~5 kHz, ~−10 dB re the first
   transverse wave, scaling with the blow, through the body like the strings; (b) a structure thump with modes
   near 95–330 Hz, limited to ~2 kHz, with a long (0.2–0.5 s) low ringing in the treble; (c) no knock noise
   above ~2–3 kHz. Measures: N6 knock, N4 percussive, E4 per register and velocity.
2. **Per-strike variation** (class V): at equal pitch and velocity the model's notes vary 30–70 % as much as the piano's in spectrum, attack, early decay and onset timing (level: 70–105 %).
3. **Hall T60** set from P4 per year.
4. **Phantoms** from (j−1, j+1) pairs as well, and strong enough up to R6 at loud velocities (N12, 11.3); N9 on
   loud bass notes.
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

## 12. Phase 3 (2026-09-30)

### 12.1 The order

Agreed in discussion, revising 7.5: the machinery that fits single notes comes with the first physics, not after it,
because the current loss cannot see much of that physics (the attack term starts at 400 Hz; below 200 Hz the band
term's window is 341 ms, so a 95–330 Hz thump is outside the one and smeared by the other). Each item is set from a
measure where the measure inverts directly, fitted on isolated notes where it does not, and fitted on music only
where notes cannot show it.
0. Fixes that move the baseline: per-key B; the strike comb's sign (12.2, 12.3).
1. Set from measurement: hall T60 from P4 (10.4 item 3); limit the body filter's Q (11.4).
2. Fit on isolated notes: per-note terms aligned to N0, each with a bias gate; first the velocity map (item 5), then
   the attack in three parts (item 1) with a per-register velocity law fitted after the map, and the phantoms
   (item 4).
3. Per-strike variation (item 2), its spread set from N11 rather than fitted (median-seeking terms would shrink it).
- Alongside 1–3: T1–T3, from partial tracks fitted with phase (two or three close components per partial: the
  double decay and the beating in one fit), pooled across years where 2018 is too thin.
4. One training run on music: per-key unison mistuning with per-key long-window terms (class K), the sympathetic
   bank for R2–R3 (item 6), the whole-excerpt level term (after the attack, which moves 4–8 kHz), texture.
5. All years.
Each step is checked on the bench with re-renders of the round-2 control; the training comes once, after step 3.

### 12.2 Per-key B

The mined B is flat through the wound bass and rises where the plain strings begin: ~5e-5 up to MIDI 42, then
7–8e-5 at 43–45, 1.1e-4 at 48, 1.3e-4 at 50–52 (2018, 232 reliable notes, 1–21 per key). B was frozen in round 2
from the start, so the smoothness regulariser never touched it; `apply_mined_priors` set it from seven per-register
medians interpolated over the keys, which cut the knee: the round-2 models' B is 15–20 % low from MIDI 43 to 63
(mean log ratio −0.158 against the per-key medians, 21 keys). It now takes, per key, the median over the reliable
notes within ±1 key (widened until 5 notes are in; ±1 predicts held-out pieces better than ±2).
- Held out by piece (a third of the pieces, 73 notes): median |log error| 0.104 with register medians, 0.035 per key;
  on MIDI 43–63 the bias goes from +13 % to +0.2 %.
- Independently, on the bench's evaluation notes (partials 10–30 found near the table, 0.13–0.53 s): the recordings'
  partials sat +4.8 to +5.8 cents sharp of the old table at MIDI 43–63 (n 12–30 per range); against the new table
  −0.3 to +0.3 cents, with the IQR down from ~3.7 to ~2.1 cents. Unchanged within 0.5 cents elsewhere.
- On the bench's other measures (`runs/phase3/step0/bench/`, whose partial table now comes from per-key B for every
  source) only per-partial values move: N8 drops of single partials 5–8 by 0.7–1.4 dB in R3–R5 and N2 partial 15 by
  −1.8 dB in R3, in both directions (partly the old model's high partials sitting off the new table's windows); N9
  −0.7…−0.9 dB. Band-level measures move by under 0.6 dB.
The tuning (cents) keeps its register medians: per key, the mined cents scatter 2–4 cents around the model's smooth
curve at some keys (MIDI 52–56 −4.5…−6.0 vs −1.2…−1.5; 71, 73 +3.6 vs +1.6…+2.1), on 3–12 notes each. Not acted on.

### 12.3 The strike comb's sign

The model's bridge force used `sin(n pi x0)`, with x0 from the agraffe: that is the force at the agraffe end. At the
bridge end it is `(-1)^(n+1) sin(n pi x0) = sin(n pi (1 - x0))`: the same level for every partial, but the first
transverse pulse reaches the bridge (1 − x0) T/2 after the strike instead of x0 T/2 (A1: ~9 ms instead of ~2.6 ms on a
synthetic sum with the contact ramp; C3 3.8 vs 1.6; C4 1.8 vs 1.1). Config `bridge_end_comb` (off, so older
checkpoints render as before); the phantoms keep their signs.

Evidence, on the round-2 control with per-key B, with and without the flag:
- **Magnitude measures barely move:** under 1 dB in every register except N4 percussive at 8 kHz in R1–R2 (−3.1 dB,
  against a +14.7 dB gap). As expected: the partial levels are the same.
- **N0 moves in the bass:** +4.15 ms in R1–R2 (41 notes), +0.9 ms in R3, nothing above. The model's bass onset
  was 3.2 ms *earlier* than the recordings' (R3: −0.2 ms); with the flag +0.9 ms. A register-dependent delay between
  the Disklavier's MIDI time and the strike would read the same way, so this alone is not decisive.
- **The string delay** (`scripts/attack_waveforms.py`, `runs/phase3/step0/attack_waveforms/`): from the first
  1.5–8 kHz arrival to where the 150–2000 Hz envelope first reaches half its one-period maximum, as a fraction of the
  period. It does not depend on the MIDI timing. Validation: in R2 the model reads 0.30 [0.23, 0.32] and, with the
  flag, 0.64 [0.48, 0.71] (14–15 notes; 0.34 apart, 0.38 expected); **in R3 both read ~0.6: blind**. The recordings
  in R2 read 0.59 [0.51, 0.70] (17 notes, all nearer the bridge-end value), the same at mf–f (5 notes) and ff (12
  notes), so the key-bottom thump that precedes the strike at ff does not explain it. Paired over 15 R1–R2 notes:
  recording − flag +0.01 of a period [−0.12, +0.23], recording − control +0.22 [−0.12, +0.37]. The waveforms
  (`waves.svg`) show the same: on A0 and C♯2 the recordings' tone starts near (1 − x0) T/2.
- Limits: 112 of 147 recorded notes fail the first-arrival test (they need a 20 dB rise within 20 ms; most bass
  notes in context rise 9–17 dB over what is already ringing), so R1–R2 rest on 15–18 notes; nothing is seen in R3
  and above, where the two readings differ by under 2 ms.
- A phase measure (the envelope projected on the period) was tried first and dropped: the models' body filter puts
  its 150–2000 Hz energy ~30 ms later than its 1.5–8 kHz energy (energy centroids 37–40 vs 6.6 ms), the narrow
  modes of 11.3, which turns that phase by more than a period in the bass.

Two readings with different confounds agree in the bass: the bridge-end sign is what the recordings show there.
Proposed: switch it on for the next training run (step 4), where the latency and the knock timing are fitted with it.

### 12.4 Step 1: the body's Q and the hall, set from measurement

Two changes on the step-0 model (round-2 control, per-key B, bridge-end comb), checked on the bench's evaluation group
(`runs/phase3/step1/`: `bench/` for the three variants, `extra_s0/`, `extra_s1/` for N12). The result is the
checkpoint `runs/phase3/step1/hall/model.pt`, the baseline for step 2.

**The body's Q** (config `body_q_max`, 50: the loss factor 0.02 the body was initialised with). Per octave band the
body FIR is made to decay after its direct arrival at least as fast as a mode of that Q, and each band keeps its
energy (`Room.limit_q`; a test puts an undamped 1158 Hz mode in a body and checks that its ringing at 0.15–0.3 s
falls ≥ 30 dB). On the fitted body: the 1158 Hz peak over its 1/3-octave mean goes from 6.6 to 1.1 dB (612 Hz 4.8 →
2.7, 2740 Hz 4.1 → 1.2); 1/3-octave levels move within ±0.5 dB in most bands (1 kHz −1.3, 4 kHz −1.2, 8 kHz +0.9…+1.6
dB on one channel or the other); the energy centroid of 150–2000 Hz falls from 37–40 to 7 ms. Q = 25 does barely more.
- N12 on pedal-up R3–R7 notes: peaks at a recurring frequency (± 1.5 Hz in ≥ 4 notes of ≥ 3 keys) go from 37 of 103
  (1158 Hz in 27 notes of 15 keys, 3155 Hz, 402 Hz) to 16 of 72 after both changes (775 Hz in 11 notes of 4 keys,
  384 Hz in 5 of 3). The recordings: 0 of 209.

**The hall** (`scripts/set_hall.py`): per octave band, the T60 multiplied by the median ratio recording/model of P4 on
the 150 calibration free decays, and the band level moved by the early-level difference / 0.67, three times.
T60 set (s): 1.78, 1.81, 1.90, 1.93, 1.60, 1.16 at 125 Hz–4 kHz (before 1.66, 1.46, 1.59, 1.53, 1.40, 0.91); 8 kHz
unchanged (no decay stays 10 dB over the floor long enough). Band levels +0.2, +1.7, +3.0, −1.9, −1.3, +3.5 dB.
On the 52 evaluation free decays (model − recording, paired medians):

| band | T60, step 0 | T60, step 1 | early level, step 0 | early level, step 1 | n |
|---|---|---|---|---|---|
| 250 Hz | −20 % | +2 % | −3.7 dB | +0.3 dB | 12 |
| 500 Hz | −15 % | −7 % | −2.5 | −2.0 | 24 |
| 1 kHz | −17 % | −4 % | −3.0 | +0.4 | 22 |
| 2 kHz | −12 % | +3 % | −1.4 | +0.2 | 18 |

(125 Hz and 4 kHz: too few evaluation decays; on the calibration group +7 % and −4 %.) The two groups differ at
500 Hz by ~1.6 dB in the early level before and after, so that band's level is uncertain by about that much.

**Side effects** (evaluation notes, model − recording):
- **Level.** N1 rises by +0.9 dB (R1–R2) to +2.3 dB (R4–R5), about half from each change: the model is now 1.6–1.7 dB
  loud in R3–R5 (step 0: −0.7…0.0). A peaky body gives most partials less than its band average, which the round-2
  fit made up with its gains; a smooth one gives them the average. Left for step 2, whose velocity-map calibration
  sets the level per key and velocity on notes. In music (P3 band levels, 24 excerpts) the gap moves from −0.4…−2.3
  dB to −1.5…+1.5 dB.
- **Rise (N4).** R4 medians: recordings 49 ms (IQR 20–91), step 0 45, with the Q cap 22, with the hall too 82. The
  body's narrow modes were what gave the model its slow build-up; the hall's level, set on the decays, slows the
  onset past the recordings (in a split on R4, the level alone +19 ms, the T60 alone +4 ms, paired medians, ~95
  notes). N4 rise is wide and validated on synthetic notes only; the audit already had it as "revisit after the knock
  change". Left for step 2 (board build-up, D2, with the attack).
- **The attack measures** (N4 percussive, N5, N6 knock) are relative to the note's 100–400 ms energy, which the hall
  raises: they all fall by ~1.3 dB in R1–R5 for that reason, not because the attack changed. The Q cap lowers 4–8 kHz
  by 1–4 dB more (most in R6–R7).
- **Releases (N10).** The slope after the dampers: step 0 −0.3 dB/s from the recordings, now −3.6 (the Q cap −8.5,
  the hall +6.4; 45 note-offs): the body's ringing had held the level up after the dampers.
- Per-partial N2 and N8 values move by 1–5 dB in both directions where partials sat on or off the old body's modes.

### 12.5 Step 2: fits on isolated notes

**The machinery** (`pianonn/notefit.py`, `scripts/fit_notes.py`, `tests/test_notefit.py`). Each bench note is
rendered in its context (1 s warm-up, 12 s lookback) and compared with its recording in windows placed at each side's
own N0: band levels (1/3 octave, 100 Hz–8 kHz, mean power per sample, so a sine reads the same level at any window
length), L1 in dB. Calibration: 819 notes of 65 pieces; evaluation: 383 of 18 (N0 found on the recording).
- **Cells count where the recording's or the model's window stands 6 dB over its own background** (330–30 ms
  before). Selecting on the recording alone biased the paired median towards "the model is too weak" by up to 7 dB
  near the background: where the recording's knock is weak, the model's excess was never counted (the 8 kHz knock
  between the partials, evaluation notes: −6.4 dB selected on the recording, n 38; +0.6 symmetric, n 77; +1.8 over
  every cell, n 632). The first fits used it; they are kept in `runs/phase3/step2/recording_selected/`.
- **A free level per piece**, averaging zero: at equal key and velocity the recordings' level differs between pieces
  by ~2 dB (sd of the per-piece medians, 2018), which no model parameter should absorb. Held out, the term splits into
  a level (median of the piece medians) and a shape (mean |difference − its piece's median|).
- **Per-key tables move by a piecewise-linear correction with knots every 8 keys.** With 1–20 notes per key, free
  tables followed single notes and the evaluation got worse.

**The velocity map** (`physics.cond_vel_map`, hammer speed per condition: 8 segments of 16 MIDI steps, monotone,
anchored at velocity 64 → 2.8 m/s; the contact time and roll-off order follow the speed), fitted with the per-key
level and brightness at mf on the early (30–100 ms) and sustain (100–400 ms) windows; `nomap` fits the same without
the map (evaluation notes, 4557 cells, dB):

| | mean abs | level | shape |
|---|---|---|---|
| step 1 | 3.90 | +1.33 | 3.51 |
| without the map | 3.70 | +0.51 | 3.46 |
| with the map | 3.65 | +0.55 | 3.40 |

The fitted speeds: 0.83, 1.17, 1.81, 2.80, 4.27, 6.34, 9.20 m/s at velocity 20, 32, 48, 64, 80, 96, 112 (the prior
1.02, 1.34, 1.94, 2.80, 4.05, 5.85, 8.45): ~20 % slower at pp, ~9 % faster at ff. The map adds 0.06 dB of shape
over per-key level and brightness alone: a small effect. Most of the gain is the level (step 1 had left the model
1.6–1.7 dB loud in R3–R5, 12.4).

**The attack** (`--fit knock`, from the velocity-map model): the knock noise's per-key spectrum, velocity slope and
decay, the key-bottom thump's gain and decay, the knock impulse's per-key level and velocity slope. Windows: the attack
(−3 to 33 ms, flat-topped as N6) re the early window in every band, below f0 too (the attack's excess over the
tone); the attack between the partials (N6's bins: the knock without the partials' own onset); the early window as a
guard. A band-level term alone let the knock noise stand in for high partials the tone lacks (first run).
- Evaluation: mean abs 3.37 → 3.09 dB, shape 3.31 → 3.02 (6843 cells); calibration 3.89 → 3.25.
- Between the partials below 1 kHz (paired medians, calibration), the model's knock was 6–10 dB too loud in R6 (now
  −0.3…+2.2), 10–14 dB in R7 (now +2.1…+6.6) and up to 3 dB in R4–R5 (now −2.2…−0.8). Left: at 1–2 kHz between the
  partials the model is 1–3.4 dB too weak in R3–R5 (4 kHz: 2.5–3.1 dB in R3–R4); the attack's excess over the tone at
  125 Hz is 1.4–2.9 dB too high in R4–R5 (both groups).
- What moved (every 6th key): the knock impulse down 7–13 dB at MIDI 24–30 and 60–108, up 1–7 dB at 36–48, with a
  steeper velocity slope; the knock noise's velocity slope up by up to 12 dB per unit of velocity (most keys 5–12),
  its decay 1.3–2.3 times longer at MIDI 30–96; the thump +3.2 dB, decaying more slowly.

**On the bench** (evaluation notes and 24 music excerpts, `runs/phase3/step2/bench/`; paired medians, model −
recording; step 1 → velocity map → attack refit):
- Level (N1): R3 +1.7 → +1.2, R4 +1.3 → +0.1, R5 +2.8 → +1.2 dB with the map (R2 +0.5 unchanged); the refit leaves it.
- Brightness (N2 centroid): R2 +209 → +66 cents, R3 +136 → +96 (+75 after the refit), R4 +68 → +50.
- The percussive part below 500 Hz in R6–R7 (N4 percussive 125–500 Hz, 14 notes): +10…+21 dB → −3…+0.2 dB with the
  refit.
- **At 8 kHz the refit went the wrong way on isolated notes:** N6 knock R2 +0.8 → +6.0, R3 +1.2 → +8.0, R4 +3.8 →
  +7.4, R5 +8.0 → +9.2 dB; N4 percussive R2 +4.0 → +12.2, R3 +7.6 → +11.7. Between the partials at 4–8 kHz the
  recordings' attack window stands a median 0.1–0.8 dB over its background (evaluation, R3–R5), so the fit counted
  6–15 notes per register there and moved the per-key 8 kHz knock by +4…+8 dB on them. N6 subtracts the background,
  which magnifies a small excess: +1 dB over the background reads as +6–8 dB.
- In music the attacks' 8 kHz flux contrast (E4) falls with the refit, towards the recordings: R2 5.1 → 4.4, R4 6.2
  → 4.1, R5 9.0 → 7.8, R7 18.5 → 11.8 dB (recordings 0.3, 0.5, 2.4, 3.6). E4 reads how abrupt the onset is, N6 the
  energy between the partials: the model's high-frequency attack is too abrupt, and the refit traded abruptness for
  energy. A lead, not checked: the knock's envelope starts with a step (`NoiseBank._env`), applied after the band
  split, which spreads every band's onset over all frequencies.
- Music band levels (P3): with the map 500 Hz +1.45 → +0.54, 1 kHz −0.23 → −1.16, 2 kHz −1.33 → −1.82 dB; the refit
  moves them by under 0.4 dB.
- **Refit with the knock's spectrum above 2.5 kHz kept** (`--knock-max-hz 2500`, `runs/phase3/step2/knock_2k5/`,
  bench in `bench_knock/`): the term held out is the same (evaluation 3.37 → 3.11 dB, shape 3.31 → 3.04), and the 8
  kHz regression halves but stays: N6 knock R2 +0.8 (map) → +6.0 (refit) → +4.5, R3 +1.2 → +8.0 → +5.5, R4 +3.8 →
  +7.3 → +4.2, R5 +8.0 → +9.2 → +8.0. The rest comes from the knock's per-key decay (1.3–2.3 times longer) and
  velocity slope, which act on every band: one decay per key cannot lengthen the low knock without lengthening the high
  one. The refits also leave the percussive part at 125–500 Hz in R4–R5 2–4 dB weak on N4 (the map: −1…+1.5), while
  the note term's attack re early reads +1.4…+2.9 dB too strong at 125 Hz there: the two measures disagree.
  `knock_2k5/model.pt` is the step-2 checkpoint.

**Bias gates fail in every fit, before fitting too.** Within register × velocity strata the term's optimum sits
+0.5–1 dB over the energy match in most bands, +1.8–3.2 dB at 4 kHz in the early window: these fits leave the model
slightly loud against an energy match. Not corrected.

### 12.6 An onset-aligned attack term for training (checked, not built)

The training loss sees the attack through 10.7 ms windows with no band below ~400 Hz, and the bass through 341 ms
windows (band term) or, bin by bin and not tied to onsets, 43–171 ms (fine term): a design choice, not a limit of
the data. Proposed: the note fit's attack window moved to music (`scripts/onset_term_check.py`): band levels (1/3
octave, 100 Hz–8 kHz) from 3 ms before to 33 ms after each note's expected sound onset, cells where the recording or
the model stands 6 dB over its background, L1 in dB; notes whose expected onsets lie within 10 ms share a window. The
**pooled** variant sums the counted windows' energy per segment and band before the log.

**Timing.** The recordings' N0 lags the MIDI onset by 7.2 − 0.27 (pitch − 60) − 0.07 (velocity − 64) ms (382
evaluation notes; R2 +14 ms, R5 +5, R6 +1); the model follows the same law (7.15, −0.24, −0.05). Per note the
residual IQR is −3.8…+4.2 ms, and 46 % of notes differ between model and recording by more than 5 ms. Windows sit at
MIDI + this delay. On the model against itself (64 validation segments, 884 onset groups), a 5 ms shift costs the
term 0.85 dB (pooled 0.50), about what another noise draw costs (0.54, pooled 0.28); the current attack term reads
2.1 dB for the same shift.

**Scans** (`runs/phase3/onset_term/check_k25/`, the step-2 checkpoint; each term against the recordings as one
parameter moves, the noise draw fixed; vertex of a parabola through five points, 10–90 % over 200 bootstrap draws;
curvature: the rise one step (3 dB or 10 %) from the vertex):

| parameter | band | fine | attack | onset | pooled | current loss |
|---|---|---|---|---|---|---|
| knock level, vertex (dB) | −3.7 | −2.2 | −5.8 | −4.5 | −7.0 | −5.1 |
| knock level, curvature | 0.003 | 0.003 | 0.021 | 0.014 | 0.013 | 0.015 |
| knock below 400 Hz, vertex | flat | −2.9 | −5.5 | −4.9 | −5.9 | −4.2 |
| knock below 400 Hz, curvature | 0.001 | 0.000 | 0.005 | 0.006 | 0.005 | 0.003 |
| knock impulse, vertex (dB) | +5.0 | −3.2 | −4.0 | −4.1 | −5.4 | +2.0 |
| knock impulse, curvature | 0.002 | 0.004 | 0.001 | 0.007 | 0.005 | 0.004 |
| contact time, vertex (%) | +8.0 | +5.5 | +6.2 | +2.8 | +2.8 | +6.9 |

- **The new term sees the knock impulse where the current loss reads almost nothing**: its change comes from below
  400 Hz (+0.18 dB at +6 dB), where the attack term has no band; the current loss's minimum for the impulse (+2 dB) is
  set by the band term (+5), against every other view (−3.2…−5.4).
- **The current attack term also moves with the knock below 400 Hz**, through its bands above 1.6 kHz. Probably the
  knock's step onset (12.5): a louder low band adds a broadband click. So the current loss sees the low knock only
  through a side effect of the model.
- **Music and isolated notes disagree on the knock's level**: every attack-aware term puts it 4.5–7 dB below where
  the note fit left it (on the unfrozen refit too: −4.4…−5.8). Not explained: the note fit's positive bias (12.5), the
  bench's soft skew (ff 67 of 1003 calibration notes) and masking in music are candidates.
- The contact time: the onset terms put it +2.8 % from the note fit's value, the current loss +6.9 %.

**Bias gate: fails.** Per onset (velocity strata of the group's loudest note), the L1 optimum sits off
the energy match below 1 kHz (−0.2…−1.7 dB), +0.8…+3.3 at 1.6–4 kHz, −1…−3.7 at 5–8 kHz. Pooled per segment: −1.2…+1.4 dB from 100
Hz to 4 kHz (median −0.46), but −2.5…−2.9 at 6.4–8 kHz. A likely part of it (not checked): the model's attacks vary
less than the piano's (N11), and a median and an energy mean part ways when two spreads differ. Pooling over a whole
batch, or pairing the term with the whole-excerpt level term of step 4, would shrink it; untested.

Counts: a band counts in 16–48 % of onset groups (recording alone 13–34 %), most below 200 Hz and at 4–6 kHz.

**Built, pooled over the batch** (`pianonn.losses.OnsetLoss`; `train.py --onset-weight`, default 0; four tests in
`tests/test_losses.py`: a 6 dB change reads 0.602 exactly; only the expected onset's window counts; a 150 Hz burst at
the onset reads 4.8 dB where the attack term reads 0.3; with the target's attacks spread by 8 dB and the prediction's
not at all, the optimum sits at the energy match, +5.40 dB against +5.34, where a per-window median would give
+1.78). The check now drives this class and stores every window's powers (`scripts/onset_term_check.py`,
`runs/phase3/onset_term/pooled/`: 128 validation segments, 1780 onset groups, scans on the first 64):
- **Bias gate: pooling helps, but does not pass at the step-4 batch.** Largest band bias for batches of k segments:
  2.09 dB (k = 1), 1.49 (2), 1.03 (4), 1.09 (8); median over bands −0.44, −0.11, −0.04, −0.20. At k = 2: 635 Hz
  −1.5, 1–1.3 kHz +1.0…+1.4, 2–3.2 kHz +1.0…+1.3, 8 kHz −1.2 (bootstrap 10–90 % ranges ~±0.7 dB).
- **The spread explains much of it.** Within velocity strata the recordings' attack levels spread more than the
  model's at 4–8 kHz (sd 10.5–10.9 against 7.2–9.1 dB at 4–6.4 kHz) and below 800 Hz, less at 1–2 kHz; the bias per
  window has the sign 0.115 (sd_rec² − sd_mod²) predicts in 16 of 20 bands (smaller in size). Step 3 (per-strike
  variation) is the fix at the source.
- **Pooling within velocity strata makes it worse** (2.7 dB at k = 2): each pool holds fewer attacks.
- **The knock's level is not identifiable by any band-level term at onsets.** Its optimum moves with the pooling:
  per window −4.5 dB (the typical attack's high bands too loud), batches of 2 flat, batches of 4 and one global pool
  (the energy match) louder (the loud attacks' 6–8 kHz too weak). At 4–8 kHz the attack window holds the partials'
  onset as well as the knock, and a knock level fills in for high partials the loud notes lack: the confound the note
  fit met (12.5), solved there by the attack re the early window and by the energy between the partials. Likely,
  not checked in music.
- What holds under every pooling: the contact time's minimum at +1.3…+3.1 % (curvature 0.39–0.63 dB per 10 %, the
  current loss 0.38 with its minimum at +6.9 %), and the knock impulse's at −2.5…−4.5 dB where it has one (the current
  loss +2.0, set by the band term).

Before the term goes into a training run: the attack relative to the same onset's early window (30–100 ms), which
cancels the tone's own level errors, and a running pool across steps (per band, decaying sums of both sides' counted
power; the sign from the running totals, the step from the batch), whose optimum is the energy match whatever the
batch size. Built and checked in 12.8.

### 12.7 Step 3: per-strike variation

**How much the piano varies** (`scripts/strike_spread.py`, `runs/phase3/step3/`). On the bench's notes, per register
and measure, each source's values become residuals: a linear fit on pitch and velocity, then each piece's median
removed (the pieces differ by ~2 dB in level at equal key and velocity, 12.5: a session's offset, not a strike's). The
residuals split into a per-key part (voicing, the same at every strike of the key) and a per-strike part, from pairs of
notes of the same key (robust sds; 10–90 % ranges over 200 bootstrap draws of pieces). Calibration group, 1003 notes,
the step-2 model; per-strike sd, recording / model, and the excess sqrt(rec² − model²):

| measure | R2 | R3 | R4 | R5 | R6 |
|---|---|---|---|---|---|
| N1 level (dB) | 1.56 / 1.02, +1.2 | 1.70 / 1.29, +1.1 | 1.51 / 1.45, +0.4 | 2.00 / 2.17, −0.9 | 2.33 / 2.59, −1.1 |
| N2 slope (dB/oct) | 1.27 / 1.71, −1.1 | 1.80 / 1.58, +0.9 | 2.01 / 1.27, +1.6 | 1.93 / 1.13, +1.6 | 1.66 / 1.08, +1.3 |
| N5 attack 2 kHz (dB) | 3.07 / 1.63, +2.6 | 3.47 / 1.82, +3.0 | 3.14 / 1.65, +2.7 | 3.53 / 1.71, +3.1 | 2.97 / 2.09, +2.1 |
| N8 drop 1–4 (dB) | 2.18 / 1.48, +1.6 | 1.49 / 1.25, +0.8 | 2.12 / 1.10, +1.8 | 2.72 / 1.26, +2.4 | 2.89 / 1.94, +2.1 |
| onset (ms) | 5.65 / 2.49, +5.1 | 6.32 / 5.24, +3.5 | 5.28 / 3.31, +4.1 | 8.15 / 2.13, +7.9 | 4.59 / 0.93, +4.5 |

(n 51–186 per cell; the attack at 1 and 4 kHz reads like 2 kHz, +1.9…+3.1 dB.)
- **The excess is per strike, not per key.** The per-key part is ~0 in the recordings for every measure except the
  early decay in R2 (2.1 dB): the piano's notes differ from strike to strike, not by key.
- **The dimensions are nearly independent in the piano.** Rank correlations of the residuals (evaluation group, R3–R5,
  242 notes): level–slope −0.16, level–drop +0.24, slope–attack +0.17, every other |ρ| ≤ 0.12. So a jitter of the
  velocity (which moves level, brightness and knock together) is not the piano's variation; each dimension is drawn
  on its own.
- **Onset timing:** 4.5–5.1 ms (sd) with the pedal up, half or down, smaller at *ff* (3.5 ms) than at *p* (5.7), as the
  model's own detection noise is (0.6 vs 2.8 ms); neighbouring notes of a piece do not share it (ρ −0.08…+0.08 at any
  distance), so it is not audio/MIDI drift. Whether it is the strike's timing or N0 reading real attacks less
  precisely than the model's, this measure cannot tell.

**The model's dimensions** (config `strike_*`; `NeuralPhysicalPiano.strike_offsets`): per note a zero-median normal
offset (clipped at 2.5 sd), drawn afresh at every render, of the level (strings, knock impulse and noise together),
the brightness (log contact time), the attack's noises (knock noise, thump, impulse), the decay rates (log), a decay
tilt (log rate per octave re 1 kHz: the high partials' early decay) and the sound's onset; one sd for every key or one
per register. **Brightness and decay keep the note's energy over 0–0.3 s**: probes showed the contact time alone moves
the level by 10.8 dB per unit of log (in the piano level and brightness barely correlate), so the physics rescales each
note's partials to keep their early energy, and the level is a dimension of its own (tests: level within 0.3 dB while
the centroid moves 50+ cents).

**Setting the spreads.** Probes on the calibration notes, one dimension at a time (level 2 dB, contact time 0.15,
knock 6 dB, decay 0.2, tilt 0.2 per octave, onset 4 ms; `runs/phase3/step3/probes*`), give each measure's added
per-strike variance per unit of each dimension's (pooled over registers):
- **The knock barely reaches the attack at 1–4 kHz:** N5 attack 0.03–0.08 dB per dB of knock at 1–2 kHz, 0.17 at 4 kHz
  (N6 between the partials: 1.0). There the attack window is the partials' own onset.
- Brightness: N2 slope 5.5 dB/oct per unit, N5 attack 7–20 dB (the sustain window holds other notes and the room);
  decay: N8 4.6 dB, N5 attack at 0.5–1 kHz 3.8; tilt: N8 4.7, N5 attack at 2–4 kHz 3.4–3.9; onset: 1.2 ms per ms.

The spreads that match the recordings' per-strike excess variance (non-negative least squares over N1, N2 slope, N5
attack 1/2/4 kHz, N6 knock 2 kHz, N8 and onset, weighted by their bootstrap sds, pooled over R2–R6; the level per
register, whose excess is confined to R2–R3): **level 1.12 dB in R2–R3 (0 from R4), brightness 0.159 (±17 % contact
time), knock 0, decay 0.248 (±28 %), tilt 0.258 per octave, onset 3.29 ms.** The knock gets nothing: no target needs
it once brightness and decay vary. The fit falls short on the attack at 1–2 kHz (excess variance 2.5–10.9 dB²,
fitted 2.4–4.1) and overshoots it at 4 kHz (6.4–8.9, fitted 10.6): no dimension moves the 1–2 kHz attack alone.
Checkpoint `runs/phase3/step3/spread/model.pt` (the step-2 weights with these spreads).

**Check on the evaluation notes** (459, held out; `runs/phase3/step3/bench/`, one draw per note). The bench's spread
ratio (model / recording IQR of residuals after a fit on pitch and velocity), R2 / R3 / R4 / R5, step 2 → step 3:

| measure | step 2 | step 3 |
|---|---|---|
| N1 level | 0.99 / 0.76 / 0.58 / 1.62 | 0.81 / 0.80 / 0.76 / 1.69 |
| N2 slope | 0.96 / 0.66 / 0.69 / 0.95 | 1.56 / 0.97 / 0.79 / 0.99 |
| N5 attack 2 kHz | 0.57 / 0.70 / 0.90 / 0.69 | 0.73 / 0.73 / 0.85 / 0.69 |
| N8 drop 1–4 | 0.49 / 0.72 / 0.43 / 0.56 | 0.60 / 0.74 / 0.64 / 0.69 |
| onset | 0.37 / 0.65 / 0.33 / 0.21 | 1.01 / 0.94 / 1.04 / 0.74 |

Per strike (piece medians removed): the attack at 2 kHz still varies less than the piano's (model 2.2–2.8 dB, recordings
3.0–4.3; excess +2.7…+3.9 → +1.7…+3.6); the level excess in R2 goes from +1.0 dB to −0.3; the slope's in R3–R4 from
+1.5 to +1.2 dB/oct, while R2 now varies more than the piano (−1.6); N8's excess in R3–R5 from +1.1…+2.2 to
+0.8…+1.9 dB.
- **The medians stay:** per note, step 3 − step 2 has medians within ±0.5 dB for level, attack and early decay and
  within 30 cents for the centroid; the 8 kHz percussive part rises by +1.7 dB (R2) and +1.1 (R3). The model −
  recording paired medians move more where their differences are skewed (N2 centroid in R2 +69 → +204 cents, from a
  mean well above the median): added spread moves a skewed difference's median towards its mean.
- **Music** (24 excerpts): E4 and P3 move within their noise.
- **In music the attack levels' spread barely moves** (12.8): at 4–6.4 kHz the model's sd goes from 6.7–8.6 to
  7.2–9.0 dB against the recordings' 10.0–10.7, and most of that gap is the tone's (relative to the early window the
  spreads are 1.9–2.2 against 2.6–3.1). A steeper brightness-vs-velocity law in the piano than in the model would
  widen the spread within velocity strata the same way; not checked.
- **The training loss** against the recordings rises with the variation (512 validation segments): band 4.32 → 4.54,
  fine 6.58 → 6.69, attack 3.76 → 4.00 dB. A random model loses to its median under L1 terms (the texture penalty,
  4.1). Between two draws of the model the terms read band 3.0, attack 2.8 dB (1.1, 0.5 without).

**For step 4 (proposed):** train with the variation off and render with it on. The spreads are set, not fitted, and
the offsets have zero median, so the fitted parameters are the medians either way; with the draws on during training,
the frame-aligned terms would see the timing jitter and could pull the attack's shape to blur it (likely, not
checked). A sign of it: with the draws on, the current loss's minimum for the contact time moves from +6.9 % to
+9.0 % (12.8's scans on the step-3 model).

### 12.8 The onset term: relative form and running pool

Built (`OnsetLoss(relative=True, pool_decay=...)`, `train.py --onset-relative --onset-pool`, two tests: a level
change of the whole note reads nothing in the relative form where the absolute reads 6 dB; trained on a stream of
batches whose target attacks vary by 8 dB, a level offset settles at +5.8 dB with the running pool against an energy
match of +6.2, and at +0.8 with the batch's own pool). The check (`scripts/onset_term_check.py`, now storing the early
window's powers too) ran on 512 validation segments (6725 onset groups) with the step-2 model
(`runs/phase3/onset_term/rel_k25/`) and the step-3 model (`rel_s3/`).

**Bias gate** (largest band bias, dB; the running pool simulated on random orders of the batches of 2, decay per
step; about 8, 16 and 32 batches):

| model, form | k = 2 | k = 8 | k = 16 | running 0.875 | running 0.9375 | running 0.969 |
|---|---|---|---|---|---|---|
| step 2, abs | 3.16 (fail) | 1.25 | 0.83 | 0.96 | 0.87 | 0.69 (pass) |
| step 2, rel | 1.27 (fail) | 1.06 | 0.81 | 0.67 | 0.48 | 0.28 (pass) |
| step 3, abs | 2.82 (fail) | 0.94 | 0.76 | 0.34 | 0.29 | 0.11 (pass) |
| step 3, rel | 1.21 (fail) | 1.00 | 0.59 | 0.44 | 0.39 | 0.28 (pass) |

The running pool passes for both forms and both models, the medians over bands within ±0.12 dB. On these 512 segments
the batch of 2 fails by more than on 128 (3.16 against 1.49, at 8 kHz): the variation of step 3 does not fix it; the
pool does.

**What each form sees** (scans on 64 segments, the step-2 model; curvature one step from the vertex):
- **The relative form is flat on everything scanned:** +6 dB of knock moves it by 0.014 dB (k = 2), the contact time
  (as intended: a common brightness cancels) and the impulse by under 0.03. In music the knock is a small share of
  the attack window's energy in most bands.
- **The absolute form, pooled over all scanned segments**, keeps the contact time's minimum at +2.1 % (curvature 0.63;
  batches of 2 +1.6 %; the current loss +6.9 %) and pushes the knock louder (+6 dB: −0.08, no vertex in range).
- Energy-weighted over all onsets, model − recording (step 2): the attack re the early window is −0.7…−1.8 dB from
  635 Hz to 6.4 kHz and −2.9 dB at 8 kHz, within ±0.3 dB below 500 Hz; the absolute attack's errors at 1–1.6 kHz
  (+1.5…+3.5 dB) and 6–8 kHz (−2.3…−5.4) are in the early window as well (+2.3…+4.4, −2.2…−2.8): the tone's, not the
  attack's. The model's attack stands about 1 dB too little over its tone above 600 Hz; the knock barely moves that.

**For step 4 (proposed):** the absolute form with the running pool (`--onset-pool 0.97`, about 33 batches of 2): it
passes the gate and sets the contact time and the knock impulse where the current loss does not. The relative form
passes too but moves nothing the run would fit; it stays a measure. The knock's level in music remains unset by
either (12.6).

### 12.9 Step 4: one training run on music

**What was built for it** (`pianonn/train.py`): `--init-from` starts from a checkpoint's weights and model config
(fresh optimiser, step 0, no data initialisation: the recording chain is already set); `--strike-train` (off by
default) draws the per-strike variation in training, otherwise it is off in training and validation and on in the audio
dumps, and the checkpoints keep the spreads in their config for rendering; `--lr-warmup` (each parameter's rate rises
from 0 over that many steps from when it first trains: the start, or stage 2 for the tables and the residual);
`--lr-decay-at` / `--lr-final` (a half cosine from that point to the end). `scripts/compare_runs.py` now takes config
options after the mode, as the other scripts do.

**Why the warm-up and the lower rate** (`runs/phase3/step4/lr_checks/`, `notes.md`). From the step-3 model, at the
round-2 rate (1e-3), the physics' validation loss rose from 0.898 to 0.91–0.95 within 20–100 steps (16 validation
segments); at 3e-4 without decay it rose too, to 0.937 at step 180. Put back one parameter at a time, the rise is in
the level parameters (condition contact time, mic gain, condition gain, velocity curve, per-key gain: +0.003…+0.008
each), which learn at 20 times the base rate in dB and follow each batch's pieces (at equal key and velocity the pieces
differ by ~2 dB, 12.5). With the rate decayed at the end, a 150-step run beat the start on both splits (64 segments
each, physics only: validation 0.898 → 0.884 and 0.888 → 0.871 on two sets, training 0.860 → 0.832). So the run
warms up over 200 steps, uses 5e-4 and decays over its second half.

**The run** (`runs/phase3/step4/train/`, `chain.sh`; launched 2026-09-30 21:50): 2018, from
`runs/phase3/step3/spread/model.pt` (per-key B, bridge-end comb, body Q ≤ 50, hall from free decays, velocity map,
knock refit, the per-strike spreads), 450 min, batch 8, lr 5e-4 with the step-2 per-unit scales, warm-up 200 steps;
stage 2 (partial gains, colouration, the residual; physics rate ×0.3) and the cosine decay to ×0.05 both from half-way;
frozen: B, tuning, and the hall's T60 and levels (`room.raw_log_t60`, `band_log_gain`, `log_gain`); the onset term,
absolute, weight 0.5, running pool 0.97 (12.8); the per-strike variation off in training. Validation on 64 fixed
segments every 250 steps. `post.sh` then renders the listening set and runs the comparisons below.

Not in this run, of the items 12.1 lists for step 4: per-key unison mistuning with long-window terms, the sympathetic
bank (small batches only), a whole-excerpt level term, and texture (the GAN). Each needs a term or a check first.

**The run as it went** (2026-09-30 21:50 to 2026-10-01 05:21): 17,840 steps in 450 min, stage 2 from step 9,498. At
step ~5,290 one batch ran out of memory under the 0.6 cap; the process printed its traceback and hung instead of
exiting, so the restart loop never fired. It was resumed from step 5,250 with the cap at 0.8 and a watchdog in
`chain.sh` (no log output for 10 min: kill and resume); about 5 min of wall clock were lost, none of the budget.
Validation (64 segments, `val_curve.svg`): the start 0.898; in stage 1, at the constant rate, it wandered between 0.80
and 0.90 from check to check while its lows fell (0.838 by step 500, 0.812 by 4,000, 0.801 at 5,750); once the decay
began, 0.797–0.808. The end: physics 0.802, with the residual 0.808. `best.pt` is step 11,500 (0.792 with the
residual), picked out of 72 noisy checks; the evaluation uses `last.pt`, with `best.pt` alongside in the distances.

**Distances on 96 test excerpts** (`compare.md`; 2 noise seeds, per-strike variation off; the round-2 control is the
base of the paired differences, mean ± 2 se over excerpts, both seeds agreeing in sign in every cell):

| model | total | band | fine | attack | MR-STFT | log-mel (dB) |
|---|---|---|---|---|---|---|
| round-2 control (physics) | 0.719 | 0.377 | 0.619 | 0.375 | 1.105 | 3.35 |
| step 3, the start (physics) | 0.773 | 0.415 | 0.657 | 0.387 | 1.197 | 3.71 |
| **step 4, physics** | **0.691** | 0.367 | 0.597 | 0.350 | 1.062 | **3.19** |
| step 4, with the residual | 0.686 | 0.364 | 0.595 | 0.347 | 1.066 | 3.17 |
| step 4 `best.pt`, physics | 0.693 | 0.368 | 0.600 | 0.351 | 1.063 | 3.21 |

Step 4 against the round-2 control: total −0.028 ± 0.010, band −0.010 ± 0.005, fine −0.022 ± 0.008, attack
−0.025 ± 0.008, log-mel −0.16 ± 0.06 dB. Steps 1–3 had cost 0.053 on music (they were set on decays and isolated
notes); the run recovers that and goes 0.028 past round 2. The round-2 residual branch read 0.702 / 3.27 dB on the same
excerpts (`docs/round2_results.md`); the residual adds 0.005 here, and at the end of the run it reads slightly worse
than the physics alone on validation: it adds little.

**The note bench** (459 evaluation notes, held out; 24 music excerpts; the per-strike variation on;
`runs/phase3/step4/bench/`), step 3 → step 4, model − recording paired medians:
- **Brightness** (N2 centroid): R2 +204 → +7 cents, R3 +160 → −72, R4 +41 → −5; the slope within ±1 dB/oct in
  R2–R5.
- **Level** (N1): R2–R5 within −1.0…+0.3 dB (step 3 +0.1…+1.6); by velocity −0.1, −0.1, 0.0 at p, mp–mf, mf–f
  (step 3 +0.4, +0.4, +1.7). R6 −1.8 (11 notes).
- **The attack:** the 8 kHz percussive excess (N4) +8.8…+14.3 → +5.1…+9.5 dB in R2–R6; between the partials at 8 kHz
  (N6) R3–R4 +4.3/+4.9 → 0.0/+0.1, R5 +10.4 → +4.4; the 4 kHz percussive part at p +11.9 → +2.1 and at mp–mf +5.5 →
  +2.5; the 250 Hz thump in R3–R5 −4.4/−2.7/−4.3 → −2.5/−0.2/−0.5 (R2 −4.0 → −3.4); N5 attack re sustain at 0.5–8
  kHz within −1.0…+2.8 dB in R2–R6 (step 3 −3.1…+2.5); N6 knock at 1 kHz by velocity −1.3/−4.0/−5.7 → −1.1/+0.2/−2.2.
- **Phantoms** (N9): R4 −1.5 → +0.4, R5 −3.6 → −1.0, R3 −1.8 unchanged, R2 −4.7 → −6.4. The run raised the phantom
  gain table by +6.5, +10.9, +6.4 dB in R4–R6 and barely below (`moved.py`).
- **Early decay** (N8 drop 1–4) within −0.4…+1.4 dB in R2–R6 (step 3 −0.1…−3.1).
- **Rise** (N4): R3 +39 → +12 ms, but R4 +9 → −17 and R5 +3.5 → −16: the onset now builds faster than the recordings'
  in R4–R5 (N4 rise is wide and validated on synthetic notes only, 12.4).
- **Spread** (variation on): the onset's spread ratio fell from 0.94–1.04 to 0.64–0.74 in R2–R5 (not looked into: a
  sharper model attack would make its own onset detection tighter); N5 attack at 2 kHz R3–R5 0.73–0.85 → 0.81–0.93.
- R6 (11 notes) got worse in the low bands of the attack (N4 percussive 250 Hz +5.8 → +14.2, N6 knock 125/250 Hz
  −1.0/+0.4 → +6.2/+10.2 dB; 500 Hz +10.3 → +2.1).

**In music** (24 test excerpts):
- E4: the 8 kHz flux contrast falls towards the recordings' everywhere (R2 4.1 → 1.2, recordings 0.3; R3 3.4 → 1.0,
  0.5; R4 5.3 → 2.0, 0.5; R5 7.9 → 5.9, 2.4): the high attack is less abrupt, still more than the piano's. At 250 Hz
  the model's contrast is lower than the recordings' (R2 1.7 vs 4.1, R5 1.6 vs 4.9; step 3 3.1, 2.2).
- **Level.** In energy the level is right: on the 96 test excerpts (`train/eval/report.md`) the broadband median is
  −0.1 dB (IQR −1.4…+1.4), every octave band from 63 Hz to 8 kHz within ±0.7 dB, soft / middle / loud excerpts +0.3 /
  −0.9 / −0.4 dB. P3's mean of the compressed envelope (amplitude^0.3 of 5 ms power, which weights the quiet moments)
  reads the model below the recordings by 0.8, 1.3, 2.0, 1.0, 0.5 dB at 250 Hz–4 kHz (125 Hz +0.1, 8 kHz −0.1; step 3
  −0.9…+1.2): the model's quiet stretches (tails, between notes) are quieter than the piano's, its loud ones not. Not
  looked into further. The 8 listening excerpts (RMS, variation on) come out 1.2–2.8 dB under the recordings at mean
  velocity 32–58 and 0.4–1.2 dB over at 64–83. The other texture statistics (P3 CV, modulation, correlation) stay
  within their earlier differences.

**`scripts/evaluate.py`** (`train/eval/report.md`) loads the checkpoint as it renders, with the per-strike
variation on: physics 0.723, with the residual 0.721, log-mel 3.40 dB (the round-2 control, which has no variation,
read 0.719 there). The same model with the variation off reads 0.691 (above): the variation costs 0.032, the 2–6 %
expected of a random model under median-seeking terms (12.7). Its other tables: the loss terms' level bias passes
(total −0.44 dB, band −0.45 with 4 kHz −1.1, fails by 0.1 dB); the fitted floor and hum stay within ~2 dB of the test
pieces' silence; the residual's per-note outputs are small (knock −0.10…−0.12 of its ±1 bound, everything else ≤ 0.06).

**What the run moved** (`runs/phase3/step4/moved.py`, 2018's rows): the phantom gains (above); the knock impulse −1…−6
dB in R2–R3 and R5–R7, +3.5 in R1 and R4; the knock noise −0.1…−0.6 dB with a steeper velocity slope; the level
redistributed with velocity (the velocity curve +4 dB at the softest segment, −2 at the loudest, the microphone gain
−1.9 / −2.4 dB); the damping tilt and the b3 loss term lower in R1–R5. The hall, B and the tuning were frozen.

**Listening** (`runs/phase3/step4/listen/`, `listen_long/`, `listen_45s/`, each with an `index.html` from
`scripts/listen_page.py`): 8 × 12 s test excerpts soft to loud and the demo score, 2 × 20 s, 3 × 45 s; the recording,
the round-2 control, the step-3 model and step 4, rendered as the model renders (the per-strike variation on). Not yet
listened to by the owner.

## 13. The residual (2026-10-01)

The context net (`ContextNet`, `pianonn/synth.py`) adds 0.005 to the step-4 model on test (12.9). How it is built
(read in the code): a one-layer GRU (128) over the piano roll (onset velocities, key-down curves, three pedals) with 12 s
of history, so it sees the MIDI before the window; it does not see what the physics renders (no partial levels, decay
state or spectrum, and a key embedding of its own rather than the physics' per-key values). Its per-note outputs are
read once, at the note's onset frame, and stay fixed for the note's life; its only time-varying outputs are 16 band
gains (±6 dB) on the whole dry mix and a noise path. Every output is a bounded knob the physics already has (gain,
contact time, decay, tilt, six log-f bumps, knock, a slow attack noise). In step 4 it trained only in stage 2, which
began together with the cosine decay (~8,300 steps at a falling rate), under the residual budget, alongside per-key
tables that take whatever is the same at every strike of a key.

### 13.1 The ceiling of its output language

`scripts/residual_ceiling.py` (`runs/residual/ceiling/`): the step-4 model frozen, the per-strike variation off, its
context net replaced by free outputs per test excerpt in the net's own language and bounds, fitted by Adam on the
training loss (150 steps, a fresh noise draw per step, no budget) and scored on render-noise seeds the fit never saw.
24 test excerpts (the first 24 of `compare.md`'s), mean ± 2 se over excerpts, variant − physics:

| variant | total | band | fine | attack | log-mel (dB) |
|---|---|---|---|---|---|
| physics (absolute) | 0.7015 | 0.3663 | 0.5978 | 0.3716 | 3.25 |
| the trained net | −0.006 ± 0.006 | −0.003 ± 0.003 | −0.003 ± 0.004 | −0.006 ± 0.004 | −0.04 ± 0.04 |
| free per-note outputs | −0.136 ± 0.019 | −0.076 ± 0.011 | −0.053 ± 0.007 | −0.095 ± 0.018 | −0.66 ± 0.10 |
| free per-frame outputs | −0.219 ± 0.015 | −0.128 ± 0.011 | −0.070 ± 0.009 | −0.146 ± 0.020 | −0.83 ± 0.07 |
| both | −0.268 ± 0.020 | −0.155 ± 0.015 | −0.106 ± 0.011 | −0.173 ± 0.023 | −1.05 ± 0.09 |

(The residual's paths on at zero output: +0.001 ± 0.002. The fits were still falling slowly at 150 steps, so these
ceilings are, if anything, low.) For scale: step 4 gained 0.028 over the round-2 control.

- **The output language is not what holds the residual back.** Per-note corrections alone, fixed over each note's life,
  could take off ~20 times what the net does. The per-frame variant is a generous bound: free band gains every 5 ms can
  trace an excerpt's envelope in a way no predictor from MIDI could.
- **The gain is per note, not a shared bias.** Of each per-note output's variance, 8–23 % is one value per excerpt
  (`outputs.md`; the knock 23 %). Every note given the medians of the fitted outputs scores +0.009 ± 0.014 against the
  physics (`constant.md`): no gain. (The medians per output are not the best joint constant; a fitted constant was not
  tried.)
- **Simple context features barely rank the corrections** (595 notes struck in the loss windows, Spearman): |ρ| ≤ 0.14
  against velocity, pitch, keys down, the gap to the previous onset and to the same key's previous onset, except the
  contact time against the same key's gap (−0.22: re-struck keys want to be brighter) and the size (not the sign) of
  the spectral bumps against pitch and velocity (−0.63, −0.38). The fitted per-note level spreads by 3.1 dB (sd); the
  piano's own strike-to-strike level spread on isolated notes is 1.5–2.3 dB (12.7). So part of the ceiling is likely
  strike-to-strike variation no context can predict; how much, this check cannot say.
- The decay and tilt corrections sit at their bounds for 35 % and 42 % of notes (per-note variant).

What is open is how much of the ceiling a predictor from the context can reach. **This ceiling answers a different
question** (review 5, 4.2): the free outputs were fitted to each excerpt's realised recording, so they include the
strike-to-strike variation no context predicts. The fitted per-note level spreads 3.1 dB (sd) where the piano's own
spread at equal key and velocity is 1.5–2.3 dB, so a quarter to a half of that output's variance is unpredictable by
construction, and the features' |ρ| ≤ 0.14 says the predictable part is small. A reachable ceiling needs a fit that
cannot see the realisation (free outputs fitted on training pieces, predicted from context on held-out ones); not
done. The owner chose to go straight to (2) below rather than retrain the current net: a predictor that sees the
physics' own render and each note's state, with per-note outputs that vary over time (13.2).

### 13.2 A residual that sees the physics (first run)

Built (`pianonn/residual.py`, config `residual_kind=aware`; tests in `tests/test_model.py`). Per control frame (20 ms) over
the rendered window, for every note that sounds there:
- **What it sees:** the note's expected energy in 8 octave groups of partials (62.5 Hz–8 kHz), computed from the physics'
  modal parameters without the residual (amplitudes, decay rates, the damper integral, re-strikes; no oscillators
  rendered, no gradient), the same summed over all notes, the difference of the two, the note's age, whether its key is
  held, its damper, the pedals, its key and velocity, the excerpt's level, and a GRU's summary of the MIDI history
  (12 s).
- **How:** two layers of attention across the sounding notes at each frame, each followed by a GRU over the note's own
  frames (causal).
- **What it changes:** per note, a gain curve for each octave group of partials (±12 dB), applied inside the oscillator
  bank (`osc_bank` with group weights, its analytic backward extended and gradchecked): the note's spectral shape over
  time. Also the onset-time outputs of `ContextNet.NOTE`, read at the note's first frame, and the per-frame mix outputs
  of `ContextNet.FRAME`. Every output layer starts at zero; 470k parameters.

**The run** (`runs/residual/aware/`): the step-4 physics, room and noise frozen (the residual attack noise `noise.att`
trains), the residual from scratch, no budget, lr 1e-3 (5e-4 for the residual), warm-up 100 steps, cosine decay over
the second half, 30 min = 1,000 steps at 1.8 s per step (10.8 GB). Validation, 64 segments, physics 0.7199: with the
residual 0.7161, 0.7064, 0.7077, 0.6987, 0.7052 at steps 200–1000 (the checks move by about ±0.005).

**On 96 test excerpts** (`compare.md`, 2 seeds, variation off; difference from physics alone, mean ± 2 se):

| | total | band | fine | attack | log-mel (dB) |
|---|---|---|---|---|---|
| the GRU residual (step 4, ~8,300 steps) | −0.005 ± 0.004 | −0.003 ± 0.002 | −0.002 ± 0.002 | −0.004 ± 0.003 | −0.02 ± 0.02 |
| the aware residual, `last.pt` (1,000 steps) | −0.010 ± 0.004 | −0.006 ± 0.002 | −0.005 ± 0.002 | −0.006 ± 0.004 | −0.06 ± 0.03 |
| the aware residual, `best.pt` (step 800) | −0.004 ± 0.006 | −0.003 ± 0.003 | −0.004 ± 0.002 | −0.000 ± 0.005 | −0.01 ± 0.03 |

- Twice the old residual's gain, in an eighth of its steps. Against 13.1's per-note ceiling (−0.136 on the first 24
  of these excerpts) it is 7 %, but that ceiling includes what no predictor can reach (13.1), so the reachable
  fraction is unknown.
- `best.pt`, picked on validation, is worse on test than `last.pt`: at this size the 64 validation segments cannot rank
  checkpoints.
- Whether more training would take it further, this run cannot say: validation is flat within its noise after step
  400. Not yet looked at: which outputs it uses, and how large its curves are.

**In-sample** (`compare_train.md`: 96 excerpts of the training pieces, the same seeds; the windows are not those the
run drew, the pieces are): the aware residual −0.016 ± 0.004 (band −0.009, fine −0.007, attack −0.012, log-mel −0.09
dB), the GRU residual −0.007 ± 0.004. On the training pieces it gains about 1.6 times what it gains on test (the GRU
1.4 times): small either way. The residual is far from fitting even the pieces it trained on, so after 1,000 steps the
gap to the ceiling is not overfitting; whether it is too few steps, too little capacity or what cannot be predicted
from the context, this does not separate.

### 13.3 Capacity only: a memorisation test

This shows what the network can express, not what it carries over to new pieces: any network of this size fitted to 8
excerpts of 2 s would be expected to get close to a per-excerpt free fit (review 5, 4.2).

`scripts/residual_ceiling.py --variants aware --max-batches 1` (`runs/residual/memorise/`): the aware residual of 13.2,
from its 30-min weights, trained on the first 8 test excerpts alone (400 steps, lr 1e-3, a fresh noise draw per step),
then scored on the noise seeds of 13.1, against the free outputs fitted to the same 8 excerpts (13.1's first batch):

| | total | band | fine | attack | log-mel (dB) |
|---|---|---|---|---|---|
| physics | 0.727 | 0.376 | 0.592 | 0.406 | 3.41 |
| the aware residual as trained (13.2) | 0.714 | 0.370 | 0.590 | 0.393 | 3.33 |
| free per-note outputs | 0.592 | 0.302 | 0.545 | 0.306 | 2.78 |
| free per-frame outputs | 0.501 | 0.252 | 0.529 | 0.233 | 2.60 |
| both | 0.452 | 0.226 | 0.493 | 0.207 | 2.39 |
| **the aware residual, fitted to these 8** | **0.489** | 0.247 | 0.514 | 0.227 | 2.47 |

Fitted to them, the network goes past the free per-note and per-frame outputs and gets 86 % of the way to both
together (−0.238 ± 0.028 of −0.275), its loss still falling at 400 steps. So its capacity and output path (20 ms
control, ±12 dB groups, its features) are not what holds it at −0.010 on new excerpts. What is left is whether the corrections carry over from piece to piece: too few steps or too
little data for that, or corrections no context predicts (strike-to-strike variation). A longer run that tracks the
training and test pieces side by side would separate the first from the last.

## 14. Phase 4: the attack in three parts, training hygiene, a cheap sympathetic bank (2026-10-01)

After review 5 (`docs/reviews/review_5_phase3.md`) the owner chose: rebuild the attack, add a gain per piece and the
whole-excerpt level term, turn on a cheap sympathetic bank for the bass and tenor, then one training run, the bench
and a listening set. No control run and no per-change ablations: the bench attributes the physics constituent by
constituent, and a control could only change how much of step 4's gain is credited to steps 1–3, which would not
change what is kept.

### 14.1 What was built

- **A gain per training piece** (`train.py --piece-gain`; `notefit.PieceLevels`): a free level in dB per training
  piece, averaging zero, applied to the render before every loss term, learning at 50 times the base rate (each of
  2018's 70 training pieces is in about one batch in nine); not used in validation or evaluation. At equal key and
  velocity the pieces differ by ~2 dB (12.5); without it the level parameters followed each batch's pieces (12.9).
- **The whole-excerpt level term** (`PianoLoss(level_weight=)`, `train.py --level-weight`): L1 of the log of each band's
  energy summed over the excerpt, in the band term's bands. Checked on a synthetic target whose energy comes in bursts
  against a steady prediction of equal energy (`tests/test_losses.py`): the per-frame band term's optimum is below
  −6 dB (the edge of the scan), the level term's +0.1 dB.
- **The attack in three parts** (config `attack_model="parts"`, `synth.NoiseBank`; 10.4 item 1, review 5 3.2): the
  knock noise (its per-key spectrum rolled off above 2.5 kHz at −40 dB/oct), the key-bottom thump (its own spectrum up
  to 2 kHz, a level per register), and a string-borne precursor (up to 5 kHz, a decay of 0.3–5 ms, its own velocity
  slope). Each part has, per band, a smooth rise of at least half a period of the band's centre, then an exponential
  decay; the knock's and the thump's decays are per register (knots at MIDI 21, 65, 108) and band, the thump's up to
  0.5 s for the treble board's low ringing. The step onset after the band split is gone. Each kernel keeps the energy
  of the step-onset exponential with the same decay. Rendered as event trains per band and register convolved with
  the kernels (FFT): no per-note envelopes, so long decays are free and the step uses less memory than step 4 (9 vs 11
  GB). Tests: the converted knock keeps the old one's energy per band within 2 %; nothing above 10 kHz (−40 dB re 0.5–2
  kHz); blocks render as one pass; gradients reach every new parameter.
- **A cheap sympathetic bank** (config `symp_lo_midi`, `symp_hi_midi`, `symp_max_hz`, `symp_decimate`): only the
  strings of MIDI 21–59 respond, their first 16 partials below 2.5 kHz, at a quarter of the sample rate (the drive
  low-passed by a 128-tap Kaiser FIR and decimated, the response interpolated back; the filters' histories carry
  across blocks). Against the full-rate bank on the same keys: energy within 1 dB. With it on, a training step takes
  2.2 s instead of 1.5.

### 14.2 The attack fitted on isolated notes

`scripts/fit_notes.py --fit parts --init-attack-parts` (`runs/phase4/attack_fit/`): from step 4, the parts set from its
knock and thump, then fitted on the 819 calibration notes (600 steps; the knock windows of 12.5 plus the energy between
the partials at 30–100 and 100–400 ms). Evaluation notes (383), mean abs over 7,700–7,900 cells:

| model | mean abs (dB) | shape (dB) |
|---|---|---|
| step 4 (`step4_baseline/`) | 3.064 | 2.972 |
| step 4, attack converted to the parts, before the fit | 3.073 | – |
| fitted, the precursor started at the knock's dark spectrum (`fit_dark_precursor/`) | 3.012 | 2.927 |
| fitted, the precursor started flat above 1 kHz (`fit/`, the one used) | 2.997 | 2.905 |

- The conversion and the caps cost nothing on isolated notes (+0.009 dB); the fit gains 0.07 dB over step 4.
- Between the partials in the attack window, R3–R4 read 4–7 dB weak at 4–8 kHz in step 4 already and still do; in R5
  the knock's cap took away energy step 4 had there (8 kHz: +2.9 → −3.5 dB after the fit). The precursor does not
  fill 4–8 kHz: the fit lowered its 4 kHz level and shortened it to 0.8 ms (velocity slope 76 dB/u).
- **The treble thump's long low ringing did not appear:** its low-band decays stay at 20–25 ms in every register
  (bound 0.5 s); the notes' 100–400 ms windows do not ask for it.

### 14.3 One run on music

`runs/phase4/train_run/` (`chain.sh`): from `attack_fit/fit/model.pt` with step 4's per-strike spreads, the parts, B,
the tuning and the hall frozen, the bank on, the piece gains and the level term (weight 0.5), the onset term as in step
4; stage 1 only (no tables, no residual), 50 min = 1,322 steps, lr 5e-4, warm-up 200, cosine decay over the second
half. Validation (64 segments; its total includes the new term): 0.928 at the start, 0.909–0.949 along the way, 0.923
at the end. The piece gains end with an sd of 1.5 dB.

**On 96 test excerpts** (`compare.md`, 2 seeds, variation off), phase 4 − step 4: total −0.0027 ± 0.0026, band −0.0015
± 0.0012, fine −0.0018 ± 0.0014, attack −0.0015 ± 0.0027, log-mel −0.02 ± 0.01 dB (both seeds agree). A small gain,
at the edge of the resolution.

**On the bench** (`bench/report.md`, evaluation notes and 24 music excerpts, variation on; model − recording, step 4
→ phase 4):
- The high attack on isolated notes comes down where it was worst: N4 percussive 8 kHz R2 +7.0 → +3.9, R5 +8.4 → +6.9,
  R6 +9.5 → +3.7 dB (R3–R4 unchanged at +5 to +6); N6 knock 8 kHz R6 +10.9 → +5.4, 4 kHz R6 +6.8 → +3.1, 250 Hz R6
  +10.2 → +3.3 dB. The knock at 1 kHz in mf–f: −2.2 → −1.7 dB.
- **In music the attack is as abrupt as before** (E4, 8 kHz flux contrast; recordings 0.3, 0.5, 0.5, 2.4, 4.2, 3.6 dB in
  R2–R7): 1.2 → 1.2, 1.0 → 1.1, 2.0 → 2.1, 5.9 → 5.5, 7.6 → 6.4, 8.8 → 9.6. The smooth rises and the knock's cap did
  not change it, so the abrupt high onset in music comes from elsewhere: the partials' own onset, the knock impulse, or
  the onset jitter and timing; not checked.
- **The pedal halo is unchanged** (E1, down − up between the partials): R2 −2.5 → −3.7 dB, R3 −3.2 → −3.1 (recordings
  +1.9, 0.0). The bank is far too quiet to matter: on 16 test excerpts its output sits 32–45 dB under the strings,
  pedal down or up, and training moved its gain by under 1 dB. Its coupling (a gain of 0.02–0.04 on the share of
  each string's losses that goes through the bridge; resonances ~1 Hz wide against the drive's inharmonic partials)
  gives a response a band-level loss cannot see, so the gain gets no gradient worth the name. Setting its level from
  E1 (a scan of the gain on the bench) is the obvious next step; not done.
- Texture statistics (P3) and level are unchanged within a few tenths of a dB.

Listening: `listen/` (8 × 12 s, soft to loud, and the demo) and `listen_long/` (2 × 20 s), step 4 against phase 4,
the per-strike variation on.

## 15. Isolated tenor notes, partial by partial (2026-10-01)

The owner, listening to `samples/phase4/listen_long/0_ab.wav`: the lower notes played slowly under a texture (D3–B2,
MIDI 47–50, velocity 64–74, pedal down) sound like "a complex synth wave with a custom attack envelope"; the decay,
the timbre and the attack differ. Notes inside music overlap others and the pedal, so the measurement is on isolated
notes of the same register (R3) and the excerpt stays the listening check.

**What was built.** `measures.partial_profile` (T1/T2 of the design): each partial's level over the note (a 40 ms
Hann window every 5 ms at the partial's own frequency, so level changes up to ~12 Hz are followed), and per partial
the peak time, the decay over 50–350 ms and over 0.5–1 s (fitted jointly with a sinusoid at the partial's beat rate
where the window holds a full cycle: the strings of a unison start in phase, and a line alone read a 6 Hz beat of
±2.3 dB on −10 dB/s as −18), the fluctuation around a cubic in time (dB rms), its strongest rate (1.2–12 Hz) and its
periodicity (the share of the fluctuation at that one rate). `measures.non_tonal`: the energy away from the partials
per octave band, re the note's energy in the same window. Both are checked on synthetic notes with known decays,
knee, beat and irregular fluctuation (`tests/test_measures.py`). `scripts/note_profile.py` finds the notes, renders
them with the model in context and writes the report, figures and A/B files.

**The notes** (`runs/phase4/r3_profile/`): 2018, MIDI 47–59, velocity ≥ 50, no other onset 0.3 s before or 1.0 s
after: 56 (49 from training pieces; isolated notes are rare in this repertoire: 4 at mf–f stay clear for 1.6 s).
42 sound to 1 s (key held or pedal down); 14 are released earlier and count up to their release. Model: phase 4,
per-strike variation on. Medians over notes; n per cell 10–56.

**What it found** (model − recording, paired medians unless noted):
- **The average note is close.** Each partial's level re the note's at 50, 300 and 900 ms is within ±2 dB for most
  of partials 1–12 (`tracks.png`); the beating is as strong (fluctuation 0.5–0.8 dB rms on both sides), as fast
  (2–3 Hz) and as (ir)regular (periodicity 0.3–0.4 on both), in as many partials (20–50 %).
- **The notes are too much alike.** Across notes, the spread (IQR) of each partial's decay after 0.5 s is 6–24 dB/s in
  the recordings and 3–8 dB/s in the model, for 11 of 12 partials; the spread of the two-stage contrast is 1.5–3 times
  smaller in partials 7–12, and of the fluctuation in partials 10–12 4–5 times smaller. Every model note of the
  register decays its partials in much the same way; the piano's do not. The per-strike variation scales all of a
  note's decays together (`strike_log_decay`, `strike_decay_tilt`), and a partial's decay in the model is a smooth
  function of its number. A candidate, not checked: in the piano a partial's decay depends on the bridge's
  admittance at its frequency, which is peaky (the board's modes), so each note's partials land on different peaks;
  the recordings' decays plotted against absolute frequency across notes would show it.
- **The fundamental fades too fast:** partial 1 decays at −14.6 against −6.4 dB/s over 50–350 ms (paired −8.4, n
  30) and ends ~4 dB low at 0.9 s; partial 6 too slowly after 0.5 s (paired +9.4, n 15; +5.7 dB at 0.9 s).
- **The middle partials peak too early:** partials 2–4 and 8 reach their maximum 15–35 ms sooner than in the
  recordings (the piano's 3rd and 4th peak at 68 and 105 ms, the model's at 40 and 25); the spread is large (IQR
  40–110 ms). Together with N4's faster rise (R3 −10 ms) this is the attack building up too fast.
- **Between the partials:** in the attack window 2.7 dB weak at 500 Hz; afterwards 1–2.7 dB too much at 125–500 Hz
  and 2–4 kHz.

**What this does not see.** The first ~20 ms (the 40 ms window smears the onset; N4 rise and N6 cover it coarsely);
beats slower than ~1.2 Hz and the knee beyond 1 s (the notes are clear for only 1 s); the stereo image per partial
(each partial radiates from a different part of the board; `channel_measures` is per band); and everything the
context adds (pedal halo, overlapping notes), which these notes exclude by design. The recordings' spread includes
measurement noise (earlier notes ringing under the pedal, the room), but the model's renders carry the same contexts.

Listening: `samples/r3_notes/` (12 of the notes sounding to 1 s, soft to loud; each recording and the model's render,
its level matched over the first 0.5 s; `ab.wav` plays every pair in turn).
