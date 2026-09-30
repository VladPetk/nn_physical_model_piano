# What makes up a piano tone: constituents, time course, perception, measurement

Literature search for the neural/physical piano synthesizer project (Disklavier recordings, MAESTRO).
Date of search: 2026-09-30. Budget-limited: 12 web searches and about 10 page/PDF pulls, so this is a
map with evidence labels, not an exhaustive review.

## 0. How to read this report

Evidence labels used on every claim:

- **read**: I saw the text or abstract stating it (source ID in brackets, e.g. [R1]). Where the claim
  was in a paper that itself cites someone else, I say "read (secondary)".
- **cited**: from memory or second-hand, not checked in this session. Treat as a lead, not a fact.
- **mine**: my own inference or design suggestion, not from the literature.

Already covered by the project and only cited here (not re-read): KTH "Five Lectures on the Acoustics
of the Piano" (Askenfelt ed.), Weinreich 1977, Conklin 1996, Hall's hammer papers, Bank et al.,
Chaigne & Askenfelt 1994, DDSP-Piano (Renault et al.), Bader & Plath, Ege & Boutillon, Rigaud et al.

What I actually read (details and URLs in section 7):

| ID | Source | What was read |
|----|--------|---------------|
| R1 | Goebl, Bresin, Fujinaga 2014, touch quality in piano tones | full text, extracted with grep (abstract, intro, results) |
| R2 | Simionato & Fasciani 2024/25, sines-transient-noise piano | full text, extracted with grep |
| R3 | Simionato et al. 2024, physics-informed differentiable piano | summary from a fetch tool (secondary quality) |
| R4 | Bank & Lehtonen 2010, perception of longitudinal components | abstract-level text from search result |
| R5 | Järveläinen, Välimäki, Karjalainen 1999/2001, inharmonicity audibility | abstract-level text from search results |
| R6 | Peeters et al. 2011, Timbre Toolbox | full text, definitions section via grep |
| R7 | Bank, Avanzini, Borin, De Poli, Fontana, Rocchesso 2003, research overview | full text, several passages via grep |
| R8 | Valiente et al. 2024, una-corda model | abstract and intro |
| R9 | Lehtonen et al. 2007, sustain-pedal effects | abstract-level text from search result |
| R10 | McDermott & Simoncelli 2011, sound textures | abstract-level text from search results |
| R11 | Turian & Henry 2020, spectral distances bad at pitch | abstract |
| R12 | Vahidi et al. 2023, mesostructures | abstract |
| R13 | Galembo et al. (bass range: phases; inharmonicity vs envelope) | search-result summaries only |
| R14 | Bensa et al. 2005, perceptive and cognitive evaluation of a piano synthesis model | search-result summary only |
| R15 | Bank & Sujbert 2005, longitudinal vibrations | search-result summary only |

Bottom line up front: a **physics-only model of string + soundboard omits at least four
constituents that the literature ties to audible or measured effects**: the touch precursor and
key/action noises (audible, R1), the longitudinal/phantom partials in the bass up to about C5
(audible, R4), the pedal-dependent change of decay in the mid-range (measured, R9), and the
treble knock (called the most important issue for high notes, R7). The one neural piano paper with
a listening test (R2) reports the **attack** as where the model fails perceptually; R3 lists the
omitted noisy components as a limitation.

---

## 1. Constituents of a piano tone along the sound-production chain

Convention: register bands are approximate: bass = A0 to about C3, mid = C3 to C5, treble = C5 to C7,
top = highest notes (undamped). These are my bands, chosen to match where the sources give cut-offs
(C5 for longitudinal audibility, A3 for the modelling cut-off in R4).

### 1.1 Master table

| # | Constituent | Physics (one line) | Magnitude / time scale | Register and velocity dependence | Perceptual relevance | How measured (short) | Evidence |
|---|-------------|--------------------|------------------------|----------------------------------|----------------------|----------------------|----------|
| 1 | Finger-key touch precursor | Finger hits or presses the key; key/felt/frame noise before the hammer moves | Occurs 20-30 ms before the tone (Askenfelt 1994, "touch precursor"); in Goebl's stimuli 20-200 ms before the sound. Peak SPL of the noise (100 Hz low-passed) about 47 dB (pressed touch) to 56-64 dB (struck touch), as measured in that setup | Depends on **touch**, not on hammer velocity: tones with identical hammer velocity but different touch were distinguishable. Louder for struck touch and varies by pianist (7 dB between two pianists) | Audible in a controlled test: about half the participants identified touch type; with the noise cut out, identification fell to chance | Peak of the 100 Hz low-passed signal before onset; spectrogram of the pre-onset interval | **read** [R1]; the 20-30 ms figure is **read (secondary)** from Askenfelt 1994 / Koornhof & van der Walt 1994 via R1 |
| 2 | Key-bottom thump (key hits the keyframe/keybed) | Key stops on felt at the end of travel | Occurs "almost simultaneously" with hammer-string contact, so usually masked by the tone | Tested at E7 and F7 with three dynamics; the pitch-by-dynamics interaction came from the two soft-dynamics pairs (F7 soft was discriminated best) | Musicians discriminated tones with vs without key-bottom impact, but the authors say the cue may instead be other action sounds (e.g. key release) | Needs key-position sensor (Bosendorfer CEUS in R1) to know when it happened; audio alone shows extra energy 0.15-0.25 s in one pair | **read** [R1] |
| 3 | Let-off/escapement, hammer rebound, backcheck catch | Mechanical clicks of the action | Not quantified in anything I read | Likely velocity-dependent (mine) | Not found | Not found | **cited** (generic knowledge of the action); nothing read that quantifies it |
| 4 | Hammer-string contact | Nonlinear felt: force about K*compression^p, p about 2.5-4 from bass to treble; hysteresis | Contact of "a few milliseconds"; example force pulse about 2 ms (R7). Hammer speed 1-6 m/s | Contact shortens and spectrum brightens with velocity (harder felt response); p and contact time change with register. Multiple contacts of the hammer on the string are reported for treble notes | Sets spectral tilt and its velocity dependence (loudness vs brightness coupling) | Fit hammer parameters from spectra at several velocities; envelope of partials vs strike position comb | **read** [R3, R7] for p range, 2 ms and 1-6 m/s; multiple contacts and velocity-brightness rule are **cited** (Hall; Chaigne & Askenfelt; Askenfelt lectures) |
| 5 | Strike-point comb | The string is struck at a fraction of its length, suppressing partials whose node is at that point | Notches in partial amplitudes (strike point about 1/8-1/7 of length in most pianos, **cited**) | Position varies with register (mine) | Contributes to timbre; not found tested alone | Compare partial amplitude ratios to |sin(k*pi*x0/L)| | **cited** |
| 6 | Transverse partials with inharmonicity | Stiff string: f_k = k f0 sqrt(1 + B k^2) | B from 0.0002 (bass) to 0.4 (treble) as quoted in R3 (the treble figure looks large; check against Rigaud et al. before using) | B grows steeply with register; audible effect strongest for low f0 | Detectability depends strongly on f0 and is easier at low f0 (thresholds measured at 55 to 1108.7 Hz). R7 adds that beating between misaligned high partials is probably the cue, and that partials above a band could be made harmonic "without relevant perceptual consequences" | Peak-picking then least squares fit for B and f0 (R2 uses pairwise formula on 6 partials; Rigaud et al. 2013 is a parametric estimator) | **read** [R3, R5, R7, R2] |
| 7 | Unison strings: mistuning, beating, coupled-mode double decay | Two or three strings per note coupled through the bridge; slight mistuning and excitation differences give a fast "prompt" decay then a slow "aftersound", with beats | Amplitude modulation overlaid on exponential decay; two-stage decay: faster early, slower later. Not quantified in the sources I read | Single string in the lowest octave (no beating); two strings in the low-mid; three strings above. Beat pattern depends on tuning, so it varies note to note and piano to piano | No formal psychoacoustic test found (see 3); used by tuners as "unison quality" | Fit partial envelopes with one vs two exponentials; estimate beat frequency from spectral doublets | **read** [R7, R2, R8] for mechanism and string counts; "no formal test found" comes from a weak search summary |
| 8 | Two polarisations of each string | Vertical polarisation couples strongly to the soundboard (fast decay); horizontal decouples (slow decay) | Adds a second decay stage on top of the unison effect | Present in all registers, strength depends on bridge and soundboard | Not tested separately | Same two-stage decay fits; cannot separate from unison mistuning with audio alone (mine) | **read** [R2] |
| 9 | Una corda effect | Shift of the action so the hammer strikes fewer strings; the free string is driven sympathetically | Model with 2 strings and reduced soundboard: with all strings struck there is beating; with una corda the decay is more even, no beating | Grand pianos only; note-dependent | Described as a "softer and duller" tone | Compare partial envelopes with and without pedal | **read** [R8] |
| 10 | Longitudinal vibration, precursor, longitudinal modes | Tension modulation and string stretching generate longitudinal waves; they travel much faster than transverse waves and are transmitted to the bridge | Small amplitude relative to transverse, but give "metallic" character to low notes and play a role in the attack | Bass and low mid. Audible up to C5 (523 Hz); a synthesizer that models them only up to A3 (220 Hz) was judged sufficient by listeners | Real: audible up to C5 | Look for spectral peaks that are not on the transverse inharmonic series | **read** [R4, R15, R2]; "faster than transverse" is **cited** |
| 11 | Phantom partials | Nonlinear mixing: partial at 2*f_j (even, "forced") and at f_j +/- f_m (odd, "free") | Double-frequency phantom decays with about half the decay time of the source partial | Increase with amplitude (quadratic nonlinearity, so faster than linear in velocity, **mine**), strongest in the bass | Part of the same audible longitudinal content (see 10) | Peak-picking at sum/difference frequencies; compare with modelled transverse series | **read** [R2, R3, R15] |
| 12 | Pitch glide (tension modulation) | Large amplitude stretches the string, raising initial pitch, then settles | Not quantified in the sources I read | Larger at high velocity and in bass strings | Not tested | Track f0 in the first 100-300 ms | **cited** (Conklin 1996; not verified here) |
| 13 | Bridge/soundboard admittance and radiation | Soundboard's driving-point admittance controls energy loss (decay times) and radiation | Soundboard response high modal density at high frequency; R7 models the low band separately from about 2.2 kHz | Shapes long-term spectral envelope; strongly register-dependent | Bass timbre difference between large grands and uprights was attributed to inharmonicity, but a test showed spectrum bandwidth matters more | Frequency-dependent decay rate per partial; radiation measured with impulse or shaker | **read** [R7, R13 summary]; soundboard modal details are in the sources the project already covers |
| 14 | Soundboard build-up and the "knock" | Attack noise caused by the hammer impact exciting the soundboard directly | "Simulating the attack noise (the knock) of the tone is the most important issue" for **high** piano notes | Treble; independent of the longitudinal effect | Attack fails perceptually in the neural models tested | Transient/percussive component via HPSS; short-window spectral difference | **read** [R7 for high-note statement; R2 for HPSS] |
| 15 | Sympathetic resonance from undamped strings | Strings not held by dampers vibrate when driven by the bridge or the air | With sustain pedal: decay times of partials increase in the **mid range**, not bass and treble; free string register increases sympathetic vibration | Mid-range for pedal; top treble has no dampers (**cited**) | Audible as sustain; no threshold found | Decay time vs pedal state; remove partials and see what survives | **read** [R9]; no-damper top **cited** |
| 16 | Duplex/aliquot scaling | Short unstruck string segments beyond the bridge resonate at partials of the played note | Not quantified in anything I read | Treble | Not found | Not found | **cited** |
| 17 | Dampers and release sounds | Damper lands on the string; key release clicks | Release sounds may have been the cue in R1's key-bottom stimuli (extra energy 0.15-0.25 s after onset in one pair) | Damper sound depends on note and release velocity | Possibly audible | Time from key-off to level drop (T_release) per partial | **read (weak)** [R1]; the rest **cited** |
| 18 | Pedal noise | Sustain pedal mechanism thumps | Not found | | | | **not found** |
| 19 | Room and microphone | Reverberation, room modes, microphone position and colouration | Not found for MAESTRO | | | T60 by backward integration or blind estimation | **cited** (Schroeder integration); project already has a room module |

Note on row 3 and 18: I did not find a source that quantifies these. They are listed only so the
suite has a slot for them.

### 1.2 What the neural piano papers say they omit or fail on

- R2 (sines/transient/noise): the model has three sub-modules: quasi-harmonic (sinusoids with inharmonicity,
  phantom partials, beating, double decay), transient (HPSS "percussive" component, generated in the
  DCT domain), and noise (filtered Gaussian noise trained on the residual). Their own listening test
  (MUSHRA, 20 participants) shows the **attack** is where most inaccuracy is; high partials are over-predicted.
  **read** [R2].
- R3 (physics-informed differentiable, earlier by the same group): omits "noisy components" that
  matter for attack realism, has poor accuracy for high partials, and omits pedal and key coupling.
  **read** [R3, summary].

### 1.3 Things a physics-based synthesizer would commonly miss (mine, based on the above)

1. The touch precursor: 20-30 ms before onset, level clearly above the room noise in a close setup,
   and it changes with the pianist's touch, not with MIDI velocity. Note that MAESTRO is human-played
   on a Disklavier, so these sounds are in the audio (mine: I did not verify the microphone distance).
2. Longitudinal and phantom partials below about C5.
3. The mid-range sympathetic contribution under the pedal, which changes decay times.
4. Treble knock.
5. Release sound: key/damper noise after key-off.
6. Beat and double-decay structure that is different for each note (a statistic, not a curve to match).

---

## 2. The time course of a note

Only numbers that I read are given; the rest are qualitative and flagged.

### 2.1 Phases

| Phase | Window (typical) | What dominates | Level / detail | Evidence |
|-------|------------------|----------------|----------------|----------|
| Pre-onset / precursor | 20-30 ms before hammer-string contact (up to 200 ms in the stimuli of R1) | Finger-key noise, key/frame noise | Peak of the 100 Hz low-passed noise about 47-64 dB SPL in R1's setup, depending on touch and pianist; level of the tone itself not compared | **read** [R1] |
| Hammer contact | About 2 ms in the example force plot | Nonlinear felt force, force pulses from reflected waves on the string | Sets initial spectrum | **read** [R7] |
| Attack (0-50 ms) | First tens of ms | Broadband transient (knock), longitudinal precursor, bridge/soundboard build-up, key-bottom thump | The transient "decays rapidly" (R2); listeners find the attack the weakest part of the neural models | **read** [R2]; key-bottom "almost simultaneous" **read** [R1] |
| Prompt sound (early decay) | From the end of the attack until the slope changes | Transverse partials, fast decay stage set by vertical polarisation and unison mismatch | Not quantified here | **read** (mechanism) [R2, R7]; times **not found** |
| Aftersound | After the transition, up to several seconds (bass longer) | Slow decay stage (horizontal polarisation, antiphase unison motion), beating | Not quantified here | **read** (mechanism) [R2, R7]; times **not found** |
| Sustain with pedal | While pedal down | Extra decay time in mid-range partials; sympathetic strings | See row 15 | **read** [R9] |
| Release | From key-off | Damper landing, decay of the partials, key release noise | Not quantified | **cited** |

### 2.2 Per-register dominance (what I can support)

| Register | Dominant constituents | Evidence |
|----------|-----------------------|----------|
| Bass (single or dual strings) | Inharmonicity easiest to hear (beats between misaligned high partials); longitudinal/phantom partials; bandwidth of the spectrum; relative phases of partials | **read** [R5, R7, R4, R13 summary] |
| Mid | Unison beating and double decay; pedal-dependent decay increase; hammer nonlinearity | **read** [R9, R7]; hammer nonlinearity **cited** |
| Treble | Attack noise (knock); fewer audible partials; shorter sustain; top notes undamped | **read** [R7]; top undamped **cited** |

The lowest octave has only one or two strings per note; "two or three ... except for the lowest octave"
is R7's wording (**read**).

### 2.3 Velocity dependence

- Touch noise does **not** follow hammer velocity (R1).
- Hammer nonlinearity: spectral brightness and contact time depend on velocity (**cited**).
- Phantom/longitudinal content grows faster than transverse amplitude (mine, from the quadratic mixing
  described in R2 and R3; not measured here).

---

## 3. Perceptual relevance

### 3.1 Table

| Constituent | What is known about audibility | Numbers / JNDs found | Evidence |
|-------------|-------------------------------|----------------------|----------|
| Inharmonicity | Detectability strongly depends on f0, easier at low f0. A simple threshold model as a function of f0 was fitted | Tested at 55 to 1108.7 Hz (five f0). I did not obtain the threshold values | **read** [R5] |
| Inharmonicity vs spectrum | In the piano bass range, spectral bandwidth/envelope mattered more for timbre than the level of inharmonicity | No numbers | **read (summary)** [R13] |
| Relative phases of partials, bass | Phase matters for pitch and timbre in the piano bass range; authors frame the question as whether it is strong enough to matter for real instruments | No numbers | **read (summary)** [R13]; authors (Galembo, Askenfelt, Cuddy, Russo, 2001?) **cited** |
| Longitudinal components | Audible up to C5 (523 Hz); modelling to A3 (220 Hz) judged sufficient by listeners | Note limits only | **read** [R4] |
| Longitudinal/phantom, informal | Important to the attack and responsible for the metallic character of low notes | | **read** [R15 summary] |
| Touch precursor | Half of participants identified touch when noise present; chance when removed. Pressed touches were identified better than struck; struck touches with the noise removed were misjudged as pressed | 68% vs 59% correct (pressed vs struck) with noise; 57% vs 41% without | **read** [R1] |
| Key-bottom sound | Musicians discriminated pairs with and without it, but the cue may be other action sounds | Not reduced to a threshold | **read** [R1] |
| Unison beating, double decay, aftersound | Considered important by Weinreich and tuners ("unison quality"). Formal psychoacoustic tests: I found none in this search (absence not proven) | None | **cited**; weak search summary |
| Sustain pedal | Changes decay in mid-range | No perceptual test read | **read** [R9] (physical effect) |
| Piano timbre semantics | Semantic descriptors of piano timbre: bright, dry, dark, round, velvety (from a study with 17 pianists) | | search summary; source attribution uncertain, possibly Bernays & Traube; **cited** |
| Bensa et al. | Subjective evaluation of a piano synthesis model linking parameters to semantic descriptors | | **read (summary)** [R14] |
| General timbre space | Attack time, spectral centroid, spectral flux are the classic dimensions (Grey 1977; McAdams et al. 1995). The Timbre Toolbox implements descriptors for them | See 4.1 for definitions | **cited** for the dimension claim; **read** [R6] for the descriptor definitions |

### 3.2 JNDs and thresholds I can quote

- Pure-tone frequency JND is used in R7 as a bound for how far modelled partials may deviate from the
  target (dashed bounds in their Fig. 4); the numeric bound was not extracted. **read** [R7].
- Level changes: R7 states that a less than 1 dB change of resonator amplitudes was found to be inaudible.
  **read** [R7].
- Reverberation time JND is about 5% (ISO 3382 convention). **cited**; not checked.
- Modulation (beating) detection thresholds: general psychoacoustics gives detection at a few percent
  modulation depth for slow rates; not looked up. **cited**.
- Touch noise and key-bottom sounds: thresholds unknown, only presence/absence tests (R1).

### 3.3 Practical reading of the perception evidence (mine)

- The best-supported audible items are: the attack (all neural models fail here, R2), the touch precursor
  (R1), longitudinal content up to C5 (R4), and bass inharmonicity (R5).
- Items with plausible but untested audibility: unison beating/double decay, release, pedal noise.
  These should be measured, but their loss weight cannot be justified from perception data.

---

## 4. How researchers measure each constituent

### 4.1 Global descriptors and definitions (Timbre Toolbox, R6, **read**)

| Descriptor | Definition (as in the paper) | Settings |
|------------|------------------------------|----------|
| Attack start/end | Peeters' "weakest effort" method: a set of energy thresholds; start and end are chosen where the effort to go from one threshold to the next is weak | Energy envelope |
| Log-attack-time (LAT) | log10(t_end - t_start) | one value per note |
| Attack slope | Weighted mean of local energy slopes during the attack (weights peak at 50% threshold) | |
| Decrease slope | Exponent of a decreasing exponential fitted to the energy envelope from its maximum (log-domain regression) | |
| Temporal centroid | Centre of gravity of the energy envelope | |
| Effective duration | Time the envelope is above 40% of its maximum | 40% threshold |
| Modulation of the envelope | Residual energy after removing the decay model, then DFT, peak in 1-10 Hz gives amplitude and frequency; 0 if no peak | Useful as a beat detector at the note level (mine) |
| Spectral variation (flux) | 1 minus the normalised correlation of consecutive spectra | STFT 23.2 ms Hamming window, hop 5.8 ms |
| Noisiness | Ratio of noise energy to total energy (harmonic model) | |
| Harmonic spectral deviation, odd/even ratio, tristimulus | Standard | |

### 4.2 Constituent-by-constituent measurement table

Column "Suggestion" is my design choice (mine) unless stated.

| Constituent | Descriptor | Method used in the literature | Window / setting | Blind spot | Evidence |
|-------------|------------|-------------------------------|------------------|------------|----------|
| Inharmonicity B and f0 | B; f0; deviation of partials in cents | Pairwise formula from six partial frequencies, B = mean over pairs (30 combinations per note per velocity); Rigaud parametric estimator; Bensa parameter fitting | R2 uses partials 1-6; "cent loss" on the first six partials | Peak errors at high partials; B estimate itself approximate | **read** [R2]; Rigaud, Bensa **cited** |
| Partial amplitude envelopes | Amplitude per partial vs time | Sinusoidal tracking or bandpass envelope | Long windows for bass | Overlap of doublets | **cited**; R7 says envelopes of recorded partials are extracted first, then fitted |
| Decay rate per partial | Decay rate vs frequency: linear or polynomial fit (R7 uses a fit of the decay rate vs partial index and loss-filter design) | Regression on log envelope | | | **read (partial)** [R7] |
| Two-stage decay | Initial rate, late rate, transition time, level at transition | Two-exponential fit to the envelope; doublet in spectrum | | Cannot separate unison from polarisation (mine) | **cited**; mechanism **read** [R7, R2] |
| Beating | Beat frequency, modulation depth per partial | Spectral doublets (Bensa: pair of exponentially damped sinusoids per mode pair); envelope modulation | Window must be longer than 1/beat frequency | Slow beats (<1 Hz) need long analysis (mine) | **read (summary)** [Bensa summary]; R7 says a coupled pair is a pair of damped sinusoids |
| Longitudinal / phantom partials | Peaks off the inharmonic series; peaks at 2*f_j and f_i +/- f_j; their decay (phantom about half) | Peak-picking with the transverse series removed | Bass only | Easily confused with sympathetic strings and noise | **read** [R2, R3] for structure and decay ratio |
| Attack / rise time | LAT, attack slope | R6 | Energy envelope | Blind to spectral content of the attack | **read** [R6] |
| Transient vs harmonic vs noise separation | Three-part decomposition | HPSS (Driedger et al. 2014) with margin 8; noise = residual = spectrogram minus harmonic minus transient | | Depends on separation quality (R2 says so) | **read** [R2] |
| Touch precursor level | Peak of 100 Hz low-passed signal before onset | Manual excerpt before onset | | Needs onset knowledge; depends on microphone | **read** [R1] |
| Touch spectrogram | Difference spectrogram between conditions | 256-sample FFT windows, 90% overlap, 16 kHz audio, differences below 3 dB left blank | | | **read** [R1] |
| Key-bottom sound | Not from audio alone: key position and velocity sensor | CEUS system | | | **read** [R1] |
| Pedal / sympathetic resonance | Decay time of partials with and without pedal; effect of removing partials | Analysis of recorded tones; string-register model with 12 strings for the lowest tones | | | **read** [R9] |
| Room T60 | Reverberation time | Backward integration (Schroeder) or blind estimation | | Not measurable directly on MAESTRO without a reference | **cited** |
| Release time | Time from key-off to given level drop, per partial | Not found | | | **not found** |

### 4.3 Objective metrics used in the neural piano papers and their blind spots

| Metric | Where used | What it can miss |
|--------|-----------|------------------|
| Multi-resolution STFT loss | R2: window sizes 256-4096 for the quasi-harmonic part, 32-256 for the transient, 32-512 for noise; normalised by target norm | Phase-insensitive: cannot see relative phases of partials, which matter in the bass (R13 summary); sensitive to timing shifts (R12) |
| RMS envelope MAE | R2, R3 | Does not see the spectral content of the attack |
| Cent loss on the first partials | R2, R3 | Only the first six partials |
| B_MSE, F loss | R3 | Only inharmonicity and pitch |
| Listening test (MUSHRA, 20 participants) | R2 | Exposed weakness of the attack that the numeric losses did not show: the STFT and RMS losses looked good while listeners found attack errors |
| Multi-scale spectral loss variants (spectral convergence, log-mag, linear-mag) | DDSP literature (R16 summary) | Over-smoothing: L1/L2 spectral losses average rapid spectral changes and can sound muffled (search-summary statement) |
| Spectral distances for pitch | Turian & Henry (R11) | Many audio distances have poor sense of pitch direction (tested on stationary sinusoids) |

Other facts: Turian & Henry's abstract says the task is trivial for humans and hard for common audio
distances (**read** [R11]). Vahidi et al. say spectrogram loss is sensitive to timing misalignment and
that neural synthesizers capture only local amplitude variations up to about 100 ms
(**read** [R12]).

---

## 5. Losses that match statistics or distributions rather than frames

| Idea | Source | Relevance to piano (mine) | Evidence |
|------|--------|--------------------------|----------|
| Texture statistics of an auditory model: marginal moments of subband envelopes, correlations between channels, modulation-band power. Per-channel statistics alone produced poor textures; adding cross-channel correlations made them realistic | McDermott & Simoncelli 2011, Neuron 71 | Fits the stochastic constituents: key/action noise, hammer noise, room tail, pedal noise. Not suited to tonal partials, where frame-level match is meaningful | **read** [R10] |
| Time-frequency scattering (joint time-frequency, time-invariant, multiscale) as a loss in place of a spectrogram loss | Vahidi et al. 2023 (R12); Andén, Lostanlen, Mallat (JTFS); differentiable versions by Muradeli et al. | Robust to small timing offsets in the attack | **read** [R12]; JTFS papers **cited** |
| Transient plus spectral synthesis for percussive audio | Shier et al. 2023 (arXiv:2309.06649) | Transients as a separate module | title only; **cited** |
| Multi-scale spectral loss revisited | Schwar & Muller 2023 | Discusses variants of the multi-scale loss | title only; **cited** |
| Sinusoidal frequency estimation by gradient descent | Hayes et al. (spectral loss local minima) | Explains why frequency parameters are hard to learn from spectral losses; supports a separate cent-loss on partials | **cited** |
| HPSS-guided training targets | R2 | Gives targets for transient and noise modules | **read** [R2] |
| Onset-weighted losses | DDSP review mentions listeners are more sensitive to artefacts near the onset (pre-echo), although most spectrogram energy is in sustain/release | Argues for extra weight or extra term on the first 50-100 ms (mine) | search-result summary only; the underlying source is not identified (it concerns audio restoration of pitched sounds); treat as **cited** |

Practical notes (mine):

- A distribution-matching loss belongs on the constituents with random phase or random timing (noise, touch
  precursor, room), not on the partials.
- Any attack loss should first be checked for timing sensitivity (R12); consider aligning the onset or using
  scattering-type invariants.
- None of the sources I read gives a validated loss for piano noise specifically.

---

## 6. Cross-cutting notes and gaps

1. **Evaluate on the constituents, not on the sum.** R2 shows a good total STFT loss with a perceptual failure in the attack.
2. **Per-note statistics are needed for unison effects.** The beat rate and two-stage decay vary from note to note
   (tuning) and cannot be matched frame by frame; compare distributions across notes (mine).
3. **Touch is not velocity.** Any model that maps MIDI velocity to everything cannot reproduce the
   precursor (R1). For MAESTRO the pre-onset region contains real touch noise (mine).
4. **Register matters.** Longitudinal up to C5 (R4), knock in the treble (R7), pedal effect in mid-range (R9),
   inharmonicity strongest in the bass (R5).
5. **Unknowns that block a principled weighting:** typical dB levels of the aftersound, phantom partials, knock and
   precursor relative to the tone; JNDs for beat depth and decay time in piano tones.

---

## 7. References and evidence status

Sources read in this session (full-text or abstract):

- R1 W. Goebl, R. Bresin, I. Fujinaga (2014). Perception of touch quality in piano tones. J. Acoust. Soc. Am. 136(5), 2839-2850. doi:10.1121/1.4896461. PDF: https://iwk.mdw.ac.at/goebl/papers/GoeblBresinFujinaga2014-JASA-PianoTouchQuality.pdf . **Full text read (grep-targeted).**
- R2 R. Simionato, S. Fasciani (2024; arXiv v3 2025). Sines, transient, noise neural modeling of piano notes. Front. Signal Process. doi:10.3389/frsip.2024.1494864; arXiv:2409.06513, https://arxiv.org/abs/2409.06513 . **Full text read (grep-targeted).**
- R3 R. Simionato, S. Fasciani, (and co-authors; author list not verified) (2024). Physics-informed differentiable method for piano modeling. Front. Signal Process. 3, doi:10.3389/frsip.2023.1276748. https://www.frontiersin.org/journals/signal-processing/articles/10.3389/frsip.2023.1276748/full . **Read via a summarising fetch tool; numbers should be rechecked.**
- R4 B. Bank, H.-M. Lehtonen (2010). Perception of longitudinal components in piano string vibrations. J. Acoust. Soc. Am. 128(3), EL117-EL123. https://pubs.aip.org/asa/jasa/article/128/3/EL117/598974 . **Abstract-level text from search result (page returned 403).**
- R5 M. Järveläinen, V. Välimäki, M. Karjalainen (1999), Audibility of inharmonicity in string instrument sounds, and implications to digital sound synthesis, Proc. ICMC, Beijing; and (2001) Audibility of the timbral effects of inharmonicity in stringed instrument tones, Acoust. Res. Lett. Online 2(3), 79-84. https://pubs.aip.org/asa/arlo/article/2/3/79/123666 . **Abstract-level text from search results (page returned 403).**
- R6 G. Peeters, B. L. Giordano, P. Susini, N. Misdariis, S. McAdams (2011). The Timbre Toolbox: extracting audio descriptors from musical signals. J. Acoust. Soc. Am. 130(5), 2902-2916. https://www.mcgill.ca/mpcl/files/mpcl/peeters_2011_jasa.pdf . **Full text read (definitions section).**
- R7 B. Bank, F. Avanzini, G. Borin, G. De Poli, F. Fontana, D. Rocchesso (2003). Physically informed signal processing methods for piano sound synthesis: a research overview. EURASIP J. Appl. Signal Process. 2003(10), 941-952. https://home.mit.bme.hu/~bank/publist/jasp03.pdf . **Full text; several passages read.** (Author list from memory of the paper header; verify.)
- R8 P. M. Valiente, G. Squicciarini, D. Thompson, D. O. Norris, C. Hernandez (2024). Modelling the una-corda effect in pianos. J. Phys.: Conf. Ser. 2909, 012036. doi:10.1088/1742-6596/2909/1/012036. https://generic.wordpress.soton.ac.uk/isvr-new/wp-content/uploads/sites/422/2024/12/una-corda-effect-in-pianos.pdf . **Abstract and introduction read.**
- R9 H.-M. Lehtonen, H. Penttinen, J. Rauhala, V. Välimäki (2007). Analysis and modeling of piano sustain-pedal effects. J. Acoust. Soc. Am. 122(3), 1787-1797. https://pubs.aip.org/asa/jasa/article-abstract/122/3/1787/853180 . **Abstract-level text from search result.**
- R10 J. H. McDermott, E. P. Simoncelli (2011). Sound texture perception via statistics of the auditory periphery: evidence from sound synthesis. Neuron 71(5), 926-940. https://www.sciencedirect.com/science/article/pii/S0896627311005629 . **Abstract-level text from search results.**
- R11 J. Turian, M. Henry (2020). I'm sorry for your loss: spectrally-based audio distances are bad at pitch. ICBINB@NeurIPS 2020. arXiv:2012.04572, https://arxiv.org/abs/2012.04572 . **Abstract read.**
- R12 C. Vahidi, H. Han, C. Wang, M. Lagrange, G. Fazekas, V. Lostanlen (2023). Mesostructures: beyond spectrogram loss in differentiable time-frequency analysis. arXiv:2301.10183. **Abstract read.**
- R13 A. Galembo, A. Askenfelt, L. L. Cuddy, F. A. Russo, work on the piano bass range: "Effects of relative phases on pitch and timbre in the piano bass range" (J. Acoust. Soc. Am., 2001; PubMed 11572374) and "Perceptual relevance of inharmonicity and spectral envelope in the piano bass range" (Acta Acustica united with Acustica, 2004). https://pubmed.ncbi.nlm.nih.gov/11572374 ; https://www.researchgate.net/publication/225284530 . **Only search-result summaries seen; author lists and years from memory.**
- R14 J. Bensa and co-authors (author list not verified). Perceptive and cognitive evaluation of a piano synthesis model. In: Computer Music Modeling and Retrieval (CMMR 2004), LNCS 3310, Springer 2005. https://link.springer.com/chapter/10.1007/978-3-540-31807-1_18 ; HAL: https://hal.science/hal-00088055 . **Search-result summary only.**
- R15 B. Bank, L. Sujbert (2005). Generation of longitudinal vibrations in piano strings: from physics to sound synthesis. J. Acoust. Soc. Am. 117(4), 2268-2278. https://pubs.aip.org/asa/jasa/article/117/4/2268/541382 . **Search-result summary only.**
- R16 Review of DDSP for music and speech synthesis (Hayes et al.), arXiv:2308.15422. **Search-result summary only.**

Pointers found but not read:

- B. Bank, J. Chabassier (2019). Model-based digital pianos: from physics to sound synthesis. IEEE Signal Process. Mag. 36(1), 103-114. https://ieeexplore.ieee.org/document/8588429/ (**best next read for a full constituent list**).
- H.-M. Lehtonen, doctoral thesis (2010), Analysis, perception, and synthesis of the piano sound, Aalto. https://aaltodoc.aalto.fi/items/7d986ec9-91b0-4ddc-a106-f3dade146608
- Master's thesis, Analysis and parametric synthesis of the piano sound (TKK 2005). http://lib.tkk.fi/Dipl/2005/urn007876.pdf
- F. Rigaud, B. David, L. Daudet (2013). A parametric model and estimation techniques for the inharmonicity and tuning of the piano. J. Acoust. Soc. Am. 133(5), 3107. https://pubs.aip.org/asa/jasa/article-abstract/133/5/3107
- J. Bensa, S. Bilbao, R. Kronland-Martinet, J. O. Smith III (2003/2005). Parameter fitting for piano sound synthesis by physical modeling. J. Acoust. Soc. Am. 118(1), 495. https://pubs.aip.org/asa/jasa/article/118/1/495/540250 . (Author list from memory.)
- Resynthesis of coupled piano string vibrations based on physical modeling. https://www.researchgate.net/publication/247404895
- Study on double decay of individual partials of piano sound: preliminary results on the "unison quality". J. Acoust. Soc. Am. 108(5) Suppl., 2592 (2000). https://pubs.aip.org/asa/jasa/article/108/5_Supplement/2592/552888 (authors not confirmed).
- Similarity of piano tones: a psychoacoustical and sound analysis study. Applied Acoustics (2018). https://www.sciencedirect.com/science/article/abs/pii/S0003682X18303463
- Decay rates of piano tones. https://www.researchgate.net/publication/372593181_Decay_Rates_of_Piano_Tones (2023?; not opened)
- Motor origins of timbre in piano performance. PNAS 2025. https://www.pnas.org/doi/10.1073/pnas.2425073122 (not opened)
- Schwar, Muller (2023). Multi-scale spectral loss revisited. https://www.researchgate.net/publication/375676965
- Shier et al. (2023). Differentiable modelling of percussive audio with transient and spectral synthesis. arXiv:2309.06649
- Fletcher, Blackham, Stratton (1962). Quality of piano tones. J. Acoust. Soc. Am. 34. (Reference list of R7 confirms the citation; content not read.)
- Grey (1977); McAdams, Winsberg, Donnadieu, De Soete, Krimphoff (1995), Psychol. Res. 58; Driedger, Muller, Disch (2014) HPSS. **cited**.

---

## 8. Open questions I could not answer within the budget

1. Typical **levels** (dB re the main tone) of: aftersound vs prompt sound, phantom partials, longitudinal precursor,
   knock, finger-key noise, key-bottom thump, per register and velocity.
2. **Numerical thresholds**: inharmonicity audibility values from R5 (only the trend was seen); beat-depth and
   decay-rate JNDs for piano tones; audibility of double decay (no formal test found).
3. **Pitch glide** magnitude vs velocity and register.
4. Contact time vs register and velocity, and how often multiple hammer contacts occur in the treble (only the
   general statement is cited).
5. Whether unison mistuning distributions have been published per note (would allow a distribution-level loss).
6. Whether any published work measures **release time** and **damper noise** from recordings.
7. Whether **MAESTRO's** microphones capture finger-key noise and pedal noise at levels above the room (needs a
   project-side check; the source data are in the audio).
8. Read Bank & Chabassier 2019 and Lehtonen's thesis for a full constituent list with numbers; open the
   "unison quality" 2000 abstract and "Decay rates of piano tones".
9. Loss functions validated on piano noise/transients specifically (only generic percussive and DDSP work found).
10. Perceptual validation of the objective descriptors: which of R6's descriptors correlate with listener
    ratings for piano tones in particular.
