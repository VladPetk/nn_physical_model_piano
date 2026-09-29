# Literature priors: soundboard/room, dampers, pedals, sympathetic resonance, noises, Disklavier/MAESTRO

Scope: everything except string and hammer. Key index k = MIDI - 21 (0 = A0, 87 = C8). Sample rate in code: 24 kHz, hop 120 (200 Hz control rate).

## 0. How to read this report (evidence status)

Web access was partial. `WebSearch` worked, but `WebFetch` was blocked (arxiv, KTH, Wikipedia, USPTO, AIP, Semantic Scholar, etc.). So I could read only search-result abstracts and snippets, never the full papers. Every value carries one of three tags:

- **[V]** appeared in a search-result snippet or abstract this session (the source is named). Numbers are still second-hand snippets, not read from the PDF.
- **[D]** derived by me from formulas I state, with the arithmetic shown. Checkable, but it is my modelling, not a measurement.
- **(unverified, from memory)** my recollection or engineering judgement. Treat as a prior to be learned, not a fact.

Confidence: high / medium / low. No citation below is invented. Where I am unsure of the exact paper I say so.

One correction to the brief: **Podlesak & Lee (1988) is about dispersion / string stiffness in piano strings** [V snippet], not the soundboard "knock" or precursor. The relevant papers for the key-bed thump and the touch precursor are Askenfelt & Jansson (1990-93) and Askenfelt (SMAC 1993, "Observations on the transient components of the piano tone"). For longitudinal precursors and phantom partials see Conklin 1996 Part III, Bank & Sujbert (JASA 2005, from memory), and Chabassier/Chaigne/Joly.

Sources actually surfaced (titles/abstracts only):
- Ege & Boutillon, "Synthetic description of the piano soundboard mechanical mobility" (ISMA 2010, arXiv 1210.5688).
- Ege, Boutillon & Rebillat, "Vibroacoustics of the piano soundboard: (Non)linearity and modal properties in the low- and mid-frequency ranges", J. Sound Vib. 2013.
- Boutillon & Ege, "Vibroacoustics of the piano soundboard: Reduced models, mobility synthesis, and acoustical radiation regime", J. Sound Vib. 2013.
- Giordano, "Mechanical impedance of a piano soundboard", JASA 103(4), 1998.
- Conklin, "Design and tone in the mechanoacoustic piano", Parts I-III, JASA 99/100, 1996.
- Weinreich, "Coupled piano strings", JASA 62(6), 1977.
- Lehtonen, Penttinen, Rauhala, Valimaki, "Analysis and modeling of piano sustain-pedal effects", JASA 122(3), 2007.
- Lehtonen et al., "Analysis of the part-pedaling effect in the piano", JASA-EL 126(2), 2009.
- Askenfelt & Jansson, "From touch to string vibrations I-III" (KTH lecture pages and JASA 1990-93).
- Goebl & Bresin, "Measurement and reproduction accuracy of computer-controlled grand pianos", JASA 114(4), 2003.
- Goebl, Bresin & Galembo, "Touch and temporal behavior of grand piano actions", JASA 2005.
- Tan, Kohlrausch, Hornikx, "Sympathetic vibration in a piano", ICA 2019.
- Southampton ISVR, "Modelling the una-corda effect in pianos" (2024 PDF; author names not seen).
- Hawthorne et al., MAESTRO / Wave2Midi2Wave, ICLR 2019.
- Chabassier, Chaigne, Joly, "Modeling and simulation of a grand piano", JASA 134(1), 2013.
- Fletcher & Rossing, *The Physics of Musical Instruments* (from memory only).
- A Frontiers in Signal Processing 2023 paper, "Physics-informed differentiable method for piano modeling" (doi 10.3389/frsip.2023.1276748). Related prior art; I did not read it and do not know the authors.

---

## 1. Soundboard, bridge and radiation

### 1a. Parameter table

| symbol | meaning | units | value / range | source | conf. |
|---|---|---|---|---|---|
| Y_b(f) | bridge (driving-point) mobility, normal to board | m/s/N | Only four published bridge measurements exist: Nakamura 1983, Wogram 1980, Conklin 1996, Giordano 1998. Giordano's C4 curve (upright) has mean impedance ~1500 kg/s, i.e. mean mobility ~ -63.5 dB re 1 m/s/N | [V] Ege & Boutillon 2010 intro | medium (upright, one point) |
| \|Z_b\| mid-range grand | mean bridge impedance, treble/mid bridge | kg/s | 1000-2500 (mobility -60 to -68 dB re 1 m/s/N) | extrapolated from the above; (unverified, from memory) for grands | low |
| Z_bass / Z_treble | bass-bridge vs mid/treble-bridge impedance | ratio | bass bridge is higher/stiffer, so mean mobility is "much lower" at bass-string coupling points. Guess Z_bass ~ 2-5x Z_mid, i.e. 6-14 dB | [V] direction only (Ege et al.); factor is (unverified, from memory) | low |
| f_c | crossover between "homogeneous plate" and "rib-strip" regime | Hz | ~1.1 kHz. Below it the board behaves like an isotropic clamped plate; above it modal density rises to that of the strips between ribs | [V] Ege/Boutillon 2013 | high |
| Y_mean(f) | mean mobility (Skudrzyk mean-value theorem) | m/s/N | Depends only on mass M, modal density n(f) and loss factor eta(f). For a plate Re Y = n(f)/(4M) with n in modes/Hz (= pi n(omega)/(2M)). Below f_c it is flat (infinite-plate value 1/(8 sqrt(D m''))); above f_c mobility rises (impedance falls) | [V] statement of theorem; the plate formula is [D] (checked against 1/(8 sqrt(D m'')) using n = (A/2) sqrt(m''/D)) | high (form) |
| n(f) | modal density below f_c | modes/Hz | ~0.05-0.1 (mean spacing 10-20 Hz). [D] from plate theory: A = 1.5 m^2, D ~ 460 N m, m'' ~ 4 kg/m^2 gives 0.07 modes/Hz. Rises above 1.1 kHz | [D]; the rise is [V] | low-medium |
| eta | soundboard modal loss factor | dimensionless | 1-3 % over several kHz, mean ~2 %, large scatter, no strong systematic frequency trend | [V] Ege et al. | high |
| T60_sb(f) | soundboard-alone modal decay | s | T60 = 6.91/(pi eta f) = 2.2/(eta f). With eta = 2 %: 60 Hz 1.8 s, 100 Hz 1.1 s, 200 Hz 0.55 s, 500 Hz 0.22 s, 1 kHz 0.11 s, 2 kHz 55 ms, 4 kHz 28 ms | [D] from eta [V] | medium |
| modal overlap | eta f n(f) | - | ~1 near 700 Hz-1 kHz (eta = 0.02, n = 0.07). Below that, peaks are individually resolved (visible resonant bumps); above it the response is diffuse | [D] | low-medium |
| f_1 (first modes) | lowest soundboard modes, big grand | Hz | first mode roughly 60-70 Hz; output falls steeply below it | [V] snippet (ICMC paper on hammer impact; author not verified) | medium |
| sigma(f) | radiation efficiency | - | fairly efficient 100 Hz-1 kHz, very efficient above ~1.4 kHz, smooth "transition range" 1-1.6 kHz. Consequence: partials below ~1.1 kHz radiate less efficiently and therefore decay less rapidly than upper ones | [V] Ege/Boutillon 2013 | high |
| coincidence f_crit | frequency where bending wavelength = air wavelength | Hz | ~1-1.6 kHz for the soundboard's along-grain direction (the transition range above) | [V] as "transition range"; the mechanism is standard plate acoustics | medium |
| bridge-loss decay | how Y_b sets string decay | 1/s | See 1b (derived). Example C4: alpha_bridge ~0.7 /s per string, ~2 /s in-phase for a trichord, i.e. T60 3-10 s | [D] | medium (form) / low (numbers) |
| Z_string | string characteristic impedance sqrt(T mu) | kg/s | C4 (T ~ 700 N, d ~ 1 mm steel): ~2 kg/s per string. Z_string/Z_bridge ~ 1e-3-4e-3 | [D]; T and d (unverified, from memory) | medium |

### 1b. Bridge admittance sets partial decay [D, standard]

- A string terminated by the bridge admittance Y_b has bridge reflection coefficient r = (1 - Z_s Y)/(1 + Z_s Y) ~ 1 - 2 Z_s Y.
- Only Re Y dissipates; Im Y shifts partial frequencies (this is where the "inharmonic-ish" bridge detuning comes from).
- One bridge reflection occurs per round trip, and the round trip lasts 1/f_1. So the bridge-loss amplitude decay rate of partial n is

      alpha_bridge,n = 2 * Z_s * Re{Y_b(f_n)} * f_1     (per string; multiply Y by N_in-phase for the in-phase mode)

- Weinreich (1977) [V]: the bridge admittance couples the strings of one note into one dynamical system. The in-phase mode loads the bridge with N times the admittance and decays fast ("prompt sound"). Out-of-phase modes, and the horizontal polarization, load it far less and give the slow "aftersound". This is what `mode 0 = prompt_ratio x faster` in physics.py stands for.
- Since Re Y_mean is flat below 1.1 kHz, alpha_bridge,n ~ constant in n and proportional to f_1. Individual partials that fall on a soundboard resonance peak decay faster, and partials in a valley decay slower.
- This is visible mainly for bass and tenor keys, where the mode spacing is larger than the partial spacing. Above ~1 kHz the response is diffuse and the partial-to-partial scatter averages out.
- Practical use for the priors:
  - `prior_log_b1` (per key) could be cross-checked as alpha_bridge ~ 2 Z_s Re Y f_1. That is the "physical floor" of the prompt-sound decay.
  - Bass strings: Z_s is large (heavy wound strings) and Y_b is small at the stiff bass bridge; both push toward slow decay.
  - The per-partial `partial_gain` and a per-key random "soundboard comb" of alpha scatter (a few dB in amplitude, more in the bass) could be initialised from the Rayleigh statistics of a modal sum.

### 1c. What linear filter bridge-force -> sound pressure at 1-3 m looks like

Qualitative shape [D from the above plus standard radiation theory; magnitudes are (unverified, from memory)]:

- Below f_1 (~60-70 Hz): steep high-pass. Output falls sharply below the first board mode [V]. Model as 12-18 dB/oct.
- 70-300 Hz: sparse modes, resolved peaks 10-20 dB above valleys. Bass-note fundamentals A0-C2 (27-65 Hz) are radiated 15-30 dB below their neighbouring partials 3-8; the perceived pitch of the lowest octave comes largely from partials 3-10. The current code has no such roll-off (see diagnosis).
- 300 Hz-1.1 kHz: modal overlap approaches 1, response becomes a smoother "mean mobility" flat plateau with statistical peak/valley fluctuation (Rayleigh-like, +-6-10 dB).
- 1.1-1.6 kHz: radiation efficiency saturates; mobility starts to rise (impedance falls), which is partly offset by the falling string-force spectrum.
- Above ~2-4 kHz: little modal structure. The board response is diffuse and very short (T60 tens of ms from 2.2/(eta f)). The high-frequency tilt comes from the string/hammer spectrum, air absorption, mic and hall.
- Directivity: low frequencies roughly omni (rim + board radiating as a dipole/monopole mix). High frequencies beam out of the open lid and radiate downwards to the floor. This gives lid/floor comb reflections 1-3 ms after the direct sound. (unverified, from memory.)
- The soundboard's own impulse response is short. It is dominated by 60-300 Hz modes with T60 0.5-1.8 s (weakly radiating, string-loaded) and by <100 ms decays above 500 Hz. The 1-2 s "tail" a listener hears is the room plus the string-register reverberation (sympathetic resonances), not the board.

### 1d. Recommended initialisation of the learned "soundboard + room" FIR

**Structure.**

1. Split the IR into a short body FIR (`body`, 0-300 ms, 24 kHz -> 7200 taps) and a parametric hall tail. Learning band-wise T60 and level of a fixed random-noise carrier (envelope parameters only) is far more identifiable than 24k free taps.
2. Alternatively keep one long FIR, but initialise it as the sum below and add a smoothness/decay regulariser (penalise |h(t)| growth relative to the initial envelope).

**Body part (bridge -> pressure), per condition, initial values.**

Frequency-domain magnitude target (dB re the 200 Hz-1 kHz plateau; a prior on shape):

| f (Hz) | 20 | 30 | 40 | 55 | 70 | 100 | 200-1000 | 2000 | 4000 | 8000 | 11000 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dB | -40 | -30 | -20 | -10 | -4 | 0 | 0 (with +-6 dB modal fluctuation) | -1 | -3 | -7 | -12 |

- The hall/mic HF slope above 4 kHz is the least certain part. Keep it learnable.
- Modal part below 1.1 kHz: draw ~60-80 modes with frequency density 0.07 /Hz between 60 and 1100 Hz (random, roughly Poisson with some repulsion). Amplitude of mode m: Gaussian.

      h_body(t) = sum_m a_m exp(-t/tau_m) sin(2 pi f_m t + phi_m),   tau_m = 1/(pi eta f_m),  eta = 0.02 (range 0.01-0.03)

  (T60 = 6.91 tau = 2.2/(eta f): 100 Hz 1.1 s, 300 Hz 0.37 s, 1 kHz 0.11 s.) Cap tau at 0.25 s.
- Above 1.1 kHz: Gaussian noise multiplied by a frequency-dependent decay, T60(f) = 2.2/(eta f) (eta = 0.02: 2 kHz 55 ms, 4 kHz 28 ms), via a small STFT/filter-bank envelope or by summing a few subband-noise layers. Smoothly cross-fade at 1.1 kHz.
- Normalise so the body's mean |H| over 200-1000 Hz is 0 dB. Include a first (direct) arrival at 3-6 ms (1-2 m of sound travel, 2.9 ms/m). In the current code the direct peak sits at sample 0. A causal IR then cannot represent "sound arrives earlier than the aligned onset", so a few ms of pre-delay leaves room for the fitted alignment.
- Optional: 2-3 discrete lid/floor reflections at 2, 5 and 9 ms, -6 to -10 dB relative to the direct sound. (unverified, from memory.)

**Hall tail (initial).**

| f (Hz) | 125 | 250 | 500 | 1000 | 2000 | 4000 | 8000 |
|---|---|---|---|---|---|---|---|
| T60 (s), generic mid-size hall | 2.0 | 1.8 | 1.7 | 1.6 | 1.45 | 1.2 | 0.8 |

- Provenance: concert halls typically 1.8-2.2 s at mid frequencies, occupied, with low-frequency values as high as ~2.4 s at 125 Hz when mid is 1.8 s [V snippet, generic room-acoustics site]. I shaded slightly lower for recital-sized halls.
- Both the values (unverified, from memory, hall-specific) and the shape are priors only. Because MAESTRO venues are unknown, learn T60(f) per year.
- Onset: tail starts ~20-40 ms after the direct sound, with a gentle build-up (first 30 ms echo density low).
- Level: initial direct-to-late energy ratio ~0 dB (range -6 to +6). Rationale: mics in concert recordings sit near or beyond the critical distance (a few m). (unverified, from memory.) Current code has D/R = +2.5 dB, but with a white spectrum.
- Length: >= 2.0-2.5 s at T60 = 1.7 s (to reach -60 dB the FIR must be >= T60; ideally 1.2 x T60). The current `ir_seconds = 1.0` is shorter than any concert hall.
- Sampling: 24 kHz limits the tail bandwidth to 12 kHz (fine for a first pass).
- Stereo: MAESTRO audio is stereo [V], and the code renders mono. When stereo is added, decorrelate the tail across channels (different noise seeds), and pan the direct sound by register (bass toward one side, treble toward the other; orientation depends on the mic setup, unknown).

---

## 2. Room / MAESTRO recording setup

| item | value | source | conf. |
|---|---|---|---|
| Audio format | uncompressed, CD quality or higher: 44.1-48 kHz, 16-bit PCM stereo | [V] MAESTRO paper/dataset page snippet | high |
| Size / span | over 200 h (v3: ~200 h; v1 172 h per one snippet), 2004-2018, 10 competition years in v3 (2004, 2006, 2008, 2009, 2011, 2013, 2014, 2015, 2017, 2018); v1 has nine | [V] snippets; year list from the repo config | high |
| MIDI/audio alignment | ~3 ms | [V] | high |
| MIDI content | key strikes, velocities, sustain pedal positions (paper text says sustain; the presence of CC66/CC67 in the files is (unverified, from memory) and should be checked in the data) | [V] partially | medium |
| Piano | Yamaha Disklavier (concert-quality acoustic grand with MIDI capture/playback); the exact model per year (CFIIIS/CFX/other) is not documented in my sources (unverified, from memory) | [V] general statement | medium |
| Venue | Competition finals are broadcast from Ted Mann Concert Hall (Univ. of Minnesota, Minneapolis) per a snippet, but early rounds ("virtual auditions") were recorded at ~10 locations around the world on Disklaviers, and the per-year venue for each MAESTRO recording is NOT documented in what I could read | [V] partial | low |
| Mic placement | not documented in what I could access; assume a stereo pair/hall mics plus room, distance unknown (metres) | - | low |

Consequences for modelling:

- The condition-specific IR must absorb differences in instrument, hall and mic, all mixed together. Initialising all years with the same generic prior and letting the IR learn is the only defensible approach.
- Realistic prior spread across years: mid-band T60 1.2-2.4 s; direct-to-reverberant -10 to +6 dB; different LF room modes.
- Ask (open question): does anything exist in the MAESTRO metadata, or in Magenta/Piano-e-Competition materials, on mic setup? I could not check.

---

## 3. Dampers

### 3a. Which top keys have no dampers

| item | value | source | conf. |
|---|---|---|---|
| Undamped top section (typical grand) | "the top octave and a bit more"; ~G6 and above undamped on many instruments, exact note varies by piano | [V] snippets (Mark Goodwin Pianos; forum posts) | low-medium |
| Common rule of thumb | top ~18-20 keys undamped, i.e. last damped key about E6-G6 (MIDI 88-91, k = 67-70) | (unverified, from memory) | low |
| Yamaha CFX exact last damped key | Not found in any source I could reach. Do not treat E6 (current code) as verified; the plausible range is E6-G6 | - | low |
| Current code | `HIGHEST_DAMPED_MIDI = 88` (E6, k = 67), 20 undamped keys | physics.py | inside the plausible range |

Recommendation:
- Keep 88 or move to 90 (F#6); the difference is at most 2-3 keys.
- Better: make the boundary data-driven. In MAESTRO, take non-pedalled notes and measure how the level falls after MIDI note-off. Notes that do not drop after release are undamped. That gives the exact key for the MAESTRO pianos and the damper rate per key directly.
- Make the last damped keys weakly damped (the top 4-6 dampers are small and light and act less strongly). A ramp such as has_damper = 1 for k <= 62 falling to 0.3 at k = 67 is more realistic than a step. (unverified, from memory)

### 3b. Damping time constants after release

I could not find any published measurement of damped-string T60 by register in searchable snippets (a general statement only: "vibrations rapidly decay when the damper contacts the string"). The values below are engineering priors. Use the `alpha_damp` convention in `_osc_bank`: amplitude decay exp(-alpha_d t) added to the free decay; T60 = 6.91/alpha_d.

| key k (note) | alpha_d1 at partial 1 (1/s) | implied damped T60 of fundamental (s) | conf. |
|---|---|---|---|
| 0 (A0) | 8 | 0.86 | low |
| 12 (A1) | 10 | 0.69 | low |
| 24 (A2) | 14 | 0.49 | low |
| 36 (A3) | 20 | 0.35 | low |
| 48 (A4) | 28 | 0.25 | low |
| 60 (A5) | 36 | 0.19 | low |
| 67 (E6, last damped) | 40 | 0.17 | low |

Point list (key_index, alpha_d1): `[(0,8),(12,10),(24,14),(36,20),(48,28),(60,36),(67,40)]`. Current code uses 25 /s for every key (T60 0.28 s). That is fine for the mid-range but ~3x too fast for the bass, where released bass notes audibly linger, and ~1.5x too slow for the top. (unverified, from memory / judgement)

Frequency dependence, per partial [D + judgement]:
- A felt damper is roughly a resistive termination at position x_d from a string end. The energy removed from mode n scales with the local mode amplitude, so alpha_d,n ~ alpha_inf * sin^2(n pi x_d / L), averaged over the finite felt width.
- For x_d/L ~ 0.03-0.1 (bass, damper near an end) the fundamental is damped least and alpha_d grows roughly as n^2 until n ~ L/(2 x_d) ~ 5-15, then saturates (averaging over the felt width smooths the sin^2 nodes).
- Practical form: alpha_d,n = alpha_d1 * n^p with p ~ 0.5-1 for n <= ~8, saturating at ~3-6 x alpha_d1.
- Current code: (f_n/f_1)^0.3, i.e. a much weaker rise. Suggest the prior tilt 0.3 -> 0.6 and widen the learnable range to +-0.5.
- Consequence: after release the fundamental of bass notes (partial 1-2) audibly rings on for a few tenths of a second after the upper partials have gone. (unverified, from memory)
- Damper engagement is not instantaneous. The felt closes on the string over ~5-20 ms, and the audible cut-off is not a step. The code's frame-rate linear interpolation (5 ms) already provides a ramp.

### 3c. Damper lift timing relative to key motion

| item | value | source | conf. |
|---|---|---|---|
| Time key start -> key bottom | ~25 ms at forte (hammer ~5 m/s) up to ~160 ms at piano (~1 m/s) | [V] Askenfelt & Jansson | high |
| Hammer/string contact vs key bottom | hammer contacts 12 ms BEFORE key bottom at piano (1 m/s); 3 ms AFTER key bottom at forte (5 m/s) | [V] | high |
| Damper leaves string | when the key has moved ~1/4-1/3 of its dip (3-4 mm of ~10 mm), so before the hammer arrives at the string in all cases | (unverified, from memory; typical regulation spec) | low |
| Damper returns to string on release | as the key returns through roughly the same height (~3-4 mm above rest); delay after the MIDI note-off (which comes from a key-position sensor) is unknown, plausible range -20 to +60 ms | (unverified, from memory / reasoning) | low |

Recommendation:
- Add a learnable per-condition damper delay `dt_damp` (init +15 ms, bounds -30 to +80 ms) and, ideally, velocity-dependent: slow releases give a longer delay. Apply it by shifting `key_down`'s release time.
- Damper felt on the string: smooth ramp 5-20 ms (bass slower).
- When the sustain pedal rises, the damper rail carries all dampers up together; on release the dampers fall under gravity within ~10-30 ms (asymmetric lag: attack ~10 ms, release ~20-40 ms). (unverified, from memory)

---

## 4. Sustain pedal

### 4a. CC64 -> damper lift

| item | value | source | conf. |
|---|---|---|---|
| CC64 range | continuous 0-127; "127 = dampers fully off the strings" means the dampers are completely clear before the pedal reaches the bottom of its travel | [V] Sweetwater/forum-style snippet | medium |
| Half-pedal / part-pedal | continuous control. Lehtonen et al. 2009 identify three phases in part-pedaling: initial free vibration, damper-string interaction, final free vibration; part-pedaling leaves a prolonged soft tail of several seconds after release | [V] | high |
| Where dampers start to lift | roughly when the pedal has travelled ~20-35 % of the stroke, fully clear at ~55-75 %. Some lost motion in the linkage | (unverified, from memory) | low |
| Equivalent CC64 thresholds | lift starts CC ~25-45 (0.2-0.35), completely off CC ~70-95 (0.55-0.75). Midpoint ~0.4-0.5, 10-90 % span ~0.25 | (unverified, from memory) | low |
| Current code | logistic, theta 0.45, width 0.06 (10-90 % from 0.32 to 0.58) | physics.py | close to the recollection |
| Per-key spread | regulation spread, and bass vs treble dampers differ; suggest per-key jitter of theta +-0.03-0.05 (a learnable per-key offset with smoothness) | (unverified, from memory) | low |
| Hysteresis | the lift-on threshold (pedal down) is higher than the lift-off threshold (pedal up) by ~5-10 % of the stroke (friction, felt) | (unverified, from memory) | low |
| Damping vs lift | Damper felt pressure, not only gap. The extra decay rate should fall faster than linearly as the damper rises: alpha_d,eff = alpha_d * (1 - lift)^p with p ~ 2-3 (felt compression is nonlinear; hammer-felt exponents 2.5-3.5 by analogy) | (unverified, from memory / analogy) | low |

Implication for code: `engagement = (1 - key_down)(1 - lift)` is linear in (1 - lift). Use (1 - lift)^2.5 (or pass the exponent as a learnable parameter, init 2.5, bounds 1-4), and per-key theta jitter.

Disklavier and MAESTRO note: MAESTRO CC64 is the raw sensor value, not a calibrated damper height. Also note that Lehtonen 2009 measured damper height with a dial gauge against pedal position on a real piano; the Disklavier's own CC64 to damper-height map is instrument-specific and, as far as I could find, not published (a Yamaha patent on identifying the "half-pedal region" exists, but I could not open it).

### 4b. Whole-instrument resonance when all dampers lift

From Lehtonen et al. 2007 [V snippet]:
- The sustain pedal lengthens the decay time of partials in the middle range of the keyboard but not for bass and treble tones.
- The string register response was studied by removing partials from recorded tones. When the register is free to vibrate, the amount of sympathetic vibration increases.
- Their synthesis used 12 string models corresponding to the lowest tones (as I read the snippet: only the bass strings matter most for the resonance halo; a lower bound on how many resonators are needed).
- The audible effect is a reverb-like halo whose spectrum sits in the fundamentals/low partials (partials 1-6) of the whole register (from memory, qualitative).

Values (all (unverified, from memory) except where stated):
- Overall level of the halo relative to the struck note: -20 to -35 dB at the note's own low partials; more at partials that coincide with the struck note's partials. It builds up over 0.1-1 s and decays with the free string decay (seconds).
- With repeated notes/chords the halo raises the apparent noise floor by +2 to +6 dB in the 100-1000 Hz range.
- After pedal release, the halo collapses in ~0.3 s (damper rates of 3b).

### 4c. Pedal noise

| item | value | conf. |
|---|---|---|
| Pedal press: thump/rustle of pedal mechanism and damper rail moving (low-frequency, 50-300 Hz, plus soft felt-on-metal sizzle) | ~ -35 to -45 dB rel. an mf note, 30-100 ms, amplitude ~ pedal speed | (unverified, from memory) low |
| Pedal release: dampers falling onto the strings and the rail hitting its stop ("damper slap") | ~ -30 to -40 dB rel. mf note, broadband 100 Hz-3 kHz, 20-80 ms | (unverified, from memory) low |
| Modartt (Pianoteq) forum discussion mentions a dedicated pedal-noise model that responds to half-pedaling (title of thread only, no numbers) | [V] title | - |
| NoiseBank: no pedal noise at all | current code | - |

Recommendation: add a "pedal event" generator. Trigger when lift crosses 0.5 up or down, amplitude proportional to |d lift/dt| (clipped), low-pass shaped (corner ~800 Hz up, ~1.5 kHz down).

---

## 5. Sympathetic resonance

| item | value | source | conf. |
|---|---|---|---|
| Coupling path | strings are coupled only through the bridge (and soundboard). A driven string mode acts as a resonator whose bandwidth is alpha/pi | Weinreich 1977 [V]; standard | high |
| String-mode bandwidth | with alpha ~ 0.5-3 /s, bandwidth alpha/pi ~ 0.15-1 Hz, so only partials that coincide within about 1 Hz respond | [D] | medium |
| Peak transfer (coupled-mode theory, [D]) | da_s/dt = -alpha_s a_s; da_r/dt = -(alpha_r + i Delta) a_r + g a_s with g ~ sqrt(kappa_s kappa_r) and kappa the bridge-loss rate. For exactly equal frequencies and equal decay rates the driven response peaks at t = 1/alpha with ratio g/(e alpha). With kappa/alpha ~ 0.3-0.7 (prompt mode) that is about -10 to -15 dB. Off resonance by Delta it drops as alpha/\|Delta\|: 2 pi x 1 Hz detuning against alpha = 2 /s gives another ~-10 dB. Typical audible coincidences (octave, twelfth, unison strings) land at roughly -25 to -40 dB re the struck partial | [D]; the -25 to -40 dB range is also (unverified, from memory) | low-medium |
| Which partials respond | those of undamped strings that coincide (within ~1 Hz) with a partial of the source. Octave and 2nd-octave strings respond best (most coincidences), then twelfths and double octaves. The lowest 4 partials of each free string are the strongest resonances | Lehtonen 2007 [V] ("register free -> more sympathetic vibration") | medium |
| Frequency-coincidence dependence | matching is set by tuning to ~1 Hz, so the effect depends on stretch tuning and inharmonicity of the other agent's parameters | [D] | medium |
| Decay | the responding string rings at the prompt (in-phase) rate, since the bridge drives the in-phase mode; in the code `key_modes["alpha"][...,0]` is the prompt mode, which is physically correct | Weinreich [V] + [D] | medium-high |
| Sustain pedal effect on decay | mid-keyboard partials decay more slowly with pedal, bass and treble do not | [V] Lehtonen 2007 | high |
| Undamped top section (without pedal) | with pedal up only the top ~18-20 strings (last damped key E6-G6) respond, so a low chord makes the top strings shimmer softly at coinciding high partials of the low notes (their 8th-24th partials). Typical level -35 to -50 dB re struck note | (unverified, from memory) | low |
| Duplex scaling / aliquot | non-speaking string segments between bridge and hitch pin (or aliquot string on some makes) tuned to a harmonic of the string (typically 2nd-4th partial of the speaking length, differing by register) ring sympathetically and add "shimmer" at high partials; effect small, ~ -30 to -40 dB. Whether the Yamaha CFX uses duplex scaling I could not verify | (unverified, from memory) | low |
| Current bank | 88 keys x 4 partials; steady-state gain at resonance exp(log_gain) = 0.05 (-26 dB); gin = gain * alpha/sr; driven by the summed dry string signal; uses the key's own prompt-mode alpha and damper state | synth.py | - |

Review of the current SympatheticBank (in my scope):
- Unit gain 0.05 is a defensible order of magnitude. The bank does one-pole complex resonators. With a decaying drive of equal alpha the peak response is gain * alpha * t * e^{-alpha t}, i.e. gain * alpha / e. Because gin carries an alpha factor, the response of the *slowly decaying bass* strings is very small: at k = 0 (alpha ~ 0.25) the peak is 0.05 * 0.25 / 2.7 = -47 dB, versus -29 dB at alpha = 2. By coupled-mode theory the ratio is g/(e alpha) ~ kappa/(e alpha), which depends on kappa/alpha (a nearly register-independent fraction of the decay that is bridge loss), not on alpha. So the gain should NOT scale with alpha; use gin = gain_k * (alpha/sr) with gain_k ~ kappa_k/alpha_k ~ 0.15-0.4 for prompt-dominated keys and less for bass. Simplest fix: initialise `log_gain` per key at log(0.15) and make gin independent of alpha.
- The struck note's own resonator is included. It sees its own dry signal. Physically this is just the note's own decay, so it double counts: exclude the source key's own partials from the bank (or accept a small +0.4 dB).
- Only 4 partials per key: fine for the pedal-halo level (Lehtonen 2007 needed the lowest 12 strings in a synthesis model [V]), but it misses the high-partial shimmer of the top section (partials 8-24 of undamped top strings). If cost allows, take partials 1-4 for k <= 50 and 1-8 for the top 20 keys.
- The drive should be bridge force, and force ~ slope of the string, so the bridge signal has a +6 dB/oct tilt versus displacement (see item 10). Whether `strings` is a displacement, velocity or force analogue is not specified in physics.py.

---

## 6. Una corda (soft pedal, CC67)

| item | value | source | conf. |
|---|---|---|---|
| Mechanism (grand) | soft pedal shifts the whole action (keyboard + hammers) sideways to the right; the hammer misses the leftmost string, striking 2 of 3 (trichord) or 1 of 2 (bichord) strings. The remaining strings are struck with a less-used felt region; double-strung and wound strings (bichords / monochords) are struck off-centre | [V] Tan et al. ICA 2019 abstract; Wikipedia snippet; Southampton una-corda PDF abstract | high |
| Lateral shift | ~5-6 mm on a concert grand ("about a quarter inch"), about one string spacing | (unverified, from memory) | low-medium |
| Level change | "softens the sound by as much as one dynamic level" (~ -3 to -6 dB) | [V] Wikipedia snippet (qualitative) | medium |
| Level change, trichord [D] | strike vector (1,1,0) vs (1,1,1): in-phase mode amplitude 2/sqrt3 vs 3/sqrt3 = 0.667, i.e. -3.5 dB in the prompt sound; the missing string does not add to the force on the bridge | [D] | medium |
| Level change, bichord (off-centre) | strike vector (1, w) with w ~ 0.5-0.8: prompt amplitude (1 + w)/2 of full, i.e. -0.9 to -3.5 dB | [D]; w is (unverified, from memory) | low |
| Aftersound [D] | with strike vector (1,1,0), the amplitude of the out-of-phase eigenmode e2 = (1,1,-2)/sqrt6 is 0.816 versus 0 for a normal trichord strike (e1 = (1,-1,0)/sqrt2 gets 0). Aftersound / prompt amplitude = 0.816/1.155 = 0.71 (-3 dB) versus roughly -15 to -25 dB in a normal strike (there it is nonzero only because of detuning and strike mismatch). Bichord with one string struck: prompt and aftersound modes have equal amplitude (ratio 1.0, 0 dB) | [D] | medium (geometry), low (real pianos) |
| Time behaviour (una corda) | the unstruck string starts silent and its level rises as power flows soundboard -> string; the struck string has a longer aftersound (more even decay, less pronounced beating in spectrograms) | [V] Southampton abstract | high (qualitative) |
| Timbre | darker: fresh, softer felt region -> lower hammer stiffness -> lower cutoff. Hammer-agent domain; guess fc x 0.6-0.8 | qualitative [V]; number (unverified, from memory) | low |
| Current code | `soft_gain_db = -3`, `soft_log_fc = log(0.75)`, `soft_log_after = log(1.5)` (aftersound amplitude x1.5), applied to all keys equally | physics.py | gain, fc plausible; after is too weak |

Recommendations:
- soft_gain_db: -3.5 dB for k in trichord range (k >= 28), -2 dB for bichords (10 <= k < 28), 0 dB for monochords (k < 10, where una corda only shifts the felt region; some grands do not shift the bass at all in the sense that the string is struck off-centre). Make it per-register (three learnable values, init as here).
- soft_log_after: replace the constant 1.5 by register values. Trichords: aftersound/prompt amplitude ratio init ~0.7 (base ratio 0.2 -> multiply by ~3.5, i.e. log 3.5 = 1.25); bichords: ~1.0 (multiply by 5, log 1.6); monochords: 1.0. Keep bounds wide.
- Also let the aftersound decay be *slower* under una corda (the struck string's transverse component has a longer aftersound [V]); multiply aftersound alpha by ~0.8 when soft is on (unverified, from memory).
- soft_log_fc: 0.75 is fine (0.6-0.8).
- Soft pedal depression is not instantaneous: the action shift takes ~50-100 ms. Treat CC67 as a continuous 0-1 blend (the code does: `soft` is read at note onset).
- The reported effect is stronger on early pianos than on modern ones (qualitative, [V] Wikipedia snippet); it is modest on a concert grand.

---

## 7. Sostenuto (CC66): how to model it correctly

Mechanics (standard; (unverified, from memory) but very well established):
- The sostenuto rod catches the dampers of the keys that are **held down at the moment the pedal is pressed**. Those dampers stay off the strings until the sostenuto pedal is released, whatever the keys do afterwards.
- Keys played after the pedal is engaged are not caught and are damped normally. The sustain pedal is independent, and either one alone is enough to keep a damper up.
- Sostenuto acts only on keys that have dampers (nothing for the undamped top keys). It is essentially binary: it engages once the pedal passes a catch position, ~50 % of travel (a threshold at CC66 ~ 64).

Implementation (per key k, at frame t):

    S_press(t)   = 1 if CC66 crosses 0.5 upward at t
    latch_k(t)   = 1 while (CC66 > 0.5) since the last upward crossing t_p, if key_down_k(t_p) = 1 else 0
                   (latch is set once at the crossing time and cleared when CC66 falls below ~0.4)
    engagement_k = (1 - key_down_k) * (1 - lift_sus)^p * (1 - latch_k)

- If a latched key is re-struck the latch persists (its damper never re-touches the string).
- Notes struck when the pedal is already down but which were never down at t_p get latch 0.
- Implementation in the current code: `key_rolls` already builds per-key `key_down[B,88,F]`, so latch_k can be computed as a cumulative "hold" mask, `latch = cummax over frames after crossing`, with `key_down` sampled at the crossing frame. The README lists this as a TODO.
- Whether MAESTRO's CC66 is populated must be checked in the data.

---

## 8. Mechanical noises

### 8a. Timing and levels

All timing numbers are from Askenfelt & Jansson (KTH lecture pages, JASA 1990-93) and Askenfelt SMAC 1993, via snippets [V]. Level and spectrum numbers are not in the snippets I saw and are (unverified, from memory) unless stated.

| event | timing relative to hammer-string contact (= MIDI note-on, roughly) | level rel. note | spectrum | duration | conf. |
|---|---|---|---|---|---|
| Finger-key touch ("touch precursor", starts the key motion) | precedes the string tone by ~20-30 ms at mf/f (precursor duration 20-25 ms); key travel time to key bottom 25 ms (forte) to 160 ms (piano) [V] | very weak, -40 to -50 dB rel tone (much weaker than the string precursor) [V "much weaker" only] | low-frequency thud radiated by the key bed, < ~500 Hz | ~10-20 ms | timing high, level low |
| Key-bottom impact ("thump", key on the stop rail/keybed) | at piano (1 m/s) the thump comes 12 ms AFTER hammer contact; at forte (5 m/s) 3 ms BEFORE. Linear interpolation: dt_kb = +12 - 3.75 (v_h - 1) ms | roughly -25 to -35 dB rel peak tone at ff; relatively louder at pp (-10 to -20 dB) because it scales weaker with velocity than the tone | keybed/case, broadband peak 100-500 Hz, falls ~12 dB/oct above ~600 Hz to below noise at 3-4 kHz | ~30-60 ms (T60), much of it low frequency | timing high [V], rest low |
| Hammer knock through the case/soundboard (the "soundboard knock") | ~0-2 ms after contact (bridge moves before the transverse waves have reached it via longitudinal/precursor waves; precursor arrival within ~0.3-0.5 ms for the longitudinal wave on a ~1-2 m string, ~5000 m/s) | -30 to -45 dB | mainly < 200 Hz thud plus broadband click (the longitudinal "zing" ~ 1-5 kHz) | 10-30 ms | low |
| Key release: key returns on the front rail/felt, backcheck lets go | at note-off +0-30 ms | -40 to -50 dB rel note (ff) | 100-600 Hz | 10-30 ms | low |
| Damper touchdown on the string ("damper noise") | at note-off + dt_damp | -35 to -45 dB rel the note just released (partly masked by the tone) | mid-band 200 Hz-2 kHz | 10-40 ms | low |
| Pedal | see 4c | -30 to -45 dB | LF + broadband | 30-100 ms | low |

Sources for the structure: Askenfelt & Jansson pages [V]; Askenfelt SMAC 1993 [V title]; Chabassier et al. 2013 note that their simulation reproduces phantom partials and precursors from string nonlinearity [V]; Conklin 1996 Part III on phantom partials [V]. I could not find level numbers for the key-bed thump; the -25 to -35 dB figure is a plausible guess to be learned.

### 8b. The current NoiseBank against these numbers

- `knock_log_tau = log 0.02`: env is on *power* exp(-2 d/tau), so the amplitude is exp(-d/tau) and T60_amp = 6.9 tau = 138 ms. Real thump T60 is ~30-60 ms, so the init is ~3x too long. Use tau = 6-8 ms (T60 ~ 45 ms).
- `release_log_tau = log 0.04`: T60 = 276 ms. Damper touchdown/key release is ~20-40 ms. Use tau = 4-6 ms.
- `knock` bands start at -3.0 with a slope of -1.5 (log-amplitude) over 32 log-spaced bands from 40 Hz to 12 kHz (~8.2 octaves). That is only ~13 dB total across the range, i.e. ~-1.6 dB/oct, almost white noise. Real thump is dark: use a roughly 2-pole low-pass at 500-800 Hz, so -12 dB/oct above the corner and a 40-60 Hz high-pass. In the current 32-band table (band centres log-spaced from 40 Hz), initial values in log-amplitude relative to the first band: 0 up to 600 Hz (bands 0-~13), then falling by 1.39 nats per octave (12 dB/oct = factor 4 in amplitude), reaching about -6 nats (-52 dB) at 12 kHz. The release noise should be lower-passed at ~1.5 kHz.
- Onset of the knock at `onset` is right for the hammer/soundboard knock; add the key-bottom thump at onset + dt_kb(v) with dt_kb = +12 ms (pp) to -3 ms (ff), and the touch precursor at onset - (25 to 150) ms with -40 dB weight.
- Velocity dependence in the current code: amplitude ~ exp(2(u - 0.6)) over the velocity range, i.e. ~16 dB from u = 0.1 to 1.0, versus 36 dB for the tone (40 dB/unit u). So the noise-to-tone ratio falls by ~20 dB with velocity, which matches the qualitative statement that touch noise is relatively louder at pp. Fine.
- Add a pedal-noise generator (4c).
- Level calibration: rather than absolute numbers, at init render one mid-range note and scale `knock` so the integrated energy in the first 60 ms is ~ -25 dB (u = 1) / -12 dB (u = 0.2) relative to the tone's energy in the same window (low confidence, learnable).

---

## 9. Yamaha Disklavier specifics

| item | value | source | conf. |
|---|---|---|---|
| Onset timing accuracy | Disklavier recording: onset within about +-12 ms (better than its reproduction); reproduction -20 to +30 ms (larger timing errors for soft tones). Bosendorfer SE better | [V] Goebl & Bresin 2003 snippet | medium-high |
| MAESTRO alignment | ~3 ms audio-to-MIDI after the authors' alignment (per-piece; residual velocity-dependent timing, if any, is not known) | [V] MAESTRO | high |
| Dynamic range of reproduction | Disklavier reproduction flattens the extremes (soft tones too loud, loud tones too soft); hammer velocities above ~3.5 m/s cannot be reproduced by the solenoids (this concerns playback; MAESTRO is human-played recordings, so the recorded velocities are not clipped, but this shows the calibration table is not linear) | [V] | high |
| MIDI velocity vs level | Different pitches give different velocity-to-SPL curves; higher pitches are louder at the same MIDI velocity (so the code's per-key `raw_vel_slope`/`gain_db` are warranted). Roughly linear in dB over the mid velocity range | [V] Goebl & Bresin snippet | medium |
| MIDI velocity vs hammer velocity | Rough prior: v_h[m/s] ~ 5.5 (vel/127)^1.5, giving vel 20 -> 0.34, 64 -> 2.0, 100 -> 3.8, 127 -> 5.5 m/s. Fits "hammer velocity ~5 m/s at forte" and "1 m/s at piano" [V Askenfelt]; the exponent is my guess | (unverified, from memory) | low |
| Hammer velocity to key velocity | hammer ~5x key velocity; peak key velocity in forte seldom exceeds 1 m/s | [V] Askenfelt & Jansson | high |
| Note-on stamp | recorded when the hammer/key sensors trip; onset ~ hammer-string contact plus a few ms sensor latency. The key needs 25-160 ms from rest to key bottom depending on velocity, so if the sensor trips early in the key travel, a velocity-dependent residual (up to tens of ms at pp) is possible | reasoning (unverified, from memory) | low |
| MIDI note-off vs damper contact | note-off is from the key sensor as the key returns; the damper touches the string when the key is a few mm from rest. Sign and size of the offset are unknown; plausible -20 to +60 ms (see 3c) | - | low |
| CC64 meaning | continuous pedal-position sensor value 0-127 (half-pedaling is recorded); commonly interpreted on/off at 64 by transcription work, which discards half-pedal information | [V] | high (that it is continuous) |
| Soft/sostenuto in MAESTRO | presence of CC67/CC66 in the MIDI files: (unverified, from memory), must be checked | - | low |

Recommendations:
- Add a learnable per-condition onset offset with a velocity term: onset' = onset + d0 + d1 (1 - u), d0 in [-8, +8] ms, d1 in [0, 20] ms, init 0.
- Keep `velocity/127` as `u` but add a per-condition velocity-curve exponent. The current `cond_vel_slope` (+-10 dB/unit) roughly does this.

---

## 10. Other things outside strings/hammers that literature says matter

1. **String -> bridge force is the slope, not the displacement.** F_bridge = T dy/dx at the bridge, so relative to displacement the bridge-force spectrum has a factor n (+6 dB/oct). Plate velocity = Y_b F. Radiated pressure ~ velocity x f (for a small source, +6 dB/oct) up to ~1 kHz, then ~ sqrt(sigma) x velocity. `physics.py` does not say whether `a_n` is displacement, velocity or force. For a hammer-velocity excitation the partial displacement goes as sin(n pi x0)/n, the bridge force as sin(n pi x0) (flat), so the overall spectral tilt is then determined by the hammer lowpass only. Check that the chosen `hammer` rolloff of 1.5 (=-9 dB/oct above fc) plus this tilt produces a realistic slope. Missing +6 dB/oct or wrong 1/n here changes the timbre from bright/thin to dull.
2. **Longitudinal precursor and phantom partials** (Conklin 1996 Part III; Chabassier et al. 2013 [V]: phantom partials and precursors arise from string nonlinearity) give the "attack bloom" that a linear model lacks. Phantom partials at f = f_a + f_b (and 2 f_a - f_b) are largest at ff in bass/tenor. README already lists it as TODO.
3. **Case / rim / lid.** Rim and cavity resonances under the board add low-frequency colouring. The lid is a reflector, and the piano radiates directionally. Hard to prior; leave to the body FIR with register dependence (see below).
4. **Register-dependent body response.** Bass strings couple through the bass bridge (stiffer, lower mobility, different modes); tenor/treble through the long bridge. A single per-condition FIR cannot capture that the filter seen by C2 differs from C6. First-order fix: two body FIRs (bass bridge k < ~30, treble bridge k >= 30) or a low-order per-key EQ on `dry` before the IR.
5. **Stereo and radiation directivity** (MAESTRO is stereo; the code is mono at 24 kHz): pan by register; decorrelated tail. Also the "board radiates more at HF toward the lid" effect (low priority).
6. **Sample-rate cap**: 24 kHz mono drops the 12-20 kHz "air" and hammer noise; minor for the toy-guitar issue.
7. **Onset alignment and pre-delay** (see section 9 and 1d).
8. Related prior art on differentiable/learned piano models to cite: DDSP-Piano (Renault, Roebel & Mignot, cited in the README), and the Frontiers 2023 paper "Physics-informed differentiable method for piano modeling" [V title only; I did not read it].

---

## (b) Recommended prior values as (key_index, value) point lists

| parameter | point list | notes |
|---|---|---|
| alpha_d1 (damper decay at partial 1, 1/s) | `[(0,8),(12,10),(24,14),(36,20),(48,28),(60,36),(67,40)]` | replaces flat 25; T60 from 0.86 s to 0.17 s; conf. low |
| damper partial exponent p | `[(0,0.7),(30,0.6),(60,0.5)]` | code's 0.3 is too small; saturate at 3-6x alpha_d1 |
| damper strength ramp (has_damper) | `[(0,1),(62,1),(67,0.3),(68,0)]` (soft edge) or keep the step at k = 67-70 | check with MAESTRO data |
| soft pedal gain (dB) | `[(0,0),(9,0),(10,-2),(27,-2),(28,-3.5),(87,-3.5)]` | [D]; conf. medium-low |
| soft pedal aftersound multiplier | `[(0,1),(9,1),(10,5),(27,5),(28,3.5),(87,3.5)]` | [D]; keep wide bounds |
| soft pedal fc multiplier | constant 0.7 | conf. low |
| sympathetic gain (linear, at resonance, independent of alpha) | `[(0,0.08),(12,0.12),(30,0.15),(60,0.15),(67,0.15),(87,0.10)]` | conf. low; keep learnable |
| sustain-pedal lift midpoint theta | 0.42 (+-0.04 per-key jitter) | code 0.45; conf. low |
| sustain-pedal width | 0.05-0.08 | code 0.06 fine |
| damping-vs-lift exponent | 2.5 | replaces linear (1 - lift) |
| body FIR magnitude (Hz, dB) | `[(20,-40),(30,-30),(40,-20),(55,-10),(70,-4),(100,0),(1000,0),(2000,-1),(4000,-3),(8000,-7),(11000,-12)]` | not per-key; conf. low-medium |
| hall T60 (Hz, s) | `[(125,2.0),(250,1.8),(500,1.7),(1000,1.6),(2000,1.45),(4000,1.2),(8000,0.8)]` | learnable per year; conf. low |
| knock/thump amplitude tau (s) | `[(0,0.010),(40,0.007),(87,0.005)]` | current 0.02 constant; conf. low |
| release noise tau (s) | 0.005 | current 0.04 |
| thump timing dt_kb (ms), by hammer velocity v_h (m/s) | `[(1,+12),(5,-3)]` (interpolate linearly) | [V] |

---

## (c) Formulas / modelling recommendations for the code

1. **Body + room.** Replace `0.02*randn*exp(-6.9 t/0.8)` (white tail, T60 0.8 s, delta) by:
   - `h = delay(3-6 ms) * h_body + gain_tail * h_hall`, with `h_body` from the modal recipe (1d) and `h_hall` = noise shaped by a 7-band T60(f) table and an onset ramp.
   - Length 2.0-2.5 s (48-60k taps at 24 kHz) or a hybrid short FIR + parametric tail.
   - Keep `ir` learnable but add L2 regularisation toward the prior envelope, or parameterise the tail envelope only.
2. **Body magnitude at init**: HP at 60-70 Hz (12-18 dB/oct), 0 dB plateau 100 Hz-2 kHz, gentle roll-off above.
3. **Bridge-loss consistency check**: alpha_bridge,n = 2 Z_s Re Y_b(f_n) f_1 (times N for the in-phase mode). Use it to sanity-check `prior_log_b1` and the `prompt` factor.
4. **Damper**: `alpha_d,n = alpha_d1(k) * min(n^p, n_sat^p)` (p ~ 0.6, n_sat ~ 6); effective decay uses `(1 - lift)^p_lift` with p_lift ~ 2.5; per-condition delay `dt_damp` init 15 ms; per-key `theta` jitter.
5. **Sostenuto latch**: as in section 7 (per-key latch set at the pedal crossing, keys held down at that instant).
6. **Una corda**: per-register soft gain/aftersound (section 6), not a global constant.
7. **Sympathetic bank**: gin independent of alpha, drop the source key's own resonator (or accept), more partials for the top section.
8. **NoiseBank**: tau 5-8 ms, dark spectrum (low-pass 500-800 Hz, 12 dB/oct), key-bottom event at onset + dt_kb(v), touch precursor, pedal-event noise, release noise low-passed at ~1.5 kHz.
9. **Data-driven calibration before training** (cheap, high value):
   - measure the damper boundary key and per-key damper rates from non-pedalled MAESTRO notes;
   - estimate pedal thresholds from notes released under partial CC64 values;
   - estimate per-year T60(f) of the hall from the decay after loud staccato chords with pedal up;
   - fit the initial body EQ from long-term average spectra vs the synthesised dry signal.

---

## (d) Ranked "toy guitar" diagnosis (outside strings/hammers)

Rank by plausibility of causing the character, with the fix.

1. **Near-delta impulse response, no soundboard body, no hall (highest confidence it is the main cause).** `out["audio"] = fft_convolve(dry, ir)` with `ir[0] = 1` plus a 0.02-amplitude white decaying tail. The dry output is a bank of undamped-looking harmonic decaying sinusoids in which each partial is only shaped by the hammer/comb model. There is no body formant structure, no 60-300 Hz resonant weight, no diffuse high-frequency smear and no hall. Listeners hear this as a plucked string with no body: a toy guitar/harp. The tail is white with T60 0.8 s (energy 0.56 of the direct, D/R = +2.5 dB), which adds a hissy "spring-reverb" wash but not a hall. Fix: 1d (modal body + frequency-dependent hall T60 + length >= 2 s).
2. **No LF roll-off / radiation filter.** With no high-pass at the first board mode (~60-70 Hz) the bass fundamentals A0-C2 come out at full strength, where a real piano radiates them 15-30 dB below partials 3-8. Result: unnatural, "plucky", sub-heavy bass. Fix: HP 60-70 Hz at 12-18 dB/oct in the body target.
3. **Noise bank spectra and time constants.** Nearly white knock (~-1.6 dB/oct) with T60 ~140 ms (release T60 ~280 ms), versus a dark, ~45 ms thump. This makes every note onset a hissy "tick" like a pick/plastic-key click rather than a thud from the key bed. Fix: 8b.
4. **Unshaped partial spectrum (displacement/force tilt).** Whether the string signal is treated as displacement or bridge force changes tilt by 6 dB/oct; and no radiation efficiency, i.e. no gentle transition at ~1.1-1.6 kHz. A too-bright or too-dark balance reads as toy. Fix: item 10.1 (choose the convention explicitly; put the compensating tilt in the body target).
5. **Damper decay too uniform.** 25 /s at every key (T60 0.28 s) chops the bass off ~3x too fast and lets treble notes ring ~1.5x too long after release, and the weak partial tilt (0.3) leaves high partials too long. This gives the "staccato synth" feel of bass notes and takes the sustain/release contrast out of the piano. Fix: section 3b.
6. **Sympathetic bank and pedal.** Gain scaling with alpha silences bass strings' resonance (-47 dB); only 4 partials, no top-section shimmer, own-note double counting. The pedal "halo" that makes a pedalled piano sound big is therefore weak. Fix: section 5.
7. **Una corda under-modelled** (aftersound x1.5 instead of ~x3.5-5): affects only passages with soft pedal; small global effect.
8. **24 kHz mono.** Limits the air/space and stereo image; secondary.

---

## (e) Open questions

1. Exact last damped key on the MAESTRO pianos (Yamaha CFX/CFIIIS/other by year). Measure from data (3a).
2. Which model of Disklavier and which hall per year, and mic distance. Ask whether any Piano-e-Competition or Magenta metadata documents this.
3. Numerical bridge mobility of a *grand* (all published: Giordano is an upright at C4). Ege & Boutillon's grand data are in figures I could not open. If access opens up, read arXiv 1210.5688 and 1305.3057, and Giordano 1998 directly, to replace the 1000-2500 kg/s guess and the bass/treble factor.
4. Pedal calibration: how does Disklavier CC64 map to damper height (thresholds, hysteresis)? A Yamaha patent on identifying the half-pedal region exists (US 8,933,316 by number from a search title; contents not read).
5. Level and spectrum of key-bed thump vs tone: Askenfelt & Jansson and SMAC 1993 have the figures but I could not open them.
6. Are CC66 and CC67 present in MAESTRO MIDI? If not, sostenuto and una corda are moot for training and only useful for rendering.
7. Is the note-off/damper offset positive or negative? A cheap test: fit `dt_damp` per year on data.
8. Does `strings` represent displacement, velocity or force (item 10.1)? This decides the sign and size of the tilt correction inside the IR.
9. Duplex scaling on the Yamaha CFX: not verified.
10. Whether to fold a register-dependent body (bass vs treble bridge) into version 1 or keep one FIR per year.
