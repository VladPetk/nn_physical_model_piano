# Physical parameters: requirements (v3)

This is the spec the physics prior (`pianonn/physics.py`, `pianonn/synth.py`, `pianonn/room.py`)
must meet before any MAESTRO fitting.

v3 fixes what the second review ([`reviews/review_2_calibration.md`](reviews/review_2_calibration.md)) showed was wrong in v2:
- The v2 "prompt T60" metric misread the recordings. Real mid-range notes lose about 20 dB in their first
  second, three strings' worth of bridge loss, as the literature predicted; v2's model lost 5 dB. Decays are
  now fitted to measured decay profiles.
- Partials were picked by list position, not by number.
- The stereo mics were summed to mono, which comb-filters.
- The treble inharmonicity search locked onto distortion components.
- A NaN gradient for soft treble notes.

v2, and this version, replace most of v1's from-memory values with values taken from two kinds of source:

- **Literature, read.** The KTH *Five Lectures on the Acoustics of the Piano* (Conklin, Askenfelt &
  Jansson, Hall, Weinreich, Wogram), arXiv papers (Ege & Boutillon et al.) and Zenodo papers
  (DDSP-Piano, Bank et al., Bader & Plath). See [`literature/round2_sourced.md`](literature/round2_sourced.md).
  The round-1 reviews, which were memory-based, are kept in [`literature/`](literature/) for history.
- **Measurement.** 260 isolated notes (pp/mf/ff) of the University of Iowa Steinway model B,
  close-miked, analysed with `pianonn/calibration.py`. The model's rendered notes go through the
  same code. See [`calibration_iowa.md`](calibration_iowa.md).

Evidence tags:
- **L**: read in a source (pointer given).
- **I**: measured on the Iowa recordings.
- **D**: derived from stated physics.
- **M**: memory or engineering judgement (still unsourced).

Everything is a *prior*. Each value gets a bounded learned offset that MAESTRO fitting will
refine. The Iowa piano is a 6'11" Steinway B, not the MAESTRO Yamaha Disklaviers. So
per-instrument quantities are priors to be re-fitted per MAESTRO year (inharmonicity in the
bass, stretch, decays), while mechanisms and shapes transfer.

Key index `k = MIDI - 21`. Landmarks: A0 = 0, C1 = 3, C2 = 15, C3 = 27, C4 = 39, A4 = 48,
C5 = 51, C6 = 63, C7 = 75, C8 = 87. Point lists are `key_curve` inputs, `(k, value)`.

## 1. Strings

| parameter | prior | evidence |
|---|---|---|
| Inharmonicity B | B(m) = exp(0.0926 m − 13.64) + exp(−0.0847 m − 5.82), m = MIDI pitch: 5.1e-4 at A0, minimum 1.4e-4 near k = 23, 3.3e-4 at C4, 2.85e-3 at C6, 2.6e-2 at C8 | **L**, second-hand: Rigaud, David & Daudet (DAFx 2011), as quoted and used by DDSP-Piano (Zenodo 8386706, eq. 4); the original paper was not read. **I** agrees within 5 % from C4 to C5 (C4 0 %, A4 +5 %, C5 +1 %) and is 34 % lower at C6 (1.9e-3 vs 2.85e-3; treble fits are less reliable). The Iowa bass is 1.3–2× lower, and the bass asymptote is piano-specific per Rigaud. The treble tracker is constrained to ×/÷4 of this curve. |
| Stretch (cents re A4, of the *sounding* fundamental f₁ = f₀√(1+B)) | `(0,-16) (3,-15) (15,-4) (27,0) (39,-1) (48,0) (51,0) (63,6) (75,14) (87,25)` | **I** (C8 extrapolated). v1's memory-based Railsback curve (−30 / +25 at A0 / A7) was about twice as wide. Tuners tune what sounds, so the model computes f₀ = f₁/√(1+B). |
| Strings per key | 1 for k < 8, 2 for k < 26, 3 above | **L** only "except the lowest octave" (Bank et al.); break points M |
| Unison mistuning | 0.2–2 cents per aftersound mode, same in cents for all partials, with sign and size drawn at random per key (seeded) | **L**: Weinreich, *mistuned.html*, Fig. 8. Tuners' mistuning "varied randomly from note to note" (Kirk 1959, as quoted). A shared pattern lines up the beat nulls of neighbouring keys. |
| Aftersound loss b1 (1/s) | `(0,0.101) (3,0.101) (15,0.126) (27,0.171) (39,0.23) (48,0.248) (51,0.332) (63,0.6) (75,0.583) (87,1.5)` | **I**: fitted with R and the aftersound amplitude to the measured decay profiles (`scripts/fit_decays.py`). A0 rests on 3 notes (partly mislabelled), so it takes C1's values. |
| Loss b3 (per Hz²) | `(0,2.5e-7) (20,2.5e-7) (30,1.2e-7) (55,1.0e-7) (63,5e-8) (87,2.5e-8)` | M. The treble values are low confidence: the treble recordings' "partials" are partly non-string components. |
| Prompt (in-phase) mode | **Additive** bridge loss: α_prompt,n = α_after,n + (R − 1)·b1. R = `(0,6.15) (3,6.15) (15,7.67) (27,8.26) (39,12.2) (48,14.7) (51,17.3) (63,5.1) (75,7.1) (87,5.4)` | Form **L**: Weinreich (in-phase motion of N strings loads the bridge N× harder; one string decays at 8 dB/s, so a trichord's prompt stage is about 24 dB/s); Ege & Boutillon (mean bridge mobility roughly frequency-independent, so the loss does not depend on n). R is **I** (fitted). |
| Aftersound amplitude per mode | `(0,0.11) (3,0.11) (15,0.154) (27,0.076) (39,0.042) (48,0.025) (51,0.04) (63,0.04) (75,0.006) (87,0.007)` | **I** (fitted). The treble aftersound almost vanishes: C7 decays nearly single-slope. |

Measured decay profile the prior must reproduce (**I**). Values are the level, in dB re peak, of the power-summed decay
partials, both channels, medians over mf and ff and ±3 semitones. The decay partials are n = 2..5 below C3 (the
recordings often lack the fundamental there) and n = 1..4 above, the same for model and recordings.

| key | 0.5 s | 1 s | 2 s | 4 s | 8 s | 16 s | notes |
|---|---|---|---|---|---|---|---|
| C1 | −3 | −6 | −10 | −18 | −27 | −33 | 8 |
| C2 | −5 | −10 | −12 | −22 | −24 | −35 | 14 |
| C3 | −5 | −10 | −22 | −24 | −32 | −45 | 14 |
| C4 | −12 | −21 | −25 | −31 | −45 | −55 | 14 |
| A4 | −15 | −26 | −29 | −38 | −53 | −64 | 14 |
| C5 | −20 | −21 | −29 | −37 | −53 | −70 | 14 |
| C6 | −15 | −22 | −36 | −47 | −69 | – | 13 |
| C7 | −19 | −34 | −53 | −66 | – | – | 14 |

Three points on this table:
- The fast early drop in the mid-range (C4 −21 dB at 1 s) is the three-string prompt stage. v2 had it about
  3× too slow, because it trusted a per-partial energy-decay-curve metric whose median fell between the two
  stages.
- A0 and C8 rest on 2–3 notes and are not used as targets.
- Treble targets (C5–C7) are low confidence (see b3).

Phase 2 (not implemented; there are no data to check against yet):
- pitch glide at *ff*;
- longitudinal modes at 12–20× f₀ (**L**: Conklin, *longitudinal.html*, Fig. 32; the E1 string's
  longitudinal mode at about 600 Hz is about 20 dB below its neighbouring partials). v1's
  k·c_L/(2L) estimate was 3× too high in the bass.
- phantom partials.

## 2. Hammer and excitation

| parameter | prior | evidence |
|---|---|---|
| Bridge-force partial amplitude | a_n ∝ **sin**(nπx₀/L)·\|F̂(f_n)\|, signed, with no 1/n | D (standard). **L**: Hall's C4 spectrum has the strike-position dip at n = 8 (*compare.html*, Fig. 15) |
| Strike position x₀/L | `(0,0.122) (27,0.122) (39,0.121) (49,0.115) (54,0.108) (59,0.100) (69,0.090) (79,0.075) (87,0.065)` | **L**: Conklin, *whereshould.html*, Fig. 11 (contemporary grand) |
| Contact time T_c at mf (2.8 m/s), ms | `(0,3.7) (15,3.0) (27,2.8) (39,2.1) (51,1.45) (63,1.1) (75,0.6) (87,0.5)` | **L**: Askenfelt & Jansson, *stricont.html*, Fig. 7 |
| Hammer speed vs MIDI velocity | v_h = 2.8·exp(0.023·(vel − 64)) m/s, so p ≈ 40 → 1.6, mf = 64 → 2.8, f ≈ 90 → 5.1, ff ≈ 115 → 9.0 | M, anchored on the dynamic labels of **L** (Askenfelt Fig. 6). It makes "mf" mean the same thing (MIDI 64 = 2.8 m/s) in contact time, roll-off order, key-bottom timing and calibration. |
| T_c vs hammer speed | T_c = T_c,mf·(v_h/2.8)^−0.2 | **L**: slope −0.19 at C4 (Askenfelt Fig. 6) |
| Force spectrum | Smooth envelope \|F̂\| = (1 + (fT_c/0.59)^(2q))^−½, computed in the log domain (softplus) so partials far above Nyquist cannot overflow. −3 dB at 0.59/T_c, then −6q dB/oct. The ideal half-sine's nulls are not modelled: they slide across partials with T_c and made treble brightness fall with velocity. | D; pulse shape **L** (Hall Fig. 18: smooth, slightly skewed bells) |
| Roll-off order q at mf | `(0,1.3) (27,1.3) (39,2.35) (48,3.2) (51,4.0) (63,4.7) (75,5.3) (87,5.3)`, learnable ×0.4–2.5 | C4 = 2.35 is **L**: Hall Fig. 15, C4 slopes −18 / −15 / −11 dB/oct at vel 30 / 64 / 110; the model gives −18.0 / −14.8 / −11.3. It also reproduces the Iowa C4 near-field slope within 1 dB/oct, independently. The rest is a smooth curve rising from bass to treble, through orders fitted to the Iowa slopes (**I**, both channels, through the body). Because it is microphone- and instrument-specific, it has a wide bound. |
| q vs hammer speed | q = q_mf·(v_h/2.8)^−0.225: harder blows drive the felt into its stiff, nonlinear range and sharpen the pulse | **L**: Hall (felt exponent p = 2.2 bass → 3 treble, Fig. 20). Fitted jointly with the C4 order. **I**: mf→ff centroid ×1.20–1.25 at C4/A4 (model ×1.24–1.27). |
| Level vs velocity | learned dB slope, about 40 dB across the MIDI range | **L**: 33 dB pp→ff on the pianist-accessible scale (Askenfelt, *keybott.html*, Fig. 5). Iowa's dynamics carry no MIDI velocities, so MAESTRO calibrates this. |
| Attack | Partials ramp in over T_c, with phases referenced to the pulse centre | D. The slower rise measured in the recordings (8–16 ms mid-range) comes mainly from soundboard build-up: the model with its body gives 20 ms at C4 and 11 ms at C6, against 13 and 9 ms measured. |
| Partial count | 96 | Bass bandwidth (**L**: the C2 spectrum reaches about 4 kHz at mf, Askenfelt) |

## 3. Soundboard, radiation and room (one set per MAESTRO year)

- **Body FIR** (0.3 s, learnable), initialised from:
  - Measured low modes of a 2.90 m concert grand at 62, 90, 105, 127, 187, 222, 245 and 325 Hz (**L**: Wogram, *modal.html*).
  - Above them, random modes at 0.07 modes/Hz up to the plate/rib-strip transition at 1.35 kHz (**L**: Steinway D model, Boutillon et al., arXiv 1210.3948).
  - Loss factor η = 0.02, i.e. T60 = 2.2/(ηf), capped at T60 0.7 s. The finished grand soundboard T60 is about 0.6 s (**L**: Bader & Plath 2020); v1's cap of 1.7 s made low modes ring and bass attacks swell.
  - Diffuse response above the transition.
  - Magnitude envelope `(20,-40) (30,-30) (40,-20) (55,-10) (70,-4) (100,0) (1000,0) (2000,-3) (4000,-6) (8000,-11) (11000,-15)` dB. The low-frequency corner comes from the first mode near 60 Hz (**L**); the high-frequency droop follows Wogram Fig. 5 (upright; low confidence).
  - A 3 ms pre-delay.
- **Body and hall are in series**: `ir = body ⊛ (δ + hall)`.
- **Hall** (parametric, learnable per year): octave-band noise with T60 `(125,2.0) (250,1.8) (500,1.7) (1k,1.6) (2k,1.45) (4k,1.2) (8k,0.8)` s, direct-to-reverberant ratio about 0 dB, 2.5 s long. M: the MAESTRO paper says only that the "microphone setup varied between competition years"; DDSP-Piano likewise learns one reverb per recording environment.

## 4. Dampers and pedals

| parameter | prior | evidence |
|---|---|---|
| Undamped keys | full damper strength up to k = 62, 0.3 at k = 67, none from k = 68 | M. Verify on MAESTRO. |
| Damper rate at partial 1 (1/s) | `(0,8) (12,10) (24,14) (36,20) (48,28) (60,36) (67,40)`; α_d,n = α_d1·min(n, 6)^0.6 | M. **L** consistency check: C4 broadband after damper contact is about 290–500 dB/s (Askenfelt, *measure.html*, Fig. 3); the prior gives about 510 dB/s for n ≥ 6. |
| Damper delay after note-off | +15 ms, fixed | **L**: first damper contact about 18 ms after the finger releases (Askenfelt Fig. 3). Not learnable (hard key timing), so it is calibrated per year on MAESTRO. |
| Sustain lift | logistic, θ = 0.42, width 0.06; damping ∝ (1 − lift)^2.5 | M; half-pedalling exists (**L**: Lehtonen et al. 2009, as quoted) |
| Sostenuto | latch the keys held when CC66 crosses 0.5, until it drops; approximate at the start of a training window | Standard mechanics. **L**: CC64/66/67 are present in the MAESTRO MIDI (DDSP-Piano). |
| Una corda | gain 0 / −2 / −3.5 dB (1 / 2 / 3 strings); aftersound ×1 / ×5 / ×3.5; brightness via T_c ×1.43 | D (strike-vector geometry) |
| Sympathetic resonance | 88 keys × 4 partials; drive ∝ κ = α_prompt − α_after; each key's own strings excluded; gain re-levelled to a −32 dB halo after the prompt stage got faster. **Off by default**: it costs about 90 % of a CPU training step. | D; **L** (Lehtonen 2007, Weinreich) |

## 5. Mechanical noises

| event | timing | spectrum and envelope | level |
|---|---|---|---|
| Hammer / soundboard knock | at the strike | flat to 600 Hz, −12 dB/oct above; τ 10 → 5 ms (bass → treble) | −25 dB (ff) to −12 dB (pp) re the tone over the first 60 ms (M, consistent with an informal −25 dB measurement on a KTH sound example) |
| Key-bottom thump | re the strike: +12 ms (p), +0.5 (mf), −2.5 (f), −5 ms (ff), interpolated in hammer speed (reachable now that ff ≈ 9 m/s) | same spectrum, τ 8 ms | M |
| Damper noise | note-off + damper delay | low-passed at 1.5 kHz, τ 5 ms | −40 dB re the released note (M) |
| Pedal noise | when the damper rail moves | low-passed at 800 Hz, τ 30 ms | −35 dB re an mf note (M) |

Key-bottom timing is **L** (Askenfelt & Jansson, *keybott.html*, Figs. 4–5). At f/ff the key
bottoms out *before* the hammer reaches the string, so a few milliseconds of sound before
the strike are correct.

Not modelled yet, from **L**:
- a 20–30 ms touch precursor before struck (staccato) notes;
- hammer-shank resonances in the knock (about 250 Hz mid-range).

## 6. Acceptance checks (`python -m pianonn.diagnostics`, report in [`diagnostics_prior.md`](diagnostics_prior.md))

Against the Iowa recordings, analysed identically, at C1, C2, C3, C4, A4, C5, C6 and C7:

1. **Decay profile**: model median over the same neighbouring keys and velocities as the recordings, each point ±6 dB.
2. **Early spectral slope** at mf, rendered through the soundboard body with the same 60 dB noise floor, ±5 dB/oct.
3. **Stretch** ±4 cents. *Regression check*: the prior copies these values.

A0 and C8 are not checked: 2–3 notes, partly mislabelled.

Against the literature and physics:

4. Effective B vs Rigaud at C4 and C6. *Regression check*: the prior is this curve.
5. Effective B vs Iowa (×2) at C2 and C4. Independent of the prior.
6. C4 slope over partials 1–10 at vel 30/64/110 vs Hall's −18/−15/−11 ±2 dB/oct. *Regression*: fitted to it.
7. Brightness rises monotonically with velocity at C2, C4, C6 and C7.
8. C4 partials 2–6 in −30..+3 dB and partial 10 ≤ −30 dB. v1's memory-based sanity range (M), kept.
9. A0 and C2 radiated fundamentals are ≥ 10 dB below the strongest partial.
10. Damper release; C8 undamped; pedal halo (with the bank switched on for the check).
11. Noise levels (section 5); nothing more than 6 ms before the strike; the ff key-bottom thump lands before the strike.

Removed since v2:
- The prompt/aftersound T60 and knee checks. The underlying metrics were unreliable on recordings (review 2).
  The decay profile replaces them.
- v1's "C4 centroid vel 120/vel 40 in 1.1–2" (M). It is replaced by check 6 (sourced). With Hall's slopes, the
  ratio is 2.1.

Current status: **47/50**. The failures are the C5, C6 and C7 decay profiles, off by 7–11 dB at 1–4 s. These
treble targets are low confidence, and fitting single-note beat patterns to them would overfit, so they are left
failing.

## 7. Open items

- **Bass rise time.** The model (with body) is slower than measured: C2 67 ms against 8 ms. The soundboard's low modes are probably too strong.
- **Level vs velocity per key** (ff − mf): the model gives 17–30 dB (most in the treble, where a harder, shorter
  blow excites a fundamental that sits above the hammer corner), Iowa 8–16 dB. Iowa has no MIDI velocities, so this
  is fitted on MAESTRO.
- **Treble decays** (C5–C7): low-confidence targets; re-derive on MAESTRO with a treble-aware tracker.
- **To calibrate on MAESTRO:** damper delay and damper boundary key, hall T60 per year, bass inharmonicity per year.
