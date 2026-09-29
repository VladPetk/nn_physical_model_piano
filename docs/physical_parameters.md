# Physical parameters: requirements (v2)

This is the spec the physics prior (`pianonn/physics.py`, `pianonn/synth.py`, `pianonn/room.py`)
must meet before any MAESTRO fitting. v2 replaces most of v1's from-memory values with
values taken from two kinds of source:

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
| Inharmonicity B | B(m) = exp(0.0926 m − 13.64) + exp(−0.0847 m − 5.82), m = MIDI pitch: 5.1e-4 at A0, minimum 1.4e-4 near k = 23, 3.3e-4 at C4, 2.85e-3 at C6, 2.6e-2 at C8 | **L**: Rigaud, David & Daudet (DAFx 2011), as used by DDSP-Piano (Zenodo 8386706, eq. 4). **I** agrees from C4 to C5 within 10 %. The Iowa bass is 1.3–2× lower, and the bass asymptote is piano-specific per Rigaud. |
| Stretch (cents re A4, of the *sounding* fundamental f₁ = f₀√(1+B)) | `(0,-16) (3,-15) (15,-4) (27,0) (39,-1) (48,0) (51,0) (63,6) (75,14) (87,25)` | **I** (C8 extrapolated). v1's memory-based Railsback curve (−30 / +25 at A0 / A7) was about twice as wide. Tuners tune what sounds, so the model computes f₀ = f₁/√(1+B). |
| Strings per key | 1 for k < 8, 2 for k < 26, 3 above | **L** only "except the lowest octave" (Bank et al.); break points M |
| Unison mistuning | 0.2–2 cents, same in cents for all partials. Strings lock (no beats) below about 0.3 Hz of mistuning (mid-range). | **L**: Weinreich, *mistuned.html*, Fig. 8 |
| Aftersound loss b1 (1/s) | `(0,0.062) (15,0.067) (27,0.118) (39,0.216) (48,0.25) (51,0.30) (63,0.40) (75,0.6) (87,1.0)` | **I**, solved from the measured aftersound T60s |
| Loss b3 (per Hz²) | `(0,2.5e-7) (20,2.5e-7) (30,1.2e-7) (55,1.0e-7) (63,5e-8) (87,2.5e-8)` | M in the bass and mid; **I** requires it to fall in the treble, otherwise b1 < 0 |
| Prompt (in-phase) mode | **Additive** bridge loss: α_prompt,n = α_after,n + (R − 1)·b1. R = `(0,4.3) (15,3.7) (27,3.5) (39,3.0) (48,3.5) (63,2.8) (75,3.0) (87,3.0)` | Form **L**: Weinreich (in-phase string motion loads the bridge; single-string 8 dB/s vs < 2 dB/s) and Ege & Boutillon (mean bridge mobility roughly frequency-independent, so the extra loss does not depend on n). R is **I**. v1's R = 1.5–1.7 in the bass gave no knee; the recordings show a clear two-stage decay in the bass too. |
| Aftersound amplitude per mode | `(0,0.06) (20,0.06) (27,0.06) (36,0.09) (48,0.09) (55,0.10) (63,0.12) (75,0.06) (87,0.05)` | **I**, fitted to the measured aftersound T60 and knee (see section 6) |
| Double decay on all partials | kept for every partial | **I** contradicts the claim that it is "mostly the lowest harmonics" (Bank et al.): mid-range partials 5–8 show a *stronger* aftersound (knee about −13 dB) than partials 1–4 |

Measured decays the prior must reproduce (**I**; medians over partials 1–4, mf and ff):

| key | A0 | C1 | C2 | C3 | C4 | A4 | C5 | C6 | C7 | C8 |
|---|---|---|---|---|---|---|---|---|---|---|
| prompt T60 (s) | 26 | 20 | 27 | 16 | 10 | 5.8 | 5.7 | 5.0 | – | – |
| aftersound T60 (s) | 113 | 110 | 93 | 51 | 26 | 18 | 14 | 9.1 | 4.6 | 3.1 |

Notes on the table:
- v1's memory-based table was wrong in the bass: it had the aftersound at 20–30 s where the
  measured values are about 100 s, and no knee.
- The mid-range agrees with Weinreich's single-string E♭4: prompt 8 dB/s and aftersound
  1.8 dB/s, i.e. T60 7.5 s and 33 s (**L**).
- These are *effective* decays: what an analysis reports, beating included. The recordings and
  the model are measured with the same code.

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
| T_c vs hammer speed | T_c = T_c,mf·(v_h/2.8)^−0.2, with v_h = 5.5·(vel/127)^1.4 m/s | **L**: slope −0.19 at C4 (Askenfelt Fig. 6). The v_h map is M; MAESTRO fits the level law. |
| Force spectrum | Smooth envelope \|F̂\| = (1 + (fT_c/0.59)^(2q))^−½: −3 dB at 0.59/T_c, then −6q dB/oct. The ideal half-sine's nulls are not modelled: they slide across partials with T_c and made treble brightness fall with velocity. | D; pulse shape **L** (Hall Fig. 18: smooth, slightly skewed bells) |
| Roll-off order q at mf | `(0,1.5) (27,1.5) (39,2.1) (48,1.9) (51,2.6) (63,3.2) (75,4.5) (87,4.5)` | C4 = 2.1 is **L**: Hall Fig. 15, C4 slopes −18 / −15 / −11 dB/oct at pp / mf / ff; the model gives −17.8 / −14.2 / −11.9. The rest is **I**: early spectral slope of the radiated near-field sound. It is gentle in the bass and steep in the treble, where the contact outlasts half the string period (**L**: Askenfelt Fig. 8). |
| q vs hammer speed | q = q_mf·(v_h/2.8)^−0.2: harder blows drive the felt into its stiff, nonlinear range and sharpen the pulse | **L**: Hall (felt exponent p = 2.2 bass → 3 treble, Fig. 20). The exponent is fitted to Hall's C4 slopes. **I**: mf→ff centroid ×1.22 at C4/A4 (model ×1.21). |
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
| Sympathetic resonance | 88 keys × 4 partials; drive ∝ κ = α_prompt − α_after; each key's own strings excluded. **Off by default**: it costs about 90 % of a CPU training step. | D; **L** (Lehtonen 2007, Weinreich) |

## 5. Mechanical noises

| event | timing | spectrum and envelope | level |
|---|---|---|---|
| Hammer / soundboard knock | at the strike | flat to 600 Hz, −12 dB/oct above; τ 10 → 5 ms (bass → treble) | −25 dB (ff) to −12 dB (pp) re the tone over the first 60 ms (M, consistent with an informal −25 dB measurement on a KTH sound example) |
| Key-bottom thump | re the strike: +12 ms (p), +0.5 (mf), −2.5 (f), −5 ms (ff) | same spectrum, τ 8 ms | M |
| Damper noise | note-off + damper delay | low-passed at 1.5 kHz, τ 5 ms | −40 dB re the released note (M) |
| Pedal noise | when the damper rail moves | low-passed at 800 Hz, τ 30 ms | −35 dB re an mf note (M) |

Key-bottom timing is **L** (Askenfelt & Jansson, *keybott.html*, Figs. 4–5). At f/ff the key
bottoms out *before* the hammer reaches the string, so a few milliseconds of sound before
the strike are correct.

Not modelled yet, from **L**:
- a 20–30 ms touch precursor before struck (staccato) notes;
- hammer-shank resonances in the knock (about 250 Hz mid-range).

## 6. Acceptance checks (`python -m pianonn.diagnostics`, report in [`diagnostics_prior.md`](diagnostics_prior.md))

Against the Iowa recordings, analysed identically, at A0, C1, C2, C3, C4, A4, C5, C6 and C7:

1. Prompt T60 and aftersound T60 (medians over partials 1–4) within ×/÷1.35.
2. Aftersound knee **of the fundamental** within ±4 dB. For partials 2–4 the knee fit saturates at about −13 to −16 dB, in the recordings and the model alike, so they are not used.
3. Early spectral slope at mf, radiated through the soundboard body, within ±5 dB/oct.
4. Stretch re A4 within ±4 cents.

Against the literature and physics:

5. Effective B within ×1.5 of Rigaud at C4 and C6, and within ×2 of Iowa at C2 and C4.
6. Brightness rises monotonically with velocity at C2, C4, C6 and C7; the C4 centroid rises 1.1–2× from vel 40 to vel 120.
7. A0 and C2 radiated fundamentals are ≥ 10 dB below their strongest partial.
8. Damper release; C8 undamped; pedal halo (bank switched on for the check).
9. Noise levels as in section 5; nothing more than 6 ms before the strike; the ff key-bottom thump lands before the strike.

Current status: **62/65**. The three failures are the bass knees of the fundamental: A0 −17 against −29 dB, C2 −22 against −28, C3 −21 against −27. There, the knee metric responds non-monotonically to the aftersound amplitude: at A0 no amplitude reaches −29 dB while also keeping the aftersound T60 in range. The aftersound T60 check is the more robust of the two, so it wins.

## 7. Open items

- **Bass rise time.** The model (with body) is slower than measured: C2 67 ms against 8 ms. The soundboard's low modes are probably too strong.
- **Level vs velocity per key** (ff − mf): the model gives 16–28 dB, Iowa 8–16 dB, but Iowa has no MIDI velocities. Fit on MAESTRO.
- **To calibrate on MAESTRO:** damper delay and damper boundary key, hall T60 per year, bass inharmonicity per year.
