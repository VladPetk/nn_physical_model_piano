# Calibration against recorded piano notes (Iowa Steinway B)

## Data

- University of Iowa Electronic Music Studios, *Musical Instrument Samples*, Piano: a Steinway &
  Sons model B (6'11"), recorded in 2001.
- 260 isolated notes, each at pp, mf and ff, left to ring for 10–54 s.
- Two Neumann KM 84 microphones 8" above the bass and treble strings; 16-bit / 44.1 kHz stereo;
  a non-anechoic room.
- Download: `https://theremin.music.uiowa.edu/MISpiano.html` (free to use). The files live in
  `data/iowa/`, which is not committed.

What this data is good for:
- **Good for** string physics: inharmonicity, tuning, prompt and aftersound decay, spectra, and
  how the spectrum changes with dynamics. The close microphones hear the strings and soundboard,
  not a hall.
- **Not good for** room and radiation, or absolute level against MIDI velocity: the dynamics are
  pp/mf/ff, not MIDI numbers.
- **Not the same instrument** as the MAESTRO Yamaha Disklaviers, so instrument-specific values
  are priors to be re-fitted.

## Method (`pianonn/calibration.py`)

Per note:
1. **Onset** by level threshold.
2. **Partials** by a grid search over (f₀, B) of a stiff-string comb against the spectrum. The
   search is robust to a weak fundamental and stray peaks. It is followed by peak refinement,
   regression of fₙ²/n² on n², and rejection of outliers more than 5 cents off.
3. **Decays** per partial, from a heterodyne power envelope with a noise floor estimated before
   the onset:
   - prompt T60 from the energy decay curve, −3 to −13 dB;
   - aftersound T60 and knee from a line fitted after the level drops 25 dB, down to 10 dB above
     the noise floor.
4. **Early spectrum** (10–200 ms), noise-compensated at the tracked partials; from it the spectral
   slope in dB/oct and a partial-based centroid.

Quality control:
- A fit's B is trusted only with at least 8 partials and a residual under 3 cents. In the treble,
  distortion in the recording chain adds exact integer harmonics that mimic B ≈ 0.
- Notes more than 50 cents off their label are flagged as suspect. There are 15: `ff.A0` and
  `ff.B0` sound about 80 cents sharp (probably mislabelled), plus many top-octave pp/mf files.

Aggregation: medians over partials 1–4, over mf and ff (decay does not depend on blow force,
per Hundley et al. as quoted in arXiv 1212.2323), then over ±3 semitones around each landmark.

The model is rendered at the landmark keys (MIDI velocity 64 = "mf", 110 = "ff"). Decays and
spectra use the dry strings rendered for 45 s, with a −140 dB noise floor. The same code then
analyses those renders.

## Recording vs current prior

| quantity | source | A0 | C1 | C2 | C3 | C4 | A4 | C5 | C6 | C7 | C8 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| inharmonicity B (reliable fits) | recording | 2.51e-04 | 2.15e-04 | 1.19e-04 | 1.23e-04 | 3.04e-04 | 7.53e-04 | 9.86e-04 | - | - | - |
| inharmonicity B (reliable fits) | model | 4.55e-04 | 4.55e-04 | 1.74e-04 | 1.53e-04 | 3.30e-04 | 8.57e-04 | 8.57e-04 | 2.85e-03 | - | - |
| tuning, cents re ET | recording | -13.1 | -11.7 | -1.1 | +2.8 | +1.6 | +3.0 | +2.6 | +8.7 | +16.9 | +44.5 |
| tuning, cents re ET | model | -15.4 | -15.4 | -3.9 | +0.0 | -1.0 | -0.2 | -0.2 | +6.0 | +14.0 | +25.0 |
| prompt T60, median partials 1-4 (s) | recording | 26.4 | 19.8 | 26.7 | 15.7 | 10.3 | 5.8 | 5.7 | 5.0 | 17.9 | 27.4 |
| prompt T60, median partials 1-4 (s) | model | 26.4 | 26.4 | 24.6 | 13.3 | 8.5 | 5.8 | 5.8 | 5.1 | 2.7 | 1.8 |
| aftersound T60, median partials 1-4 (s) | recording | 113 | 110 | 93 | 51 | 26 | 18 | 14 | 9 | 5 | 3 |
| aftersound T60, median partials 1-4 (s) | model | 112 | 112 | 83 | 43 | 23 | 16 | 16 | 9 | 5 | 3 |
| aftersound knee, partials 1-4 (dB re peak) | recording | -24 | -24 | -21 | -21 | -21 | -20 | -14 | -13 | -15 | -12 |
| aftersound knee, partials 1-4 (dB re peak) | model | -25 | -25 | -23 | -19 | -16 | -16 | -16 | -12 | -13 | -10 |
| aftersound knee, partials 5-8 (dB re peak) | recording | -27 | -25 | -20 | -18 | -13 | -13 | -7 | -35 | - | - |
| aftersound knee, partials 5-8 (dB re peak) | model | -20 | -20 | -18 | -19 | -15 | -12 | -12 | -32 | - | - |
| early spectral slope, mf (dB/oct) | recording | -1 | 1 | -4 | -5 | -14 | -17 | -26 | -27 | -33 | - |
| early spectral slope, mf (dB/oct) | model | -4 | -4 | -7 | -9 | -14 | -16 | -16 | -19 | -26 | - |
| rise time 10-90 %, mf (ms) | recording | 54 | 35 | 8 | 11 | 16 | 13 | 13 | 9 | 7 | 11 |
| rise time 10-90 %, mf (ms) | model | 2 | 2 | 3 | 9 | 5 | 2 | 2 | 2 | 2 | 2 |
| slope change mf -> ff (dB/oct) | recording | +1.3 | +0.2 | +1.0 | +1.2 | +4.6 | +2.7 | +6.9 | +3.2 | -12.7 | - |
| slope change mf -> ff (dB/oct) | model | +1.0 | +1.0 | +1.4 | +1.6 | +2.2 | +0.9 | +0.9 | +0.8 | -13.2 | - |
| peak level ff - mf (dB) | recording | +16 | +13 | +8 | +8 | +12 | +11 | +13 | +15 | +15 | +11 |
| peak level ff - mf (dB) | model | +16 | +16 | +17 | +17 | +18 | +19 | +19 | +22 | +25 | +28 |
| harmonic centroid ff / mf | recording | x1.04 | x1.03 | x1.04 | x1.09 | x1.22 | x1.22 | x1.25 | x1.05 | x1.01 | x1.05 |
| harmonic centroid ff / mf | model | x1.08 | x1.08 | x1.11 | x1.14 | x1.21 | x1.20 | x1.20 | x1.14 | x1.06 | x1.03 |

Notes on reading the table:
- The model's "C1" column repeats A0: the model is only rendered at landmark keys, and C1 sits
  within ±3 semitones of A0.
- The model's slope row here is the dry string signal. The acceptance check in
  `pianonn.diagnostics` includes the soundboard body, as the near-field recording does, and
  matches within ±5 dB/oct in every register.

## What changed because of this data

| quantity | v1 prior (memory) | measured here | now |
|---|---|---|---|
| bass aftersound T60 | 20–30 s | about 110 s (A0–C1), 93 s (C2) | measured |
| bass double decay | none (R = 1.5–1.7) | clear knee, R ≈ 4 | measured |
| mid prompt T60 | C4 7.3 s, A4 5.0 s (a model artefact of beating) | C4 10 s, A4 5.8 s | measured |
| stretch | A0 −30, C7 +25 cents | A0 −16, C7 +14 cents | measured |
| treble brightness | C6 partial 2 at −8 dB | C6 partial 2 at −26 dB | per-key hammer roll-off |
| velocity brightening | ×1.0 roll-off, +2 dB at n = 8 mf→ff | ff/mf centroid ×1.22 at C4/A4 | velocity-dependent roll-off, fitted to Hall |

## Known limitations

- **Bass knee metric.** For the fundamental of A0–C3 it responds non-monotonically to the
  aftersound amplitude, so the bass knee checks fail while the aftersound T60 matches.
- **Bass rise time.** The model is too slow (see `physical_parameters.md`, section 7).
- **Treble prompt T60 (C7, C8)** could not be measured reliably: there are few partials, and
  the pp/mf files are often suspect.
