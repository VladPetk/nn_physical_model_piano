# Calibration against recorded piano notes (Iowa Steinway B)

## Data

- University of Iowa Electronic Music Studios, *Musical Instrument Samples*, Piano: a Steinway & Sons model B
  (6'11"), recorded in 2001.
- 260 isolated notes at pp, mf and ff, each left to ring for 10–54 s.
- Two Neumann KM 84 microphones 8" above the bass and treble strings; 16-bit / 44.1 kHz stereo; a non-anechoic
  room.
- Source: `https://theremin.music.uiowa.edu/MISpiano.html`, free to use. Stored in `data/iowa/`, not committed.

What the data can and cannot tell us:
- Good for string physics: inharmonicity, tuning, how notes decay over time, spectra, and how the spectrum changes
  with dynamics.
- Not good for room and radiation, or for absolute level against MIDI velocity (the files are labelled pp/mf/ff,
  not with MIDI numbers).
- A different instrument from the MAESTRO Yamaha Disklaviers, so instrument-specific values are priors to be
  re-fitted there.

## Method (`pianonn/calibration.py`, v3 after review 2)

- **Channels.** Both channels are combined in *power* in every spectrum and envelope. A mono sum comb-filters: in
  review 2, a C2 partial's decay read 73 s from the mono sum but 6–8 s on either channel alone.
- **Onset** by level threshold.
- **Partials.** A grid search over (f₀, B) of a stiff-string comb, with B restricted to ×/÷4 of Rigaud's curve (so
  treble distortion harmonics cannot win with B ≈ 0). This is followed by peak refinement, regression of fₙ²/n² on
  n², and outlier rejection.
- **Decay profile** (the primary decay target): power of the decay partials, chosen **by number** (n = 2..5 below
  C3, where recordings often lack the fundamental; n = 1..4 above), noise-subtracted, smoothed over 0.1 s. It is
  read at 0.5, 1, 2, 4, 8 and 16 s in dB re peak, and set to NaN within 6 dB of the noise floor.
- **Two-segment fit** of the same envelope with a free breakpoint (prompt T60, aftersound T60, knee). Reported for
  information only: mid-range envelopes have three phases (fast drop, beating plateau, slow tail), which two
  segments cannot capture.
- **Early spectrum** (10–200 ms): noise-compensated partial levels, spectral slope (dB/oct) and a partial-based
  centroid.
- **Quality flags.** B is "reliable" with at least 8 partials and residuals under 3 cents. A note is "suspect" if it
  lies more than 50 cents off its label: 12 notes, including `ff.A0` and `ff.B0`, which sound a semitone high.
- **Aggregation.** Medians over mf and ff and over ±3 semitones around each landmark. The last row of the table
  gives the number of notes behind each landmark: A0 and C8 rest on 2–3 notes and are not used as targets.
- **Model.** The model is rendered like a recording and analysed by the same code:
  - 0.3 s of silence first;
  - dry strings for 45 s, for decays;
  - strings through the soundboard body, for spectra;
  - white noise 60 dB below the peak, so that weak partials are censored the same way;
  - MIDI 64 = "mf", 110 = "ff".

  The model's "C1" column repeats A0, because only landmark keys are rendered.

## Recording vs current prior

| quantity | source | A0 | C1 | C2 | C3 | C4 | A4 | C5 | C6 | C7 | C8 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| inharmonicity B (reliable fits) | recording | 2.50e-04 | 2.16e-04 | 1.17e-04 | 1.23e-04 | 3.26e-04 | 7.57e-04 | 9.55e-04 | 1.87e-03 | - | - |
| inharmonicity B (reliable fits) | model | 4.55e-04 | 4.55e-04 | 1.74e-04 | 1.53e-04 | 3.24e-04 | - | - | - | - | - |
| tuning, cents re ET | recording | -13.2 | -12.8 | -1.2 | +3.6 | +1.7 | +3.1 | +2.7 | +8.4 | +17.1 | +11.1 |
| tuning, cents re ET | model | -15.5 | -15.5 | -3.2 | +0.3 | -0.3 | +0.9 | +0.9 | +6.3 | +14.0 | - |
| decay profile at 0.5 s (dB re peak) | recording | 1 | -3 | -5 | -5 | -12 | -15 | -20 | -15 | -19 | -33 |
| decay profile at 0.5 s (dB re peak) | model | -2 | -2 | -3 | -5 | -10 | -15 | -15 | -11 | -16 | - |
| decay profile at 1 s (dB re peak) | recording | -1 | -6 | -10 | -10 | -21 | -26 | -21 | -22 | -34 | -53 |
| decay profile at 1 s (dB re peak) | model | -4 | -4 | -7 | -12 | -20 | -27 | -27 | -21 | -35 | - |
| decay profile at 2 s (dB re peak) | recording | -5 | -10 | -12 | -22 | -25 | -29 | -29 | -36 | -53 | -70 |
| decay profile at 2 s (dB re peak) | model | -9 | -9 | -15 | -21 | -31 | -36 | -36 | -33 | -50 | - |
| decay profile at 4 s (dB re peak) | recording | -10 | -18 | -22 | -24 | -31 | -38 | -37 | -47 | -66 | - |
| decay profile at 4 s (dB re peak) | model | -17 | -17 | -23 | -27 | -30 | -38 | -38 | -48 | -66 | - |
| decay profile at 8 s (dB re peak) | recording | -19 | -27 | -24 | -32 | -45 | -53 | -53 | -69 | - | - |
| decay profile at 8 s (dB re peak) | model | -26 | -26 | -27 | -34 | -40 | -52 | -52 | - | - | - |
| decay profile at 16 s (dB re peak) | recording | -29 | -33 | -35 | -45 | -55 | -64 | -70 | - | - | - |
| decay profile at 16 s (dB re peak) | model | -34 | -34 | -36 | -47 | -60 | - | - | - | - | - |
| prompt T60, two-segment fit (s) | recording | 18.3 | 18.2 | 14.9 | 10.2 | 12.8 | 10.0 | 10.0 | 3.5 | 1.7 | 0.7 |
| prompt T60, two-segment fit (s) | model | 15.8 | 15.8 | 8.0 | 5.9 | 3.0 | 2.3 | 2.3 | 2.8 | 1.6 | - |
| aftersound T60, two-segment fit (s) | recording | 55 | 68 | 81 | 52 | 51 | 40 | 23 | 10 | 4 | 2 |
| aftersound T60, two-segment fit (s) | model | 62 | 62 | 52 | 37 | 27 | 27 | 27 | 10 | 5 | - |
| knee, two-segment fit (dB re peak) | recording | -10 | -21 | -23 | -26 | -35 | -36 | -31 | -21 | -23 | -17 |
| knee, two-segment fit (dB re peak) | model | -19 | -19 | -18 | -20 | -24 | -30 | -30 | -20 | -16 | - |
| early spectral slope, mf (dB/oct) | recording | -1 | 1 | -4 | -5 | -15 | -19 | -20 | -29 | -37 | - |
| early spectral slope, mf (dB/oct) | model | 3 | 3 | -4 | -7 | -15 | -19 | -19 | -29 | -32 | - |
| rise time 10-90 %, mf (ms) | recording | 46 | 52 | 20 | 12 | 15 | 13 | 9 | 9 | 6 | 7 |
| rise time 10-90 %, mf (ms) | model | 62 | 62 | 67 | 56 | 20 | 27 | 27 | 11 | 6 | 5 |
| slope change mf -> ff (dB/oct) | recording | -2.4 | +0.8 | +1.0 | +1.4 | +6.0 | +2.3 | +4.3 | +5.1 | -8.2 | - |
| slope change mf -> ff (dB/oct) | model | +1.0 | +1.0 | +1.5 | +1.9 | +3.5 | +2.2 | +2.2 | +5.1 | +7.1 | - |
| peak level ff - mf (dB) | recording | +16 | +12 | +8 | +9 | +12 | +11 | +13 | +15 | +15 | +10 |
| peak level ff - mf (dB) | model | +17 | +17 | +18 | +19 | +20 | +20 | +20 | +27 | +30 | +35 |
| harmonic centroid ff / mf | recording | x1.10 | x1.04 | x1.04 | x1.10 | x1.20 | x1.25 | x1.28 | x1.09 | x1.02 | x1.05 |
| harmonic centroid ff / mf | model | x1.06 | x1.06 | x1.13 | x1.20 | x1.24 | x1.27 | x1.27 | x1.07 | x1.05 | x1.02 |
| notes behind each decay value | recording | 2 | 8 | 14 | 14 | 14 | 14 | 14 | 13 | 14 | 3 |

## Fitting

`scripts/fit_decays.py` fits the aftersound loss b1, the prompt ratio R and the aftersound amplitude at each
landmark, from an analytic decay profile (beats averaged). The fitted profiles are within a few dB of the targets.

Rendered notes are then checked against the targets, with the model aggregated like the recordings: medians over
neighbouring keys and velocities. They pass at C1–C4, A4 and C7. C5–C7 were less reliable to begin with (see
`physical_parameters.md`, section 6).

## Per-partial fit (v4)

v3 calibrated decays on partials 2–5 (bass) or 1–4 only, and the spectrum on the slope of the first 12 partials.
Listening showed a harpsichord-like bass. `pianonn.calibration.partial_tables` now measures every partial below
10 kHz: its attack level and its level at 0.5/1/2/3/5/8/12/16 s (noise-compensated; below the noise it becomes
an upper bound), stored in `data/iowa_partials.json` (254 notes). `scripts/compare_partials.py` renders the
model at the same keys and dynamics and measures it with the same code. It showed, for v3, in the bass and tenor:

- attack spectrum 15–25 dB too loud above 3.5 kHz (with 4–8 dB too little at 0.4–1.8 kHz);
- partials above ~1 kHz decaying far too fast (1.7–3.5 kHz: −48 dB at 3 s against −35 dB recorded);
- the same prompt-stage loss for every partial.

Two one-pass fits followed (analytic, seconds each):
- `scripts/fit_spectra.py`: hammer roll-off order per key plus a second corner (x₂, q₂). Error 15.0 → 9.8 dB rms.
  A smooth body-EQ correction was fitted too; it came out below 2 dB to 7 kHz, so the body prior is unchanged.
- `scripts/fit_decays.py`: b1, b3, R, aftersound amplitude per key, a global exponent p for the aftersound loss
  over frequency, and one bridge-conductance curve g(f). Per-partial level error 18 → 11 dB rms (median 6.4 → 4.9).

After the fit, rendered and measured (`compare_partials.py`), the bass/tenor attack above 3.5 kHz is within ~5 dB
and the partial decays track the recordings to within a few dB up to 8 s in most bands. Known misses: the C2–B2
partials at 110–220 Hz decay much faster on this Steinway (a soundboard feature), and the mid-register low
partials still fall ~5 dB too fast in the first second. These fits are a starting point for MAESTRO, not a
final calibration.

## Review history

- **Review 2** ([`reviews/review_2_calibration.md`](reviews/review_2_calibration.md)) found that v2's per-partial
  energy-decay-curve "prompt T60" misread recordings. Real mid-range notes fall about 20 dB in the first second,
  while the v2 model fell about 5 dB.
- It also found partial selection by list index, the mono-sum comb filtering, the unconstrained treble tracker, and
  a NaN gradient. All are fixed in v3.
