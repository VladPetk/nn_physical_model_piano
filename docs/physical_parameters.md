# Physical parameters: requirements

This is the spec the physics prior (`pianonn/physics.py`, `pianonn/synth.py`) must meet before
any MAESTRO fitting. It condenses two literature reviews, kept verbatim in
[`literature/strings_hammer.md`](literature/strings_hammer.md) and
[`literature/body_dampers_pedals.md`](literature/body_dampers_pedals.md).

**Evidence level.** The reviewers could search the web but could not open papers (fetch was
blocked). Values are tagged:
- **S**: confirmed by a search snippet or abstract.
- **D**: derived from stated physics.
- **M**: from memory or engineering judgement.

Everything here is a *prior*. Each value gets a bounded learned offset and is refined on
data. Where the two reviews disagreed or were internally inconsistent, the decision is noted.

Key index `k = MIDI - 21`. Landmarks: A0 = 0, C2 = 15, C3 = 27, C4 = 39, A4 = 48, C6 = 63,
C7 = 75, C8 = 87. Point lists are `key_curve` inputs, `(k, value)`.

## 1. Why the prior sounds like a toy guitar (combined ranking)

1. **No soundboard body and no hall.** The IR is a delta plus a 0.8 s white tail.
2. **Hammer spectrum too bright.** It rolls off at -9 dB/oct from cutoffs up to 3.5 kHz; the
   real force pulse is -12 dB/oct from a few hundred Hz. The cutoff also moves 5.4× with
   velocity instead of 1.5–2×.
3. **Bass fundamentals too strong.** There is no radiation high-pass around 60–70 Hz, where a
   real piano radiates A0–C2 fundamentals 15–30 dB below partials 3–8.
4. **Decay shape wrong.** Decays are too short, and the double decay is too weak and comes too
   early.
5. **Attack noise wrong.** The knock is near-white and 3–6× too long (a click rather than a
   thud).
6. **Smaller items:**
   - flat damper rate;
   - strike position constant across the keyboard;
   - 64 partials cap the bass at about 2.6 kHz;
   - treble inharmonicity 3–6× too low;
   - weak una corda and a weak pedal halo.

## 2. Strings

| parameter | prior | evidence |
|---|---|---|
| Inharmonicity B, log10 | `(0,-3.52) (6,-3.66) (14,-3.85) (22,-3.96) (29,-3.92) (33,-3.70) (39,-3.42) (48,-3.07) (56,-2.74) (63,-2.46) (75,-2.00) (87,-1.55)`. That is B = 3e-4 at A0, 1.1e-4 minimum at the bass break, 3.8e-4 at C4, 3.5e-3 at C6 and 2.8e-2 at C8. | Shape S (Rigaud 2013); values D/M. The reviewer proposed 4e-2 at C8; the string-scale formula gives 2–3e-2, so the decision is -1.55. |
| Stretch, cents | `(0,-30) (12,-18) (24,-8) (36,-3) (48,0) (60,5) (72,13) (84,25) (87,30)` | M (Railsback) |
| Strings per key | 1 for k < 8, 2 for k < 26, 3 above | M, ±2 keys |
| Unison mistuning | 0.2–2 cents, same in cents for all partials (so beats grow with n). Keep the current (+0.6, -0.4) init. | M (Weinreich, Kirk) |
| Aftersound decay b1 (1/s) | `(0,0.17) (15,0.25) (27,0.22) (39,0.25) (48,0.30) (63,0.55) (75,0.70) (87,1.3)` | Refit so the rendered notes land in the measured T60 table below |
| Loss b3 (1/s per Hz²) | 2.5e-7 for k < 20 (wound strings), 1.2e-7 for k ≥ 30 | M (Chaigne & Askenfelt 1994) |
| Prompt (in-phase) extra bridge loss | **Additive**: α_prompt,n = α_after,n + (R − 1)·b1, with R = 1.5 for monochords, 1.7 for bichords, ramping 2.5 → 4 from k = 26 to 39, then 4. The bichord value follows from the C2 targets below (15–20 s prompt against 20–30 s aftersound); the reviewer's rough 2–4 contradicts them. | S (Weinreich: 8 dB/s vs < 2 dB/s). The additive form is D. It replaces the current multiplicative ratio, which made high partials decay far too fast. |
| Aftersound amplitude per mode | `(0,0.10) (39,0.06) (87,0.06)`, i.e. the knee sits 20–35 dB down | M |

Target fundamental T60s the prior must reproduce (M). Values are prompt / aftersound:

| key | A0 | C2 | C3 | C4 | A4 | C6 | C7 | C8 |
|---|---|---|---|---|---|---|---|---|
| T60 (s) | 25–40 | 15–20 / 20–30 | 10–15 / 25–40 | 6–8 / 20–35 | 5–6 / 15–30 | 2.5–3.5 / 8–15 | 1.5–2 / 3–6 | 0.7–1.2 |

Partial 10 of C4 should have T60 of about 3–4 s.

Measurement convention: these are *effective* decays, i.e. what an analysis of a recording
reports, including beating. Diagnostics measure the prompt T60 from the energy decay curve
(−3 to −13 dB) of the fundamental. In the treble the aftersound modes beat against the prompt
mode within the first half-second, so the measured early decay is shorter than 6.91/α_prompt
(C7: 1.65 s measured against 2.0 s analytic). The b1 prior is tuned so the *measured* values
land in range, as they would on a real piano.

Phase 2 (low confidence, all D/M; do not implement until there are data to check against):
- pitch glide at *ff* (1–5 cents, time constant 1/(2α_prompt));
- longitudinal modes at k·c_L/(2L) (A0 about 1.3 kHz, C4 about 4.1 kHz);
- phantom partials at f_n + f_m (-30 to -45 dB at mf, amplitude scaling with v²).

## 3. Hammer and excitation

| parameter | prior | evidence |
|---|---|---|
| Bridge-force partial amplitude | a_n ∝ **sin**(nπx₀/L) · \|F̂(f_n)\|. No 1/n (that factor belongs to displacement). Keep the sign; do not take abs. `strings` is therefore a bridge-force signal. | D (standard; Hall) |
| Force spectrum | Smooth envelope of a half-sine pulse of length T_c: \|F̂(f)\| = 1/sqrt(1 + (fT_c/0.59)⁴), i.e. -3 dB at 0.59/T_c then -12 dB/oct. **Deliberate deviation**: the reviewers proposed the ideal half-sine with nulls filled to ε = 0.15. That form divides by zero at fT_c = 0.5, and its nulls slide across partials as T_c changes, so brightness falls with velocity in the treble (measured at C6). Real pulses are skewed and their nulls are largely filled, so the smooth envelope is the better prior. | D; pulse shape S (Askenfelt) |
| Contact time T_c at mf (ms) | `(0,3.5) (15,3.0) (27,2.4) (39,1.9) (51,1.5) (63,1.1) (75,0.8) (87,0.6)` | S (about 2 ms typical); per-key values M |
| Velocity dependence | Hammer speed v_h = 5.5·(vel/127)^1.4 m/s, so about 0.4 m/s at pp and 5.5 m/s at fff. T_c = T_c,mf·(v_h/2)^(−0.25). | S (1–5 m/s, T_c ±20 % p→ff); exponent D/M |
| Level vs velocity | Keep the learned dB slope, about 40 dB across the MIDI range, per key. | S (Goebl: roughly linear in dB, pitch-dependent) |
| Strike position x₀/L | `(0,0.125) (15,0.12) (27,0.115) (39,0.11) (48,0.105) (56,0.095) (63,0.088) (75,0.075) (87,0.06)` | S (Conklin: just under 1/8 in the bass, 1/12–1/17 at the top) |
| Attack | Partials ramp in over T_c (raised cosine), phase referenced to the pulse centre (T_c/2), instead of starting all partials at full amplitude at the onset | D |
| Partial count | 96. A0 then reaches about 5 kHz once inharmonic stretch is included. | Bandwidth argument |

## 4. Soundboard, radiation and room (one set per MAESTRO year)

- **Body FIR** (0.3 s, learnable), initialised from:
  - Magnitude envelope, as `(Hz, dB)` points: `(20,-40) (30,-30) (40,-20) (55,-10) (70,-4) (100,0) (1000,0) (2000,-1) (4000,-3) (8000,-7) (11000,-12)`. (S: first mode 60–70 Hz; the rest D/M.)
  - Below 1.1 kHz: about 0.07 modes/Hz with random amplitudes and loss factor η = 0.02, i.e. T60 = 2.2/(ηf): 1.1 s at 100 Hz, 0.11 s at 1 kHz. Above 1.1 kHz: diffuse noise with the same T60(f). (S: η 1–3 %, plate-to-rib-strip transition at 1.1 kHz.)
  - A 3 ms pre-delay, so learned alignment can move either way.
- **Body and hall are in series**: `ir = body ⊛ (δ + hall)`. The hall hears what the soundboard radiates, so the 60–70 Hz radiation high-pass shapes the reverberant field too. (A parallel sum let the hall bypass the high-pass: the response at 27.5 Hz was only -4.5 dB instead of about -35 dB.)
- **Hall** (parametric, learnable per year): an octave-band noise tail with T60 per band starting from `(125,2.0) (250,1.8) (500,1.7) (1k,1.6) (2k,1.45) (4k,1.2) (8k,0.8)` s, onset at about 20 ms, direct-to-reverberant ratio about 0 dB, total length 2.5 s. (M; the venues are undocumented.)
- **Why parametric:** 7 T60s and 7 gains per year are far more identifiable than 24k free FIR taps.

## 5. Dampers and pedals

| parameter | prior | evidence |
|---|---|---|
| Undamped keys | Dampers up to k = 67 (E6). Soft edge: full strength to k = 62, 0.3 at k = 67, none from k = 68. Verify the boundary on MAESTRO. | M |
| Damper rate at partial 1 (1/s) | `(0,8) (12,10) (24,14) (36,20) (48,28) (60,36) (67,40)`, i.e. released A0 lingers about 0.9 s and A4 dies in about 0.25 s | M |
| Damper rate vs partial | α_d,n = α_d1 · min(n, 6)^0.6 | D/M |
| Sustain lift | Logistic, θ = 0.42, width 0.06. Damping scales with (1 − lift)^2.5 rather than linearly. | M; half-pedalling S (Lehtonen 2009) |
| Damper delay after note-off | +15 ms, fixed (`PianoConfig.damper_delay`). **Deviation**: it acts through hard frame-level key timing, which has no gradient, so instead of learning it we calibrate it per year from MAESTRO (section 8). | M |
| Sostenuto (CC66) | Latch the dampers of keys held at the moment the pedal crosses 0.5 upward; hold them until the pedal falls. Engagement = (1 − key_down)·(1 − lift)^2.5·(1 − latch). Known approximation: if a training window starts with the pedal already down, the latch takes the keys held at the window start (the true press happened earlier). The 1 s warm-up before the loss region absorbs most of this. | Standard mechanics |
| Una corda gain (dB) | 0 for monochords, −2 for bichords, −3.5 for trichords | D |
| Una corda aftersound multiplier | ×1 for monochords, ×5 for bichords, ×3.5 for trichords; brightness ×0.7 | D (strike-vector geometry) |
| Sympathetic resonance | All 88 keys × 4 partials; each key's damper state controls its decay. Drive ∝ the string's bridge-coupling rate κ = α_prompt − α_after (reciprocity: it gains energy through the bridge at the rate it loses it), so the resonant gain is G·κ/α, the bridge's share of the string's losses. (An earlier √α drive was wrong: it made the gain vary as 1/√α across the register.) Exclude each key's own strings from its drive. Target halo about -25 to -40 dB re the struck partial. | D (coupled modes); S (Lehtonen 2007) |

## 6. Mechanical noises

| event | timing | spectrum and envelope | level |
|---|---|---|---|
| Hammer / soundboard knock | at onset | Flat to 600 Hz, then -12 dB/oct. Amplitude time constant 10 ms (bass) to 5 ms (treble). | about -25 dB (ff) to -12 dB (pp) re the tone over the first 60 ms (M) |
| Key-bottom thump | onset + 12 ms at pp, −3 ms at ff, linear in v_h | Same dark spectrum, τ ≈ 8 ms | M |
| Damper / release noise | note-off + damper delay | Low-passed around 1.5 kHz, τ ≈ 5 ms | M |
| Pedal noise | when lift changes; amplitude ∝ \|d lift/dt\| | Low-passed around 800 Hz, τ ≈ 30 ms | about -35 dB (M) |

Timing is S (Askenfelt & Jansson); levels and spectra are M.

Noises are rendered in the time domain (band-split white noise under sample-accurate
envelopes), so the 5–10 ms decays and the thump timing are realised exactly and nothing
sounds before the hammer. An STFT-based version put 36% of the knock's energy before the
onset. Band levels are spectral densities, and the initial constants are calibrated so the
levels above hold at C4 (checked by diagnostics).

## 7. Acceptance checks (`python -m pianonn.diagnostics`)

The ranges are exactly those of this document; the current report is in
[`diagnostics_prior.md`](diagnostics_prior.md).

1. Prompt T60 of the fundamental (energy-decay-curve fit) inside the section 2 ranges at A0, C2, C4, A4, C6, C7 and C8.
   Aftersound T60 at C2, C4, A4, C6 and C7 is read from the model's mode parameters and labelled as such:
   beating between the aftersound modes makes a render-based fit meaningless.
2. Measured effective B at C4 is within 1.5× of 3.8e-4; at C6 within 1.5× of 3.5e-3.
3. At C4 mf, partials 2–6 fall roughly 0 to -30 dB re partial 1, and partial 10 is ≤ -30 dB.
4. Brightness (spectral centroid) rises monotonically with velocity at C2, C4, C6 and C7; at C4 it rises 1.1–2× from vel 40 to vel 120.
5. In the radiated signal, A0 and C2 fundamentals sit ≥ 10 dB below their strongest partial.
6. Released-note decay to -60 dB: A0 ≥ 0.5 s, A4 0.2–0.4 s. C8 released and held are identical (undamped).
7. The pedal halo is -25 to -40 dB re the strings with pedal, and at least 10 dB weaker without.
8. Noise: knock -25 ± 3 dB (ff) and -12 ± 3 dB (pp) re the tone; damper noise -35 to -45 dB re the released note;
   pedal-press noise -35 ± 5 dB re an mf note; under 1 % of the noise energy before the hammer strikes.

## 8. Data-driven calibration once MAESTRO is available (both reviews recommend this)

- Calibrate the damper delay after note-off per year.

- Estimate B, tuning and prompt/aftersound decays per key by partial tracking at known MIDI pitches.
- Find the damper boundary and per-key damper rates from notes released without pedal.
- Estimate hall T60(f) per year from decays after loud staccato chords.
- Check whether CC66 and CC67 are present in the MIDI at all.
