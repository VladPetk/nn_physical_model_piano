# Review: applying docs/physical_parameters.md to the code (commit 8287ac4)

## Verdict

Most of the spec's tables were copied into the code faithfully. The core machinery is sound:
- `pytest`: 20/20 pass.
- Gradients are finite.
- The `torch.where` branches in `half_sine_spectrum` are safe.
- Block rendering matches single-pass rendering to about -90 dB with every new feature on (per-key exclusion, sostenuto, damper delay, pedals, noise). The largest relative error was 3.3e-5 on the strings and 0 on the noise, at block sizes 0.37 s and 2 s.

The "23/23 pass" result overstates what was achieved, for three reasons:
- The checks' target ranges were widened where the model misses the spec.
- Two checks read back the model's own parameters.
- Several spec rows are never measured, and three of those are badly off:
  - Pedal noise is about 31 dB too loud.
  - The hall path bypasses the radiation high-pass.
  - Treble brightness falls as velocity rises.

The sqrt(α) sympathetic drive rests on an arithmetic slip in the literature review.

## Findings

### Major

**M1. The hall bypasses the radiation high-pass** (`room.py:12-13, 110, 146-150`).
- Body and hall are summed in parallel. The hall's lowest band has no lower edge, because of the `if i > 0` at line 110, so the hall is flat down to DC.
- Measured, in dB relative to the body's 200–1000 Hz plateau:

  | frequency | body alone | full IR (body + hall) |
  |---|---|---|
  | 27.5 Hz (A0) | -30.2 | -3.4 |
  | 41 Hz | -17.7 | -1.7 |

- So the spec's cause #3 ("bass fundamentals too strong") is mostly undone. The A0 check passes at only 12 dB, where the literature says 15–30 dB.
- Physically the hall is excited by the radiated sound, not by the bridge force.
- Fix: put the hall in series (`ir = body + hall ∗ body`, or approximate that by shaping the hall carriers with `BODY_TARGET_DB`), and give band 0 a lower edge.

**M2. The sympathetic drive ∝ sqrt(α) is physically wrong** (`synth.py:75-78, 102`; spec §5).
- The bank takes in a bridge-force signal and outputs one. By reciprocity, the input and output coupling each go as sqrt(κ_r), so the transfer goes as κ_r, where κ_r is the bridge-loss rate.
- In the code the transfer at resonance is G·sqrt(α)/α = G/sqrt(α). That gives -16.7 dB at A0, -16.5 dB at C4 and -30.1 dB at C8, a spread set by α rather than by coupling.
- The review's claim that "∝ α starves the bass" is an arithmetic slip. The peak of G·α·t·e^(-αt) is G/e, not G·α/e. So the old ∝α drive was already α-neutral at resonance.
- Fix: `gin = G * (alpha[...,0] - alpha[...,1]) / sr`, i.e. κ = the additive bridge term, then retune G to the halo target. Correct spec §5 to match.

**M3. Pedal noise is about -4 dB re an mf note; the spec says about -35 dB** (`synth.py:133, 166-174`).
- Measured with a sustain step on C4 at vel 64: pedal noise over 70 ms is -4.3 dB re the note's RMS over its first 300 ms.
- MAESTRO has thousands of pedal steps, so every one would start out as a loud thump. No check covers this.
- Fix: lower `self.pedal` by about 3.5 nats (from -1.0 to -4.5) and add an acceptance check.

**M4. Acceptance ranges were widened to pass** (`diagnostics.py:20-21`; `tests/test_model.py:145-146`).

| key | spec prompt T60 | range used in the check | measured |
|---|---|---|---|
| A0 | 25–40 s | 20–40 s | 23.2 s (analytic 23.0) |
| A4 | 5–6 s | 5–6.5 s | 5.46 s |
| C7 | 1.5–2 s | 1.2–2.0 s | 1.3 s |

- Against the spec's own ranges the check is 21/23.
- Spec rows the checks leave out also miss:
  - C2 prompt: 13.8 s against 15–20 s.
  - C8: 0.67 s against 0.7–1.2 s.
- So the spec's b1 list does not reproduce its own T60 table.
- At C6 and C7 the EDC measurement reads about 12–15 % below the analytic value (2.62 vs 2.96 s; 1.30 vs 1.52 s). Beating between the prompt and aftersound modes biases the -3 to -13 dB fit.
- Fix: restore the spec ranges and add C2, C3, C8 and C4 partial 10 (analytic 3.73 s, which passes). Then refit, e.g. b1 = 0.184 at A0, about 0.23 at C2 and about 1.9 at C8, or amend the spec table explicitly.

**M5. Treble brightness falls with velocity; fill=0.5 does not fix it above C4** (`physics.py:55-66`).
- I swept velocity 20→127 on hammer spectrum × comb:

  | key | fill | centroid (Hz) | steps where centroid decreases |
  |---|---|---|---|
  | C3 | 0.15 | — | 40/199 |
  | C3 | 0.5 | — | 2/199 |
  | C4 | 0.15 | — | 63/199 |
  | C4 | 0.5 | — | 28/199 |
  | A4 | 0.5 | — | 88/199 |
  | C6 | 0.5 | 2945 → 2293 | — (darker as velocity rises) |

- The diagnostics table shows the same: C6 centroid 2537 / 2070 / 2153 Hz at vel 40 / 80 / 120.
- Cause: f1·T_c > 0.6 from about A4 upward (1.15 at C6, 1.68 at C7), so the fundamental sits in the pulse's sidelobes. That also costs -14 to -25 dB of level, which the loudness prior then has to make up (see m1).
- The fill=0.5 choice is justified for k ≤ 39 but is neither documented in the spec nor sufficient above that.
- The spec's own formula is wrong: sqrt(cos² + ε²)/|1-4x²| has a pole at fT_c = 0.5 (+11.8 dB at 0.49, infinite at 0.5). The code's envelope-relative fill is the right repair, but it is undocumented.
- Fix: use a monotone magnitude for the velocity dependence, e.g. 1/sqrt(1+(fT_c/0.59)^4), which has -3 dB at 0.59/T_c and -12 dB/oct. The envelope-only variant I tested gave 0/199 decreasing steps at A4, C6 and C7. Keep the lobe structure, if at all, as a static per-key shape. Document the change in the spec.

**M6. The knock starts before the hammer; the specified τ and thump timing cannot be realised** (`synth.py:145-150, 176-180`).
- Noise is synthesised with a centred 512-sample (21 ms) istft on a 5 ms frame grid.
- Averaged over 20 seeds, 36.5 % of the knock energy arrives before the onset: -7.4 dB at -5 ms and -39 dB at -10 ms.
- The thump's +12 to -3 ms timing spans only 3 frames and is smeared by the window.
- The knock check integrates from the onset, so it cannot see this.
- Fix: synthesise the transient events in the time domain (sample-rate envelope × filtered noise), or use a short FFT for them.

### Minor

- **m1. Loudness prior** (`physics.py:133-137`), which is not in the spec.
  - It equalises total decay energy, not early loudness, and its ±30 dB clamp is hit at k ≥ 71 (17 keys).
  - Radiated RMS over 0–300 ms at vel 64: A0 -54.9, C4 -46.0, C6 -44.8, C8 -64.8 dB.
  - Fix: equalise energy over about 0.3 s (amp²(1-e^(-0.6α))/2α) once M5 is fixed, which removes most of the treble deficit.
- **m2. Knock level against velocity**: the noise-to-tone spread from vel 25 to 120 is 24.7 dB; the spec says about 13 dB (-12 to -25 dB).
  - The ±8 dB tolerance hides it. The vel 120 value of -32.5 dB passes with 0.5 dB to spare.
  - Fix: `knock_vel` about 3.8 (currently 2.0), re-centred; tighten the tolerance to ±4 dB.
- **m3. Damper delay** is a fixed `cfg.damper_delay` (`config.py:245`, `synth.py:302`). The spec asks for it to be learnable per year, bounded -30 to +80 ms. This deviation is undocumented, apart from a README TODO.
- **m4. Sostenuto at the segment start** (`synth.py:221-230`, `data.py:64-78`).
  - Pedal curves start at t0, so a pedal already down fires a "rising edge" at frame 0 and latches whatever keys are held there.
  - Full renders latch at the real press time, so training and render diverge whenever CC66 exists.
  - Fix: compute the latch over the whole piece during data prep, or start the pedal curves at t0 minus the lookback.
- **m5. Noise and block rendering after training**: noise selection uses a hard 1 s window (`synth.py:280`), but `knock_log_tau` and `release_log_tau` are unbounded. With τ = 0.4 s, block and single-pass noise differ by 5 % (relative max). Fix: bound the τ values or derive the window from them.
- **m6. 96 partials cost +29 % per training step**: forward+backward for B=2, 3 s, 24 notes on CPU takes 62 s with 64 partials and 80 s with 96. For pitches 40–90 only 35.5 % of partial slots are below 0.48·sr; the rest are computed and multiplied by zero. Fix: a per-register partial count, or gather only the valid partials.
- **m7. Circular or weak checks.**
  - "Aftersound T60, model" is 6.91/α read back from the prior. It is labelled "model", but it verifies a table lookup, not the rendered signal.
  - "C8 undamped" tests a buffer that is 0.
  - "B_eff" searches ±1.5 % around the model's own frequencies, so it only tests the plumbing.
  - The halo (-27 dB) sits at the loud edge of its target range.

### Nits

- Release noise ignores the sostenuto latch (`synth.py:275`).
- `pedal_log_power` is unbounded; the review suggested 1–4.
- Strings use a piecewise-constant damper integral, while the sympathetic bank uses linearly interpolated engagement, about half a frame apart.
- The sign of the half-sine lobes is discarded; only its magnitude is kept.

## Spec rows → implemented?

| row | status |
|---|---|
| §2 B, stretch, strings/key, unison, b3, aftersound amplitude | yes |
| §2 b1 values | yes (they miss the spec's own T60 table at A0, C2, C8: silent) |
| §2 prompt, additive with R | yes |
| §2 T60 table | deviation-silent (checks widened) |
| §2 C4 partial 10, 3–4 s | yes (3.73 s, not checked) |
| §3 signed comb, no 1/n | yes |
| §3 force spectrum, ε 0.15 | deviation, undocumented in spec (fill 0.5; spec formula diverges) |
| §3 T_c, v_h, q=0.25, strike position, attack ramp, 96 partials | yes |
| §3 level against velocity | yes, plus an unspecified loudness equalisation (m1) |
| §4 body FIR (envelope within 3.4 dB, η, modes, 3 ms pre-delay) | yes |
| §4 hall (T60s, onset, D/R 0 dB, 2.5 s) | yes; parallel topology bypasses the radiation high-pass (M1) |
| §5 damper edge, rates, tilt, lift/θ/power | yes |
| §5 damper delay, learnable per year | deviation-silent (fixed) |
| §5 sostenuto | yes (m4) |
| §5 una corda gain, aftersound, brightness | yes |
| §5 sympathetic: 88×4, exclude own, sqrt(α) | yes, but the spec's physics is wrong (M2) |
| §6 knock spectrum and τ | parameters yes; realised timing no (M6); level slope off (m2) |
| §6 thump timing | parameters yes; not realisable at the 5 ms / 21 ms grid (M6) |
| §6 release noise | yes |
| §6 pedal noise shape | yes; level deviation-silent (M3) |
| §2 phase-2 items | not implemented, as the spec intends |
