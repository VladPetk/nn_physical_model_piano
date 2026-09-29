# Review, round 2: Iowa calibration and spec v2 (commits 7ca07b2, 78402bc)

## Verdict

**Not ready to use as a MAESTRO prior yet.** Tests pass (20/20), and the diagnostics reproduce 62/65 exactly. Two problems block it:

1. **NaN gradients.** A C7 note at velocity 30 gives NaN gradients. After `clip_grad_norm_`, every gradient in the model is NaN.
2. **The decay calibration fits an artefact.** The "prompt T60" metric does not measure the early decay on real recordings. It reports about 10 s at C4. Measured directly, the recordings fall about 22 dB in the first second (a T60 of about 2.5–3 s). The model, fitted to the metric, falls only 5 dB. This is the N× trichord effect that `round2_sourced.md` itself predicted.

Several targets also compare different partials in the recording and in the model. The three "bass knee" failures come from this and from a metric floor, not from the model.

Reproduction scripts are in `scratchpad/rv/`.

## Critical

**C1. NaN gradients from overflow in the hammer roll-off.** `pianonn/physics.py:71` and `:207-209`.
- The problem: `(f*tc/0.59)**(2*order)` overflows to inf in float32. This happens for partials far above Nyquist, which are still evaluated before the `ok` mask. At inf, `d/d order = 0 * inf = NaN`.
- Evidence (`rv/nan_grad.py`), at the prior values:
  - NaN for k=87 at vel ≤ 54, k=75 at vel ≤ 35, k=63 at vel ≤ 9, and k=39 at vel 1.
  - Full model, MIDI 96 at vel 30: `raw_order` and `raw_order_vel` go NaN. `clip_grad_norm_` (train.py:109) then returns NaN, which poisons every parameter.
  - The v1 code used a fixed order of 2 and did not overflow.
  - `test_forward_backward` checks neither parameter, and it uses a small config.
- Fix:
  - Compute the spectrum in the log domain: `hammer = exp(-0.5*softplus(2*order*log(f*tc/0.59)))`. Alternatively, clamp `f` to 0.5·sr before the hammer.
  - Add a test that the gradients are finite for all 88 keys × velocities 1–127, with `raw_order` and `raw_order_vel` in the list.

**C2. "Prompt T60" (EDC −3…−13 dB) is not the prompt decay on recordings.** The calibration at `calibration.py:168` and `physics.py:94` (R) is built on it.
- Evidence (`rv/early_decay.py`, `rv/early_all.py`). I measured the level of partials 1–4 summed in power, 0.1 s smoothing, dB re peak. Median over all mf/ff notes, at 1 s after the onset:

  | register | recordings at 1 s | model at 1 s |
  |---|---|---|
  | A1–G#2 | −12 dB | C2 −2 dB |
  | A2–G#3 | −12 dB | C3 −3 dB |
  | A3–G#4 | −22 dB (IQR −25…−20) | C4 −5 dB, E4 −6 dB |
  | A4–G#5 | −20 dB | A4 −8 dB |

- The per-partial EDC values in a recording are bimodal. C4 mf gives `[2.1, 14.4, 5.7, 24.2]` s, and their median of about 10 s describes neither stage. The EDC can also land on a beat re-rise.
- In the treble the EDC gives prompt > aftersound (C7: 20 s against 4.7 s; C8: 28 s against 3 s), which is physically impossible. The author dropped C7 rather than question the metric.
- On an ideal double exponential the EDC works: Tp 2.5 s with a −24 dB knee reads 3.3 s. The failure is in real envelopes (beating, mic position).
- Consequence:
  - R (3–4.3) and the whole "prompt T60" row of the spec are artefacts.
  - Spec line 53 ("agrees with Weinreich E♭4 8 dB/s") compares against a **single string**. `round2_sourced.md` says a trichord should reach about 24 dB/s, T60 about 2.5 s. The recordings confirm that.
- Fix:
  - Replace the prompt metric with a two-segment fit on the power-summed envelope of partials 1–4, or with the level at 0.5/1/2 s.
  - Re-solve R. In the mid-range expect R ≈ 8–10.
  - Add the new metric as an acceptance check.

**C3. "Partials 1–4" and "knee of the fundamental" are taken by list index, not by partial number.** `calibration.py:278, 370`; `diagnostics.py:161, 168`.
- In the recordings the fundamental is not tracked in 44 of 64 notes at A0–G#2. Every A0 and C2 landmark note starts at n = 2 or 3. The dry model always starts at n = 1.
- So the bass T60 targets are partials 2–5 against the model's 1–4, and the "fundamental" knee targets at A0 and C2 (−29, −28) are the knee of partial 2 or 3.
- The model's knee changes with partial: at A0 it is −17 / −21.5 / −28.9 for n = 1/2/3.
- Fix: select partials by `decay_partials` value (for example n ∈ 2..5 for both), and compare knees on the same n.

## Major

**M1. The knee estimator has a floor near −25 dB and depends on the prompt rate.** `calibration.py:182-201`.
- The fit starts where the smoothed level crosses −25 dB, and extrapolating back only raises the intercept.
- Synthetic true → estimated knee: −29 → −23.5, −27 → −22.9, −25 → −19.5.
- The 0.5 s smoothing also lowers the "peak" more when the prompt is fast, as in the recordings, than in the slow model.
- The same fit window biases the aftersound T60: an ideal 113 s reads 91 s when the knee is at −29 dB.
- The bass targets (−27 to −29) sit at or below the floor. That, together with C3, explains the "non-monotonic" behaviour. Both b1 and the aftersound amplitude were tuned against these coupled, biased metrics.
- Fix: a two-segment regression with a free breakpoint, and the noise floor subtracted.

**M2. Summing the stereo channels to mono makes comb filtering and mic-position effects dominate.** `calibration.py:298` (`x.mean(1)`).
- Per-channel runs (`rv/chan.py`):

  | note | quantity | L | R | mono sum |
  |---|---|---|---|---|
  | C4 mf | slope (dB/oct) | −19.9 | −11.1 | −15.8 |
  | C4 mf | partial-1 prompt T60 (s) | 2.4 | 15.8 | 2.1 |
  | A4 mf | slope (dB/oct) | −15.3 | −20.1 | −22.8 (outside both) |
  | A4 mf | B | reliable 7.5e-4 | reliable 7.7e-4 | 5.8e-4, "unreliable" |
  | C2 mf | partial-2 prompt T60 (s) | 6.3 | 7.6 | 72.8 |

- Fix: analyse each channel and average the results, or use L²+R² power envelopes. Report the L/R spread as the measurement uncertainty.

**M3. The small, mislabelled bass sample drives the "A0" targets.**
- The A0 landmark (keys 21–24) comes from 3 notes on 2 keys (mf.B0, mf.C1, ff.C1). There is no mf.A0 or mf.Bb0.
- ff.A0 and ff.B0 sound a semitone high. ff.B0 correlates 0.93 with ff.C1, so it is probably the same take. pp.C1 is −111 cents (a B0).
- Bootstrap 90 % intervals of the target medians: A0 Tp 18.5–34 s, Ta 66–147 s, C3 Tp 9–27 s. These are wider than the ×/÷1.35 tolerance, so passing is partly luck.
- Fix: report the intervals, and treat A0 as extrapolated.

**M4. The treble "partials" are not string partials.**
- At C6–C7 the tracked peaks fit B = 3e-4…1e-3 with 10-cent residuals, against 3e-3…9e-3 expected. The tracker locks onto near-harmonic components (distortion or phantom partials).
- Partials 3–5 sit 45–80 dB down, which is near the noise floor. They are censored by SNR in the recordings but not in the model, whose floor is −140 dB.
- The treble slope targets, the treble hammer order (4.5), "b3 must fall in the treble" and the C7–C8 aftersound all rest on these components.
- Fix:
  - Constrain the treble tracker to Rigaud-range B.
  - Tag these rows M or low confidence.
  - Apply the same SNR censoring to the model.

**M5. Undocumented goalpost moves and circular checks.**
- Removed checks:
  - C4 partials 2–6 in −30…+3 and partial 10 ≤ −30. Both would still pass.
  - Analytic aftersound T60.
- The C1 knee and the C7 prompt are skipped silently. Spec §6 says A0–C7 for all four checks.
- The Rigaud targets moved from 3.8e-4 / 3.5e-3 to the prior's own values, so those checks, like the stretch checks (the prior copies the table, ±4 cents), are circular. They test only that the analysis round-trips.
- Say so in §6, or label them as regression tests.

## Minor

- **Key-bottom timing does not match the spec.** In `synth.py:120-127` the 9 m/s knot is unreachable (`hammer_velocity` ≤ 5.5). ff therefore gets at most −2.5 ms, not the spec's −5 ms. MIDI 64 gives +4.9 ms, not the "mf +0.5". Fix: map by MIDI velocity, or rescale the knots.
- **"mf" means two different things.** The physics uses 2.8 m/s (≈ vel 80), while calibration and diagnostics use vel 64. The per-key order was therefore fitted about 1.06× off its reference.
- **The early-spectrum window includes the knock.** The 10–200 ms window (`calibration.py:204`) takes in the knock and board transient, and the model's slope check uses a 1e-7 floor. The SNR censoring is asymmetric (see M4).
- **B agreement is overstated.** Spec §1 says Iowa agrees "within 10 %" at C4–C5. It is 8 % at C4, 14 % at A4 and 13 % at C5.
- **The Rigaud reference is second-hand.** The L tag is via DDSP-Piano: the literature doc says "I did not read [35] itself".

## Nits

- `_moving_average` divides by k at the edges. This is harmless as used.
- The body-cap comment says "T60 ~0.6 s" while the code gives 0.69 s. The cap applies below 159 Hz, and it is correct.

## Transfer to MAESTRO

What to keep as priors and what to loosen:

- **Keep:** Rigaud B (treble), strike position, T_c, the Hall C4 order anchor and its velocity law, and the unison mistuning range.
- **Loosen or re-derive:**
  - R and the prompt stage (C2). Start from the N× bridge loss.
  - Per-key order away from C4 (M2, M4). Near-field, mic-specific ±4 dB/oct, and degenerate with the learned per-year body FIR. Use a smooth monotone curve with a wide bound.
  - Treble b3 (M4).
  - Bass aftersound: 110 s and 0.06 amplitude, from 3 notes. In hall recordings under sustain pedal this sets the bass "wash", so check it on MAESTRO.
  - Stretch: it is Steinway-B specific. Keep the ±30 cent bounds, but don't treat the ±4-cent checks as validation.

## Spec v2 rows: traced?

| row | traced? | note |
|---|---|---|
| B formula | yes (partly for the I claim) | code = spec; L second-hand; "10 %" overstated |
| Stretch | yes | = measured re A4; C8 25 chosen over the measured +41.5 (stated as extrapolated) |
| Strings per key | yes | L + M, honest |
| Unison mistuning | yes | Weinreich Fig. 8 |
| b1 | partly | = code; I from a biased aftersound fit (M1) |
| b3 | partly | = code; treble I from non-string partials (M4) |
| R / prompt form | no | = code; I is an EDC artefact (C2); Weinreich comparison misapplied |
| Aftersound amplitude | partly | = code; tuned to biased metrics (M1) |
| Double decay, all partials | partly | knee estimator bias |
| Decay table | yes as numbers, no as physics | matches the JSON; C2, C3 |
| Force partial amplitude, strike, T_c, T_c(v) | yes | code = spec = literature doc |
| Force spectrum | yes | D |
| Order q at mf | partly | C4 reproduced exactly (−17.8/−14.2/−11.9 at vel 30/64/110); other keys mic-specific I |
| q vs speed, centroid ×1.22 | yes | |
| Level vs velocity, partial count | yes | |
| Attack (20/11 ms with body) | partly | not in the committed outputs |
| Body (modes, 0.07/Hz, 1.35 kHz, cap, envelope) | yes | Bader "band not stated" |
| Hall | yes | M |
| Dampers, delay, pedals, una corda, sympathetic | yes | |
| Noises | yes | |
| Key-bottom timing | partly | ff −5 ms unreachable; vel 64 ≠ +0.5 ms |
| §6 status 62/65 | yes | reproduced; C1 knee and C7 prompt silently skipped |
