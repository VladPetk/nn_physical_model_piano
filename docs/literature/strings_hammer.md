# Literature priors: strings and hammer of a concert grand (for `pianonn/physics.py`)

Scope: strings and hammer excitation. Key index k = MIDI - 21 (A0 = 0, C1 = 3, A1 = 12, C2 = 15, A2 = 24, C3 = 27, A3 = 36, C4 = 39, A4 = 48, C5 = 51, A5 = 60, C6 = 63, A6 = 72, C7 = 75, C8 = 87).

## 0. Verification status (read first)

Web access was mostly blocked. WebFetch was refused (egress proxy) for euphonics.org, arxiv.org, kth.se, bme.hu, espci.fr and kent.edu. Only WebSearch snippets were available, and none of them contained the big parameter tables (Chaigne & Askenfelt Tables I/II, Rigaud et al. fitted values, Conklin scale data).

Facts confirmed by a search snippet (marked **[snippet]**):
- Hall & Askenfelt static hammer tests: F = K u^p with p between 2.2 and 3.5 for real voiced hammers (Hertz would be 1.5). Quoted in Stulov's papers.
- Conklin (KTH "piano design factors" page): the textbook strike point is 1/7–1/9 of L. In modern pianos d/L in the bass is a little less than 1/8, falls gradually up to about A4 (k = 48), then falls quickly. In the best grands the top treble is 1/12 to 1/17.
- Askenfelt & Jansson: hammer-string contact is about 2 ms in typical playing. Contact duration varies about +/-20 % between p and ff around mf, and shortens as level rises. Contact is short compared with the fundamental period in the bass and long in the treble. Final hammer velocity is about 1 m/s (piano) to about 5 m/s (forte). Key travel time is about 160 ms at piano and 25 ms at forte.
- Weinreich (KTH page on coupled strings): the prompt sound falls at about 8 dB/s and the aftersound at less than one quarter of that rate. The cause is the coupling of the 2–3 strings of a unison. The in-phase mode radiates strongly and the out-of-phase modes weakly.
- Rigaud, David & Daudet, JASA 133(5), 3107 (2013): 2-parameter model of B(m) from string-design physics (two asymptotes, bass and treble) and a 4-parameter tuning model. The snippet gave no numbers.
- Bank & Sujbert, JASA 117(4), 2268 (2005): phantom partials sit at sums and differences of transverse partial frequencies. Longitudinal vibration is responsible for the "metallic" character of low notes and for the attack.
- Conklin 1996 (JASA 99, 3286, Part I hammers; JASA 100, 1286, Part III strings and scale design) covers longitudinal modes, wound strings, inharmonicity and phantom partials.

Everything else is from memory and marked **(unverified, from memory)**. Confidence: high / medium / low.

Note on the sample rate: the code runs at 24 kHz with 64 partials. Partial 64 of A0 is only about 1.8 kHz, and of C2 about 4.2 kHz. Bass notes therefore lose the 2–10 kHz "bite" that real bass notes have (see 11).

---

## 1. Inharmonicity B

Formula: f_n = n f0 sqrt(1 + B n^2), with B = pi^3 E d^4 / (64 T L^2) for a plain round wire. Here d is the wire diameter, T the tension, L the speaking length, and E about 2.0e11 Pa for piano steel (E, d, T and L are as in Fletcher & Rossing; standard formula, high confidence). For wound strings only the core stiffens the string while the winding adds mass. Effective B is therefore of order (core d)^4 / (total-mass-dependent T).

Scale data. I could not verify any exact set for a Steinway D, Yamaha CFX or Conklin scale.

| Item | Value | Confidence |
|---|---|---|
| Plain wire E | 2.0e11 Pa (unverified, from memory) | medium |
| Wire d, treble plain | about 0.8–1.0 mm (Roslau gauges) | medium |
| Wire d, tenor plain | about 1.0–1.35 mm | low |
| Wound bass core d | about 1.6–2.0 mm; total 6–8 mm | low |
| Tension per string | about 650–800 N (plain), about 700–1000 N (bass); total plate load about 20 tonnes+ | medium |
| Speaking length L, A0 | about 1.95–2.0 m (9' grand, Steinway D) | medium |
| L, C2 | about 1.2–1.5 m | low |
| L, C4 | about 0.62–0.65 m | medium |
| L, A4 | about 0.39–0.41 m | medium |
| L, C6 | about 0.19–0.21 m | medium |
| L, C7 | about 0.098–0.10 m | medium |
| L, C8 | about 0.05 m | medium |
| Total mass per string M_s, C4 | about 3.9 g (from L = 0.63 m, T about 670 N, f0 = 261.6) | medium |
| Chaigne & Askenfelt (1994) strings | C4: L = 0.63 m, M_s = 3.93 g, T about 670 N, B about 3.8e-4. C7: L about 0.10 m, M_s about 0.47 g. C2: L about 1.9 m, M_s about 35 g. The C2 and C7 tensions do not reproduce f0 exactly from my recalled masses, so treat C2 and C7 as unreliable. | low |

Derived B from the scale formula (my calculation). The wire diameters and tensions are assumptions:

| Key | B computed |
|---|---|
| A4 | 8e-4 |
| C6 | 2.3e-3 |
| C7 | 5.7e-3 |
| C8 | 2.3e-2, before the clamped-end correction |

Clamped ends (Fletcher): f_n ~ n f0 sqrt(1 + B n^2) (1 + 2/xi + ...) with xi = L sqrt(T/EI). At C8, xi is about 21, so the effective B measured on real C8 is larger than the bulk formula, plausibly 3e-2–1e-1 (low confidence).

### Measured values
Range quoted in a search snippet: B about 2e-4 (bass) to a very large value in the top treble. The snippet's "0.4" is not credible for a concert grand and is possibly an upright or top-of-scale value.

Shape (medium confidence, Rigaud et al. 2013 structure; slopes from memory):
- log B is piecewise linear in key with two asymptotes, ln B_bass falling toward the bass break and ln B_treble rising.
- The treble slope is about 0.09–0.115 ln-units/semitone (a factor of about 3–4 per octave; low-medium confidence). This is consistent with constant T and d, where L^-2 alone gives a factor of 4 per octave.
- The minimum is around k = 22–32 (F2–D3). It is about 1–1.5e-4 on a big grand and higher on short grands or uprights.
- The first plain strings above the break jump up by about 1.5–2x relative to the last wound strings (real, not in the smooth model).
- Between-piano variation is a factor of about 2 in B (larger pianos have lower B).

### Recommended prior, log10 B (fits `prior_log_B` as `key_curve` points)
```
(0,-3.52) (6,-3.66) (14,-3.85) (22,-3.96) (29,-3.92) (33,-3.70)
(39,-3.42) (48,-3.07) (56,-2.74) (63,-2.46) (75,-2.00) (87,-1.40)
```
This is B = 3.0e-4 at A0; 1.1e-4 at k = 22; 3.8e-4 at C4; 8.5e-4 at A4; 3.5e-3 at C6; 1.0e-2 at C7; 4e-2 at C8. Confidence is medium for keys 0–48, low-medium above.

Current prior (log10): (0,-3.6) (25,-4.0) (45,-3.7) (65,-3.1) (87,-2.0). Compared with the recommended curve:
- Bass and tenor are about right (within 1.5x).
- C4 is 2.4x too low (1.6e-4 vs 3.8e-4).
- The treble is 3–6x too low: C6 6e-4 vs 3.5e-3, C7 2.5e-3 vs 1e-2, C8 1e-2 vs 4e-2.
- Effect: partial n = 8 at C6 is stretched by 0.5 * B * n^2 = 0.11 (about 190 cents) in reality but only about 33 cents in the code.

---

## 2. Tuning stretch (Railsback curve)

Railsback (1938), as reproduced in Fletcher & Rossing and elsewhere (values approximate, unverified, from memory). Deviation of the fundamental from equal temperament with A4 = 440:

| Key | Deviation |
|---|---|
| A0 (0) | -30 cents (up to -35) |
| A1 (12) | -18 |
| A2 (24) | -8 |
| A3 (36) | -3 |
| A4 (48) | 0 |
| A5 (60) | +5 |
| A6 (72) | +13 |
| A7 (84) | +25 |
| C8 (87) | +30 (up to +35) |

Confidence is medium. Modern electronic-tuning-device tunings and Rigaud's fitted curves (which use a 4-parameter model where octave type moves from 2:1 toward 4:2 to 6:3) give similar shapes. Disklavier concert grands are tuned by technicians for recording, so expect the extremes at the mild end of this range (about -25 / +25).

Recommended `prior_cents` points: `(0,-30) (12,-18) (24,-8) (36,-3) (48,0) (60,5) (72,13) (84,25) (87,30)`.

Current: `(0,-20) (24,-5) (48,0) (66,4) (87,25)`. It is about 10 cents too flat at A0, 3 cents at A2, and too small at A5–A6 (2 vs 5, 10 vs 13). This is a small error. The tuner sets partial pitches by ear, so the fundamental deviation in the bass is partly a consequence of B.

Code note: the code applies `cents` to f0 (the "ideal" frequency) and B enters separately. Measured tuning curves refer to the fundamental of the stiff string, f_1 = f0 sqrt(1+B). That differs by less than 0.5 * B (under 1 cent for B < 1e-3, up to 5–10 cents at C8). Ignore it.

---

## 3. Strings per key

Approximate layout of a modern 9' concert grand (Steinway D, Yamaha CFX; unverified, from memory):

| Item | Value | Confidence |
|---|---|---|
| Monochord (single, wound) | about k = 0–7 (A0–E1), possibly to k = 9 | low-medium |
| Bichord (wound) | about k = 8–25 | low |
| Trichord | from about k = 26 (B2/C3) to k = 87 | low-medium |
| Total strings | about 230 (Steinway D quoted at about 243; Yamaha CFX uncertain) | low |

The bass and tenor break on a 9' grand is around F2–C3 (k about 21–27). The trichord start matches the switch to plain strings, roughly.

The code uses `k < 10 → 1`, `k < 28 → 2`, else 3. That is within about 2 keys of the layout above, which is good enough. I recommend `k < 8 → 1`, `k < 26 → 2`, else 3.

Monochords still have two polarisations (code already clamps n_active to at least 2). In the bass, vertical polarisation couples to the bridge and decays faster; horizontal decays slower. Ratio: unverified, from memory, about 2–6 (low).

---

## 4. Unison mistuning

| Parameter | Value | Source | Confidence |
|---|---|---|---|
| Mechanical string-to-string mistuning in a unison in normal concert tuning | about 0.2–2 cents typical; beat rates 0.1–2 Hz. Tuners aim for slight detuning "to sweeten" and avoid dead unisons | Weinreich 1977, Kirk 1959 (JASA 31, tuning preferences for piano unison groups; details unverified, from memory) | low |
| Effect | Bridge coupling pulls strings with mistuning below the coupling width together, so the effective (observable) detuning in the eigenmodes is smaller than the mechanical one | Weinreich | medium |
| Audible effect | Beating and slow amplitude modulation at 0.1–2 Hz on partials, and the aftersound. A perfectly tuned unison gives a short single-slope decay ("dead"); detuning of a few tenths of a Hz gives a two-slope decay | Weinreich | medium |
| Scaling | Mistuning is roughly constant in cents. The beat rate rises proportionally with frequency, so upper partials beat faster | | medium |

Code prior: modes 1..M-1 at (+0.6, -0.4, +0.2, -0.1) cents relative to mode 0. That is inside the plausible range. Recommend a per-key random spread of about 0.3–1.5 cents (sigma about 0.7 cents) for the bass and treble, since it is what a tuner leaves. Use the same cents value for all partials of a string, so the beat rate grows with n.

---

## 5. Loss and decay

Conventions: amplitude decay exp(-alpha t), T60 = 6.91 / alpha. Chaigne & Askenfelt use sigma_n = b1 + b3 omega_n^2 (b1 in s^-1, b3 in s). The code's b3 is the coefficient of f^2 (Hz^-2), so b3_code = 4 pi^2 b3_CA.

| Parameter | Value | Source | Confidence |
|---|---|---|---|
| C&A (1994) b1, C2 | 0.5 s^-1 | (unverified, from memory) | low |
| C&A b1, C4 | about 1.1 s^-1 | (unverified, from memory) | low |
| C&A b3, C2 | 6.25e-9 s (b3_code about 2.5e-7 Hz^-2) | (unverified, from memory) | low |
| C&A b3, C4 | about 2.7e-9 s (b3_code about 1.1e-7 Hz^-2) | (unverified, from memory) | low |
| C&A b1, b3, C7 | values I do not remember | | |

Consequence: at C4 the fundamental has T60 of about 6 s, and partial 10 (2.6 kHz) about 3.8 s. The code's b3 = 3e-7 is close to the C&A value for C2 and about 2.7x too large for C4.

### Measured T60 of the fundamental, concert grand, dampers up (approximate; unverified, from memory; low-medium)
Askenfelt, Fletcher et al. (1962), Bank & Valimaki (2003) measured examples. The prompt figures below are the early decay and the aftersound figures the late slope.

| Key | Prompt T60 | Aftersound T60 |
|---|---|---|
| A0 | 25–40 s (not separable) | (same) |
| C1 | about 25 s | (same) |
| C2 (bichord) | 15–20 s | 20–30 s |
| C3 | 10–15 s | 25–40 s |
| C4 | 6–8 s | 20–35 s |
| A4 | 5–6 s | 15–30 s |
| C5 | 4–5 s | 12–25 s |
| C6 | 2.5–3.5 s | 8–15 s |
| C7 | 1.5–2 s | 3–6 s |
| C8 | 0.7–1.2 s | (about the same) |

Higher partials: T60 falls roughly like 1/f to 1/f^1.5 through the audible band. Partial 10 at C4 about 3–4 s, and partials above 5 kHz under 1 s (unverified, from memory). Wound-string losses are higher, and high partials of bass notes decay faster than the b1 + b3 f^2 form suggests. Plan on a learned b3 per key.

### Double decay (Weinreich 1977; Sci. Am. 1979)
- Weinreich, KTH lecture page **[snippet]**: prompt about 8 dB/s, aftersound under one quarter of that (that is, at most about 2 dB/s). This is presumably for a mid-range trichord.
- Physical picture: alpha_prompt = alpha_int + N gamma_b, alpha_after = alpha_int, where gamma_b is the per-string bridge-loss rate and alpha_int the internal + air losses. In the mid-treble the bridge dominates, so the ratio prompt/after is about 3–8 for trichords, about 2–4 for bichords and about 1–2 for monochords (plus polarisation splitting).
- Level of the aftersound relative to the initial: the knee usually appears 20–35 dB down, after about 2–6 s for mid notes (unverified, from memory, low). The knee time is t_k = ln(1/a) / (alpha_p - alpha_a). With a = 0.05, alpha_p = 1.0, alpha_a = 0.3: 4.3 s.
- Weinreich's theory (bridge admittance, coupling): for identical strings the in-phase mode has gamma = N gamma_b, and the N-1 other modes have gamma_b -> 0. Mistuning delta mixes them once |delta| is comparable to N gamma_b / (2 pi) (in Hz). With gamma_b of order 0.3–1 s^-1, that is about 0.05–0.3 Hz. This is the same order as tuner mistuning. Bridge admittance real part is the other agent's domain.

Current code comparison:
- `prior_log_b1`: (0,0.25) (40,0.5) (60,1.0) (87,5.0) s^-1 as alpha_after; prompt = 3x that.
- At C4: alpha_after about 0.5 (T60 = 14 s) and prompt about 1.5 (T60 = 4.6 s). The real C4 aftersound T60 is 20–35 s and the prompt 6–8 s, so both decay about 1.5x too fast, and the ratio 3 is too small (should be about 4–6).
- At C6 and C7 the code's decays are about 2x too fast compared with the table above.
- The aftersound amplitude 0.2 per aftermode (2 modes, -14 dB each) is higher than the real 20–35 dB down (0.02–0.1).

Recommended `prior_log_b1` (alpha_after in s^-1, T60_after = 6.91/alpha): `(0,0.20) (15,0.28) (27,0.25) (39,0.30) (48,0.35) (63,0.60) (75,1.3) (87,3.0)`, with prompt ratio (1 + exp(...)) about 5 for k >= 26, about 3 for k 8–25, about 1.5 for k < 8. The code computes `prompt = 1 + 2*exp(raw)`, so the prior offset becomes `log(4)` for trichords (giving 1 + 4 = 5).

Recommended b3: `log(1.2e-7)` for k >= 26 (T60 at 3 kHz about 3 s for C4) and `log(2.5e-7)` for k < 20. Replace the constant `log(3e-7)`.

Polarisation: a monochord's two polarisations are two modes with different bridge coupling, which double-decay in the same way (unverified, from memory). Horizontal polarisation is excited only if the hammer hits off-axis, so it starts 10–20 dB down (low).

---

## 6. Hammer

Model: F = K [delta^p] (power law; Hunt & Crossley or Hall) with felt hysteresis (Stulov 1995: F = K [u^p - (eps/tau) integral of u^p exp(-(t-s)/tau) ds]; details of eps and tau unverified, from memory; eps about 0.5–0.9 and tau about 1–10 ms are the ranges I recall, low confidence).

| Parameter | Value | Source | Confidence |
|---|---|---|---|
| p | 2.2–3.5 for real voiced hammers **[snippet]** (Hall & Askenfelt); Hertz 1.5 | Hall & Askenfelt 1988; Stulov | high |
| p (C&A), C2 / C4 / C7 | about 2.3 / 2.5 / 3.0 | Chaigne & Askenfelt 1994 (unverified, from memory) | low |
| K (C&A), C2 / C4 / C7 (N/m^p) | about 4e8 / 4.5e9 / 1e11 (units differ because p differs). At 1 mm compression that gives about 50 N / 140 N / 100 N. | (unverified, from memory) | low |
| Hammer mass (C&A), C2 / C4 / C7 | about 4.9 g / 2.97 g / 2.2 g | (unverified, from memory) | low-medium |
| Hammer head mass, real Steinway-type | bass about 10–13 g (incl. shank contribution), tenor 5–7 g, treble 2–3 g | Conklin 1996 Part I; Fletcher & Rossing (unverified, from memory) | low |
| Felt stiffness trend | Increases from bass to treble and with compression (hammers in the treble are smaller and harder; voicing changes it a lot, by a factor of 10 or more) | Hall 1987; Conklin | medium |
| Hammer/string mass ratio, C&A | about 0.14 (C2), about 0.75 (C4), about 4.7 (C7) | derived from my recalled masses | low |
| Hammer velocity | about 0.5–1 m/s (pp) to about 5 m/s (ff); up to 6–7 m/s for fff | Askenfelt & Jansson **[snippet]** | medium-high |
| Contact duration | about 2 ms typical **[snippet]**; +/-20 % between p and ff around mf **[snippet]** | Askenfelt & Jansson 1990–1993 | medium-high |

Contact time vs key (unverified, from memory, low-medium): bass about 3–4 ms at mf, mid about 1.5–2.5 ms, upper treble about 0.6–1 ms. Askenfelt & Jansson **[snippet]** say contact is short compared to the fundamental period in the bass and long in the treble (contact spans more than one period at C7).

### Velocity dependence
- For a power-law felt against a rigid wall, T_c = 2 (x_max / v) * B(1/(p+1), 1/2) / (p+1), with x_max = ((p+1) m v^2 / (2K))^(1/(p+1)). That gives T_c proportional to v^-(p-1)/(p+1): exponent -0.39 (p = 2.3), -0.43 (p = 2.5), -0.50 (p = 3.0). I computed the constant: T_c = 2.74 x_max/v (p = 2.3), 2.70 (p = 2.5), 2.62 (p = 3).
- With the C&A parameters, the rigid-wall estimate at v = 2 m/s is about 0.8–1.2 ms for C2, C4 and C7, shorter than the measured 1.5–2.5 ms because the string yields (the hammer stays in contact through reflections). Use measured T_c as the prior target, and the formula only for velocity scaling.
- Measured: T_c changes only about +/-20 % from mf to p/ff **[snippet]**, weaker than v^-0.43 over v = 1 to 5 m/s (which would be a factor 2). Use an effective exponent of about -0.2 to -0.3 (low-medium confidence).

### Force pulse spectrum
- The force pulse is approximately half-sine (skewed and asymmetric due to hysteresis, and with secondary bumps from string reflections). For a half-sine of duration T_c: |F(f)| proportional to |cos(pi f T_c)| / |1 - (2 f T_c)^2|. I computed:
  - The -3 dB point is at f T_c = 0.59.
  - The first null is at f = 1.5/T_c.
  - The asymptote is -12 dB/octave (1/f^2), with nulls at f = (k + 1/2)/T_c.
  - Level relative to DC at f T_c = 1, 2, 3, 4, 6, 8 (mid-lobe): -9.5, -23.5, -31, -36, -43, -48 dB.
- Real force pulses fill the nulls. Hall's measurements show smooth roll-off with weak notches, so use a "filled" version (see section C).
- Spectral centroid: proportional to 1/T_c, so scales with v^(+0.2 to +0.4). Between pp and ff it moves the -3 dB point by about 1.5–2x, and high partials (near 3–5 kHz) grow faster than the fundamental with dynamic level (roughly +1.5–2 dB per dB of hammer speed compared with +1 dB per dB for the fundamental; unverified, from memory, low).
- The current code's fc corresponds to T_c: A0 about 2.2 ms, k = 40 about 1 ms, k = 87 about 0.22 ms. Its velocity scaling is fc proportional to exp(2 * u0) = 5.4x over the velocity range, more than the measured 1.5–2x.

Recommended contact-time prior at mf (v about 2 m/s), for use with the half-sine spectrum: `(0,3.5) (15,3.0) (27,2.4) (39,1.9) (51,1.5) (63,1.1) (75,0.8) (87,0.6)` ms. Confidence: low-medium.

Equivalent fc for the code's existing form (1 + (f/fc)^2)^(-r/2) with r = 1.5: fc = 0.78 / T_c: k = 0: 220; k = 15: 260; k = 39: 410; k = 63: 710; k = 87: 1300 Hz. That is substantially darker than the current prior (350, 800 at k = 40, 3500 at k = 87).

Velocity mapping (my suggestion, not from a source): v_h = 5.0 * (vel/127)^1.3 m/s gives about 0.45 m/s at vel 20, about 2 m/s at vel 64, and 5 m/s at vel 127. MAESTRO's Disklavier velocity to hammer speed relation is not something I could verify (belongs to the Disklavier agent).

---

## 7. Striking position x0/L

Source: Conklin **[snippet]**: bass a little under 1/8 (about 0.12), gradually decreasing to about A4 (k = 48), then quickly decreasing; best modern grands end at 1/12 to 1/17 in the top treble. C&A used about 0.12 for their three notes (unverified, from memory, low).

Recommended `x0/L` points (medium in the bass and middle, low above): `(0,0.125) (15,0.12) (27,0.115) (39,0.11) (48,0.105) (56,0.095) (63,0.088) (75,0.075) (87,0.06)`.

In the code, `x0 = 0.12 * exp(bounded(raw_strike, 0.7))`: a constant 0.12. Replace 0.12 by a `key_curve`. Note the strike position is measured from the agraffe (hammer end); the comb is symmetric under x0 -> L - x0 for |sin|. Real hammer contact is not at one point; the effective x0 differs somewhat from the felt centre.

Also, a well-known consequence: the comb null falls at partial n = round(L/x0): with 0.12 that is partial 8 (0.96 pi, |sin| = 0.13), and 0.06 at C8 gives partial 16 (irrelevant at 24 kHz).

---

## 8. Partial amplitudes at the bridge vs displacement

Derivation (standard result: Hall 1986–87 "Piano string excitation" series; Fletcher & Rossing; I re-derived it here so it is high confidence for the ideal-string case).

Modal equation for a string of mass M, y = sum q_n sin(n pi x / L), force F(t) at x0:

q_n'' + omega_n^2 q_n = (2/M) F(t) sin(n pi x0 / L)

After the force pulse ends:
- displacement amplitude: |q_n| = (2 / (M omega_n)) |sin(n pi x0/L)| |F_hat(omega_n)|
- string velocity amplitude: (2/M) |sin(n pi x0/L)| |F_hat(omega_n)| (no 1/n)
- bridge force amplitude (T times end slope, using (n pi/L)/omega_n = 1/c and T/c = Z): A_n = (2Z/M) |sin(n pi x0/L)| |F_hat(omega_n)| = 4 f0 |sin(n pi x0/L)| |F_hat(omega_n)| (no 1/n)

where F_hat(omega) = integral F(t) exp(i omega t) dt and F_hat(0) = J, the impulse (about m_h v_h (1+e)).

So the answer to the question asked: yes, for the force at the bridge (and for string velocity) the partial amplitude is proportional to |sin(n pi x0/L)| times the hammer spectrum with no 1/n factor. Displacement carries an extra 1/n (that is the plucked-string-like envelope, wrong for bridge force). The code's `hammer * comb` has the right form for bridge force. The radiated sound then multiplies by soundboard admittance and radiation (other agent's domain).

Sanity check: C4, J about 0.008 N s, sin(pi * 0.11) = 0.34, A_1 = 4 * 262 * 0.008 * 0.34 = 2.8 N transverse bridge force. Plausible (unverified).

Corrections:
1. Inharmonic string: factor 1/sqrt(1 + B n^2) (from c_n rising with n). Negligible except in the treble.
2. Finite hammer width w (force spread over the contact patch): multiply by sinc(n pi w / (2L)) = sin(x)/x with x = n pi w / (2L). First null at n = 2L/w. For w of about 5–10 mm and L = 0.63 m, n about 125–250 (way above the audible range at 24 kHz). It matters only for the high partials of the bass on real strings and is negligible here (unverified estimate of w, from memory; Hall 1987 / Chaigne & Askenfelt say it acts as a mild low-pass).
3. Hammer contact is not instantaneous relative to the reflected waves: while the hammer is in contact, reflections from the agraffe return after 2 x0 / c = T0 (x0/L) * 2 / ... = T0 * 2 x0/L (a quarter to a fifth of the fundamental period for x0/L about 0.1–0.125). In the bass T_c (3–4 ms) is larger than that (about 3.8 ms at C2). This modifies the force pulse (it ends earlier) and moves the spectral null pattern away from the ideal comb (Hall; Chaigne & Askenfelt).
4. The code's `.abs()` on the comb throws away the sign. Real modal amplitudes alternate sign as sin changes sign, which changes the summed waveform (peakiness) but not the spectrum magnitude. It is a minor issue (use the signed sin as amplitude with phase 0, or fold into `partial_gain`).

Model-derived (not measured) bridge-force partial levels for a half-sine with filled nulls and comb, relative dB to the strongest partial, n = 1..10:
- C2 (T_c 3 ms, x0 0.12): -6, -1, 0, -3, -7, -14, -28, -45, -33, -25
- C4 (T_c 1.8 ms, x0 0.11): 0, -7, -23, -17, -28, -26, -36, -40, -66, -46 (with a smoother two-pole low-pass instead: 0, -1, -4, -7, -11, -15, -20, -28, -51, -33)
- C6 (T_c 1 ms, x0 0.09): 0, -9, -14, -21, -34, -29, -27, -34 (two-pole: 0, -5, -9, -12, -15, -19, -22, -27)

These are bridge-force envelopes at mf, before soundboard filtering, and they fall between "half-sine with holes" (worst case) and "smooth". Real partials are typically between the two columns.

---

## 9. Nonlinear string effects

### Pitch glide (tension modulation)
- Kirchhoff-Carrier: T(t) = T0 + (EA/2) <y_x^2> (spatial mean of the squared slope). The mean part shifts every mode by roughly the same fraction: Delta f / f = (EA / (4 T0)) <y_x^2>. For a single mode of amplitude a: Delta f / f = EA pi^2 a^2 / (8 T L^2). Then cents = 1731 * Delta f / f.
- My estimates (derived, unverified): C4 ff (v_rms of the string about 2 m/s, EA about 1.6e5 N, T about 680 N, c about 330 m/s) gives about 3.7 cents. At v_rms 0.7 m/s (mf) about 0.45 cents. Bass C2 ff about 2 cents. So expect about 1–5 cents at ff, under 1 cent at mf, scaling with v^2.
- Time constant: the shift follows amplitude squared, so it decays at exp(-2 alpha t), tau_glide = 1 / (2 alpha_prompt); with alpha_p about 1 s^-1, about 0.5 s. Conklin, Fletcher and Bank (2005) discuss this; I do not have a measured number from a source.
- Perceptual: 1–5 cents is at or below pitch-discrimination thresholds but interacts with beats; low priority except for bass ff.

Implementation: f(t) = f * (1 + g e^(-t/tau_g)), g = 2.9e-3 * (v_h/5)^2 (about 5 cents at ff; my estimate, scale per key by (EA/T)/c^2), tau_g = 1/(2 alpha_prompt).

### Longitudinal modes and phantom partials
- Longitudinal speed c_L = sqrt(E/rho) about 5100–5200 m/s for steel (high confidence). f_L,k about k * c_L / (2L): A0 (L 1.95 m) about 1.3 kHz; C2 (about 1.2 m) about 2.2 kHz; C4 (0.63 m) about 4.1 kHz; C6 (0.2 m) about 13 kHz (above the audible range at 24 kHz). Wound strings have lower effective c_L because the winding adds mass but no axial stiffness (about 2.5–4.5 km/s, low).
- Phantom partials (Conklin 1997; Bank & Sujbert 2005 **[snippet: at sums and differences of transverse partials]**): frequencies f_n + f_m (and |f_n - f_m|), including 2 f_n. With inharmonicity, f_n + f_m < f_(n+m) by roughly f0 * B * 3 n m (n+m) / 2, so they fall slightly flat of the real partials, which produces additional beating/roughness in bass and tenor (Conklin: may differentiate timbre between pianos).
- Levels: phantom amplitude grows as the product of the source partial amplitudes (2 dB per dB of the transverse level, so strongly velocity dependent). Typical values (unverified, from memory, low): -30 to -45 dB relative to the transverse partials at mf, -15 to -25 dB at ff, in the bass and tenor (below about C4). Longitudinal free modes have a fast decay (T60 about 0.1–1 s) and act as a "ping"/metallic component of the bass attack.
- Register: strongest in A0 to about C4, weak above (Bank & Lehtonen, JASA 128(3), EL117 (2010), audibility study; the exact register limits are unverified).

Implementation: partial pairs with freq f_n + f_m for n, m in 1..8; amplitude = g_ph * a_n * a_m, g_ph per key, with a velocity-squared law; plus 1–3 free longitudinal modes at k * c_L / (2L) with T60 of about 0.3 s.

---

## 10. Measured partial spectra (sanity check)

I could not find numerical measured tables through the search tool. What follows is qualitative recall (unverified, from memory, low confidence) of Fletcher, Blackham & Stratton (1962) and Askenfelt/Jansson spectra:
- C2 (mf, microphone): the fundamental is weak (soundboard rolls off below about 100 Hz). The strongest partials are 3–8 (200–500 Hz), with about -20 dB at partial 12, and a clear dip near the comb null (n = 8). This is the bridge-force shape above plus the soundboard's high-pass.
- C4 (mf): fundamental and partial 2 strongest; partial 3 about -8 to -15 dB, partials 4–6 about -15 to -25 dB, partials 7–10 about -30 to -40 dB, with the comb null visible around n = 9.
- C6 (mf): the fundamental dominates; partial 2 about -10 to -15 dB, partial 3 about -20 to -25 dB, above that about -35 dB or lower.

Check for the code: the model output at mf, in bridge-force terms, should fall between the two columns of the model-derived tables in section 8. The current code gives partial 16 at C4 about -22 dB below partial 1 (before the comb), well above the real -40 to -50 dB level.

---

## 11. Other things the literature says matter (string/hammer domain) that the code lacks

1. **Bandwidth in the bass**: 64 partials at 24 kHz caps A0 at about 1.8 kHz and C2 at about 4.2 kHz. Real bass notes have strong partials well above 2 kHz. Raise `n_partials` for k < 30 (128–256) or add a noise/comb "bite" component.
2. **Attack timing**: real partial amplitudes ramp up over the contact time (1–3 ms), with phases that follow the force-pulse midpoint (linear phase, delay about T_c/2). The code starts all partials at zero phase at the onset (an impulse-like click).
3. **Longitudinal free modes and phantom partials** (section 9), essential for bass and tenor notes.
4. **Pitch glide** at ff (section 9).
5. **Hammer force pulse asymmetry (hysteresis)**: the pulse is skewed with a fast rise and slower decay, which fills the spectral nulls (Stulov 1995, Hall).
6. **Unison strings not hit identically**: hammer misalignment, felt grooves, and unequal string heights mean the strings receive different initial amplitudes and small timing offsets (about 0.1 ms). This sets the aftersound amplitude (Weinreich). The code's `after` parameter covers it.
7. **Duplex scale / sympathetic aliquot**: treble strings behind the bridge ring sympathetically. It adds a shimmering component in the treble (unverified, low). Others are handling sympathetic resonance.
8. **Non-simultaneous hammer strike across the strings of a unison** and their resulting phase relations at the onset.
9. **Frequency-dependent loss from bridge coupling**, resonance peaks of the bridge/soundboard giving partial-by-partial irregular decay (other agent).
10. **Re-strikes on a still-vibrating string** (already in the README to-do list).
11. Separate the "hammer velocity mapping" per Disklavier (other agent).

---

## (a) Summary table of parameters

| # | Parameter | Symbol, units | Value(s) | Source | Confidence |
|---|---|---|---|---|---|
| 1 | Inharmonicity | B (-) | A0 3e-4; k22 1.1e-4 (min); C4 3.8e-4; A4 8.5e-4; C6 3.5e-3; C7 1e-2; C8 4e-2 | Rigaud 2013 structure; my scale calculation; C&A C4 (unverified) | medium (k < 50), low above |
| 2 | Stretch | cents | A0 -30; A2 -8; A4 0; A5 +5; A6 +13; A7 +25; C8 +30 | Railsback 1938 (unverified, from memory) | medium |
| 3 | Strings/key | (-) | 1 for k < 8; 2 for 8–25; 3 for k >= 26 (about 230 strings) | (unverified, from memory) | low-medium |
| 4 | Unison detune | cents | sigma about 0.7; range 0.2–2; beats 0.1–2 Hz | Weinreich 1977; Kirk 1959 (unverified) | low |
| 5a | Fundamental decay, prompt | T60 (s) | C2 15–20; C4 6–8; C6 2.5–3.5; C7 1.5–2; C8 0.7–1.2 | (unverified, from memory) | low-medium |
| 5b | Aftersound | T60 (s) | C4 20–35; C6 8–15 | Weinreich [snippet: under 1/4 of the prompt rate] | low-medium |
| 5c | Prompt/after rate ratio | (-) | about 5 (trichord), 3 (bichord), 1.5 (mono) | Weinreich (theory + snippet) | medium |
| 5d | Aftersound level | dB re initial | -20 to -35 | (unverified) | low |
| 5e | C&A loss | b1 (1/s), b3 (s) | C2 0.5, 6.25e-9; C4 about 1.1, 2.7e-9 | C&A 1994 (unverified, from memory) | low |
| 6a | Hammer mass | m_h (g) | A0 about 10–12; C2 about 5–8; C4 about 3; C7 about 2.2 | C&A 1994, Conklin (unverified) | low-medium |
| 6b | Felt exponent | p (-) | 2.2–3.5; bass about 2.3, mid 2.5, treble 3.0 | Hall & Askenfelt [snippet]; C&A (unverified) | high (range), low (per key) |
| 6c | Felt stiffness | K (N/m^p) | C2 4e8 (p 2.3); C4 4.5e9 (p 2.5); C7 1e11 (p 3.0) | C&A 1994 (unverified) | low |
| 6d | Contact time (mf) | T_c (ms) | bass 3–4; mid 1.5–2.5; treble 0.6–1 | Askenfelt & Jansson [snippet: about 2 ms typical; +/-20 % p to ff] | low-medium |
| 6e | Hammer velocity | v_h (m/s) | 0.5–1 (pp) to 5 (ff), up to 6–7 (fff) | Askenfelt & Jansson [snippet] | medium-high |
| 7 | Strike position | x0/L | bass about 0.12–0.125; A4 about 0.105; C6 about 0.09; C7 about 0.075; C8 about 0.06 | Conklin [snippet] | medium |
| 8 | Bridge partial amplitude | A_n | 4 f0 * abs(sin(n pi x0/L)) * abs(F_hat(omega_n)), no 1/n | derived; Hall 1986–87 | high (ideal string) |
| 9a | Pitch glide | cents | 1–5 at ff, under 1 at mf; tau about 1/(2 alpha_prompt) | my derivation from Kirchhoff-Carrier | low-medium |
| 9b | Longitudinal modes | f_L (Hz) | k * c_L/(2L), c_L about 5.2 km/s (steel); A0 1.3 kHz; C2 2.2 kHz; C4 4.1 kHz | Conklin 1996; Bank & Sujbert 2005 | medium (formula), low (levels) |
| 9c | Phantom partials | Hz, dB | f_n + f_m; -30 to -45 dB (mf), -15 to -25 dB (ff), bass/tenor | Bank & Sujbert [snippet]; levels (unverified) | medium (freq), low (levels) |

---

## (b) Recommended prior curves (drop into `key_curve()`)

```python
# log10 B  (code applies ln10 * key_curve(...))
prior_log_B  = [(0,-3.52),(6,-3.66),(14,-3.85),(22,-3.96),(29,-3.92),(33,-3.70),
                (39,-3.42),(48,-3.07),(56,-2.74),(63,-2.46),(75,-2.00),(87,-1.40)]

# stretch, cents (fundamental)
prior_cents  = [(0,-30),(12,-18),(24,-8),(36,-3),(48,0),(60,5),(72,13),(84,25),(87,30)]

# alpha_after  (b1, 1/s) -- T60_after = 6.91/alpha
prior_b1     = [(0,0.20),(15,0.28),(27,0.25),(39,0.30),(48,0.35),(63,0.60),(75,1.3),(87,3.0)]

# b3 in Hz^-2 (code convention, b3_CA * 4 pi^2)
prior_b3     = [(0,2.5e-7),(20,2.5e-7),(30,1.2e-7),(87,1.2e-7)]

# prompt/after decay-rate ratio (code: 1 + exp(...)); value = ratio
prompt_ratio = [(0,1.5),(7,1.5),(8,3.0),(25,3.0),(26,5.0),(87,5.0)]

# aftersound amplitude per aftermode (linear)
after_amp    = [(0,0.10),(39,0.06),(87,0.06)]

# contact time at mf (ms) -> fc = 0.78/T_c for the existing (1+(f/fc)^2)^(-r/2), r = 1.5
Tc_ms        = [(0,3.5),(15,3.0),(27,2.4),(39,1.9),(51,1.5),(63,1.1),(75,0.8),(87,0.6)]
fc_Hz        = [(0,220),(15,260),(27,325),(39,410),(51,520),(63,710),(75,975),(87,1300)]

# striking position ratio
x0_over_L    = [(0,0.125),(15,0.12),(27,0.115),(39,0.11),(48,0.105),(56,0.095),
                (63,0.088),(75,0.075),(87,0.06)]

# hammer mass, g (effective)
m_hammer_g   = [(0,10.0),(15,5.5),(39,3.0),(63,2.4),(87,2.0)]

# exponent p
p_felt       = [(0,2.3),(39,2.5),(87,3.0)]

# strings per key: 1 for k<8, 2 for k<26, else 3
```

---

## (c) Formulas the code should use

1. Frequencies: f_n = n f0 sqrt(1 + B n^2), unchanged.
2. Decay: alpha_n = b1 + b3 f_n^2 for the aftersound; alpha_prompt = alpha_n * R_prompt with R_prompt = 1 + N gamma_b / alpha_int (about 5 for trichords).
3. Bridge-force amplitude: a_n = 4 f0 * J * H(f_n; T_c) * sin(n pi x0/L) (no 1/n).
4. Hammer spectrum H (replaces the power-law low-pass), a half-sine pulse with filled nulls:
   H(f) = sqrt(cos^2(pi f T_c) + eps^2) / |1 - (2 f T_c)^2 + 1e-6|, with eps about 0.15; normalise so H(0) = 1.
   Velocity: T_c = T_c,mf * (v_h / 2 m/s)^(-q) with q about 0.25 (physical range 0.2–0.5).
   Check: it is -3 dB at f = 0.59/T_c and falls at 12 dB/oct.
5. Velocity gain: level of the fundamental proportional to v_h (about 6 dB per doubling), with brighter partials growing faster because of the T_c scaling.
6. Glide: f(t) = f (1 + g e^(-t / tau_g)), g in cents approximately 1731 * (EA/4T) * (v_rms/c)^2, tau_g = 1/(2 alpha_prompt).
7. Phantom partials: freq f_n + f_m, amplitude g_ph a_n a_m (velocity-squared law).
8. Finite hammer width: multiply by sinc(n pi w / (2L)); negligible at 24 kHz.

---

## (d) Ranked "toy guitar" diagnosis (string and hammer side, `physics.py` as written)

I have not listened to the untrained output, so this ranking is by plausibility from the numbers. The real cause may include soundboard, room and noise choices (other agent).

1. **Hammer spectrum too bright and too shallow, and uncoupled from contact time.** `hammer = (1+(f/fc)^2)^(-0.75)`, i.e. -9 dB/oct, against a real force pulse of -12 dB/oct with fc about 300–500 Hz at mid keys. At C4 the code puts partial 16 about -22 dB below partial 1 (before the comb), while real notes have -40 to -50 dB. The prior fc at k > 60 (up to 3.5 kHz) corresponds to a contact of 0.2 ms, which is shorter than the period of the top notes. Bright, long-lived upper partials are the signature of a plucked steel string. Fix: T_c prior curve, half-sine spectrum, q about 0.25 (see (b), (c)).
2. **Decays too fast in the upper half, and the double decay is too weak and too early.** After-decay is about 1.5x too fast at C4 and 2x too fast at C6 and C7. The prompt/after ratio is 3 (real 4–8), and the after level is -14 dB per mode (real -20 to -35 dB), so the knee comes too early (about 1–2 s) and is too shallow. A short, single-slope decay with no long tail is what a toy guitar does. Fix: the `prior_b1`, `prompt_ratio` and `after_amp` curves above.
3. **No attack transient: no longitudinal modes, phantom partials, glide, contact-time ramp.** The onset is a zero-phase sum of sinusoids (an impulse-like click, then smooth sustain). Real notes have a 1–3 ms ramp and, in the bass and tenor, the metallic longitudinal "ping" that gives the bass its character. Their absence leaves the attack like a clean synthetic pluck. Fix: sections 9 and 11.
4. **Bass bandwidth capped by `n_partials = 64`** (A0 to 1.8 kHz, C2 to 4.2 kHz). The low half sounds dull and boomy with no bite. Fix: more partials for k < 30.
5. **Treble B 3–6x too low** (C6 to C8), and C4 B 2.4x low. The treble partials sit near-harmonic, giving a "tuned chime/harp" quality instead of piano's slightly stretched, shimmering treble. Fix: `prior_log_B` above.
6. **Strike position fixed at 0.12 for all keys** (real 0.125 to 0.06). In the treble the comb shape is wrong (partials 3–4 over-emphasised). Fix: the `x0` curve.
7. **`abs()` on the comb** discards the alternating sign of partial amplitudes and changes the waveform (peakier). Fix: keep the sign.
8. **b3 = 3e-7 constant** makes upper partials decay about 2.7x too quickly in the mid range and does not distinguish wound from plain strings. Fix: `prior_b3` curve.
9. **Small items:** velocity scaling of fc is 5.4x over the range vs the measured 1.5–2x, stretch too small in the extremes, and the bass unison is untuned to a per-key spread.

---

## (e) Open questions

1. Real scale data for a Steinway D or Yamaha CFX (L, d, T per key, break position, wound core diameters) could not be retrieved. Conklin 1996 Part III and the Chaigne & Askenfelt Tables I/II should be checked; those are the sources most worth reading directly.
2. Rigaud et al. (2013) give fitted bass/treble B slopes and offsets for real pianos (RWC etc.). They would replace my composite B curve. Also consider estimating B directly from MAESTRO by partial tracking (already a README to-do).
3. Which of the MAESTRO Disklaviers are CFX vs C7 vs older models? B, T_c and stretch differ by 2x among them.
4. Exact contact durations vs key (Askenfelt & Jansson 1990/1993), and hammer velocity vs Disklavier MIDI velocity.
5. Numbers for the prompt/aftersound knee level and time, by key, on a real concert grand.
6. Phantom-partial and longitudinal-mode levels by key (Bank & Sujbert 2005 have measured/modelled numbers I could not retrieve).
7. Whether to model the hammer explicitly (Hunt-Crossley / Stulov contact on a modal string) instead of a filter. A modal string with a contact ODE is cheap per note and would generate the velocity-dependent spectrum, the contact-time dependence on key, and the null filling automatically. The parameters would then be (m_h, K, p, hysteresis) rather than (fc, rolloff).
8. Pitch-glide magnitude at ff is my own derivation; measured values are needed.
9. The 1/n vs no-1/n question is settled for bridge force (section 8), but the soundboard filter that follows determines the radiated shape, which is the other agent's part.
