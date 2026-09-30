# Follow-up literature search: piano tone constituents (numbers)

Date: 2026-09-30. Follow-up to `tone_constituents.md` ("the report"; its table 1.1 rows are cited as "report 1.1 row N").
Budget used: about 175K tokens (estimate; 27 searches/fetches of web pages, 12 PDFs downloaded and grepped).

Evidence labels: **read** (text seen, section or figure named), **read (secondary)** (the source I read quotes someone else), **snippet** (search-result or fetch-tool summary text only), **cited** (not checked), **mine** (my inference or arithmetic).
Two-column PDFs were extracted with pdftotext and grepped; sentences were sometimes garbled, and I say so where it matters. Every claim from a fetch-tool summary is marked **snippet**.

Main results in five lines:
1. The attack is mostly string-borne (longitudinal precursor, -9 to -14 dB re the first transverse wave at the bridge, 1-2 ms, up to 5 kHz); the structure-borne "thump" (100-330 Hz) is 25-40 dB weaker at the bridge but stronger in the microphone signal. Measured at C4 on one piano only (F1).
2. Brightness grows as hammer velocity to the power 0.4-0.5 (cutoff of the force spectrum), voiced Steinway D hammers (F9).
3. Longitudinal content is audible up to C5 at fortissimo; the useful synthesis range is up to A3 (F5). Phantom amplitude grows faster than v^2 (F4). First longitudinal mode is about 14 x f1 (F3, F4).
4. The report's "B up to 0.4 in the treble" is wrong; the top octave is about 0.01-0.03 (F13). Strike point falls from 1/8 in the bass to 1/12-1/17 at the top (F14, snippet).
5. No source gives a number for: knock level per register, re-strike removal, pitch glide in cents, prompt/aftersound rates and knee, tone-to-tone spread.

## 0. Sources read

| ID | Source | What I read, and how |
|----|--------|----------------------|
| F1 | Askenfelt (1993), Observations on the transient components of the piano tone, STL-QPSR 34(4) 15-22 (SMAC 93) | Full text (8 pages) |
| F2 | Bank (2006), PhD thesis, BME, Physics-based sound synthesis of string instruments including geometric nonlinearities | 156-page PDF, grepped: Sec. 4.3 (soundboard FIR, knock), Sec. 5.1-5.3 (regimes, glide, phantoms), Appendix; extracted passages |
| F3 | Chabassier, Chaigne, Joly (2013), Modeling and simulation of a grand piano, JASA 134(1), 648-665 | Preprint PDF, grepped: model, Table IV, Sec. IV results; extracted passages |
| F4 | Bank & Sujbert (2005), Generation of longitudinal vibrations in piano strings, JASA 117(4), 2268-2278 | Full text, grepped |
| F5 | Bank & Lehtonen (2010), Perception of longitudinal components in piano string vibrations, JASA-EL 128(3), EL117-EL123 | Full text (Aalto thesis publication VI), method and results sentences |
| F6 | Lehtonen, Penttinen, Rauhala, Välimäki (2007), Analysis and modeling of piano sustain-pedal effects, JASA 122(3), 1787-1797 | Full text, tables and results sentences (table layout partly garbled) |
| F7 | Lehtonen, Askenfelt, Välimäki (2009), Analysis of the part-pedaling effect in the piano, JASA-EL 126(2), EL49 | Abstract and method sentences |
| F8 | Askenfelt & Jansson, KTH "Five Lectures" web page, String contact duration and dynamic level | Fetch-tool summary (snippet) |
| F9 | Russell & Rossing (1998), Testing the nonlinearity of piano hammers using residual shock spectra, Acustica 84, 967-975 | PDF, abstract, intro, Sec. 5.2 extracted |
| F10 | Goebl & Bresin (2003), Measurement and reproduction accuracy of computer-controlled grand pianos, JASA 114(4), 2273-2283 | PDF, results sentences extracted |
| F11 | Weinreich, KTH "Five Lectures" web page, The coupled motion of piano strings (Fig. 7-8 text) | Fetch-tool summary (snippet) |
| F12 | Bank & Chabassier (2019), Model-based digital pianos, IEEE SPM 36(1), 103-114 | Full text (HAL preprint), grepped for constituents and missing features |
| F13 | Rigaud, David, Daudet (2013), A parametric model and estimation techniques for the inharmonicity and tuning of the piano, JASA 133(5), 3107 | PDF, grepped for B model |
| F14 | Conklin, KTH "Five Lectures" web page, Where should the hammer hit the string? | Fetch-tool summary (snippet) |
| F15 | Simionato & Fasciani (2024), arXiv:2409.06513 | Re-grepped for phantom equations only |
| F16 | Search-result summaries: Järveläinen et al. 2001 (ARLO abstract); ASA 1997 abstract on spectral centroid JND; Kazazis, Depalle, McAdams (2021) JASA 149(6) (intro only); Martin 1947 | Snippets; Kazazis intro grepped |

Downloaded but not mined for numbers: Chaigne & Askenfelt 1994 part I (F-CA1; the hammer parameter table did not survive text extraction).

## 1. Knock / attack transient

Main source: F1 (Askenfelt 1993, full text, 8 pages). Steinway C grand (7.5 ft), note C4 unless stated, staccato touch, bridge accelerometer plus a microphone. Single note, single piano: treat as one measured case, not a register survey.

| # | Finding | Number | Conditions | Label |
|---|---------|--------|-----------|-------|
| 1.1 | The attack has two transmission paths: (a) string, longitudinal wave to the bridge; (b) structure, key/keybed/rim/frame | - | F1 intro, Fig. 1 | **read** F1 |
| 1.2 | String path: longitudinal precursor arrives at the bridge about 0.2 ms after string contact, before the first transverse wave | delay 0.2 ms; lasts 1-2 ms; amplitude -9 to -14 dB re the first transverse wave at the bridge (acceleration) | C4, bridge acceleration; independent of touch | **read** F1 p.2 |
| 1.3 | Longitudinal speed vs transverse | at least 10x (F1); 10-20x (F3). C2 example: 2914 m/s vs 209 m/s | - | **read** F1, F3 (Chabassier 2013, Sec. II) |
| 1.4 | String path dominates the attack spectrum above 1 kHz and reaches to about 5 kHz. Askenfelt calls it the "bite" of the attack | with transverse motion damped, the bridge spectrum is 25 dB above the structure-only (dummy) case, extends to 5 kHz | C4, bridge acceleration | **read** F1 Fig. 4 |
| 1.5 | Structure path (dummy mass instead of strings, same contact time): low-frequency "thump", spectrum up to 2 kHz only, level about 40 dB below the string partials at the bridge | ~40 dB below string partials; 25 dB below the string precursor | C4, bridge | **read** F1 |
| 1.6 | Structure resonances that show up in bridge motion: key 290 and 440 Hz (touch), key 900 Hz (at key-bottom, ends when the hammer is caught by the backcheck 10-20 ms after contact), soundboard 100 Hz, keybed 95 and 330 Hz, rim 250 Hz, plate/frame 38 Hz | see list | grand piano | **read** F1 |
| 1.7 | In the radiated sound the string partials dominate less than at the bridge, because rim and keybed also radiate: expect more low-frequency thump in the microphone signal than in bridge vibration | qualitative | C4 | **read** F1 |
| 1.8 | Touch precursor (finger-key, before hammer contact): 20-30 ms before string contact for staccato touch; about 25 dB below the string precursor at the bridge; absent in legato touch; strongly reduced in strained touch | about 25 dB below the string precursor, i.e. about 30-40 dB below the transverse wave for the prominent staccato case (F1 text: 25 dB; 30-40 dB) | C4 | **read** F1 |
| 1.9 | Vibration at the bridge for a mezzo-forte blow on the dummy is comparable to a pianissimo tone | mf structure blow = pp string tone | C4 | **read** F1 |
| 1.10 | Treble "knock": the low-frequency body response to the treble note rings 0.2-0.5 s and is longer than a 2000-tap FIR (45 ms) can hold. In Bank's model a 0.36 s low-frequency response is added for it. The rest of the attack transient is short: the resynthesis residual was windowed to 1000 samples (23 ms at 44.1 kHz) above 4 f0 and left long below 4 f0 | 0.2-0.5 s; 0.36 s used | Bank thesis Sec. 4.3 (F2); F5 method | **read** F2, F5 |
| 1.11 | The residual after subtracting transverse and longitudinal components contains the attack transient and the low-frequency "knock" of the soundboard; for high notes this knock or "thump" is much longer than the rest of the transient | qualitative | F5 | **read** F5 |
| 1.12 | Precursor in the full-piano model: with hammer speed 4.5 m/s on C2, the longitudinal wave pushes the soundboard down first, then the transverse wave pulls it up. Chabassier's model needs both transverse and longitudinal transmission to the soundboard to get the spectral content of the precursor | v = 4.5 m/s ("forte to fortissimo") | simulation, compared with Steinway D measurements | **read** F3 |
| 1.13 | Level of the knock re the tone per register and per velocity | not found in the papers I could reach; F1 gives only the C4 case above | - | **not found** |

Reading of 1.2-1.11 for a synthesizer (**mine**): the attack has three parts with different spectra. (i) A 1-2 ms broadband burst from the longitudinal string precursor, about -10 dB re the first transverse wave at the bridge, reaching up to about 5 kHz. It is string-borne, so it scales with the blow and should be tied to the phantom/longitudinal model, not to the body FIR alone. (ii) Structure-borne thump at 100-330 Hz (soundboard, keybed, rim), 25-40 dB below the string signal at the bridge but stronger in the microphone signal than at the bridge, and long-lived in the treble (0.2-0.5 s). (iii) Key resonances at 290/440/900 Hz that belong to touch and key-bottom, 10-30 ms around the contact.

Report 1.1 row 14 (knock): **corrected in emphasis.** The report says the knock is caused by the hammer exciting the soundboard directly. F1 finds the direct structural path is the weak (-40 dB) one and the longitudinal string precursor is the stronger path to the bridge, at least above 1 kHz. The low-frequency thump in the treble is real (F2, F5) but it is the soundboard's low modes driven through the bridge and structure, and it is long, not a 1-2 ms click.
Report 1.1 row 1 (touch precursor 20-30 ms): **confirmed** at source (F1: 20-30 ms, staccato, C4, mid range).

## 2. Hammer-string contact vs velocity and register

| # | Finding | Number | Conditions | Label |
|---|---------|--------|-----------|-------|
| 2.1 | Contact duration falls from bass to treble | about 4 ms in the bass; under 1 ms at the highest treble notes | grand piano, measured by Askenfelt & Jansson | **snippet** F8 (text of Fig. 7 caption) |
| 2.2 | Contact duration vs dynamic level | varies about +-20 % around mezzo-forte over the comfortable range pp to ff | same | **snippet** F8 |
| 2.3 | In the bass the contact is short against the fundamental period (about 10 % of it); the period only becomes short against the contact at about the 10th partial | - | same | **snippet** F8 |
| 2.4 | Felt exponent p, static tests on hammers | 2.2-3.5 for hammers taken from pianos; 1.5-2.8 for unused hammers (Hall & Askenfelt) | quoted in F9 | **read (secondary)** F9 |
| 2.5 | Felt exponent p along the keyboard, dynamic tests on voiced hammers from several pianos | rises smoothly from about 2 in the bass to about 4 in the treble | F9 Sec. 2 | **read** F9 |
| 2.6 | Hammer speeds spanning the normal dynamic range | 1-6 m/s | F9 abstract | **read** F9 |
| 2.7 | Brightness vs velocity: peak frequency f_max of the residual shock spectrum of the force pulse follows f_max = a v^b, with b = 0.4-0.5 for most voiced Steinway D hammers. For a 4x velocity increase f_max rose 1.8x (hammer A0) and 2.3x (hammer F7) | b = 0.4-0.5; 1.8x and 2.3x per 4x | Steinway D hammer set, hammer on a force sensor, 1-6 m/s | **read** F9 Sec. 5.2, Table I |
| 2.8 | In the bass f_max is far above f1 (many partials); in the treble f_max is close to f1 (strong fundamental, few partials) | - | F9 abstract | **read** F9 |
| 2.9 | Nonlinearity increases toward the treble (Hall) | - | - | **snippet** (KTH Hall page) |
| 2.10 | Hammer stiffness constant K per register; tables of K, p, hammer mass, strike ratio per note (Chaigne & Askenfelt; euphonics.org) | tables are images or lost in extraction | - | **not obtained** |
| 2.11 | Multiple hammer contacts in the treble: when, how audible | - | - | **not found** (report 1.1 row 4 stays **cited**) |
| 2.12 | Brightness change per dB of level measured on a real piano over velocity | not found as a spectral-centroid number | - | **not found** |

Arithmetic on 2.7 (**mine**): 4^b = 1.8 gives b = 0.42; 4^b = 2.3 gives b = 0.60. So per +6 dB of hammer speed the cutoff of the force spectrum rises by about 1.3x (bass) to 1.5x (top). This is a property of the hammer force pulse, not of the radiated spectrum at the microphone.

Report 1.1 row 4: p range "2.5-4 from bass to treble" is **corrected** to about 2 (bass) to 4 (treble) for voiced hammers (F9), 2.2-3.5 for used hammers in static tests. Contact time "a few ms" is **quantified**: about 4 ms bass, under 1 ms top (F8, snippet).

## 3. Tone-to-tone variation

| # | Finding | Number | Conditions | Label |
|---|---------|--------|-----------|-------|
| 3.1 | Yamaha Disklavier grand, onset timing of the recorded MIDI vs the acoustic onset | mean 1.4 ms, s.d. 3.8 ms; soft tones recorded later than loud ones | five keys, two touches, accelerometers and microphone | **read** F10 Sec. III.A |
| 3.2 | Disklavier reproduction (playback) onset error | up to 20-28 ms; mean 0.3 ms, s.d. 5.5 ms | same | **read** F10 |
| 3.3 | Disklavier playback compresses dynamics: soft tones played too loudly, very loud tones too softly; well behaved only in a wide middle range | - | same | **read** F10 abstract, Fig. 6 text |
| 3.4 | Playback hammer-speed ceiling depends on pitch | G6 up to 3.5 m/s; C1 up to 2.4 m/s | Disklavier playback | **read** F10 |
| 3.5 | At the same MIDI velocity the peak SPL is higher for higher pitch, on both pianos tested | curves per pitch (Fig. 7) | recording side too | **read** F10 Sec. III.B |
| 3.6 | A human touch: the finger's first impact can give a very high peak hammer speed that decays before contact; example: max hammer speed 5.8 m/s (original) vs 5.4 m/s (playback) with nearly equal peak SPL | - | one case | **read** F10 |
| 3.7 | Same hammer speed, different touch gives distinguishable tones (report 1.1 row 1, R1) | - | - | **cited** here, read in the report |
| 3.8 | Unison strings are excited unequally by hammer irregularities, and the tuning state sets the aftersound level and beats | theory | Weinreich 1977 | **snippet** F11 (search summary of the paper) |
| 3.9 | Repeatability of level, spectrum, attack and decay of repeated strikes of one key at fixed hammer speed (dB, s.d.) | - | - | **not found** |

Caveat for MAESTRO (**mine**): F10 tested a Yamaha Disklavier grand as a recorder and as a player. MAESTRO uses the recording side with human players, so the playback compression in 3.3-3.4 does not apply, but 3.5 does: MIDI velocity is not a common level scale across notes. The touch effects in 3.6-3.7 do apply. My numbers 3.1 (timing, 3.8 ms s.d.) also bound how well audio can be aligned to MIDI onsets.

Report section 6 point 5 (unknown "spread of repeated strikes"): **still open**. Nothing I could reach measured it. The 40-80 % variation ratio the project measured has no literature number to compare with.

## 4. Longitudinal vibrations and phantom partials

| # | Finding | Number | Conditions | Label |
|---|---------|--------|-----------|-------|
| 4.1 | Longitudinal vs transverse wave speed | C2 string: 2914 m/s vs 209 m/s (ratio 13.9); "10 to 20 times" in general | Steinway D scale | **read** F3 Sec. II |
| 4.2 | First longitudinal mode | about 690 Hz for the G1 example (f1 = 49 Hz, so about 14 f1); second mode at 1380 Hz | simulation example in F4 Fig. 2; ratio of speeds in 4.1 gives the same 14 | **read** F4, F3; ratio arithmetic **mine** |
| 4.3 | Free response of a longitudinal mode decays fast; the forced response (phantoms) lasts | free mode decay about 0.15 s; phantoms decay at rates comparable to the transverse partials (1-2 s) | recorded F1 note (43.7 Hz, single string), 16 s FFT | **read** F2 Sec. 5.3, F4 Sec. IV |
| 4.4 | The strongest non-transverse peaks in the long-term spectrum are phantoms amplified by a longitudinal mode; the authors argue the forced response matters more perceptually than the free response | - | same | **read** F4 |
| 4.5 | Parentage: odd phantoms (sum) come from adjacent parents, e.g. f5+f6; **even phantoms come mainly from parents whose mode numbers differ by 2, 4, ... (for example 12+14), not from 2 f_n alone**. In the example, mode 13 was only 10 dB below modes 12 and 14, yet the largest peak was 12+14. This contradicts Conklin 1999 | - | F1 recorded note | **read** F2 Sec. 5.3.7 (Fig. 5.5), F4 |
| 4.6 | Parents relevant for the audible phantoms have mode numbers about 10-20, the ones whose sum frequency lies near a longitudinal mode | - | - | **read** F4 |
| 4.7 | Growth with amplitude: a phantom peak is quadratic in its parents' amplitudes; but the high parents (numbers 10-20) themselves grow faster than linearly with hammer speed (hammer nonlinearity), so the total is faster than quadratic. Giordano & Korty measured faster than quadratic | "faster than v^2" | - | **read** F4 |
| 4.8 | Longitudinal-to-transverse coupling is third order in the transverse amplitude | - | - | **read** F4 |
| 4.9 | Nonlinear effects noticeable when string amplitude / string diameter (ADR) exceeds 1. Simulated ADR at 0.5 / 1.5 / 3 m/s: D#1 0.57 / 1.77 / 3.59; C2 0.69 / 2.01 / 4.05; F3 0.34 / 1.04 / 2.10; C#5 0.27 / 0.87 / 1.76; G6 0.12 / 0.44 / 0.96. Conclusion: nonlinear effects noticeable in bass and mid even at moderate speed; in the treble only for strong blows | ADR grows about linearly with v | Steinway D simulation | **read** F3 Table IV, Sec. IV |
| 4.10 | Regime boundary used by Bank: nonlinear transverse component under 0.1 of the linear one (-20 dB) is "probably masked". Bank states no research exists on the perceptual significance | -20 dB | assumption | **read** F2 Sec. 5.1 |
| 4.11 | String precursor level at the bridge: -9 to -14 dB re the first transverse wave (acceleration) | -9 to -14 dB | C4, Steinway C | **read** F1 |
| 4.12 | Audibility: ABX, fortissimo tones of three pianos (two grands, one upright), components separated by resynthesis, 8 listeners, 16 trials per tone, criterion 12/16 correct (p about 0.04). Longitudinal components audible up to C5 (MIDI 72, 523 Hz); more than half the listeners heard a difference up to C4 and again near C5. Preference test: include them up to A3 (MIDI 57, 220 Hz) if 25 % of listeners find the difference significant. C1-C2: essential. Tones without them were called "more synthetic" | C5 audible; A3 sufficient | headphones | **read** F5 |
| 4.13 | Level of phantom / longitudinal content re the transverse partials, dB, per register | - | - | **not found** (Conklin 1999 and Moore's work not reachable) |

Report 1.1 row 10 ("faster than transverse", **cited**): **confirmed**, 10-20x (F3), at least 10x (F1). Row 11: "double-frequency phantom decays with half the decay time" is the arithmetic of a product of two decaying partials (F15, Eq. 13); the recorded bass note in F2/F4 shows phantoms decaying at rates **comparable** to transverse partials, so treat the "half" as a model prediction that data may not follow. Also: the report's row 11 "free (odd) / forced (even)" labelling is not how F2/F4 use the words: both odd and even phantoms are forced responses; "free" refers to the longitudinal mode's own ringing.
The project's phantom set (2 f_j and f_j + f_(j+1) only) lacks the (j, j+2) pairs that F2 finds dominant for even phantoms (**mine**: their frequencies lie within a fraction of a Hz of 2 f_(j+1) in the bass, since f_m + f_n - 2 f_k = f1 (B/2)(m^3 + n^3 - 2 k^3), so the position is nearly right and only the amplitude sum differs).

## 5. Pitch glide (tension modulation)

| # | Finding | Number | Label |
|---|---------|--------|-------|
| 5.1 | The mechanism: the mean tension follows the string's elongation; as the amplitude decays the tension and the frequency fall, so the initial pitch is higher | qualitative | **read** F2 Sec. 5.2 |
| 5.2 | Simulated C#5 fortissimo: the fundamental frequency decreases with time (Fig. 14); the text gives no numbers | - | **read** F3 |
| 5.3 | Which notes: by the ADR criterion (4.9), bass and mid at moderate speed; treble only at strong blows | - | **read** F3 |
| 5.4 | Magnitude in cents vs velocity and register; time constant; audibility | not found in any source I reached | **not found** |
| 5.5 | Order of magnitude: for one planar mode, Duffing-type shift df/f = (3 pi^2 / 32)(E S / T0)(a/L)^2. With E S / T0 of 200-300 and a/L of 1e-3 to 3e-3 this is about 0.5-4 cents at the start, falling as the square of the amplitude (so about twice as fast as the amplitude) | 0.5-4 cents | **mine**, from the Kirchhoff-Carrier tension law in F2 Eq. 5.15; not checked against a measurement; a/L and E S / T0 are my assumptions |

Report 1.1 row 12: still **cited** as far as sources go; direction and mechanism now **read**; size is only my estimate (5.5). If the estimate is right, the glide is at or below the pitch discrimination of complex tones except for loud bass notes.

## 6. Unisons, double decay, beating

| # | Finding | Number | Label |
|---|---------|--------|-------|
| 6.1 | Two explanations for two-stage decay and beating: two polarisations of each string; slightly detuned strings of a unison (except the lowest octaves). Coupling at the bridge and soundboard is the physical route | - | **read** F12 |
| 6.2 | Weinreich's coupled-string result, mid-range parameters: beats appear only if the unison mistuning is larger than about 0.3 Hz; below that the strings lock to a common frequency, giving one beat null and then a beatless aftersound whose level depends on the mistuning. Examples: 0.06 and 0.22 Hz no beats; 0.64 Hz beats (period a bit over 1.6 s) | 0.3 Hz | **snippet** F11 (page text via fetch tool) |
| 6.3 | Tuners can set the aftersound level by fine unison tuning | - | **snippet** F11 |
| 6.4 | Kirk (1959): the mistuning of trichords varied randomly from note to note | no distribution given | **snippet** F11 mention |
| 6.5 | Prompt and aftersound decay rates, knee level and knee time per register | - | **not found** (Karatsovis PhD, Southampton, "double decay rate"; Martin 1947 "Decay rates of piano tones", JASA 19(4), 535 are the leads; not opened) |
| 6.6 | Decay-time tolerance for string tones: overall decay time can change from -25 % to +40 % (equivalently 75 %-140 %) without audible effect (Järveläinen & Tolonen 2001, plucked-string synthesis, **cited** for stimulus) | -25 %..+40 % | **read (secondary)** F2 Sec. 4.2, F6 |
| 6.7 | Bank adds that smaller changes of individual partial decay times may still be audible; no research | - | **read** F2 |
| 6.8 | Fundamental T60 of one grand, no pedal: C2 9.3 s, C3 10.0 s, C4 10.3 s (from F6 Table III; other rows garbled) | 9-10 s | **read** F6 |

Report 1.1 row 7 (no quantification, "no formal test found"): **partly filled**: beat threshold 0.3 Hz (snippet) and decay-time tolerance (read, secondary). Also: the report's pointer "Decay rates of piano tones ... (2023?)" is **corrected**: it is D. W. Martin, JASA 19(4), 535 (1947) (snippet), an old paper, not a 2023 one.

## 7. Re-strike of a still-vibrating string

| # | Finding | Label |
|---|---------|-------|
| 7.1 | The restrike of an already sounding string is one of the effects that sampled pianos cannot reproduce, and commuted (linear) synthesis cannot model precisely | **read** F12 |
| 7.2 | "Key restrike" is listed among features missing from comprehensive simulation tools | **read** F12 |
| 7.3 | A hammer hitting a moving string adds energy when in phase and less (or removes it) when out of phase; harmonics are enhanced or suppressed differently at each strike | **snippet** (forum-level text; no primary source) |
| 7.4 | Measurement of how much of a ringing string's vibration a new blow removes (nats or dB) per register | **not found** |

**Mine**: with a linear string, the new hammer impulse adds a modal amplitude of fixed size and phase set by the strike time, so the sign of the change is random per partial and the mean is an energy gain, not a loss. A constant "removal" would then come from felt losses and hammer-string contact, not from the string alone. The project's prior (0.35 nats in the bass to 1.0 in the treble, i.e. amplitude x0.70 to x0.37, -3 to -8.7 dB) cannot be checked against the literature; it should be judged on MAESTRO repeated notes.

## 8. Sympathetic resonance, pedal, dampers, release

| # | Finding | Number | Conditions | Label |
|---|---------|--------|-----------|-------|
| 8.1 | Initial levels of partials change by less than 1 dB when the sustain pedal is down | < 1 dB | C2-C6, one grand | **read** F6 |
| 8.2 | T60 of the fundamental without / with pedal | C2 9.3 -> 14.8 s; C3 10.0 -> 14.3 s; C4 10.3 -> 17.0 s (+40 % to +65 %). D5 and C6 rows are garbled in my extraction | one grand | **read** F6 Table III (partly garbled) |
| 8.3 | Overall T60 at C4 with pedal is 161 % of the no-pedal value; by the 75-140 % tolerance, only C4 (of the five tones) exceeds it | 161 % | - | **read** F6 |
| 8.4 | The non-partial ("residual") signal is about 10 dB higher during 1-5 s at C4 with the pedal | +10 dB | C4 | **read** F6 Fig. 4 text |
| 8.5 | Residual energy rises 5-30 dB on Bark bands 1-14 (0-2.5 kHz) and not above | +5..+30 dB, < 2.5 kHz | C2, C4, C6 | **read** F6 Fig. 5 text |
| 8.6 | When the played string group is damped 1-2 s after onset while the pedal stays down, the signal energy drops by about 30 dB (lowest tones) to 45 dB (highest tones): the sympathetic halo from other strings is that far below | -30..-45 dB | - | **read** F6 (my reading of the sentence; it says "energy difference before and after the damping") |
| 8.7 | Part-pedalling has three intervals: initial free vibration; damper-string interaction (rapid decay, timbre changed by nonlinear amplitude limitation of the string); final free vibration at a lower decay rate | - | C1, A1, G2, G3, D4, A4, damper height steps of about 0.45 mm | **read** F7 |
| 8.8 | Duplex stringing, aliquots, una corda pedalling and dampers are listed as missing in comprehensive simulation tools | - | - | **read** F12 |
| 8.9 | Damper landing and release noise levels; pedal noise levels | - | - | **not found** |
| 8.10 | Hammer checked by the backcheck 10-20 ms after contact vibrates at about 900 Hz | - | Steinway C | **read** F1 |

Report 1.1 row 15: **confirmed and quantified** (8.1-8.5). Report says mid-range only; F6 shows fundamental T60 up by 40 % at C2 also (partial-wise pattern in the abstract still says mid-range, not checked further).

## 9. Perceptual thresholds for the measure suite

| Quantity | Threshold or tolerance | Stimulus | Source | Label |
|----------|-----------------------|----------|--------|-------|
| Inharmonicity B | ln B_thr = 2.57 ln f0 - 26.5, fitted to thresholds at five f0 from 55 to 1108.7 Hz; C#6 threshold more than 1000x that of A1. My arithmetic: B_thr about 1e-7 at 55 Hz and 2e-4 at 1109 Hz. The 55 Hz value looks too low against real bass B (about 1e-4), so the formula, as passed on by a search summary, may be garbled. Check the original before use | tones of string instruments, synthetic, duration matters | Järveläinen et al. 2001 (ARLO 2(3), 79) | **snippet** |
| Overall decay time | no audible effect for -25 % to +40 % (75 %-140 %) | plucked-string synthesis | Järveläinen & Tolonen 2001 via F2, F6 | **read (secondary)** |
| Partial / resonator level | a change under 1 dB (resonator amplitudes) and a 0.8 dB change were found inaudible | piano synthesis | F2 Sec. 4.3; R7 | **read** |
| Longitudinal components | audible up to C5 at ff (75 % correct in 16 ABX trials) | resynthesised recorded tones, headphones | F5 | **read** |
| Pedal-induced decay change | above 140 % is audible (C4 is 161 %) | recorded tones | F6, via Järveläinen & Tolonen | **read (secondary)** |
| Spectral centroid | jnd 0.153 (units not stated in the abstract text I saw) for 5-harmonic complexes on 440 Hz, three standard centroids | additive complexes | ASA 1997 abstract | **snippet** |
| Spectral centroid, instrument sounds | Wun et al. 2014 (via Kazazis 2021): discrimination above 75 % when the centroid is raised by 40 % or lowered by 24 %; identity lost at +64 % / -48 %. My extraction starts mid-sentence, so the subject is unverified | additive-synthesis instruments | F16 | **read (secondary, truncated)** |
| Reverberation time | about 5 % | - | ISO 3382 convention | **cited** (unchanged from the report) |
| Touch precursor | a precursor 30-40 dB below the transverse wave would be detectable, judging from onset-asynchrony work (Rasch 1978); not tested for pianos | - | F1 | **read (secondary)** |
| Attack / rise time; beat depth or AM detection; brightness per dB of velocity | not found | - | - | **not found** |
| Timing alignment of MIDI to audio | onset s.d. 3.8 ms (a property of the data, not a threshold) | Disklavier | F10 | **read** |

## 10. Corrections to the report and constituents it missed

| Report item | Follow-up result | Label |
|-------------|------------------|-------|
| 1.1 row 6: "B from 0.0002 (bass) to 0.4 (treble)" | **Wrong in the treble.** Rigaud et al. give the treble-bridge asymptote ln B = 0.0944 m - 13.68 (m = MIDI index). My arithmetic: B = 0.0032 at C6 (m 84), 0.0098 at C7 (m 96), 0.031 at C8 (m 108). Measured on one grand: C6 2.3e-3, D5 1.2e-3, C4 3.3e-4, C3 1.1e-4, C2 3.8e-5 (F6 Table I). So the top octave is 0.01-0.03; 0.4 is about 10-40x too high. Rigaud also warns that in the treble one peak per partial does not describe the spectrum | **read** F13, F6; arithmetic **mine** |
| 1.1 row 5: strike point "about 1/8-1/7" | Bass slightly under 1/8, decreasing slowly up to about A4 (note 49), then quickly, to about 1/12-1/17 in the treble (Conklin). 1/7-1/9 is the textbook range for the bass and mid | **snippet** F14, search summary |
| 1.1 row 14: knock caused by the hammer exciting the soundboard directly | Direct structure path is the weaker one at the bridge (about 40 dB below the strings, up to 2 kHz); the strong attack path is the longitudinal precursor. The treble "thump" is the soundboard's low modes, which ring longer than the string partials are spaced apart and are audible in the attack, 0.2-0.5 s | **read** F1, F2, F3, F5 |
| 1.1 row 3 (let-off, backcheck): "not quantified" | Backcheck catches the hammer 10-20 ms after contact; the rebounding hammer vibrates at about 900 Hz; the hammer shank vibration adds small oscillations to the hammer force that models without the shank miss | **read** F1, F3 |
| Reference "Decay rates of piano tones (2023?)" | Martin 1947, JASA 19(4), 535 | **snippet** |
| Missed constituents | (a) string-borne longitudinal precursor as a separate 1-2 ms broadband burst (F1); (b) low soundboard modes ringing after the pulse in upper notes, present even at light touch (F3); (c) key resonances at 290, 440, 900 Hz (F1); (d) key restrike, duplex stringing, aliquots, una corda, lid position, dampers as known gaps of full models (F12); (e) frame/plate resonance at 38 Hz (F1); (f) hammer-shank vibration (F1, F3) | **read** |

## What this changes (mine)

1. Knock. Model the attack as three parts, not one: (a) a 1-2 ms broadband burst carried by the string, about -10 dB re the first transverse pulse at the bridge, reaching 5 kHz, independent of touch, scaling with the blow; (b) a structure "thump" with modes near 100, 250 to 330 Hz, 25-40 dB weaker at the bridge but stronger in the microphone signal, ringing 0.2-0.5 s in the treble; (c) touch/key resonances (290, 440 Hz before, 900 Hz after contact), random per note. The project's knock (hammer force through the body FIR) matches only (b) and the weak structural path. The measured gap of 12-43 dB under the strings is of the size of the difference between the string path and the structure path (about 30 dB) in F1 (different quantities, so this is a hint, not a proof).
2. The missing 1-2 dB at 250 Hz at onset matches the rim/keybed/soundboard modes in F1 (250, 330, 100, 95 Hz). Give the body FIR a long low-frequency tail for the upper notes (Bank used 0.36 s).
3. The 8 kHz click at onsets is above what F1 found for either path (string path to about 5 kHz, structure path to 2 kHz). I would treat it as a model artefact to remove, not as a target.
4. Brightness vs velocity: the force-spectrum cutoff scales as v^0.4-0.6 (0.4-0.5 for most Steinway D hammers; higher exponents toward the treble). Use this as the check for the two-corner envelope: corner frequency proportional to v^b with b about 0.45, and larger b in the treble. Contact time: about 4 ms bass to under 1 ms top; +-20 % around mf.
5. Phantoms. Keep the a_j * a_k product law (it already carries the faster-than-v^2 growth once a_j depends on the hammer, F4) and the emphasis near a longitudinal mode; the 15 f1 assumption is close to the 14 f1 seen in F3 and F4. Add (j, j+2) parents for even phantoms (F2). Do not assume phantoms decay at twice the partial rate; F2 saw comparable rates. The -26 dB bass and -60 dB treble priors have no literature value to test them against. Audibility up to C5 at ff (F5) says the treble prior must not fall to inaudible levels below C5.
6. Re-strike: no literature number. The physics (phase-dependent add/remove) suggests the removal should be fitted on repeated notes in MAESTRO rather than fixed by prior.
7. Random per strike: the touch precursor and key resonances (not velocity-driven), unison excitation imbalance (hammer irregularity, Weinreich), and MIDI-velocity-to-level mapping that depends on pitch (F10). Mistuning: beats need more than 0.3 Hz in the mid range; below that the coupled modes lock, so a random fixed mistuning of 0-1 Hz per note is in the right range (the coupled-mode model does the locking by itself).
8. Pedal: the sympathetic halo is a residual +10 dB at C4 over 1-5 s, 5-30 dB in bands below 2.5 kHz and none above; it sits 30-45 dB below the tone itself when the played string is damped. The unused resonator bank should be limited to about 2.5 kHz and to note-dependent T60 gains of +40-65 % on the fundamental in C2-C4.
9. Error bars for the measure suite: decay-time changes under -25/+40 %, partial-level changes under about 1 dB and centroid changes under roughly 15 % (jnd 0.153, unit uncertain) are near or below the audible range; use them as tolerance bands, not as targets.
10. Fix the parameter priors: B in the top octave 0.01-0.03; strike point 1/8 in the bass falling to 1/12-1/17 at the top.

## Still open

- Level (dB re the tone) of the knock, string precursor and thump per register and velocity; F1 gives only one note (C4) on one piano.
- Level of phantom partials re the transverse partials per register (Conklin 1999, Moore's phantom-partial paper, Nakamura & Naganuma were not reachable; one NSF PAR link refused the connection).
- Pitch glide in cents (only my estimate, 5.5).
- Repeatability of strikes (level, spectrum, decay) on acoustic pianos and Disklaviers.
- Prompt/aftersound decay rates and the knee level and time by register; Karatsovis PhD and Martin 1947 not opened.
- Hammer K per register and multiple contacts in the treble; Chaigne & Askenfelt 1994 tables are images.
- Any measurement of re-strike removal; damper and pedal noise levels.
- Attack/rise-time JND, AM/beat-depth thresholds, brightness JND in piano tones; exact inharmonicity threshold values (formula from a search summary looks suspicious at 55 Hz).
- The original report's R13, R14, R15 were not re-read here.

## References

- F1 Askenfelt A. (1993). Observations on the transient components of the piano tone. STL-QPSR 34(4), 15-22. https://www.speech.kth.se/qpsr (PDF via https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=624e7d25054fb6f5e9ab864e48f432d7dc5871fd)
- F2 Bank B. (2006). Physics-based sound synthesis of string instruments including geometric nonlinearities. PhD thesis, BME. https://home.mit.bme.hu/~bank/phd/phd.pdf
- F3 Chabassier J., Chaigne A., Joly P. (2013). Modeling and simulation of a grand piano. JASA 134(1), 648-665. https://perso.ensta-paris.fr/~touze/PDF/Batwoman/chabassier-jasa.pdf
- F4 Bank B., Sujbert L. (2005). Generation of longitudinal vibrations in piano strings. JASA 117(4), 2268-2278. https://home.mit.bme.hu/~bank/publist/jasa05.pdf
- F5 Bank B., Lehtonen H.-M. (2010). Perception of longitudinal components in piano string vibrations. JASA-EL 128(3), EL117. https://pubs.aip.org/asa/jasa/article/128/3/EL117/598974 ; full text in Lehtonen's thesis publications, https://aaltodoc.aalto.fi/items/7d986ec9-91b0-4ddc-a106-f3dade146608
- F6 Lehtonen H.-M., Penttinen H., Rauhala J., Välimäki V. (2007). Analysis and modeling of piano sustain-pedal effects. JASA 122(3), 1787. (Aalto thesis publication I, same item URL as F5)
- F7 Lehtonen H.-M., Askenfelt A., Välimäki V. (2009). Analysis of the part-pedaling effect in the piano. JASA-EL 126(2), EL49. (Aalto thesis publication II)
- F8 Askenfelt A., Jansson E. From touch to string vibration: string contact duration and dynamic level. https://www.speech.kth.se/music/5_lectures/askenflt/stricont.html
- F9 Russell D., Rossing T. (1998). Testing the nonlinearity of piano hammers using residual shock spectra. Acustica 84, 967-975. https://www.acs.psu.edu/drussell/publications/pianohammer.pdf
- F10 Goebl W., Bresin R. (2003). Measurement and reproduction accuracy of computer-controlled grand pianos. JASA 114(4), 2273-2283. https://iwk.mdw.ac.at/goebl/papers/Goebl-Bresin_JASA2003_reproAccuracy.pdf
- F11 Weinreich G. The coupled motion of piano strings (KTH lecture page); paper: JASA 62(6), 1474 (1977). https://www.speech.kth.se/music/5_lectures/weinreic/mistuned.html
- F12 Bank B., Chabassier J. (2019). Model-based digital pianos: from physics to sound synthesis. IEEE SPM 36(1), 103-114. https://hal.inria.fr/hal-01894219/file/hal.pdf
- F13 Rigaud F., David B., Daudet L. (2013). A parametric model and estimation techniques for the inharmonicity and tuning of the piano. JASA 133(5), 3107. https://www.institut-langevin.espci.fr/IMG/pdf/2013_a_parametric_model_and_estimation_techniques_for_the_inharmonicity_and_tuning_of_the_piano.pdf
- F14 Conklin H. Where should the hammer hit the string? (KTH lecture page). https://www.speech.kth.se/music/5_lectures/conklin/whereshould.html
- F15 Simionato R., Fasciani S. (2024). Sines, transient, noise neural modeling of piano notes. https://arxiv.org/pdf/2409.06513
- F16 Järveläinen H., Välimäki V., Karjalainen M. (2001). Audibility of the timbral effects of inharmonicity in stringed instrument tones. ARLO 2(3), 79. https://pubs.aip.org/asa/arlo/article/2/3/79/123666 ; ASA 1997 abstract 4pPP5, https://www.auditory.org/asamtgs/asa97snd/4pPP/4pPP5.html ; Kazazis S., Depalle P., McAdams S. (2021). Ordinal scaling of timbre-related spectral audio descriptors. JASA 149(6), 3785. https://www.mcgill.ca/mpcl/files/mpcl/kazazis_2021a_jasa.pdf ; Martin D.W. (1947). Decay rates of piano tones. JASA 19(4), 535. https://pubs.aip.org/asa/jasa/article-abstract/19/4/535/763628
- Other pointers: Chaigne & Askenfelt (1994) part I PDF, https://www.math.kent.edu/~zheng/62262/piano_wave.pdf ; Goebl, Bresin, Galembo (2005) Touch and temporal behavior of grand piano actions, https://iwk.mdw.ac.at/goebl/papers/Goebl-Bresin-Galembo_JASA2005_PianoAction.pdf (not read); Karatsovis PhD, https://eprints.soton.ac.uk/333304/1/Karatsovis_PhD_document.pdf (not read).
