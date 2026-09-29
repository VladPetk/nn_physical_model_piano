## Isolated notes (dry strings)

| key | vel | f1 Hz | B_eff | partials 2..6 dB re f1 | centroid Hz | peak dBFS |
|---|---|---|---|---|---|---|
| A0 | 40 | 27.24 | 2.47e-04 | +5 +6 +6 +5 +1 | 352 | -45.7 |
| A0 | 80 | 27.24 | 2.47e-04 | +5 +7 +7 +5 +2 | 478 | -31.1 |
| A0 | 120 | 27.24 | 2.47e-04 | +5 +7 +7 +6 +3 | 625 | -16.9 |
| C2 | 40 | 65.26 | 1.16e-04 | +3 +3 +1 -3 -7 | 424 | -47.4 |
| C2 | 80 | 65.26 | 1.16e-04 | +4 +4 +3 -0 -4 | 589 | -31.8 |
| C2 | 120 | 65.26 | 1.16e-04 | +4 +4 +4 +1 -2 | 798 | -16.1 |
| C4 | 40 | 261.41 | 2.96e-04 | -5 -6 -9 -14 -20 | 734 | -49.0 |
| C4 | 80 | 261.41 | 2.96e-04 | -2 -2 -4 -8 -13 | 934 | -31.9 |
| C4 | 120 | 261.41 | 2.96e-04 | -1 +0 -1 -4 -8 | 1191 | -15.8 |
| A4 | 40 | 440.19 | 7.16e-04 | -9 -15 -21 -29 -38 | 831 | -51.5 |
| A4 | 80 | 440.19 | 7.16e-04 | -6 -10 -14 -21 -28 | 1065 | -33.0 |
| A4 | 120 | 440.19 | 7.16e-04 | -3 -5 -9 -14 -20 | 1347 | -15.3 |
| C6 | 40 | 1050.19 | 2.54e-03 | -26 -47 -66 -82 -96 | 1137 | -61.5 |
| C6 | 80 | 1050.19 | 2.61e-03 | -20 -37 -52 -66 -79 | 1247 | -36.1 |
| C6 | 120 | 1050.19 | 2.61e-03 | -15 -29 -41 -52 -64 | 1456 | -13.9 |
| C7 | 40 | 2110.00 | nan | -38 -67 -86 | 2170 | -63.4 |
| C7 | 80 | 2109.94 | nan | -32 -55 -72 | 2260 | -34.2 |
| C7 | 120 | 2109.94 | nan | -26 -45 -58 | 2441 | -9.3 |
| C8 | 40 | 4246.92 | nan | -44 | 4295 | -72.1 |
| C8 | 80 | 4246.92 | nan | -37 | 4386 | -37.8 |
| C8 | 120 | 4246.92 | nan | -29 | 4561 | -8.8 |

## Acceptance checks: 45/50 pass

| check | measured | target | result |
|---|---|---|---|
| C1 decay profile (median, +-3 keys, mf+ff), model/recording dB at 0.5/1/2/4/8/16 s | -3/-3 -6/-6 -12/-10 -17/-18 -23/-27 -33/-33 | each +-6 dB | PASS |
| C1 early spectral slope, mf, radiated (dB/oct) | 2.5 | 0.7 +-5 | PASS |
| C1 tuning re A4 (cents) [regression: the prior copies these values] | -15.8 | -15.9 +-4 | PASS |
| C2 decay profile (median, +-3 keys, mf+ff), model/recording dB at 0.5/1/2/4/8/16 s | -5/-5 -11/-10 -19/-12 -21/-22 -28/-24 -40/-35 | each +-6 dB | FAIL |
| C2 early spectral slope, mf, radiated (dB/oct) | -3.5 | -3.5 +-5 | PASS |
| C2 tuning re A4 (cents) [regression: the prior copies these values] | -4.0 | -4.3 +-4 | PASS |
| C3 decay profile (median, +-3 keys, mf+ff), model/recording dB at 0.5/1/2/4/8/16 s | -8/-5 -16/-10 -22/-22 -27/-24 -36/-32 -52/-45 | each +-6 dB | FAIL |
| C3 early spectral slope, mf, radiated (dB/oct) | -7.3 | -5.3 +-5 | PASS |
| C3 tuning re A4 (cents) [regression: the prior copies these values] | -0.4 | +0.5 +-4 | PASS |
| C4 decay profile (median, +-3 keys, mf+ff), model/recording dB at 0.5/1/2/4/8/16 s | -10/-12 -20/-21 -28/-25 -32/-31 -41/-45 -57/-55 | each +-6 dB | PASS |
| C4 early spectral slope, mf, radiated (dB/oct) | -12.0 | -15.4 +-5 | PASS |
| C4 tuning re A4 (cents) [regression: the prior copies these values] | -1.2 | -1.4 +-4 | PASS |
| A4 decay profile (median, +-3 keys, mf+ff), model/recording dB at 0.5/1/2/4/8 s | -11/-15 -24/-26 -34/-29 -39/-38 -50/-53 | each +-6 dB | PASS |
| A4 early spectral slope, mf, radiated (dB/oct) | -19.3 | -19.2 +-5 | PASS |
| A4 tuning re A4 (cents) [regression: the prior copies these values] | +0.0 | +0.0 +-4 | PASS |
| C5 decay profile (median, +-3 keys, mf+ff), model/recording dB at 0.5/1/2/4/8 s | -12/-20 -26/-21 -34/-29 -43/-37 -50/-53 | each +-6 dB | FAIL |
| C5 early spectral slope, mf, radiated (dB/oct) | -16.8 | -20.3 +-5 | PASS |
| C5 tuning re A4 (cents) [regression: the prior copies these values] | -0.3 | -0.4 +-4 | PASS |
| C6 decay profile (median, +-3 keys, mf+ff), model/recording dB at 0.5/1/2/4/8 s | -12/-15 -26/-22 -46/-36 -53/-47 -63/-69 | each +-6 dB | FAIL |
| C6 early spectral slope, mf, radiated (dB/oct) | -28.4 | -28.5 +-5 | PASS |
| C6 tuning re A4 (cents) [regression: the prior copies these values] | +5.2 | +5.3 +-4 | PASS |
| C7 decay profile (median, +-3 keys, mf+ff), model/recording dB at 0.5/1/2/4 s | -18/-19 -39/-34 -51/-53 -61/-66 | each +-6 dB | PASS |
| C7 early spectral slope, mf, radiated (dB/oct) | nan | -37.2 +-5 | FAIL |
| C7 tuning re A4 (cents) [regression: the prior copies these values] | +12.9 | +14.0 +-4 | PASS |
| C4 effective B vs Rigaud et al. [regression: the prior is this curve] | 0.000296 | 3.30e-04 (x1.5) | PASS |
| C6 effective B vs Rigaud et al. [regression: the prior is this curve] | 0.00261 | 2.85e-03 (x1.5) | PASS |
| C2 effective B vs Iowa (x2: bass is piano-specific) | 0.000122 | 1.17e-04 (x2) | PASS |
| C4 effective B vs Iowa (x2: bass is piano-specific) | 0.000285 | 3.26e-04 (x2) | PASS |
| C4 vel 80 partials 2-6 re p1 (dB) [M] | -2 -2 -4 -8 -13 | -30..+3 | PASS |
| C4 vel 80 partial 10 re p1 (dB) [M] | -32 | <= -30 | PASS |
| C4 spectral slope, partials 1-10, vel 64 (dB/oct) vs Hall (another piano) | -11.3 | -15 +-5 | PASS |
| C4 slope step vel 30 -> 64 (dB/oct) [regression: velocity law from Hall] | +3.0 | +3 +-1.5 | PASS |
| C4 slope step vel 64 -> 110 (dB/oct) [regression: velocity law from Hall] | +3.2 | +4 +-1.5 | PASS |
| C2 brightness rises with velocity (centroid Hz, vel 30/60/90/120) | 389 501 637 798 | increasing | PASS |
| C4 brightness rises with velocity (centroid Hz, vel 30/60/90/120) | 690 828 992 1191 | increasing | PASS |
| C6 brightness rises with velocity (centroid Hz, vel 30/60/90/120) | 1119 1182 1288 1456 | increasing | PASS |
| C7 brightness rises with velocity (centroid Hz, vel 30/60/90/120) | 2156 2207 2295 2441 | increasing | PASS |
| A0 radiated fundamental below strongest partial (dB) | 51 | >= 10 | PASS |
| C2 radiated fundamental below strongest partial (dB) | 18 | >= 10 | PASS |
| A0 release to -60 dB (s) | 0.61 | >= 0.5 | PASS |
| A4 release to -60 dB (s) | 0.24 | 0.2-0.4 | PASS |
| C8 released vs held (dB, undamped) | +0.0 | 0 +-1 | PASS |
| pedal halo, symp re strings (dB) [bank disabled in this model] | -31 | -40..-25 | PASS |
| halo with vs without pedal (dB) [bank disabled in this model] | 11 | >= 10 | PASS |
| knock re tone, first 60 ms, vel 120 (dB) | -24 | -25 +-3 | PASS |
| knock re tone, first 60 ms, vel 25 (dB) | -12 | -12 +-3 | PASS |
| damper noise re released note (dB) | -41 | -45..-35 | PASS |
| pedal-press noise re mf note (dB) | -37 | -35 +-5 | PASS |
| noise energy > 6 ms before the hammer strikes (fraction) | 0.000 | < 0.01 | PASS |
| ff: key-bottom thump before the hammer (fraction of noise energy) | 0.229 | > 0.005 | PASS |
