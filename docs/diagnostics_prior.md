## Isolated notes (dry strings)

| key | vel | f1 Hz | B_eff | partials 2..6 dB re f1 | centroid Hz | peak dBFS |
|---|---|---|---|---|---|---|
| A0 | 40 | 27.24 | 2.47e-04 | +5 +6 +6 +5 +1 | 351 | -41.8 |
| A0 | 80 | 27.24 | 2.47e-04 | +5 +7 +7 +5 +2 | 472 | -27.1 |
| A0 | 120 | 27.24 | 2.42e-04 | +5 +7 +7 +6 +2 | 590 | -12.2 |
| C2 | 40 | 65.26 | 1.16e-04 | +3 +3 +1 -3 -7 | 423 | -45.3 |
| C2 | 80 | 65.26 | 1.16e-04 | +4 +4 +3 -0 -4 | 586 | -29.6 |
| C2 | 120 | 65.26 | 1.14e-04 | +4 +5 +4 +1 -3 | 782 | -13.5 |
| C4 | 40 | 261.41 | 2.96e-04 | -5 -6 -9 -14 -20 | 734 | -49.0 |
| C4 | 80 | 261.41 | 2.96e-04 | -2 -2 -4 -8 -13 | 935 | -31.9 |
| C4 | 120 | 261.41 | 2.95e-04 | -0 +0 -1 -4 -8 | 1212 | -15.5 |
| A4 | 40 | 440.19 | 7.15e-04 | -9 -15 -21 -29 -38 | 831 | -51.8 |
| A4 | 80 | 440.19 | 7.16e-04 | -6 -10 -14 -21 -28 | 1065 | -33.2 |
| A4 | 120 | 440.19 | 7.16e-04 | -3 -5 -9 -14 -20 | 1365 | -15.5 |
| C6 | 40 | 1050.19 | 3.17e-03 | -26 -47 -66 -87 -124 | 1136 | -61.8 |
| C6 | 80 | 1050.19 | 2.67e-03 | -20 -37 -52 -66 -83 | 1246 | -36.4 |
| C6 | 120 | 1050.19 | 2.84e-03 | -15 -29 -41 -52 -65 | 1455 | -14.2 |
| C7 | 40 | 2109.94 | nan | -39 -70 -95 | 2168 | -64.2 |
| C7 | 80 | 2109.94 | nan | -32 -55 -74 | 2259 | -35.0 |
| C7 | 120 | 2109.94 | nan | -26 -45 -58 | 2440 | -10.1 |
| C8 | 40 | 4246.92 | nan | -47 | 4294 | -72.1 |
| C8 | 80 | 4246.92 | nan | -37 | 4385 | -37.8 |
| C8 | 120 | 4246.92 | nan | -29 | 4560 | -8.8 |

## Acceptance checks: 45/50 pass

| check | measured | target | result |
|---|---|---|---|
| C1 decay profile (median, +-3 keys, mf+ff), model/recording dB at 0.5/1/2/4/8/16 s | -3/-3 -7/-6 -12/-10 -18/-18 -23/-27 -34/-33 | each +-6 dB | PASS |
| C1 early spectral slope, mf, radiated (dB/oct) | 2.6 | 0.7 +-5 | PASS |
| C1 tuning re A4 (cents) [regression: the prior copies these values] | -15.8 | -15.9 +-4 | PASS |
| C2 decay profile (median, +-3 keys, mf+ff), model/recording dB at 0.5/1/2/4/8/16 s | -5/-5 -11/-10 -19/-12 -22/-22 -28/-24 -40/-35 | each +-6 dB | FAIL |
| C2 early spectral slope, mf, radiated (dB/oct) | -3.3 | -3.5 +-5 | PASS |
| C2 tuning re A4 (cents) [regression: the prior copies these values] | -4.0 | -4.3 +-4 | PASS |
| C3 decay profile (median, +-3 keys, mf+ff), model/recording dB at 0.5/1/2/4/8/16 s | -8/-5 -16/-10 -22/-22 -28/-24 -36/-32 -52/-45 | each +-6 dB | FAIL |
| C3 early spectral slope, mf, radiated (dB/oct) | -7.3 | -5.3 +-5 | PASS |
| C3 tuning re A4 (cents) [regression: the prior copies these values] | -0.4 | +0.5 +-4 | PASS |
| C4 decay profile (median, +-3 keys, mf+ff), model/recording dB at 0.5/1/2/4/8/16 s | -10/-12 -20/-21 -28/-25 -32/-31 -41/-45 -57/-55 | each +-6 dB | PASS |
| C4 early spectral slope, mf, radiated (dB/oct) | -12.0 | -15.4 +-5 | PASS |
| C4 tuning re A4 (cents) [regression: the prior copies these values] | -1.2 | -1.4 +-4 | PASS |
| A4 decay profile (median, +-3 keys, mf+ff), model/recording dB at 0.5/1/2/4/8 s | -11/-15 -24/-26 -34/-29 -39/-38 -50/-53 | each +-6 dB | PASS |
| A4 early spectral slope, mf, radiated (dB/oct) | -19.3 | -19.2 +-5 | PASS |
| A4 tuning re A4 (cents) [regression: the prior copies these values] | +0.0 | +0.0 +-4 | PASS |
| C5 decay profile (median, +-3 keys, mf+ff), model/recording dB at 0.5/1/2/4/8 s | -12/-20 -26/-21 -34/-29 -43/-37 -51/-53 | each +-6 dB | FAIL |
| C5 early spectral slope, mf, radiated (dB/oct) | -16.8 | -20.3 +-5 | PASS |
| C5 tuning re A4 (cents) [regression: the prior copies these values] | -0.3 | -0.4 +-4 | PASS |
| C6 decay profile (median, +-3 keys, mf+ff), model/recording dB at 0.5/1/2/4/8 s | -12/-15 -26/-22 -46/-36 -51/-47 -63/-69 | each +-6 dB | FAIL |
| C6 early spectral slope, mf, radiated (dB/oct) | -28.4 | -28.5 +-5 | PASS |
| C6 tuning re A4 (cents) [regression: the prior copies these values] | +5.2 | +5.3 +-4 | PASS |
| C7 decay profile (median, +-3 keys, mf+ff), model/recording dB at 0.5/1/2/4 s | -19/-19 -39/-34 -51/-53 -61/-66 | each +-6 dB | PASS |
| C7 early spectral slope, mf, radiated (dB/oct) | nan | -37.2 +-5 | FAIL |
| C7 tuning re A4 (cents) [regression: the prior copies these values] | +13.0 | +14.0 +-4 | PASS |
| C4 effective B vs Rigaud et al. [regression: the prior is this curve] | 0.000296 | 3.30e-04 (x1.5) | PASS |
| C6 effective B vs Rigaud et al. [regression: the prior is this curve] | 0.00267 | 2.85e-03 (x1.5) | PASS |
| C2 effective B vs Iowa (x2: bass is piano-specific) | 0.000122 | 1.17e-04 (x2) | PASS |
| C4 effective B vs Iowa (x2: bass is piano-specific) | 0.000285 | 3.26e-04 (x2) | PASS |
| C4 vel 80 partials 2-6 re p1 (dB) [M] | -2 -2 -4 -8 -13 | -30..+3 | PASS |
| C4 vel 80 partial 10 re p1 (dB) [M] | -32 | <= -30 | PASS |
| C4 spectral slope, partials 1-10, vel 64 (dB/oct) vs Hall (another piano) | -11.3 | -15 +-5 | PASS |
| C4 slope step vel 30 -> 64 (dB/oct) [regression: velocity law from Hall] | +3.0 | +3 +-1.5 | PASS |
| C4 slope step vel 64 -> 110 (dB/oct) [regression: velocity law from Hall] | +3.2 | +4 +-1.5 | PASS |
| C2 brightness rises with velocity (centroid Hz, vel 30/60/90/120) | 389 500 633 782 | increasing | PASS |
| C4 brightness rises with velocity (centroid Hz, vel 30/60/90/120) | 690 828 995 1212 | increasing | PASS |
| C6 brightness rises with velocity (centroid Hz, vel 30/60/90/120) | 1118 1181 1287 1455 | increasing | PASS |
| C7 brightness rises with velocity (centroid Hz, vel 30/60/90/120) | 2155 2205 2294 2440 | increasing | PASS |
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
