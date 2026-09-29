## Isolated notes (dry strings)

| key | vel | f1 Hz | B_eff | partials 2..6 dB re f1 | centroid Hz | peak dBFS |
|---|---|---|---|---|---|---|
| A0 | 40 | 27.24 | 5.07e-04 | +5 +7 +7 +5 +2 | 403 | -51.8 |
| A0 | 80 | 27.24 | 5.07e-04 | +5 +7 +7 +6 +3 | 609 | -37.3 |
| A0 | 120 | 27.24 | 5.07e-04 | +5 +7 +7 +6 +4 | 845 | -23.1 |
| C2 | 40 | 65.26 | 1.69e-04 | +4 +4 +2 -2 -6 | 646 | -52.4 |
| C2 | 80 | 65.26 | 1.69e-04 | +4 +5 +4 +1 -3 | 961 | -36.3 |
| C2 | 120 | 65.26 | 1.69e-04 | +4 +6 +5 +3 -1 | 1309 | -20.6 |
| C4 | 40 | 261.41 | 3.26e-04 | -10 -17 -23 -29 -37 | 521 | -55.1 |
| C4 | 80 | 261.41 | 3.26e-04 | -6 -10 -15 -21 -27 | 758 | -37.2 |
| C4 | 120 | 261.41 | 3.26e-04 | -2 -5 -9 -13 -19 | 1092 | -19.7 |
| A4 | 40 | 440.19 | 7.15e-04 | -20 -31 -41 -48 -56 | 580 | -57.5 |
| A4 | 80 | 440.19 | 7.15e-04 | -15 -23 -31 -37 -44 | 741 | -38.4 |
| A4 | 120 | 440.19 | 7.15e-04 | -10 -16 -22 -28 -34 | 1024 | -20.4 |
| C6 | 40 | 1050.26 | 2.55e-03 | -29 -49 -65 -77 -87 | 1121 | -65.3 |
| C6 | 80 | 1050.26 | 2.60e-03 | -23 -40 -53 -63 -74 | 1220 | -38.7 |
| C6 | 120 | 1050.26 | 2.60e-03 | -18 -32 -43 -52 -61 | 1417 | -15.7 |
| C7 | 40 | 2110.00 | nan | -33 -56 -76 | 2186 | -66.7 |
| C7 | 80 | 2109.94 | nan | -26 -46 -63 | 2294 | -37.6 |
| C7 | 120 | 2109.94 | nan | -21 -37 -51 | 2500 | -12.7 |
| C8 | 40 | 4246.86 | nan | -35 | 4347 | -71.6 |
| C8 | 80 | 4246.92 | nan | -30 | 4466 | -37.4 |
| C8 | 120 | 4246.92 | nan | -24 | 4653 | -8.5 |

## Acceptance checks: 47/50 pass

| check | measured | target | result |
|---|---|---|---|
| C1 decay profile (median, +-3 keys, mf+ff), model/recording dB at 0.5/1/2/4/8/16 s | -2/-3 -5/-6 -9/-10 -19/-18 -26/-27 -34/-33 | each +-6 dB | PASS |
| C1 early spectral slope, mf, radiated (dB/oct) | 1.6 | 0.7 +-5 | PASS |
| C1 tuning re A4 (cents) [regression: the prior copies these values] | -15.9 | -15.9 +-4 | PASS |
| C2 decay profile (median, +-3 keys, mf+ff), model/recording dB at 0.5/1/2/4/8/16 s | -3/-5 -8/-10 -18/-12 -22/-22 -27/-24 -36/-35 | each +-6 dB | PASS |
| C2 early spectral slope, mf, radiated (dB/oct) | -3.7 | -3.5 +-5 | PASS |
| C2 tuning re A4 (cents) [regression: the prior copies these values] | -4.2 | -4.3 +-4 | PASS |
| C3 decay profile (median, +-3 keys, mf+ff), model/recording dB at 0.5/1/2/4/8/16 s | -5/-5 -12/-10 -22/-22 -28/-24 -34/-32 -48/-45 | each +-6 dB | PASS |
| C3 early spectral slope, mf, radiated (dB/oct) | -7.2 | -5.3 +-5 | PASS |
| C3 tuning re A4 (cents) [regression: the prior copies these values] | -0.7 | +0.5 +-4 | PASS |
| C4 decay profile (median, +-3 keys, mf+ff), model/recording dB at 0.5/1/2/4/8/16 s | -9/-12 -19/-21 -29/-25 -32/-31 -40/-45 -56/-55 | each +-6 dB | PASS |
| C4 early spectral slope, mf, radiated (dB/oct) | -14.6 | -15.4 +-5 | PASS |
| C4 tuning re A4 (cents) [regression: the prior copies these values] | -1.3 | -1.4 +-4 | PASS |
| A4 decay profile (median, +-3 keys, mf+ff), model/recording dB at 0.5/1/2/4/8/16 s | -14/-15 -26/-26 -34/-29 -37/-38 -50/-53 -63/-64 | each +-6 dB | PASS |
| A4 early spectral slope, mf, radiated (dB/oct) | -19.0 | -19.2 +-5 | PASS |
| A4 tuning re A4 (cents) [regression: the prior copies these values] | +0.0 | +0.0 +-4 | PASS |
| C5 decay profile (median, +-3 keys, mf+ff), model/recording dB at 0.5/1/2/4/8 s | -17/-20 -25/-21 -32/-29 -44/-37 -49/-53 | each +-6 dB | FAIL |
| C5 early spectral slope, mf, radiated (dB/oct) | -19.6 | -20.3 +-5 | PASS |
| C5 tuning re A4 (cents) [regression: the prior copies these values] | -0.2 | -0.4 +-4 | PASS |
| C6 decay profile (median, +-3 keys, mf+ff), model/recording dB at 0.5/1/2/4/8 s | -13/-15 -31/-22 -41/-36 -46/-47 -64/-69 | each +-6 dB | FAIL |
| C6 early spectral slope, mf, radiated (dB/oct) | -28.6 | -28.5 +-5 | PASS |
| C6 tuning re A4 (cents) [regression: the prior copies these values] | +5.2 | +5.3 +-4 | PASS |
| C7 decay profile (median, +-3 keys, mf+ff), model/recording dB at 0.5/1/2/4 s | -17/-19 -37/-34 -49/-53 -59/-66 | each +-6 dB | FAIL |
| C7 early spectral slope, mf, radiated (dB/oct) | -32.2 | -37.2 +-5 | PASS |
| C7 tuning re A4 (cents) [regression: the prior copies these values] | +12.9 | +14.0 +-4 | PASS |
| C4 effective B vs Rigaud et al. [regression: the prior is this curve] | 0.000326 | 3.30e-04 (x1.5) | PASS |
| C6 effective B vs Rigaud et al. [regression: the prior is this curve] | 0.0026 | 2.85e-03 (x1.5) | PASS |
| C2 effective B vs Iowa (x2: bass is piano-specific) | 0.000174 | 1.17e-04 (x2) | PASS |
| C4 effective B vs Iowa (x2: bass is piano-specific) | 0.000323 | 3.26e-04 (x2) | PASS |
| C4 vel 80 partials 2-6 re p1 (dB) [M] | -6 -10 -15 -21 -27 | -30..+3 | PASS |
| C4 vel 80 partial 10 re p1 (dB) [M] | -43 | <= -30 | PASS |
| C4 spectral slope, partials 1-10, vel 30 (dB/oct) [regression: fitted to Hall] | -18.0 | -18 +-2 | PASS |
| C4 spectral slope, partials 1-10, vel 64 (dB/oct) [regression: fitted to Hall] | -14.8 | -15 +-2 | PASS |
| C4 spectral slope, partials 1-10, vel 110 (dB/oct) [regression: fitted to Hall] | -11.3 | -11 +-2 | PASS |
| C2 brightness rises with velocity (centroid Hz, vel 30/60/90/120) | 578 797 1046 1309 | increasing | PASS |
| C4 brightness rises with velocity (centroid Hz, vel 30/60/90/120) | 478 627 832 1092 | increasing | PASS |
| C6 brightness rises with velocity (centroid Hz, vel 30/60/90/120) | 1106 1161 1257 1417 | increasing | PASS |
| C7 brightness rises with velocity (centroid Hz, vel 30/60/90/120) | 2170 2230 2335 2500 | increasing | PASS |
| A0 radiated fundamental below strongest partial (dB) | 52 | >= 10 | PASS |
| C2 radiated fundamental below strongest partial (dB) | 19 | >= 10 | PASS |
| A0 release to -60 dB (s) | 0.62 | >= 0.5 | PASS |
| A4 release to -60 dB (s) | 0.24 | 0.2-0.4 | PASS |
| C8 released vs held (dB, undamped) | +0.0 | 0 +-1 | PASS |
| pedal halo, symp re strings (dB) [bank disabled in this model] | -32 | -40..-25 | PASS |
| halo with vs without pedal (dB) [bank disabled in this model] | 12 | >= 10 | PASS |
| knock re tone, first 60 ms, vel 120 (dB) | -25 | -25 +-3 | PASS |
| knock re tone, first 60 ms, vel 25 (dB) | -12 | -12 +-3 | PASS |
| damper noise re released note (dB) | -39 | -45..-35 | PASS |
| pedal-press noise re mf note (dB) | -35 | -35 +-5 | PASS |
| noise energy > 6 ms before the hammer strikes (fraction) | 0.000 | < 0.01 | PASS |
| ff: key-bottom thump before the hammer (fraction of noise energy) | 0.229 | > 0.005 | PASS |
