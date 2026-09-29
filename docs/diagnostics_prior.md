## Isolated notes (dry strings)

| key | vel | f1 Hz | B_eff | partials 2..6 dB re f1 | centroid Hz | peak dBFS |
|---|---|---|---|---|---|---|
| A0 | 40 | 27.24 | 5.07e-04 | +5 +7 +7 +5 +1 | 264 | -50.7 |
| A0 | 80 | 27.24 | 5.07e-04 | +5 +7 +7 +6 +3 | 415 | -36.0 |
| A0 | 120 | 27.24 | 5.07e-04 | +5 +7 +7 +6 +3 | 526 | -22.3 |
| C2 | 40 | 65.26 | 1.73e-04 | +4 +3 +0 -4 -8 | 430 | -53.4 |
| C2 | 80 | 65.26 | 1.73e-04 | +4 +5 +3 +0 -4 | 670 | -37.9 |
| C2 | 120 | 65.26 | 1.73e-04 | +5 +5 +4 +2 -2 | 845 | -23.2 |
| C4 | 40 | 261.47 | 3.27e-04 | -9 -17 -23 -30 -37 | 547 | -54.1 |
| C4 | 80 | 261.47 | 3.27e-04 | -5 -10 -16 -22 -28 | 783 | -36.2 |
| C4 | 120 | 261.47 | 3.27e-04 | -2 -7 -12 -17 -23 | 965 | -20.8 |
| A4 | 40 | 440.00 | 7.18e-04 | -10 -17 -23 -29 -36 | 983 | -53.1 |
| A4 | 80 | 440.00 | 7.18e-04 | -6 -12 -17 -22 -28 | 1311 | -34.3 |
| A4 | 120 | 440.00 | 7.18e-04 | -5 -10 -14 -19 -24 | 1540 | -18.5 |
| C6 | 40 | 1050.06 | 2.61e-03 | -20 -32 -43 -54 -65 | 1306 | -59.2 |
| C6 | 80 | 1050.06 | 2.84e-03 | -16 -25 -35 -44 -54 | 1538 | -35.3 |
| C6 | 120 | 1050.06 | 2.84e-03 | -13 -22 -30 -39 -48 | 1729 | -17.4 |
| C7 | 40 | 2110.00 | nan | -30 -53 -73 | 2228 | -63.0 |
| C7 | 80 | 2110.00 | nan | -24 -43 -61 | 2364 | -34.5 |
| C7 | 120 | 2110.00 | nan | -21 -38 -55 | 2486 | -14.6 |
| C8 | 40 | 4246.99 | nan | -36 | 4391 | -67.6 |
| C8 | 80 | 4246.92 | nan | -30 | 4527 | -34.9 |
| C8 | 120 | 4246.92 | nan | -27 | 4634 | -12.8 |

## Acceptance checks: 62/65 pass

| check | measured | target | result |
|---|---|---|---|
| A0 prompt T60, partials 1-4 (s) | 26.8 | 26.4 (x/1.35) | PASS |
| A0 aftersound T60, partials 1-4 (s) | 117.2 | 113 (x/1.35) | PASS |
| A0 aftersound knee of the fundamental (dB re peak) | -17 | -29 +-4 | FAIL |
| A0 early spectral slope, mf, radiated (dB/oct) | 2.8 | -1 +-5 | PASS |
| A0 tuning re A4 (cents) | -15.8 | -16 +-4 | PASS |
| C1 prompt T60, partials 1-4 (s) | 26.1 | 19.8 (x/1.35) | PASS |
| C1 aftersound T60, partials 1-4 (s) | 105.2 | 110 (x/1.35) | PASS |
| C1 early spectral slope, mf, radiated (dB/oct) | 0.8 | 1.5 +-5 | PASS |
| C1 tuning re A4 (cents) | -14.8 | -15 +-4 | PASS |
| C2 prompt T60, partials 1-4 (s) | 24.6 | 26.7 (x/1.35) | PASS |
| C2 aftersound T60, partials 1-4 (s) | 79.8 | 93 (x/1.35) | PASS |
| C2 aftersound knee of the fundamental (dB re peak) | -22 | -28 +-4 | FAIL |
| C2 early spectral slope, mf, radiated (dB/oct) | -4.9 | -3.5 +-5 | PASS |
| C2 tuning re A4 (cents) | -3.8 | -4 +-4 | PASS |
| C3 prompt T60, partials 1-4 (s) | 13.3 | 15.7 (x/1.35) | PASS |
| C3 aftersound T60, partials 1-4 (s) | 39.8 | 51 (x/1.35) | PASS |
| C3 aftersound knee of the fundamental (dB re peak) | -21 | -27 +-4 | FAIL |
| C3 early spectral slope, mf, radiated (dB/oct) | -8.9 | -4.7 +-5 | PASS |
| C3 tuning re A4 (cents) | +0.1 | +0 +-4 | PASS |
| C4 prompt T60, partials 1-4 (s) | 8.5 | 10.3 (x/1.35) | PASS |
| C4 aftersound T60, partials 1-4 (s) | 22.1 | 26 (x/1.35) | PASS |
| C4 aftersound knee of the fundamental (dB re peak) | -24 | -25 +-4 | PASS |
| C4 early spectral slope, mf, radiated (dB/oct) | -13.9 | -13.7 +-5 | PASS |
| C4 tuning re A4 (cents) | -0.9 | -1 +-4 | PASS |
| A4 prompt T60, partials 1-4 (s) | 6.4 | 5.8 (x/1.35) | PASS |
| A4 aftersound T60, partials 1-4 (s) | 16.9 | 18 (x/1.35) | PASS |
| A4 aftersound knee of the fundamental (dB re peak) | -22 | -24 +-4 | PASS |
| A4 early spectral slope, mf, radiated (dB/oct) | -17.9 | -17.3 +-5 | PASS |
| A4 tuning re A4 (cents) | +0.0 | +0 +-4 | PASS |
| C5 prompt T60, partials 1-4 (s) | 5.3 | 5.7 (x/1.35) | PASS |
| C5 aftersound T60, partials 1-4 (s) | 13.9 | 14 (x/1.35) | PASS |
| C5 aftersound knee of the fundamental (dB re peak) | -21 | -24 +-4 | PASS |
| C5 early spectral slope, mf, radiated (dB/oct) | -21.7 | -25.8 +-5 | PASS |
| C5 tuning re A4 (cents) | -0.1 | +0 +-4 | PASS |
| C6 prompt T60, partials 1-4 (s) | 5.1 | 5.0 (x/1.35) | PASS |
| C6 aftersound T60, partials 1-4 (s) | 9.1 | 9.1 (x/1.35) | PASS |
| C6 aftersound knee of the fundamental (dB re peak) | -16 | -16 +-4 | PASS |
| C6 early spectral slope, mf, radiated (dB/oct) | -26.4 | -26.8 +-5 | PASS |
| C6 tuning re A4 (cents) | +6.1 | +6 +-4 | PASS |
| C7 aftersound T60, partials 1-4 (s) | 6.0 | 4.6 (x/1.35) | PASS |
| C7 aftersound knee of the fundamental (dB re peak) | -13 | -14 +-4 | PASS |
| C7 early spectral slope, mf, radiated (dB/oct) | -29.8 | -32.6 +-5 | PASS |
| C7 tuning re A4 (cents) | +14.1 | +14 +-4 | PASS |
| C4 effective B (Rigaud et al.) | 0.000327 | 3.30e-04 (x1.5) | PASS |
| C6 effective B (Rigaud et al.) | 0.00284 | 2.85e-03 (x1.5) | PASS |
| C2 effective B (Iowa, x2: bass is piano-specific) | 0.000175 | 1.19e-04 (x2) | PASS |
| C4 effective B (Iowa, x2: bass is piano-specific) | 0.00033 | 3.04e-04 (x2) | PASS |
| C4 centroid vel120 / vel40 | 1.76 | 1.1-2 | PASS |
| C2 brightness rises with velocity (centroid Hz, vel 30/60/90/120) | 355 560 718 845 | increasing | PASS |
| C4 brightness rises with velocity (centroid Hz, vel 30/60/90/120) | 478 673 833 965 | increasing | PASS |
| C6 brightness rises with velocity (centroid Hz, vel 30/60/90/120) | 1240 1428 1589 1729 | increasing | PASS |
| C7 brightness rises with velocity (centroid Hz, vel 30/60/90/120) | 2192 2297 2395 2486 | increasing | PASS |
| A0 radiated fundamental below strongest partial (dB) | 54 | >= 10 | PASS |
| C2 radiated fundamental below strongest partial (dB) | 19 | >= 10 | PASS |
| A0 release to -60 dB (s) | 0.63 | >= 0.5 | PASS |
| A4 release to -60 dB (s) | 0.25 | 0.2-0.4 | PASS |
| C8 released vs held (dB, undamped) | +0.0 | 0 +-1 | PASS |
| pedal halo, symp re strings (dB) [bank disabled in this model] | -29 | -40..-25 | PASS |
| halo with vs without pedal (dB) [bank disabled in this model] | 18 | >= 10 | PASS |
| knock re tone, first 60 ms, vel 120 (dB) | -26 | -25 +-3 | PASS |
| knock re tone, first 60 ms, vel 25 (dB) | -13 | -12 +-3 | PASS |
| damper noise re released note (dB) | -41 | -45..-35 | PASS |
| pedal-press noise re mf note (dB) | -35 | -35 +-5 | PASS |
| noise energy > 6 ms before the hammer strikes (fraction) | 0.000 | < 0.01 | PASS |
| ff: key-bottom thump before the hammer (fraction of noise energy) | 0.182 | > 0.005 | PASS |
