## Isolated notes (dry strings)

| key | vel | f1 Hz | B_eff | partials 2..6 dB re f1 | T60 prompt s (EDC) | T60 after s (model) | centroid Hz | peak dBFS |
|---|---|---|---|---|---|---|---|---|
| A0 | 40 | 27.05 | 3.00e-04 | +5 +7 +7 +5 +1 | 26.7 | 40.6 | 240 | -50.8 |
| A0 | 80 | 27.05 | 2.91e-04 | +5 +7 +8 +6 +3 | 26.7 | 40.6 | 284 | -36.4 |
| A0 | 120 | 27.05 | 2.96e-04 | +5 +8 +8 +7 +4 | 26.7 | 40.6 | 314 | -22.9 |
| C2 | 40 | 64.81 | 1.37e-04 | +4 +3 +0 -4 -9 | 16.5 | 27.5 | 367 | -51.6 |
| C2 | 80 | 64.81 | 1.37e-04 | +5 +6 +4 +0 -5 | 16.5 | 27.5 | 417 | -36.6 |
| C2 | 120 | 64.81 | 1.37e-04 | +5 +6 +5 +2 -3 | 16.5 | 27.5 | 452 | -22.5 |
| C4 | 40 | 261.35 | 3.81e-04 | -4 -10 -15 -20 -25 | 7.3 | 26.8 | 851 | -50.1 |
| C4 | 80 | 261.35 | 3.81e-04 | -2 -7 -12 -17 -22 | 7.3 | 26.8 | 916 | -34.1 |
| C4 | 120 | 261.35 | 3.81e-04 | -1 -5 -10 -15 -20 | 7.3 | 26.8 | 963 | -19.6 |
| A4 | 40 | 440.19 | 8.51e-04 | -7 -14 -19 -24 -29 | 5.4 | 21.4 | 1219 | -50.2 |
| A4 | 80 | 440.19 | 8.51e-04 | -6 -12 -17 -22 -28 | 5.5 | 21.4 | 1266 | -33.8 |
| A4 | 120 | 440.19 | 8.51e-04 | -5 -11 -16 -21 -27 | 5.5 | 21.4 | 1310 | -19.3 |
| C6 | 40 | 1052.44 | 3.46e-03 | -10 -17 -26 -35 -46 | 2.6 | 10.1 | 2422 | -48.0 |
| C6 | 80 | 1052.44 | 3.46e-03 | -10 -17 -26 -35 -45 | 2.6 | 10.1 | 2440 | -31.4 |
| C6 | 120 | 1052.44 | 3.46e-03 | -9 -17 -25 -35 -45 | 2.6 | 10.1 | 2461 | -16.3 |
| C7 | 40 | 2122.95 | nan | -14 -30 -47 | 1.7 | 5.6 | 3575 | -46.3 |
| C7 | 80 | 2122.95 | nan | -14 -30 -47 | 1.7 | 5.6 | 3581 | -30.0 |
| C7 | 120 | 2122.95 | nan | -14 -30 -47 | 1.7 | 5.6 | 3587 | -14.7 |
| C8 | 40 | 4318.97 | nan | -27 | 0.7 | 2.0 | 5330 | -44.6 |
| C8 | 80 | 4318.91 | nan | -27 | 0.7 | 2.0 | 5331 | -27.7 |
| C8 | 120 | 4318.91 | nan | -27 | 0.7 | 2.0 | 5332 | -12.5 |

## Acceptance checks: 33/33 pass

| check | measured | target | result |
|---|---|---|---|
| A0 prompt T60 (s) | 26.7 | 25-40 | PASS |
| C2 prompt T60 (s) | 16.5 | 15-20 | PASS |
| C2 aftersound T60, model (s) | 27.5 | 20-30 | PASS |
| C4 prompt T60 (s) | 7.3 | 6-8 | PASS |
| C4 aftersound T60, model (s) | 26.8 | 20-35 | PASS |
| A4 prompt T60 (s) | 5.46 | 5-6 | PASS |
| A4 aftersound T60, model (s) | 21.4 | 15-30 | PASS |
| C6 prompt T60 (s) | 2.62 | 2.5-3.5 | PASS |
| C6 aftersound T60, model (s) | 10.1 | 8-15 | PASS |
| C7 prompt T60 (s) | 1.65 | 1.5-2.0 | PASS |
| C7 aftersound T60, model (s) | 5.57 | 3-6 | PASS |
| C8 prompt T60 (s) | 0.748 | 0.7-1.2 | PASS |
| C4 effective B | 0.000381 | 3.8e-04 (x1.5) | PASS |
| C6 effective B | 0.00346 | 3.5e-03 (x1.5) | PASS |
| C4 mf partials 2-6 re p1 (dB) | -2 -7 -12 -17 -22 | -30..+3 | PASS |
| C4 mf partial 10 re p1 (dB) | -44 | <= -30 | PASS |
| C4 centroid vel120 / vel40 | 1.13 | 1.1-2 | PASS |
| C2 brightness rises with velocity (centroid Hz, vel 30/60/90/120) | 349 395 427 452 | increasing | PASS |
| C4 brightness rises with velocity (centroid Hz, vel 30/60/90/120) | 830 887 929 963 | increasing | PASS |
| C6 brightness rises with velocity (centroid Hz, vel 30/60/90/120) | 2418 2430 2445 2461 | increasing | PASS |
| C7 brightness rises with velocity (centroid Hz, vel 30/60/90/120) | 3574 3578 3582 3587 | increasing | PASS |
| A0 radiated fundamental below strongest partial (dB) | 53 | >= 10 | PASS |
| C2 radiated fundamental below strongest partial (dB) | 16 | >= 10 | PASS |
| A0 release to -60 dB (s) | 0.71 | >= 0.5 | PASS |
| A4 release to -60 dB (s) | 0.25 | 0.2-0.4 | PASS |
| C8 released vs held (dB, undamped) | +0.0 | 0 +-1 | PASS |
| pedal halo, symp re strings (dB) [bank disabled in this model] | -32 | -40..-25 | PASS |
| halo with vs without pedal (dB) [bank disabled in this model] | 11 | >= 10 | PASS |
| knock re tone, first 60 ms, vel 120 (dB) | -25 | -25 +-3 | PASS |
| knock re tone, first 60 ms, vel 25 (dB) | -12 | -12 +-3 | PASS |
| damper noise re released note (dB) | -40 | -45..-35 | PASS |
| pedal-press noise re mf note (dB) | -35 | -35 +-5 | PASS |
| noise energy before the hammer strikes (fraction) | 0.000 | < 0.01 | PASS |
