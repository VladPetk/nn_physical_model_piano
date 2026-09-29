## Isolated notes (dry strings)

| key | vel | f1 Hz | B_eff | partials 2..6 dB re f1 | T60 prompt s (EDC) | T60 after s (model) | centroid Hz | peak dBFS |
|---|---|---|---|---|---|---|---|---|
| A0 | 40 | 27.05 | 2.85e-04 | +5 +7 +6 +5 +1 | 23.2 | 34.5 | 191 | -52.2 |
| A0 | 80 | 27.05 | 2.91e-04 | +5 +7 +7 +6 +2 | 23.2 | 34.5 | 234 | -37.4 |
| A0 | 120 | 27.05 | 2.96e-04 | +5 +7 +7 +6 +3 | 23.1 | 34.5 | 257 | -23.8 |
| C2 | 40 | 64.81 | 1.38e-04 | +4 +4 +1 -5 -16 | 14.4 | 27.5 | 297 | -52.8 |
| C2 | 80 | 64.81 | 1.38e-04 | +4 +5 +4 +0 -6 | 14.4 | 27.5 | 342 | -37.6 |
| C2 | 120 | 64.81 | 1.38e-04 | +5 +6 +5 +2 -2 | 14.4 | 27.5 | 371 | -23.6 |
| C4 | 40 | 261.35 | 3.81e-04 | -6 -14 -21 -22 -34 | 7.3 | 26.8 | 715 | -52.8 |
| C4 | 80 | 261.35 | 3.81e-04 | -1 -14 -15 -21 -26 | 7.3 | 26.8 | 772 | -36.5 |
| C4 | 120 | 261.35 | 3.81e-04 | +0 -7 -19 -16 -27 | 7.3 | 26.8 | 810 | -21.8 |
| A4 | 40 | 440.19 | 8.51e-04 | -15 -24 -25 -27 -32 | 5.4 | 21.4 | 997 | -53.4 |
| A4 | 80 | 440.19 | 8.51e-04 | -11 -15 -26 -27 -31 | 5.5 | 21.4 | 1079 | -36.6 |
| A4 | 120 | 440.19 | 8.51e-04 | -6 -17 -20 -24 -35 | 5.5 | 21.4 | 1114 | -21.6 |
| C6 | 40 | 1052.44 | 3.46e-03 | -9 -14 -29 -31 -48 | 2.6 | 10.1 | 2537 | -53.2 |
| C6 | 80 | 1052.44 | 3.46e-03 | -12 -21 -32 -44 -50 | 2.6 | 10.1 | 2070 | -33.3 |
| C6 | 120 | 1052.44 | 3.46e-03 | -13 -21 -30 -40 -49 | 2.6 | 10.1 | 2153 | -17.7 |
| C7 | 40 | 2122.37 | nan | -13 -29 -51 | 1.3 | 4.5 | 3334 | -46.4 |
| C7 | 80 | 2122.37 | nan | -6 -21 -41 | 1.3 | 4.5 | 4070 | -33.7 |
| C7 | 120 | 2122.37 | nan | -11 -25 -41 | 1.3 | 4.5 | 3814 | -17.4 |
| C8 | 40 | 4317.82 | nan | -24 | 0.6 | 1.6 | 5101 | -57.9 |
| C8 | 80 | 4317.37 | nan | -17 | 0.6 | 1.6 | 5818 | -43.4 |
| C8 | 120 | 4317.50 | nan | -27 | 0.6 | 1.6 | 5098 | -25.7 |

## Acceptance checks: 23/23 pass

| check | measured | target | result |
|---|---|---|---|
| A0 prompt T60 (s) | 23.2 | 20-40 | PASS |
| C4 prompt T60 (s) | 7.3 | 6-8 | PASS |
| C4 aftersound T60, model (s) | 26.8 | 20-35 | PASS |
| A4 prompt T60 (s) | 5.46 | 5-6.5 | PASS |
| A4 aftersound T60, model (s) | 21.4 | 15-30 | PASS |
| C6 prompt T60 (s) | 2.62 | 2.5-3.5 | PASS |
| C6 aftersound T60, model (s) | 10.1 | 8-15 | PASS |
| C7 prompt T60 (s) | 1.3 | 1.2-2.0 | PASS |
| C7 aftersound T60, model (s) | 4.48 | 3-6 | PASS |
| C4 effective B | 0.000381 | 3.8e-04 (x1.5) | PASS |
| C6 effective B | 0.00346 | 3.5e-03 (x1.5) | PASS |
| C4 mf partials 2-6 re p1 (dB) | -1 -14 -15 -21 -26 | -30..+3 | PASS |
| C4 mf partial 10 re p1 (dB) | -51 | <= -30 | PASS |
| C4 centroid vel120 / vel40 | 1.13 | 1.1-2 | PASS |
| A0 radiated fundamental below strongest partial (dB) | 12 | >= 10 | PASS |
| C2 radiated fundamental below strongest partial (dB) | 15 | >= 10 | PASS |
| A0 release to -60 dB (s) | 0.71 | >= 0.5 | PASS |
| A4 release to -60 dB (s) | 0.25 | 0.2-0.4 | PASS |
| C8 released vs held (dB, undamped) | +0.0 | 0 +-1 | PASS |
| pedal halo, symp re strings (dB) | -27 | -40..-25 | PASS |
| halo with vs without pedal (dB) | 11 | >= 10 | PASS |
| knock re tone, first 60 ms, vel 120 (dB) | -33 | -25 +-8 | PASS |
| knock re tone, first 60 ms, vel 25 (dB) | -8 | -12 +-8 | PASS |
