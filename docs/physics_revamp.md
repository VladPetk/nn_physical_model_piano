# Physics revamp: the design (2026-10-03, branch `physics_revamp`)

The owner's question: is the physics complete and right, and should its parts interact? A review of the code, three
literature reviews (two blind, one of the classics; notes in `scratch/lit_review/`, the owner's PDFs in `lit/`, not in
git) and the measured gaps of phases 3-6 led to seven builds, all agreed on 2026-10-03. This is their design: the
equations, what each replaces, how it starts, how it is checked, and what it costs. Labels as elsewhere: **read** (a
paper or a log), **mine** (reasoning or estimate).

## 0. Why: what the model cannot do now

The model renders each note as independent damped sinusoids (`physics.PianoPhysics.modes`): per partial an in-phase
"prompt" mode and two "aftersound" modes with independent amplitudes and a random unison detuning; one bridge force per
note, panned by a gain and sent through one body filter per microphone. Against the literature (read):

- **The unison is one coupled system** (Weinreich 1977, JASA 62). Two or three slightly mistuned strings on one bridge
  of complex admittance Y = G + iB have normal modes whose frequencies and decays follow from the mistuning, G and B.
  Mistuning larger than the bridge coupling: beats, both modes at the single-string decay rate. Smaller: no beat, a
  double decay whose depth ((eta - nu)^2 / (eta + nu)^2, nu = sqrt(eta^2 - eps^2)) changes steeply with a 1-cent change.
  The sum of the decay rates is always N times the single string's. How evenly the hammer meets the strings sets the
  aftersound; una corda is its extreme. Mid-range: Re and Im of the coupling are each ~1 /s, |Y| ~ 1e-3 s/kg, string
  impedance ~2 kg/s. A single string alone shows a double decay too: its vertical polarisation drains fast through
  the bridge, its horizontal one slowly. Woodhouse (2022, JASA 150): the string's loss is set by Re Y at its own
  attachment point; on a measured baby grand the unison's bridge loss exceeds the air damping by ~20 dB above ~160 Hz.
- **Two radiation paths.** The vertical and horizontal motions radiate through different "antennas": moving the
  microphone changes their relative phase (Weinreich); the vertical's efficiency is ~10 dB higher. The bridge's
  in-plane mobility exceeds its normal mobility by >= 10 dB above 1 kHz, and in C5-C6 the in-plane signal carried "an
  essential ingredient" of the tone that no shaping of the normal signal reproduced (Conklin 1996 II, his listening).
  Phantom partials reach the bridge in-plane, 34 dB above their normal component (Conklin 1999). The model has one path.
  The critic's easiest tell is the side signal L - R (AUC 1.00, `docs/composite_score.md` 12).
- **The admittance per key** is smooth above ~1 kHz (+-5 dB) and modal below; the strings halve its range below 200 Hz
  (Conklin II, 2.74 m grand). A board mode near a partial switches it between spring-like and mass-like loading and
  shifts its frequency (a C1 fundamental 8 cents off the B law, Conklin 1996 III). Higher mobility: louder and shorter
  (Conklin II; lore with a mechanism). The model's decay uses one conductance curve per year for every key, real part
  only, and its per-key colouration is an independent table.
- **Strike-position nulls are filled**: 15-20 dB deep in measured spectra (Hall & Askenfelt 1988; ~28 dB, Conklin I),
  not the model's zeros of sin(n pi x0).
- **Longitudinal vibration.** Phantoms are tension modulation: the bridge's longitudinal force carries every sum
  (and difference) of transverse partial frequencies, square-law in level, decaying at the sum of the parents' rates
  (Conklin 1999, monochord and piano; Bank & Sujbert 2005). Between partials 10 and 11 the pairs 1+10 ... 5+6 cluster
  within ~11 Hz below partial 11 and beat with it. The free longitudinal mode (LM1 = 11-20 f1, set by the string's
  stress; ~25-30 dB under in an E1, decay ~0.15 s) is "the most prominent pitch identifier" of some bass tones at onset
  (Conklin III). The only controlled listening test in the reviews: without longitudinal components fortissimo bass
  tones sound "synthetic" (Bank & Lehtonen 2010, 8 listeners, essential below ~A3). The model has 2f_j and f_j + f_j+1
  only, quadratic in level, and no free longitudinal mode.
- **The treble knock has pitched parts**: the hammer shank (~200-300 Hz, heard as a pitched knock in the treble) and the
  capo bar and plate (1-2 kHz modes in treble onsets) (Askenfelt & Jansson, Conklin I-II). The model's knock is smooth
  band noise; the bench found the 1 kHz knock 10 dB weak in the middle and the treble's long low ringing missing.
- **Pitch glide**: the tension rises with amplitude at onset (Conklin II); measured here at 1.5-3 cents at ff (11.3).

Our own gaps that these explain or touch (read, `docs/tone_measures.md` 15-17, review 6): the upper partials ring too
long after 1 s; the late decay varies half as much across strikes as the piano's; the mistuning never moved from its
random draw; phantoms weak in R2; the side signal.

## 1. The coupled unison core (build 1)

Per key k and partial n, the state is the complex amplitudes of every string i = 1..N of the unison in both
polarisations, x = (v_1..v_N, h_1..h_N), in a frame rotating at the partial's nominal frequency f_n:

    dx/dt = i Omega x,   Omega = diag(d_i^V, d_i^H) + zeta_V J_VV + zeta_H J_HH + zeta_X (J_VH + J_HV) + i a_n I

- J_VV is the all-ones matrix on the vertical block (the bridge moves under every string's vertical force alike), J_HH
  the same on the horizontal block, J_VH the cross block (the bridge's cross-admittance couples the polarisations).
- zeta = i c_k Y(f_n): a string's own frequency shift from the bridge, c_k = 2 f_1 Z_0 (Weinreich's eq. 8 for mode n
  gives delta omega_n = 2 i f_1 Z_0 Y). Im zeta = c G is the single-string bridge loss, Re zeta = -c B its frequency
  shift. Vertical and horizontal: Weinreich found the resistive part much larger vertically and the reactive part
  about isotropic, so G_H = r_H G_V (per key, prior r_H 0.1) and B_H = B_V; zeta_X from a small cross-admittance
  (per key, prior 0.05 |Y_V|).
- d_i = 2 pi (f_n,i - f_n): string i's own detuning, from its pitch offset c_i (cents, the unison mistuning, per key)
  and its inharmonicity offset (the strings of a unison differ slightly in B): d_i = 2 pi f_n (c_i ln2/1200 +
  B n^2 b_i / (2 (1 + B n^2))). Horizontal: d_i^H = d_i^V + d_H (prior 0; Weinreich: < 0.1 cent).
- a_n = b1 + b3 f^p: internal and air losses per string, as now.

Eigen-decomposition Omega = sum_m lambda_m r_m l_m^T (per condition, key and partial; float64) gives each mode's
frequency f_n + Re lambda_m / 2 pi and amplitude decay Im lambda_m. The strings start at rest and are struck: x(0) =
-i e (the -i makes the real signal a sine, as the oscillators now), with the excitation per string

    e_i^V = base_n sin(n pi x0_i) w_i,    e_i^H = s_H e_i^V,

base_n the note's present partial amplitude without the comb (level, hammer spectrum, partial gain, colouration),
x0_i = x0 (1 + u_i) a strike point per string (per key; prior spread ~1 %: a few mm, so the comb's nulls are filled
by the strings' differences), w_i the evenness of the blow (per key, plus a draw per strike: 2-3 numbers instead of
the present 96 x 2 per-partial aftersound draws), s_H the horizontal share (per key; prior -20 dB). Mode m then
radiates on the vertical bus with complex amplitude A_mV = (1_V^T r_m)(l_m^T x(0)) and on the horizontal bus with
A_mH = (1_H^T r_m)(l_m^T x(0)). Monochords: N = 1 (2 modes); bichords 4; trichords 6.

**Replaces:** `prior_prompt_ratio` / `raw_prompt` (the bridge loss: now N Im zeta for the in-phase mode), `raw_after`
and `prior_log_after` (the aftersound levels: now from the mistuning, evenness and s_H), `raw_unison` (the random
detuning: now c_i, started from measurement, section 9), `strike_after` and `strike_partial_decay` (the per-partial
random draws: now the blow's evenness w_i per strike). Kept: B, tuning, b1, b3, p, the hammer, levels, dampers
(alpha_damp acts on every mode of a partial), re-strike.

**Checks** (`tests/`): two strings, resistive coupling: the beat regime (both decays at the single-string rate, beat
2 sqrt(eps^2 - eta^2)) and the double-decay regime (aftersound fraction (eta - nu)^2/(eta + nu)^2), Weinreich's eqs.
20-22; the trace rule (decay rates sum to N x the single-string rate); zero coupling reduces to independent strings;
una corda (one string struck) gives the aftersound as large as the attack; gradients finite at and near the
exceptional point eps = eta (where two eigenvalues coalesce and the amplitudes cancel: computed in float64 with a
floor on the eigenvalue gap; a test sweeps eps through eta).

## 2. Two radiation paths and the image per key (build 2)

The string bank renders two buses per note: vertical (the modes' A_mV; the knock impulse and the noises join it) and
in-plane (A_mH; the longitudinal force of build 4 joins it). The body becomes one response per (bus, microphone):
`room.body [C, 2, ch, L]`, the ring-up and the Q cap per response as now. Per note, per bus and per microphone: a gain
and a delay (the source's place on the board seen from a spaced pair: |delay| <= 1.5 ms, smooth over keys), applied to
the note's bus signals after the bank (an integer shift and a 4-tap fractional delay per note: P x L x 16 operations,
small next to the bank's P x L x Q). Replaces the gain-only `pan_gains`.

Start: the in-plane responses copy the vertical ones (so nothing moves at first), delays 0, the vertical gains the
present pan. Check: the per-partial stereo measure (section 9, M5) on recordings and renders.

## 3. One admittance per key (build 3)

    Y_k(f) = P_k [ Y_c(f) + sum_m phi_m(k)^2 i omega g_m / (omega_m^2 - omega^2 + i omega omega_m / Q_m) ]

- Y_c(f): the year's smooth admittance: its real part the present conductance knots (`raw_bridge_g`, 1/6 octave), its
  imaginary part a mass-like term (smooth, a few knots).
- The modal sum: M ~ 16 board modes below ~1 kHz per year (frequency, Q, gain), each with a shape over the keyboard
  phi_m(k) (knots every 4 keys): the position dependence, and the spring/mass alternation that shifts partial
  frequencies near a mode. Passive by construction (g_m > 0). Priors: Conklin's 2.74 m grand (49, 67, 89, 113, 184,
  306 Hz in the rim; the first near 60 Hz strung), Q 20-50.
- P_k: a smooth level per key with a step allowed at the bass break (the bass bridge's mobility is lower; the key of
  the break is per year, from the data's own discontinuity or the instrument's scale).
- The same Y_k sets the loss and frequency shift (build 1: zeta = i c_k Y_k) and the radiation: the vertical bus is
  scaled by |Y_k(f_n) / Ybar(f_n)|^gamma (Ybar the mean over keys; gamma in [0, 1.5], start 0, so the start is the
  present model). The free colouration table stays, its bound halved to +-4 dB, for what the admittance does not hold.

Start: phi_m = 0 (no modal detail), Y_c's real part from the present conductance and bridge factor; the imaginary part
0. Check: in the recordings, partials that are louder than the smooth hammer model predicts decay faster (M6).

## 4. Longitudinal vibration: phantoms and the free longitudinal modes (build 4)

Not as pairs of oscillators: as the physics computes them. The longitudinal force at the bridge follows the squared
slope of the string there, y_x(L, t) = sum_n (n pi / L)(-1)^n y_n(t) (Bank & Sujbert 2005). Per note the bank
accumulates two extra sums per polarisation over its partials below sr/4 (so the products stay below Nyquist):

    S_odd(t) = sum_{n odd} n y_n(t),    S_even(t) = sum_{n even} n y_n(t)

and the force families are F_even = S_odd^2 + S_even^2 (pairs with m + n even: 2f_j and the sums, differences of equal
parity) and F_odd = 2 S_odd S_even (m + n odd: f_j + f_j+1 ...). Each drives (a) a direct quasi-static path (the
uniform-tension part; high-passed) and (b) the free longitudinal modes LM_j = j LM1 (j = 1..4; odd j from F_odd, even
j from F_even), resonators with decay ~0.15 s, per note, run as a recurrence on the note's signals. Out to the in-plane
bus (a small share to the vertical). Every pair, its cluster and its beating with the transverse partials, the
square law, the decay at the sum of the parents' rates, the velocity dependence faster than quadratic (via the hammer
spectrum) and the free ringing of LM1 then follow without a per-pair parameter.

Scale per key: kappa_k (relative to the note's mf level, as the present `ref_db`), LM1 per key (prior from the scale:
11-20 f1, steps at the bass break and the wound-to-plain change; measured, M2). Approximation: the strings of a unison
are summed before squaring (true while they move in phase, at the onset where phantoms are strongest; after it the sum
underestimates them). Replaces `_phantoms` (2 x 16 x 2 oscillators per note). Start: kappa from the present phantom
level; LM resonators at gain 0. Check: phantom clusters and LM1 on loud bass notes (M3, M2), N9/N12 of the bench.

## 5. Structural resonances in the knock (build 5)

Per note, 3-4 damped sinusoids at the onset on the vertical bus: the shank (200-300 Hz, per register) and the capo
bar and plate (1-2 kHz, treble), each a frequency, decay and level per register with the knock's velocity law. They
join the bank as a third oscillator set (no dampers). Start at -60 dB. Check: the bench's knock measures (N6, E4).

## 6. Pitch glide (build 6)

f(t) = f (1 + e_note exp(-beta tau)): the phase gains 2 pi f e_note (1 - exp(-beta tau)) / beta, one exponential per
note and sample shared by its oscillators, one multiply-add per oscillator. e_note = e_k 10^((L_note - L_ref)/10)
(square-law in amplitude), beta per key (prior: twice the in-phase decay of partial 1). Start: e_k at 0. Check: N3.

## 7. Interaction tables (build 7)

For the pairs of factors the physics above does not join, identity-initialised smooth tables in log amplitude and log
decay over (key knots x velocity knots x log-partial knots), bounded, constrained to zero mean over the axis they
add (so they hold the interaction, not a main effect another parameter already owns): decay x velocity, aftersound x
velocity, re-strike x velocity and partial, damper x velocity, decay x pedal lift. Before any run, the coherence
probe (section 10) reads which ones the loss consistently pushes.

## 8. Engine and speed

Measured today (read, `docs/speed.md`): 1.0 s per training step, ~1.1e9 oscillator-samples per batch of 8 x 3 s; the
string bank is roughly a third of the step (render + backward). Per oscillator and sample the fused kernel does one
`__sincosf`, one `__expf` and a multiply-add; the phase is float64 once per tile.

What the builds add, and how it is kept down (mine; each is measured when built):

- **Modes.** Trichord partials go from 3 oscillators to 6 (bichords 2 -> 4, monochords 2 -> 2). Pruned per batch
  after the eigen-decomposition: a mode is dropped when its energy (|A|^2 / 2 Im lambda) is under -50 dB of its
  partial's; the active modes are compacted per note before the launch (the kernel already renders a prefix per
  note). Expected: the horizontal modes of the treble and the near-degenerate pairs of a well-tuned unison drop.
- **Buses.** One oscillator feeds both buses: sin and cos of its phase are computed once, each bus adds
  A_s sin + A_c cos (two multiply-adds). The two longitudinal sums add one each, for partials below sr/4 only. Per
  oscillator-sample: one sincos, one exp, ~7 multiply-adds instead of 1: the exp and sincos dominate (mine: +30-50 %
  per oscillator-sample, to measure).
- **Phantoms** stop being oscillators: 64 per note today (2 series x 16 x 2 modes) become two accumulations and, per
  note, two squares and 4 resonators on [P, L] signals. A net saving in the bass.
- **The eigen-decomposition** is per condition, key and partial (88 x 96 small matrices per batch), not per note; per
  note only the projection of the excitation (2N x 2N per partial).
- **Glide**: one exp per note and sample, shared; **knock resonances**: 3-4 oscillators per note.
- **Stereo**: after the bank, per note: 2 buses x 2 microphones x (shift + 4 taps).
- **Memory**: the bank's per-note output grows from [P, L] to [P, 4, L] (two buses, two longitudinal sums): ~0.7 GB
  for a dense batch of 600 notes x 3 s (mine). The longitudinal sums can be squared and resonated in chunks.
- **Real time** (later, not this branch): the same modes as recursive complex oscillators (one complex multiply per
  oscillator and sample, no exp or sincos), the eigen-decomposition per key once per parameter change, the
  longitudinal path per sounding note; at ~300 active oscillators x 20 notes x 48 kHz that is ~3e8 complex multiplies
  per second (mine), within one CPU core with SIMD.

Targets: the string bank at most 1.5 x its present cost after pruning, the step at most +30 %; if the measured cost
is higher, the pruning threshold and the longitudinal partial ceiling are the first knobs.

## 9. Measurements for starting values and checks

Each validated as the project's measures are (synthetic notes with a known answer; known changes through the renderer):

- **M1 unison beat rates per key and partial** (the profile measure's fluctuation rate, on isolated notes of all years
  where one piano is clear): the starting mistuning c_i per key.
- **M2 LM1 per bass key**: a peak between transverse partials, not on their series, at the onset of loud notes.
- **M3 phantom clusters** on loud R1-R3 notes: peaks just below the partials, their levels re the parents', against
  velocity.
- **M4 the upper partials' frequencies**: offset of the recordings' peaks from the model's per partial number, above the
  mined range (the open question of the frequency review).
- **M5 stereo per partial**: inter-channel level and phase per partial and key, recordings vs renders.
- **M6 level against decay**: across keys and partials, does a partial louder than its key's smooth spectrum decay
  faster (one admittance)?

## 10. Order of work

1. The PyTorch reference: `oscbank` with buses, phases and glide; the coupled core in `physics`; the longitudinal path,
   knock resonances and buses in `synth`; per-bus bodies and the image in `room`. Config `string_model = "coupled"`
   (default `"modes"`: every old checkpoint loads and renders as before). Tests of section 1.
2. M1-M6 alongside.
3. The CUDA kernels for the new bank (`csrc/kernels.cu`), `tests/test_cuda.py` against the reference; speed measured.
4. Start from a fitted model: the new core's parameters fitted to the present checkpoint's per-partial envelopes
   (closed form on both sides, no audio: every key at mf, 0-3 s), then M1-M3 for what the old model never had; a
   render to listen to before any training.
5. The interaction tables and the coherence probe (16-32 batches, the gradient per table at identity, its agreement
   across batches and per cell).
6. One training run (~30 min stage 1, ~20 min with the residual), then the bench, N13, the critic's views and
   listening.

Starting checkpoint: `runs/loss_compare/B_comp/train/best.pt` (the composite, the latest). The long-window term
(review 2026-10-03: 75 % of the loss's weight is on notes younger than 0.5 s) is separate and comes after this.

## 11. Progress (2026-10-03)

Labels as elsewhere: **read** (measured, from a script's output), **mine** (my reasoning).

Done: steps 1, 3 and 4 of section 10 (the reference, the kernels, the fitted start). Not yet: M1–M6, the interaction
tables and the probe, the training run.

**The activity threshold** (config `activity_db`, now 70 dB by default; a checkpoint keeps the value in its config).
A note's oscillators are rendered up to the last one within this of the note's peak. B_comp rendered at 90 and 70 dB on
six held-out 2018 excerpts (soft to loud) and the demo: the difference is 62–67 dB under the render's level, its peak
−67 to −84 dBFS (read). Listened to by the owner: in the difference only hiss, and the faintest harmonic about twice
in 20 s. The two renders take the same time (the bank is not a render's bottleneck); in training the lower threshold
saves 0.04 s per step on the mode model (below).

**The kernels** (`pianonn/csrc/kernels.cu`, `pianonn/cuda_ext.py`: `BusBank`, `Longitudinal`).
- The bus bank: each oscillator's phase and decay computed once, then a sine and a cosine amplitude per bus (six:
  vertical and in-plane per microphone, the two longitudinal sums); the glide adds e(1 − exp(−βτ))/β to the phase
  of the modes, not of the knock's resonances. The radiation buses are summed per example inside the kernel
  ([B, 4, L]), so only the two longitudinal sums are kept per note ([P, 2, L]); the vertical force per key for the
  sympathetic bank likewise. Backward as the string bank's: per sample (note scalars, damper rows, curves, glide)
  and per oscillator (amplitudes, decay, frequency).
- The longitudinal force: the square of the slope, the DC blocker and the free longitudinal modes per note. A first
  version (one thread per note, sequential) needed 37 ms forward and 70 ms backward on 1,100 notes × 3 s (read): too
  few threads. Now time is cut into 512-sample segments, each run from a zero state; one pass per note carries the
  true state across segment ends and a third adds its ring. The backward does the same with the adjoint recurrences
  (backwards in time) and forward-mode tangents w = dz/da for the decay and frequencies; tiles of 32 samples staged in
  shared memory keep loads coalesced. 5.2 ms forward, 15.6 ms backward (read).
- Checks (`tests/test_cuda.py`, 3 new tests; read): against the PyTorch reference with every new parameter moved
  off its start and the residual's curves randomised, audio within −111 to −117 dB of its peak, every gradient within
  1e-5 relative except the free longitudinal modes' frequency (0.3 %: that gradient sums large terms of both signs, and
  the reference runs the recurrence per chunk); the longitudinal kernel alone within 1e-4 and its gradients within
  2e-3.

**Speed** (read; `scratch/step_speed.py`: B_comp's checkpoint, 6 fixed batches of 8 training excerpts of 2018, 1,094
notes per batch, residual and per-strike variation on; second render without gradient + render + backward, the score
left out, ~0.17 s for every model):

| | second render | render | backward | step without score | peak memory |
|---|---|---|---|---|---|
| mode model, 90 dB | 0.159 s | 0.157 s | 0.249 s | 0.565 s | 11.6 GB |
| mode model, 70 dB | 0.151 s | 0.146 s | 0.233 s | 0.529 s | 11.6 GB |
| coupled, 70 dB | 0.215 s | 0.188 s | 0.334 s | 0.737 s | 14.6 GB |

With the score the coupled step is ~0.91 s against ~0.70 s: +30 %, the target of section 8. In a 30-step smoke run
(`train.py`, validation included) the coupled model's compute was 0.81–0.93 s per step (B_comp's run: 0.75 s mean,
at 90 dB). What it took besides the kernels: the normal-mode decomposition (all keys, per condition) is computed once
per render instead of three times (the residual's view, the notes, the sympathetic bank's keys:
`physics.shared_core`), and once for both renders of a training step (the energy score's second draw has the same
parameters). What is left of the extra cost (mine, from component timings): the decomposition (`torch.linalg.eig` on
8,448 small matrices: 32 ms, 48 ms with its backward), then the bank's six outputs. A batched solver for these 2N × 2N
symmetric matrices is the next lever; at inference the decomposition happens once per piece.

**The decomposition's backward.** Two failures in tests, both fixed in `coupled.normal_modes`: (1) `torch.linalg.eig`'s
backward checks that the loss ignores each eigenvector's phase; through `inv(R)` rounding tripped it. Ω is complex
symmetric, so the left eigenvectors are the right ones transposed, l_m = r_m^T / (r_m^T r_m): no inverse, and every
amplitude is exactly invariant to the eigenvectors' scale. (2) At the start the in-plane polarisation has the
vertical's detunings, so their bridge-silent combinations coincide (gaps of 1e-4 rad/s, read); the backward divides
by the gaps. `_Eig` is PyTorch's formula with 1/gap regularised at 1e-5 rad/s. At 1e-2 rad/s the distillation below
fitted worse (2.4 against 1.0 dB rms, read): those near-degenerate modes are what it moves.

**The fitted start** (`fit_init.distill_coupled`, run by `train.py` when a coupled model starts from a mode checkpoint,
`--distill-steps`, default 1000, ~1.5–2 min). The start from the mode model's parameters (`init_coupled`) keeps each
partial's level at the strike (within 0.5 dB) but not its decay (read, every key at four velocities): +8 dB at 0.5 s
and +18 dB at 2 s in C6–C8, +3–4 dB at 2 s in the middle, −14 dB at 5 s in the bass. The likely cause (mine): the
in-plane polarisation, 20 dB down with a tenth of the bridge loss, outlasts the treble's fast vertical decay, and the
bass lacks the mode model's free aftersound levels. Fitting the coupled parameters to each partial's energy envelope
(every key, four velocities, log time grid to 8 s, weighted by energy share) brings the error from 9.7 to 1.0–1.1 dB
rms; afterwards within ~1 dB in every register, velocity and time (read).

Then the composite (energy score, physics only, the same 16 validation excerpts of 2018; read):

| | composite |
|---|---|
| B_comp | 0.619 |
| B_comp without its phantom partials | 0.638 |
| coupled, distilled, longitudinal force at the phantoms' energy | 0.834 |
| coupled, distilled, without the longitudinal force | 0.657 |
| coupled, distilled, longitudinal force 20 dB under the phantoms (the start) | 0.651–0.674 |

The spread of the last row is between distillations: the envelopes leave the beat rates and the split between the
polarisations free. At the phantoms' energy the force costs most in the between-partials and onset terms (0.368 and
0.198 against 0.293 and 0.114 without it): it carries every sum and difference tone of every mode and a spike at the
attack, the phantoms 32 sum tones (mine). Its level against the phantoms is no constant: at u = 0.6 from 27 dB under
to 39 dB over them by key, and from u = 0.3 to 0.9 it grows ~19 dB more (read). So the start puts it 20 dB under
(`coupled.long_cal_db`, a buffer set per key; the trainable level, ±20 dB, reaches parity), and the measurement of
phantom clusters against velocity (M3) is what should set it.

**Listening**: `samples/coupled_start/` (local): six held-out excerpts and the demo, the recording, B_comp and the
coupled start, all with the residual; `<i>_ab.wav` plays them in turn. Levels within 2 dB of B_comp's.

## 12. Taking stock after the first run, and the plan (2026-10-03, evening)

**Where we are** (read, from `runs/physics_revamp/coupled_run/` and `eval/`). Built: sections 1–7 (the coupled
unison, the two buses and the image, the admittance, the longitudinal force, the knock's resonances, the glide, the
interaction tables), the kernels (stage 1 +40 % per step against the mode model, clean timing), the fitted start, the
probe. One 50-min run from the start (30 min physics alone, 20 min with B_comp's residual). On the run's own 32
validation excerpts the physics went 0.685 → 0.609 and the training pieces' 32 excerpts 0.753 → 0.719; B_comp's run
ended at 0.642 and 0.733 on the same excerpts (the first 32 of each set below are the same draws). On 64 excerpts per
set (`eval/eval.md`) the order flips on the held-out pieces: coupled 0.637 against B_comp 0.614 (worse by 0.023, almost
all of it in `between_pooled`), while on the training pieces coupled is better by 0.029. In both runs the residual
helps on the training pieces (−0.024, −0.035 in-run) and not on the held-out ones (+0.011, +0.008).

**What this does and does not show** (mine). The two readings of the same comparison disagree in sign: the composite
of a small set moves by a few hundredths with which excerpts are in it, and the held-out set is 13 pieces, so its
excerpts are far from independent. Neither "the coupled model over-fits" nor "it generalises better" is shown yet.
What is shown: (1) the residual gains on the training pieces' audio (the validation excerpts of training pieces are
largely audio the run trained on) what it does not gain elsewhere, in both runs: that pattern is consistent and is the
residual's, not the new physics'; (2) the interaction tables moved to their bounds within 30 min (`decay_vel`,
`damp_vel`, `spec_vel` up to 0.64–0.76 nats in the treble, rising steadily with key; read) while the probe had found
their gradient only weakly consistent (agreement +0.02–0.08): free capacity that the loss fills fast, and the first
suspect if the held-out gap is real; (3) some of the new parameters barely moved (the knock resonances' levels
0.03 dB, the glide, the longitudinal modes' decay, the admittance's shape): either the loss does not see them or
their learning rate is too low for their scale, which needs checking before more training.

**Plan** (in this order; each step says what decides the next):

1. *A comparison with error bars.* `scripts/paired_eval.py`: 256 excerpts each of the 2018 validation, test and
   training pieces, every model on the same excerpts, terms kept per excerpt and the pooled terms' cells too, a
   bootstrap over pieces for each model's score and each paired difference, and the held-out − training difference
   of differences (the over-fitting test). Models: B_comp, the coupled start, the coupled run, physics alone and with
   their residuals. Decides whether there is a held-out gap to explain at all, and how large a difference later runs
   can resolve.
2. *Is the new physics doing what it should?* (a) Parameter audit, start → end: what moved, what hit bounds, what
   never moved and why (gradient size against learning rate). (b) Isolated notes against recordings: the note bench
   and N13 on B_comp, the start and the run: two-stage decay, beats, phantoms (N9), knock (N6), attack (N5);
   where the coupled model is closer or further than B_comp. (c) The components one at a time in rendered notes:
   that each renders what it claims (beat rates from the detunings, the in-plane bus, the glide's pitch track, the
   knock resonances' spectrum, the longitudinal sum tones).
3. *Ablations on the paired set*: the trained coupled model with each new part switched off (the tables; the
   longitudinal force; the glide; the knock resonances; the in-plane bus; the image per bus), held-out and training
   pieces. Says which part earns its place, and which part carries any held-out/training split.
4. *The residual*: why it gains on training audio only (its inputs, its capacity, its budgets; whether it memorises
   passages), and what changes: fresh or inherited, its learning rate and budgets, early stopping on its held-out
   gain.
5. *Fixes* from 2–4 (e.g. tables bounded tighter or smoother, learning rates per group, the residual's
   regularisation), with tests.
6. *The training run*, once 1–5 are clear: stage 1 physics then the residual, early stopping on held-out, the
   paired evaluation, listening, the bench and N13 afterwards.

### 12.1 Step 1: the comparison with error bars (read)

`runs/physics_revamp/paired/` (`paired_main.md`): 256 excerpts each of the 2018 validation (13 pieces), test (10) and
training pieces (63), every model on the same excerpts, physics alone unless named; ± is the bootstrap standard error
over pieces, brackets 95 % intervals of the paired difference.

| | held-out (512, 23 pieces) | training pieces (256, 63) |
|---|---|---|
| B_comp | 0.587 ± 0.019 | 0.632 ± 0.018 |
| coupled start (distilled, never trained on audio) | 0.749 | 0.724 |
| coupled run | 0.611 ± 0.023 | 0.610 ± 0.017 |
| coupled − B_comp | +0.024 [−0.004, +0.039] | −0.022 [−0.048, −0.003] |
| start − B_comp | +0.161 | +0.092 |
| B_comp with its residual − B_comp | −0.012 | −0.020 |
| coupled with the inherited residual − coupled | −0.002 | −0.007 |

The held-out − training difference of (coupled − B_comp) is +0.047 [+0.014, +0.074] (validation +0.041, test +0.036
separately): the split is real by this test. But **the start already has it** (+0.070 [+0.009, +0.118]), before any
training on audio (it was fitted to B_comp's envelopes alone), and the run improved the held-out pieces *more* than
the training pieces (−0.137 against −0.114). So the run did not over-fit; the coupled structure differs from the mode
model in a way that costs more on these held-out pieces than on the training pieces, from the start. Where: the read-
by-read terms per excerpt show no split except the onset term (+0.010 [+0.002, +0.017], unchanged by loudness,
pedal, density, velocity or register as covariates); the rest is in the pooled terms (`between_pooled` +0.035,
`pooled_exposed`, `level`). Ruled out: the Schubert sessions (6.7 h of the training audio, none held out: within
the Recital sessions alone the split is the same), the level matching (both models get the same per-piece gains
within 0.07 dB), single pieces (no piece moves a difference by more than 0.035).

Consistent in both sets (mine, from the intervals): the coupled run is better than B_comp *between* the partials
(−0.006/−0.007) and worse *at* them (`partials` +0.005/+0.007); in total no better on held-out pieces. The
residual's gain is larger on the training pieces in both models, but the difference is not significant here
(B_comp's: +0.008 [−0.009, +0.028]).

The pooled terms read set-level spectra: e.g. the energy between the partials at 240–600 Hz, ≥ 0.15 s after an onset,
is 2–5 dB above the recordings on the training pieces and within ±1 dB on the held-out ones, for both models (the
held-out pieces' recordings have more of it, read from the cells). A change that lowers it helps one set and hurts the
other; per excerpt that error grows with the recording's loudness (−2.4 dB per +10 dB) and the pedal (−1 dB pedal
down), so the model's between-partials energy does not follow loudness and pedal as the recordings' does (mine:
the halo and noise floor against what scales with the strike).

### 12.2 Step 2: is the new physics doing what it should? (read unless marked mine)

**Parameters that could not move.** Five of the coupled strings' parameters are levels in dB but trained at the base
rate (Adam moves a parameter ~lr per step: ≤ 1.5 dB over the run): the knock's resonances (`kr_db`, at −60 dB),
the free longitudinal modes (`lm_gain_db`, −40 dB), the longitudinal level (`raw_long_db`), its vertical share
(`raw_long_v`) and the in-plane share (`raw_sh`). They moved 0.01–0.6 dB: builds 4 (the free modes) and 5 (the knock's
resonances) stayed off and the longitudinal level stayed at its calibration. Fixed: they are in `train.DB_PARAMS`, and
the unison's pitch offsets (`coupled.raw_cents`, `raw_dh`) in `CENTS_PARAMS`. The glide (`raw_glide`, ≤ 0.1 % × the
level factor) moved by nothing: its measured size (N3, 11.3: 1.5–3 cents at ff) is at or below pitch discrimination
and it stays as it is. The knock's level is already 0–5 dB loud in the bench (N6 below), so there is no measured gap
for the resonances to fill either; they stay at −60 dB unless the loss moves them.

**What did move.** The unison core (pitch offsets 0.32, evenness 0.31, polarisation shares 0.3–0.6 in raw units,
mean), the image per bus (`room.raw_pan_bus`, which inherits `room.raw_pan`'s dB rate: up to 4.5), the interaction
tables (to their bounds in the treble, 12 above).

**Isolated notes** (`runs/physics_revamp/notes/bench/`, the note bench, 459 notes): the coupled run is about as close
to the recordings as B_comp in every register (levels, spectra, attack, knock, sustain within ~1 dB of B_comp's
error), phantoms too loud in R6 (+7.0 dB against +3.8). N13 was stopped to free the GPU (rerun after the main run).

**Beats** (`scratch/beat_compare.py`: the bench's 213 notes that sound freely for 1.6 s; partials 1–6, the rms of the
dB envelope around a quadratic in time over 0.25–1.6 s; recordings against single-note renders at the same pitch,
velocity and pedal). Recordings fluctuate 1.8–3.6 dB by register; B_comp 0.7–2.3, the coupled run 0.6–2.3 (R1–R6):
both models beat about half as much as the piano, the coupled one no more than the mode model. Known answer: the
coupled run with every unison's pitch offsets × 2.5 and × 4 reads R3 0.55 → 1.05 → 1.80 dB, R4 1.57 → 2.08 → 2.46,
R5 1.80 → 2.53 → 3.18 (recordings 1.76, 2.34, 3.10): the measure sees detuning, and × 4 reaches the recordings
(which also hold leakage and the room, so that is an upper bound; mine).

Why the detunings are small: the distillation fitted energy envelopes summed over the modes, so it never constrained
the beat rates, and the run then *shrank* them (pitch offsets rms 0.31 → 0.13 cents in the tenor, blow evenness 0.23 →
0.14). Mine: with `strike_evenness` 0, both draws of the energy score beat identically, so its self-distance cannot
credit a beat; against takes whose beats fall at other phases a read-by-read distance prefers a smooth envelope (the
median-seeking of 12's composite docstring), so the gradient flattens the unison. The remedy is the one the energy
score was built for: let the beats vary from strike to strike as the takes' do (the blow per string, `strike_evenness`;
a session's tuning varies too), so the score's optimum is the recordings' spread.

**The score and the beats** (`runs/physics_revamp/beats/`, `eval/paired1.md`, `paired2.md`: 128 excerpts each of
validation and training pieces, the coupled run's physics without retraining). Switching on the per-strike draw of
each string's blow (`strike_evenness`, sd of its log) improves the composite by itself: 0.25 −0.015 [−0.027, −0.008]
on validation, −0.026 on the training pieces; 0.40 −0.016 (band and partials −0.006/−0.008, between-pooled and level
worse); detunings × 2.5 with it: −0.018 (0.25) and −0.021 (0.40); × 2.5 *without* it: partials +0.003 worse (the
deterministic beats the gradient flattened). With the strike varying, the detunings that match the recordings'
fluctuation cost nothing on the score.

### 12.3 Step 3: ablations (read; `runs/physics_revamp/ablate/ablate.md`)

The coupled run's weights with one group put back to the start (the tables to zero), on the 768 excerpts of 12.1.
Positive: the group's trained change helps.

| reverted | held-out | training pieces |
|---|---|---|
| longitudinal force off | +0.014 [+0.009, +0.017] | +0.009 |
| interaction tables | +0.030 [+0.015, +0.040] | +0.026 |
| unison (pitch offsets, evenness, polarisations) | −0.011 | −0.004 |
| image per bus (pan, delay, in-plane body) | +0.065 | +0.036 |
| the mode model's parameters | +0.050 | +0.034 |
| room | +0.021 | +0.029 |

Every group's change helps on the held-out pieces at least as much as on the training pieces (none of the
differences of differences is positive beyond its interval except the image, which helps the held-out pieces
*more*): no group over-fits. The tables earn their place (mostly in `pooled_exposed`), as does the longitudinal force
(between the partials). The one group whose training hurt is the unison: the flattening above.

### 12.4 Step 4: the residual (read; `runs/physics_revamp/residual_check/`)

A fresh aware residual on the coupled run's physics (physics and room frozen), 20 min each (~400 steps, the GPU
shared), with and without the 12-s MIDI-history input (`res_history`, new config switch). On the 768 excerpts:
with the history −0.013 [−0.018, −0.005] on the held-out pieces and +0.000 on the training pieces; without it −0.006
and +0.000; B_comp's inherited residual on the coupled physics −0.002 and −0.007. The fresh residual gains on pieces it
never heard, not on the ones it trained on: no sign of memorised passages, and the history helps. The inherited one
was trained against the mode model's expected energies (an input of the aware residual) and does nothing here. For
the run: a fresh residual, history on.

### 12.5 The main run (`runs/physics_revamp/main/`)

From what 12.1–12.4 found, against the coupled run:

- the coupled strings' dB parameters at the dB rate and their cents at the cents rate (12.2);
- each string's blow drawn per strike, `strike_evenness` 0.3 (12.2: the score's own gain without training, and the
  condition for the score to stop flattening the unison);
- the unison's pitch offsets × 2.5 at the start (the run had shrunk them; × 2.5 brings the single notes' fluctuation
  within ~0.5 dB of the recordings' in R3–R5 and costs nothing on the score once the strike varies);
- the interaction tables and the longitudinal force kept (12.3); everything else from the coupled run's averaged
  weights;
- a fresh residual with its history (12.4);
- two legs, so an early stop of the physics does not skip the residual: physics alone ≤ 130 min, then the residual
  from the first leg's best ≤ 80 min; each stops when the mean of 4 validations (64 excerpts) has not improved by
  0.002 in 8 checks. Model selection reads the first 64 validation excerpts, so of the paired sets afterwards the
  test pieces are the clean held-out one.

Afterwards (`post.sh`): the paired set (B_comp, the coupled run, both legs), listening
(`samples/physics_revamp/main/`), the bench, the beat fluctuation, N13.

**Leg 1** (read; `physics.out`): 6,000 steps, 69 min, stopped early (the 4-check mean flat at ~0.594 from step
~3,500); held-out validation (64 excerpts, averaged weights) 0.616 at the start, best single check 0.589 at step 1,000
(a low of the noise), ~0.594 at the end. Leg 2 as first scripted started from `best.pt`'s *raw* weights (`--init-from`
reads `model`, not the average): 0.619 on the same excerpts. Restarted (`run2.sh`) from leg 1's averaged weights at
its end (`physics/last_ema.pt`); the evaluation uses `last.pt` of both legs (averaged), no selection on single checks.

**The unison in leg 1** (read): with the blow drawn per strike the run still shrank the pitch offsets below the treble
(rms per group of 11 keys, start → end: 0.64 → 0.26, 0.33 → 0.20, 0.38 → 0.12, 0.99 → 0.43 cents; the top two groups
grew, 1.51 → 1.97) and the evenness further. So the per-strike blow is not enough (mine): it varies the beats' depth,
not their rate, and the recordings come from sessions with other tuning states, so their beat rates vary from piece to
piece; against that a fixed rate per key still loses to a flat envelope read by read. What would fit the energy score:
a per-excerpt draw of the unison's pitch offsets (a session's tuning), applied as a first-order shift of the normal
modes' frequencies (dλ = rᵀ dΩ r / rᵀr, no new eigen-decomposition). Not built; for the owner to decide.

**Leg 2** (read; `residual.out`): from leg 1's averaged weights, the fresh residual, 80 min (4,434 steps, the time
limit); on the 64 validation excerpts the residual's gain grew to −0.027 (physics 0.607, with it 0.579).

### 12.6 The main run on the paired set (read; `runs/physics_revamp/main/paired.md`)

256 excerpts each; leg 1 = `physics/last.pt`, leg 2 = `residual/last.pt` (averaged weights); 95 % intervals of the
paired difference from B_comp's physics. Test is the clean held-out set (the early stopping read 64 of the validation
excerpts).

| | validation | test | training pieces | held-out (both) |
|---|---|---|---|---|
| B_comp physics | 0.627 | 0.605 | 0.632 | 0.587 |
| B_comp with its residual | 0.615 | 0.597 | 0.612 | 0.575 |
| coupled run, physics | 0.646 | 0.619 | 0.610 | 0.611 |
| **main, leg 1 physics** | 0.579 | **0.564** | 0.560 | **0.540** |
| main, leg 2 physics | 0.583 | 0.571 | 0.553 | 0.551 |
| **main, leg 2 with its residual** | 0.575 | **0.561** | 0.544 | **0.540** |
| leg 1 physics − B_comp | −0.048 | −0.041 [−0.082, −0.031] | −0.072 | −0.047 [−0.070, −0.026] |
| leg 2 with residual − B_comp | −0.053 | −0.045 [−0.087, −0.032] | −0.088 | −0.048 [−0.076, −0.026] |

The physics of leg 1 is better than B_comp's on every set, in every read-by-read term (band −0.008, between −0.012,
onset −0.008 on held-out, all intervals below zero) and in the pooled ones; the coupled run's deficit at the partials
is gone (−0.003). Leg 1 against the coupled run: −0.071 held-out, −0.050 on training pieces. Held-out − training
difference of (leg 1 − B_comp): +0.025 [−0.015, +0.055], no longer significant.

In leg 2 the physics, trained on at 0.3 × the rate with the partial gains and colouration unfrozen (stage 2), moved
toward the training pieces: +0.011 on held-out, −0.008 on training pieces against leg 1. The residual then gains
−0.011 held-out and −0.008 on training pieces against its own physics (no sign of memorising). Net, leg 2 with its
residual equals leg 1's physics alone on the held-out pieces (0.540 both; test 0.561 against 0.564). Mine: stage 2's
per-key partial gains (88 × partials) are the capacity that fits the training pieces; the next residual leg should
freeze the physics (train the residual alone) and keep leg 1's physics.

**Afterwards** (read; `runs/physics_revamp/main/`): the bench (`bench/`) has the main model as close as B_comp or
closer everywhere; the knock above 8 kHz +0.7…+2.9 dB (B_comp +2.5…+9.2), phantoms R6 +2.3 dB (coupled run +7.0),
R2 −2.7 (B_comp −5.2). N13 (`envelope/`): fades within ±1–2 dB to 1.5 s, as B_comp. Beats (`beats.txt`,
`scripts/beat_fluctuation.py`): still about half the recordings' fluctuation below R6 (R3 0.58 against 1.76 dB), as
the unison's shrinking predicted. The formerly frozen dB parameters at their proper rate: the longitudinal level
+0.8 dB (mean), its vertical share +2.7 dB, the knock's resonances and the free longitudinal modes stayed off (±4 dB
around −60 and −40 dB): the loss does not ask for them. Listening: `samples/physics_revamp/main/` (local; 8 test
excerpts and the demo; order recording, B_comp, coupled run, leg 1 physics, main with residual; `<i>_ab.wav`).

### 12.7 Where this leaves us, and what next (mine)

- The alarm of the first run is explained: the split between held-out and training pieces was in the structure from
  the start, not learned; no trained group over-fits; the score difference that remained came mostly from the
  flattened unison and from the dB parameters that could not move. With those fixed, the coupled physics is better
  than the mode model's on held-out pieces (−0.047), the first physics change on this branch to show that with an
  interval.
- The residual generalises once fresh (−0.011 on held-out and training pieces alike); stage 2's physics does not
  (partial gains, colouration): next residual leg with the physics frozen.
- The unison: beats still half the recordings'. Proposed: a per-excerpt draw of each unison's pitch offsets (a
  session's tuning), as a first-order shift of the normal modes (cheap); then let the score set the detunings.
- To decide with the owner: the knock resonances and free longitudinal modes (no measured gap, the loss leaves them
  off: remove, or measure M2 first?); M1–M6; the long-window term.

## 13. The attack of mid-soft lower-tenor notes (2026-10-04)

The owner, listening to `samples/physics_revamp/main/`: mid-soft lower-tenor notes (first notes of excerpt 3) sound
"dully struck", without the ring of the piano; not in ff passages; sometimes convincing, sometimes not; already so
before the revamp.

**Excerpt 3** (mine): over the whole excerpt the models are 2-3 dB weak at 2-4 kHz and 3-6 dB at 4-8 kHz (re 0.1-0.5
kHz); excerpts 4-5 (velocity 58, 64) likewise, the soft and the loud excerpts not. The gap is in single strikes: the
G#3 at 0.71 s (velocity 71, re-struck 0.33 s after the same key, pedal down) has its partials 9-24 15-26 dB stronger
in the recording than in main at 0.74-0.95 s, its fundamental 9 dB weaker. Per onset over the 8 excerpts (92 single
onsets of MIDI 45-64) the band rise at the onset matches in the median in every velocity bin.

**On 1,358 notes** (`scripts/tenor_attack.py`, `runs/physics_revamp/attack/`; MIDI 45-59, 2018, no other onset 0.25 s
before or 0.4 s after; known change: +6 dB above 2 kHz on the recording reads +5.4 to +6.0 in partials above it, 0.0
below): each partial's level re partials 1-6 at 15-50, 50-100, 100-200, 200-330 ms, per velocity bin. Brightness
(partials 7-16 re 1-6), recording / B_comp / main: level at velocity 64, MIDI 52 −30.1 / −30.2 / −29.2 dB; rise per
16 velocity steps +5.7 / +5.3 / +5.8 dB; spread after velocity and pitch (sd) 3.6 / 3.5 / 3.3 dB; partials 11-24
alike. By context (same key still sounding, fresh with pedal down or up, dense or sparse) main − recording stays
within −1.6 … +2.1 dB (n 20-111 per cell). The models' upper partials stand above their background more often than
the recording's (about 200 notes against 72), so the notes left out are not hiding a deficit.
- **The key matters:** the recording's brightness after velocity and pitch is predicted by the same key's other notes
  (leave-one-out r 0.56-0.62; by session 0.20-0.27). Per key it ranges over sd 2.2 dB, the models' over 1.2-1.3 with
  the same pattern (r 0.86 main, 0.74 B_comp): MIDI 52 +5.1 vs +3.2, MIDI 45 +2.9 vs +0.6, MIDI 56 +2.2 vs +0.4 dB.
  Three of the six notes where main is darkest are MIDI 56.
- **Not the shimmer of the upper partials** on the excerpt's G#3: the models' partials 9-24 fluctuate as much as the
  recording's or more (1-4 dB rms over 50-550 ms).

**So the level spectrum does not show the dullness on average** (mine): the models are as bright as the piano at
every velocity in 15-330 ms, with a per-key pattern too flat by ~40 %. What the owner hears is either in single
strikes far brighter than the model ever makes (excerpt 3's G#3: 15-26 dB), or in something these measures do not
see (the first 15 ms in context, where `onset_profile` reads few notes; phase or stereo). Listening set to calibrate
the measure against the ear: `samples/physics_revamp/attack/` (6 notes the measure calls darker, 6 it calls equal).

**The key-to-key pattern widened** (`runs/physics_revamp/attack/widen/make.py`: main's per-key spectral shape at
u = 0.45, re partials 1-6, minus its smooth trend over +-6 keys, times 1.7 through `physics.partial_gain`, MIDI 45-59
tapering to 0 at 40 and 64; no training). Check on the 1,358 notes (`widen/measure/`): per-key brightness (partials
7-16 re 1-6, 15-50 ms, after velocity and pitch) sd over keys 1.30 → 2.06 dB (recording 2.19), correlation with the
recording's 0.85 → 0.93, rms per key 1.28 → 0.82 dB. On excerpt 3's G#3 (note 30) it adds 1-2 dB to partials 7-24
against a 15-26 dB gap. Listening: `samples/physics_revamp/keys/` (recording, main, widened).

**Excerpt 3, notes 27-31** (the owner: one of them much louder in the recording, not in the renders). A-weighted
level of each note's own partials, 20-150 ms: note 30 (G#3, velocity 71, re-struck 0.33 s after note 28, pedal down)
is the loudest in the recording (−9.1 against −12.4 … −14.5 dB) and in main (−8.9 against −15.8 … −20.5); unweighted
it is level with notes 28-29 in the recording but 6-7 dB over them in main. The recording's note 30 is loud in partials
7-24 (10 dB over note 28's), with its fundamental 2 dB *under* what was ringing before the strike; main's is loud in
the fundamental (+7.5 dB at the strike, 8 dB over the recording's re the other notes) with partials 7-24 10 dB weaker.

**Re-strikes in general** (`scripts/restrike_spectrum.py`: the bench's re-strike runs, MIDI 45-72, 141 runs: 141 first
strikes, 218 re-strikes 0.08-0.6 s after the key's last, pedal down; per strike the change of the fundamental, of
partials 2-6 and 7-24 over -60..-5 ms, at 20 ms to min(150 ms, the next strike)). Medians, model − recording: within
±1 dB for every change, fresh or re-strike; partials 7-24 re the fundamental after the strike −1.4 to −1.7 dB in main.
The spread of (highs − fundamental change) is the same (sd 6.8 dB both on re-strikes; percentiles within 1 dB), and so
is the share of note-30-like re-strikes (fundamental ≤ +1 dB, highs ≥ +6 dB): 4.6 % recording, 3.7 % main, 4.6 %
B_comp. But strike by strike the models follow the recording on fresh strikes (r 0.75 fundamental, 0.79 highs) and not
on re-strikes (0.23 fundamental, 0.58 highs; main against B_comp 0.71). On the recording's 21 most note-30-like
re-strikes: recording fundamental +1.1 dB, highs +11.4; main +7.4, +7.0. Is that a mechanism or chance? If the hammer
met a string moving towards it, the blow would both cancel the fundamental and strike harder (brighter): then, given
velocities, gap and pitch, a re-strike's fundamental change and its brightness would be anti-correlated in the
recordings and not in the model. They are alike: r(fundamental change, highs re fundamental after) −0.39 recording,
−0.31 main, −0.26 B_comp (n 218, se ~0.07). So, at this n, chance: what a re-strike does to the ringing fundamental
depends on the phase of the old vibration at the blow (sub-millisecond timing MIDI does not carry); the models draw the
same lottery with the same odds, on other notes.

**Re-strikes, the systematic part** (the owner: is it chance, or do the recordings follow a pattern the model does
not?). Each re-strike's change regressed on what MIDI carries (velocity, the previous strike's velocity, log gap,
pitch), separately per side, 218 re-strikes, intervals from a bootstrap over runs: the recordings do follow a pattern
(fundamental change: R2 0.25; velocity +3.5, previous velocity −3.9, gap +1.3 dB per sd; highs change R2 0.15) and
main follows it (R2 0.26 / 0.17; every coefficient within 1.2 dB of the recording's and inside its interval). The
note-30 pattern, highs change minus fundamental change, is unpredictable from MIDI in the recordings too (R2 0.06;
main 0.03). The one systematic difference: partials 7-24 re the fundamental after a re-strike, main 1.3 dB under.
(With the levels before the strike as regressors, previous velocity and the fundamental's level before trade
coefficients between sides; their sums agree, −5.5 vs −5.7 dB.)

**Excerpt 3, notes 27-32 in context**: re each note's own fundamental, 20-150 ms, every model's partials 2-6 and 7-24
are 5-17 dB under the recording's on all six notes (B_comp, coupled run, leg 1, main alike). By band, main − recording
over the clip's first 1.6 s: 30-360 Hz −6 … +5 dB, 0.7-1.4 kHz −2 … −8, 1.4-2.9 kHz −4 … −10, 2.9-5.8 kHz to −17 after
note 30; 2-4 s −2.3 … −6.3; 4-12 s within 2 dB. Over the 80 single tenor onsets of the 8 listening excerpts the models
are as often brighter (excerpts 4-6) as duller (0, 3) re the fundamental.

**When is the model dull?** (`scripts/brightness_context.py`, `runs/physics_revamp/attack/context/`: 64 clips of 10 s
from 2018, B_comp and main with their residuals; per 50 ms frame the brightness, 0.7-5.6 kHz re 88-700 Hz; known
change −6 dB above 1 kHz on the recording reads −5.2 dB): overall main −0.3 dB, B_comp −0.1; by pedal, notes sounding,
pedal × notes, time since onset, velocity, register, every cell within ±1.2 dB; a regression on all of them explains 1 %
of the gap's variance. The gap varies within clips (frame sd 2.9 dB) more than between them (sd 0.9 dB over clip
medians, −3.2 … +1.9), and the two models' gaps move together (frame by frame r 0.86, per clip 0.80): where one is dull
the other is, so the dull moments come from what both get from the MIDI against what the piano did, not from either
physics.
- **Not the soft pedal.** Excerpt 3 starts at CC 82 (to 2 s), then 127, so a model that darkened too much at part pedal
  would fit. On the 1,358 notes (brightness re what velocity and pitch predict with the pedal up): the recordings darken
  by −1.6 dB at CC 64-99 and −2.3 dB at 100-127 (partials 7-16, 15-50 ms; n 40, 245), B_comp by −0.2 / −0.4, main by
  −0.2 / −1.0: the models' una corda is too weak, not too strong.

**Where this leaves the dullness** (mine): real on excerpt 3's opening and measurable there (5-17 dB re the
fundamental), shared by every model, absent on average and not tied to any context tried (velocity, key, pedal, notes
sounding, re-strike, soft pedal). What remains is variation in the recordings that the MIDI does not carry: per strike,
the models' brightness (after velocity and pitch) correlates with the recording's at 0.27-0.38 only, the key accounting
for most of that. Small systematic gaps found on the way: the per-key brightness pattern at ~60 % of its range (13),
partials 7-24 re the fundamental ~1.3-1.7 dB low after re-strikes and fresh strikes, una corda's darkening at about half.

**What tells the recording from the model** (the owner: the passage does not sound like a real piano, whatever its
band levels; `scripts/texture_tells.py`, `runs/physics_revamp/attack/tells/`: the 64 clips of `brightness_context.py`,
per 0.5 s segment and band, 2,432 segments; known change: uncorrelated noise at −15 dB per band on the recording moves
flatness +1.7 … +4.1 dB, peak share −0.2, coherence −0.02 … −0.04). Model − recording, median [95 % over clips],
separation = paired median / paired IQR:
- band level, spectral flatness, peak share: within ±0.4 dB, separation ≤ 0.13;
- fast fluctuation (10 ms level rms around a 0.25 s average): −0.24 dB at 0.5-1 kHz (separation −0.25), ≤ 0.1 above;
- **inter-channel coherence: the models' two channels are more alike** — 0.5-1 kHz +0.10 (recording 0.59; separation
  0.41 / 0.43 main / B_comp), 1-2 kHz +0.16 / +0.19 (0.56; 0.77 / 0.93), 2-4 kHz +0.09 / +0.15 (0.55; 0.40 / 0.67),
  4-8 kHz +0.01 / +0.08. Not the recording's noise floor: by quartile of the band's level the gap holds or grows in the
  loudest (1-2 kHz +0.14 / +0.17, recording 0.56). In excerpt 3's opening the same (+0.1 … +0.4 at 0.5-2 kHz).
Stationary partials through any fixed room keep a coherence of 1; what lowers it is several components within a bin
reaching the two channels differently (a unison's strings and their beats, the board's modes per partial) and a room
whose diffuse part differs between the microphones (mine). M5 (stereo per partial) was planned and not run.
Ear test: `samples/physics_revamp/mono/` (excerpts 3-4, first 4 s, stereo then mono).

**Excerpt 3's opening, one input changed at a time** (`runs/physics_revamp/attack/interv/run.py`: main renders the
first 4 s again with one change; level matched over 4-12 s of the unchanged render; mono; per note 27-32 its own
partials at 20 ms .. min(150 ms, next onset)). Mean over the six notes, model − recording, fundamental / partials 2-6 /
7-24: as is +3.3 / −3.7 / −3.7 dB; soft pedal off +3.3 / −2.1 / −1.4; soft pedal full +2.6 / −4.8 / −5.6; sustain
pedal off +1.6 / −5.9 / −7.2; earlier notes removed +4.5 / −3.3 / −0.9; velocities +15 +7.2 / +1.3 / +3.5. So the
passage's difference is the balance, the fundamental too strong against everything above it (~7 dB; notes 30 and 31
+8, +10 dB in the fundamental), which no velocity gives; the soft pedal at CC 82 adds ~2 dB to the deficit above; the
pedal and the ringing context do not make it.
- **Not the recording file.** Per file the recordings' fundamental re partials 2-6 seemed to range −9 … +11 dB at MIDI
  53-58; after one mean per key and a velocity slope (1,358 notes) the files differ by sd 1.2 dB (models 0.7-1.1),
  same-file leave-one-out r 0.20-0.27, and excerpt 3's file sits at +0.4 dB (main +0.6). The raw spread was the files'
  mix of keys and velocities.
- **Not the soft pedal in general**: within files it moves the balance either way.

**Re-strikes against the phase of the ringing string** (`scripts/restrike_phase.py`, `runs/physics_revamp/attack/phase/`;
the bench's re-strike runs at MIDI 36-84, 340 re-strikes). Per strike the fundamental's complex amplitude on the
clip's time axis before (−70..−5 ms) and after (20 ms .. min(100 ms, next strike)); the blow's own part N = after − k·old
(k = 1 and 0.5) and delta = its phase re the old vibration (offset-free: both parts pass the same body and room).
Self-test on synthetic strikes: delta within 0.04 rad (median), a built-in 3 dB dependence of the highs on cos(delta)
read as +2.94 dB. Dependences on delta after velocity, previous velocity, log gap and pitch (amplitude, 95 % over runs;
null: delta shuffled, 95 %):
- **the fundamental's change follows delta**, as superposition says: recording 5.0 dB [4.1, 5.8], main 4.4 [3.6, 5.3]
  (null 1.2): the blow adds to or cancels the ringing vibration by up to ±5 dB depending on where it was in its cycle;
  the model does the same, as strongly;
- **the highs' change does not**: 0.27 [0.09, 1.23] recording, 0.36 main (null 1.0): no sign of a harder contact on a
  string moving towards the hammer, at a resolution of ~1 dB;
- delta is spread nearly evenly on both sides (mean resultant length 0.17 / 0.28 at k = 1, 0.11 / 0.04 at k = 0.5).
- Strike by strike the model's blow follows the recording's (own fundamental r 0.69, highs change r 0.52); its own
  fundamental is 2 dB under the recording's (median); 'thin and bright' blows (own fundamental under −1 sd, highs over
  +1 sd) are 1.5-2.4 % of the recording's re-strikes and 1.2 % of main's.
- **Note 30**: delta +127° (k = 1); the recording's blow put −35.7 dB into the fundamental, 7 dB under the ringing
  vibration and 9 dB under main's blow (−26.8), and raised the highs by 16.6 dB (main 5.9). So not a cancellation: a thin,
  bright blow, one of the ~2 % of the recording's re-strikes of that kind; given velocity and pitch, the recordings'
  blow fundamental and brightness vary independently (r −0.05), so it is not a mis-recorded velocity either (that would
  move both together).

**Realism, not match** (the owner: notes 27-31 sound unlike any real piano, and physical models from MIDI can sound real,
so the question is what the model does that no piano does, judged against the range of real notes, not against this
recording; also phase 4's tenor notes, "a complex synth wave with a custom attack envelope", 15).
- **Single tenor notes are inside the real range except in stereo** (`scripts/note_realism.py`,
  `runs/physics_revamp/attack/realism/`: 482 notes of MIDI 45-60, velocity 30-85, 2018, sounding to 0.65 s; share of the
  model's notes outside the recordings' 10-90 % range, 20 % if alike): cycle-to-cycle correlation of the waveform at
  5-40 ms (14 / 6 % low / high), each partial's deviation from k f1 (≤ 23 / 21 %), its peak width re a single line
  (≤ 16 / 20 %), its fluctuation over 0.08-0.6 s (≤ 22 / 16 %): inside. Inter-channel coherence per partial: partials
  4-6 38 % above the recordings' 90 % (B_comp 46 %), 2-3 20 %, 7-9 25 %. Self-tests: one line reads width 1.02, two lines
  3 Hz apart 1.52; the cycle measure reads shape, not level (a slow ±30 % level jitter 0.992).
- **The long-term spectrum at 1/12 octave matches per channel** (`scripts/fine_spectrum.py`, `runs/physics_revamp/
  attack/fine/`: the 64 clips, 100 Hz-9.5 kHz, re each clip's total): main − recording 0.48 dB rms over bands (B_comp
  0.62). Summed to mono the same clips differ by −3 … −7 dB at 115-145, 290-430 Hz and ~1.4 kHz and +2 … +3 dB over
  1.6-8 kHz (`fine_mono_first/`): the channels' sum, not the instrument.
- **Excerpt 3's opening per channel** (0-2 s): model − recording −2 … −6 dB at 0.23-2 kHz, −8 … −16 at 2-5 kHz, +1 … +3
  under 0.2 kHz. Its brightness (0.7-5.6 kHz re 88-700 Hz) sits at the 19th percentile of 1,924 comparable real frames
  (pedal down, velocity 40-74 in the last 0.3 s, mean pitch sounding 50-65; the recording's opening at the 38th): dark,
  but not outside what pianos do; the level spectrum alone does not make it unreal.
- **The stereo image is unlike a recording's in kind** (the 8 listening excerpts; mine): in the recordings the coherent
  part arrives aligned, GCC-PHAT lag of L re R +0.04 … +0.17 ms in all 8 (a near-coincident pair), and much of the sound
  is diffuse (coherence 0.25-0.4 at 0.5-2 kHz, the coherent part within −76 … +25° there). In main the channels are one
  signal through two filters: coherence 0.5-0.75 at 0.5-2 kHz with large, frequency-dependent phase offsets (often
  +90 … +170°), and no consistent lag (−3.7 … +2.4 ms, weak peak). Where the offsets near 180° the channels cancel in
  mono: the mono dips above. Cause (read from `room.Room`): one body FIR per microphone from different random soundboard
  modes, driven by the same string signal (plus the two buses' pans and delays); in a piano each microphone hears a
  different mix of a unison's strings and the board's modes (beats and decays differ between channels) and the coherent
  direct sound arrives nearly together. Coherent but phase-shifted channels are a known "phasey", hollow cue (mine).
- **Where the phase offsets come from** (`runs/physics_revamp/attack/stereo/make.py`: main with one body FIR for both
  microphones and no bus delays; renders `samples/physics_revamp/stereo/`, physics alone `scratch/stereo_phys/`): the
  offsets stay (e.g. excerpt 3, 1-2 kHz +89° → −170°), with or without the residual (no difference). So not the two
  bodies. A sustained partial is near a stationary sine: through any two filters (two microphones in a room) it comes
  out coherent with a fixed phase difference. In main every component of a partial (the unison's normal modes) reaches
  each microphone by its bus's one path, so a partial's left/right relation is fixed by the room's filters. In a piano a
  partial's components reach the two microphones in different proportions (different board regions), and as they beat
  the left/right relation moves, which lowers the coherence (mine).
- **Measured** (`note_realism.py`, the 482 tenor notes; 40 ms frames every 10 ms over 0.08-0.6 s): within a note the
  recordings' left/right level difference per partial moves by sd 3.4 / 4.3 / 4.4 dB (partials 2-3 / 4-6 / 7-9, median),
  main's by 2.4 / 3.2 / 3.8 (29 / 35 / 27 % of its notes under the recordings' 10 %); the phase difference by a circular
  sd of 28 / 39 / 42° against 20 / 25 / 30° (18 / 36 / 27 % under); B_comp alike. The fundamental moves more in the models
  (phase 26-34° against 17°).
- **Proposed (not built)**: each mode of a partial its own radiation to each microphone (a gain and a phase per key,
  mode and microphone, fixed per instrument), so that the left/right relation moves as the modes beat; the coherent
  direct sound aligned between the channels (the recordings' lag ~0.1 ms); the diffuse part from the hall. Checked by
  the three measures here: coherence per band and per partial, the coherent part's phase, the left/right motion.
  *Superseded by 14: the measures put the gap in the direct sound's phase and the diffuse field, not in the modes.*

## 14. The stereo image (2026-10-04)

The owner: fix the stereo, and check that the stereo measures are any good.

### 14.1 The measures and their checks (`pianonn/stereo.py`, `tests/test_stereo.py`, `scripts/stereo_image.py`)

One module for the four measures (conventions: cross spectrum X_L conj(X_R); a left channel lagging by tau reads
−360 f tau and a GCC lag +tau): per 0.5 s segment and band the coherence (power-weighted), the coherent part's phase
and the level difference; per clip the GCC-PHAT lag and peak height (1 = identical channels); per partial of a note the
coherence over 7 frames and the motion of its left/right relation (40 ms frames: sd of the level difference, circular
sd of the phase difference) with its power over its surroundings. Known answers (tests, mine):
- identical channels: coherence 1, phase 0; the left delayed 0.1 ms: coherence > 0.98, phase −360 f tau within 10°
  up to 2 kHz, GCC lag 0.10 ms within 0.01, peak > 0.9;
- independent noise at SNR 0 / +6 dB: coherence (s / (1 + s))^2 = 0.25 / 0.64 within 0.03 (10 s segments); in the
  measure's 0.5 s segments independent channels read 0.105 (the estimator's floor, ~1 / K_eff), and SNR −6 / 0 / +6 /
  +12 dB read 0.14 / 0.33 / 0.68 / 0.90 for a true 0.04 / 0.25 / 0.64 / 0.89: biased up where low, as expected;
- partials through two different random filters: coherence > 0.99, motion < 0.05 dB and < 1°, whatever the filters'
  phase difference (a fixed filter does not decorrelate);
- two components 2 Hz apart reaching the microphones in different proportions: level and phase motion within 10 % of
  the closed form; partial coherence < 0.9;
- noise in a partial's bin also moves the relation: 0.2 dB / 1.5° at 31 dB over the surroundings, ~2 dB at ~11 dB.
  The recordings' tenor partials sit 3-26 dB over their surroundings (10-90 %), so the motion is also reported for
  partials 25 dB or more over them; the models' surroundings match the recordings' (medians within 1.5 dB).
On real recordings (the 64 clips of `brightness_context.py`, `--check`): the right channel delayed 0.5 ms moves the
GCC lag from +0.01 to −0.44 ms (median; the recording's own +0.1 less 0.5) and leaves the coherence as it was;
uncorrelated noise at −10 dB per band and channel lowers the coherence by 0.14-0.17; the channels' mean reads 1.
Through the renderer: one body for both microphones turns the coherent part's phase to ~0 and the lag to 0 (below);
the hall +6 dB lowers the coherence by 0.09-0.18. So the measures see what they should, at known sizes.

### 14.2 What main does (`runs/physics_revamp/stereo/measure/`; recording, median over 64 clips [95 % over clips])

| | 250-500 Hz | 0.5-1 kHz | 1-2 kHz | 2-4 kHz | 4-8 kHz |
|---|---|---|---|---|---|
| coherence, recording | 0.75 [0.72, 0.78] | 0.61 [0.53, 0.64] | 0.57 [0.53, 0.59] | 0.54 [0.52, 0.57] | 0.44 [0.32, 0.49] |
| coherence, main | 0.71 | 0.72 | 0.74 | 0.64 | 0.49 |
| coherent part's phase (share of clips within ±45°), recording | −6° (97 %) | +2° (80 %) | −32° (52 %) | −23° (27 %) | −39° (36 %) |
| same, main | +137° (2 %) | +72° (19 %) | −11° (23 %) | −3° (72 %) | +90° (8 %) |

GCC lag: recordings within ±0.25 ms in 97 % of clips (peak 0.105), main in 38 % (peak 0.079). Main's room (read from
the checkpoint, 2018): the two body FIRs correlate at −0.11 (two random soundboards, different seeds) and each channel
also rings up through its own random kernel; the direct paths' phase difference has a circular sd of 56-131° per octave
band; the hall's impulse energy is 1-7 dB under the direct sound's. In a recording both microphones hear one board.

Variants of main, no training (`stereo_image.py --variant`): **aligned** (both microphones get the left body, in-plane
body and ring-up kernel, no bus delays): phase ~0 in every band, lag 0 — but coherence up (0.72-0.80) and GCC peak
0.64: the channels now too alike; **hall +6 dB**: coherence 0.62 / 0.61 / 0.55 / 0.46 / 0.34, the phase still random;
**aligned + hall +6 dB**: 0.64 / 0.59 / 0.54 / 0.52 / 0.40 with the phase at ~0. Per partial (300 tenor notes) the
hall +6 dB variants bring partials 4-9's coherence and motion inside the recordings' range (main: 26-34 % of notes
outside on one side), but the fundamental's left/right phase moves too much (40° against the recordings' 17°): two
independent tails make the low frequencies incoherent, where two microphones a few decimetres apart hear a diffuse
field nearly in phase (coherence sinc(2 f d / c) for omnis d apart; mine).

### 14.3 Built: one board, a diffuse hall (`room.py`; config `shared_board`, `hall_mic_d`; test in `test_model.py`)

- `shared_board`: both microphones hear the left body FIR (and in-plane FIR, and ring-up kernel), each through its own
  zero-phase gain per octave band of `Q_BANDS` and bus (`room.mic_eq_db`, a dB parameter): the direct sound in phase.
  The ring-up's division at the partials stays the mean over the two kernels: one kernel's power has deep dips, and
  dividing by it blew partials up (the converted start's validation 0.75 → 1.40 on 8 excerpts; reverted).
- `hall_mic_d`: the right hall carrier mixed with the left's so their coherence is sinc(2 f d / c), real (in phase).
- Fixed on the way: `room.body_h` (the in-plane bus's FIR) learnt at the base rate, 33 x the body's FIR rate, since the
  rate rule matched only `room.body`; in main it moved 2.35 (norm; the body 0.09). Now both at 0.03.
- `runs/physics_revamp/stereo/make.py` converts a checkpoint: the shared board is the left FIR, each microphone's band
  gains keep its old direct path's octave-band energy (2018: the right +1.8-2.2 dB, flat, offsetting `mic_gain_db`).

Converted, no training (`measure2/`): microphones 0.3 m apart and the hall +6 dB come closest — coherence 0.72 / 0.60 /
0.56 / 0.49 / 0.36, phase within ±45° in 73-100 % of clips, lag 0 in all; per partial inside the recordings' range on
every measure (no side over 22 %; the fundamental's phase motion 20° against 17°). 0.15 m keeps the low bands too
coherent (0.87-0.90 at 250-500 Hz). Left over: the GCC peak 0.28 against 0.105 — above 2 kHz a recording's coherent
part is spread in phase (27-36 % of clips within ±45°), the shared board's is not.

**But the score says no, untrained** (`paired/`, 128 validation + 128 test excerpts, held-out, model − main, 95 %
over pieces): shared board alone +0.113 [+0.081, +0.139] (the level term +0.054), with the hall +3 dB +0.242, +6 dB
+0.462 (every term worse; the onset term most). A converted model has lost what training fitted per channel (each
channel's partial levels through its own random body and kernel), and a louder hall smears onsets and tails per
channel. Two things follow: only a trained model can be compared, and the per-channel score has no term that sees the
image, so training alone would keep whatever hall level suits the channels one by one. Hence a training term:
`pianonn.stereo.CoherenceLoss` (`train.py --stereo-weight`): per band the coherence of render and recording on the
same excerpts in groups of 10 frames (the measure's, in torch; it reads the numpy measure's values exactly), the batch
mean of render − recording, its absolute value averaged over the bands.

### 14.4 The check run (`runs/physics_revamp/stereo/run.sh`)

From main converted (0.3 m, hall as it was, averaged weights), 30 min (1,627 steps) with the coherence term (weight 1),
physics and residual together as main's leg 2, then `measure3/` and `paired2/`. The term (render − recording per band,
batch mean) fell from 0.10-0.18 at 1-4 kHz to within about ±0.05 in every band. Validation (64 excerpts) 0.704 → best
0.598, last 0.613 (main's leg 2 ended at 0.579); physics alone 0.688 (main 0.607): the residual takes up more (+0.075
against +0.027).

**The stereo image** (`measure3/`, same clips and notes as 14.2): coherence 0.75 / 0.62 / 0.61 / 0.56 / 0.43 against
the recordings' 0.75 / 0.61 / 0.57 / 0.54 / 0.44 (inside their 95 % intervals except 1-2 kHz, +0.04); the coherent part
in phase (within ±45° in 98-100 % of clips up to 2 kHz; the recordings 52-97 %; above 2 kHz 17-72 % against 27-36 %);
GCC lag within ±0.25 ms in every clip (recordings 97 %), peak 0.22 against 0.105 (still too alike above 2 kHz, 14.3).
Per partial (300 notes): partials 2-9 inside the recordings' range on coherence and motion (no side over 21 %; main up
to 34 %); the fundamental now too steady (coherence 0.96 against a median 0.91, 39 % of notes above the recordings'
90 %; its level motion 26 % under their 10 %): at 100-260 Hz a 0.3 m diffuse field is nearly coherent, and the
recording's fundamental moves more.

**The score** (`paired2/`, 128 validation + 128 test excerpts; model − main, 95 % over pieces): held-out +0.028
[+0.017, +0.045] (test +0.021 [+0.002, +0.034]): band +0.008, partials +0.012, level +0.023, the onset and pooled terms
within noise. Untrained the conversion cost +0.113; 30 min recovered three quarters of it. What is left is per channel:
main's right channel had its own body (and ring kernel) and the per-key levels were fitted through it; the shared board
gives the right microphone the left's fine structure with octave-band gains only. In a piano the two microphones do
hear different fine magnitude structure (different mixes of the board's regions) with the low frequencies in phase;
the model now has the phase right and the fine structure shared. Listening: `samples/physics_revamp/stereo_fix/` (the
8 excerpts of `samples/physics_revamp/main/`; recording, main, stereo).

### 14.5 Where this leaves the stereo (mine)

- Built and checked: the measures (14.1); the direct sound in phase, a diffuse hall with the right low-frequency
  coherence, a training term that sees the image. After 30 min the image is inside the recordings' range in every
  measure except the fundamental's steadiness and the HF phase spread (GCC peak).
- Cost: +0.028 on the per-channel score, after 30 min against main's ~3.5 h; whether it closes with more training or
  needs per-microphone fine structure is open. A per-microphone fine structure that keeps the phase: the right body =
  the left one through a zero-phase gain per microphone at finer resolution (e.g. 1/6 octave), or the
  board's modes shared with a gain per mode and microphone (the original proposal of 13, now for the board's modes,
  not the strings').
- The owner's question (13): whether this is what makes notes 27-31 sound unreal is for listening to decide.
