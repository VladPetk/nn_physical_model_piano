# Training speed (2026-10-02)

The composite-score training step (smoke run 2's setup, `runs/composite_train/smoke2/run.sh`) took 5 s, against
1.4 s for phase 3's step 4. This is what the step spent its time on, what was changed, and how each change was checked.
Labels as elsewhere: **read** (measured, from a log or a script's output), **mine** (my reasoning or estimate).

## 1. Where the time went

Measured on the smoke-2 checkpoint and config: 6 fixed batches of 8 training excerpts of 2018 (12 s of history, 1 s
warm-up, 2 s scored), the per-strike variation on, the residual on, a quiet RTX 3090 (read):

| | second render (no gradient) | render | score | backward | step |
|---|---|---|---|---|---|
| as it was | 1.05 | 1.18 | 0.28 | 2.02 | **4.54 s** |
| 1. allocator at its default | 0.72 | 0.72 | 0.24 | 1.64 | 3.32 s |
| 2.-3. fused string bank and sympathetic resonators | 0.24 | 0.24 | 0.25 | 0.38 | 1.11 s |
| 4. the score reads the render once | 0.26 | 0.24 | 0.17 | 0.34 | **1.01 s** |

Before the changes, in one render without gradient (1.15 s): the string bank 0.63 s, the sympathetic bank 0.21, the
noise bank 0.13, the aware residual 0.09, the rest under 0.1 (read, synchronised timers per component). Under the
profiler the string bank's forward and backward were ~40 % of the GPU's kernel time; the rest was mostly elementwise
`mul`/`sub`/`exp`/`floor` kernels over `[notes, oscillators, samples]` tensors, i.e. memory traffic, not arithmetic.
The GPU was ~90 % busy during steps (nvidia-smi), so the step was bound by GPU work, not by Python's kernel launches.

The history of the 5 s (read, the runs' logs, batch 8 throughout): phase 3 step 4 1.42 s; phase 4 2.25 s (the
sympathetic bank, the attack in parts); phase 6 3.35 s in stage 1 (the envelope term's extra renders) and 4.30 s with
the aware residual; the smoke runs ~5 s (the energy score's second render, the wider residual, the composite).

## 2. The changes

1. **The allocator.** `PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.6,max_split_size_mb:256"` (in
   `pianonn/train.py` and 16 scripts) freed and re-allocated the cache inside every step and never split a large
   block: 1.2 s of the 4.5 s (read: 4.54 → 3.32 s; the garbage collection alone 0.4 s, the split limit 0.8 s; same
   peak memory, 12.6 GB). Removed everywhere; the memory fraction (`--gpu-mem-fraction`) alone keeps the allocator
   inside the card, which is what stops Windows spilling into system RAM.
2. **The string bank as fused CUDA kernels** (`pianonn/csrc/kernels.cu`, `pianonn/cuda_ext.py`,
   `NeuralPhysicalPiano._render_strings_fused`). One launch renders every active note over the whole segment; each
   thread is a sample and loops over its note's active oscillators; the phase is formed in float64 once per oscillator
   and 128-sample tile, in float32 within it (`__sincosf`, `__expf` on reduced arguments). The backward is analytic and
   recomputes `E sin(phi)`, `E cos(phi)` in two kernels (per sample, reduced over the oscillators: the contact time,
   damper, re-strike and curve gradients; per oscillator, reduced over the samples: frequency, decay, amplitude, damper
   rate). Nothing of size `[notes, oscillators, samples]` is ever stored. The activity test is the reference's (per
   4096-sample chunk, each note's oscillators up to the last one within 90 dB of its peak, dampers included; 70 dB
   since 2026-10-03, `docs/physics_revamp.md` 11); unlike
   the reference, which renders a slice of notes up to its longest note's count, each note renders exactly its own
   (the reference rendered 2.2e9 oscillator-samples per batch for 1.08e9 that pass the test; read).
3. **The sympathetic resonators as a CUDA kernel** (same files, `SympatheticBank._block_fused`): each resonator's
   complex one-pole with its time-varying damper decay runs sequentially in its own thread, and the backward is the
   adjoint recurrence backwards in time from the stored states; no checkpoint, so no recomputation in the backward.
4. **The score reads the render once** (`CompositeLoss`, `PartialView`, `OnsetLoss`, `PianoLoss.band_list`). The energy
   score compares the render with the recording and with a second draw; each comparison recomputed the render's band
   spectra (twice, for the band and the level terms), its partial readings, the between view and the onset windows,
   all with gradient. A cache per step now holds the render's side, so its spectra are computed, and back-propagated,
   once. `validate_composite` shares it the same way.

The PyTorch code stays the definition and the fallback: on the CPU, without the compiled extension, or with
`PIANONN_FUSED=0`, the model renders as before. The extension builds on first use into `pianonn/csrc/build/`
(~1 min; nvcc 12.6 and the Visual Studio 2022 Build Tools on this machine; `cuda_ext.py` imports `vcvars64.bat`'s
environment when `cl` is not on the PATH).

## 3. Checks

- **The kernels against the reference on real batches** (read; 3 batches of 4 training excerpts, the smoke-2 model,
  the residual on, identical random draws; every oscillator rendered on both sides): the audio agrees to −105 to
  −113 dB re its peak, the strings to −100 to −113 dB; every parameter's gradient within 0.3 % (relative norm). At the
  default activity threshold the audio differs by −79 to −88 dB, the reference's padding oscillators; the gradients of
  the frozen B and of the unison detuning (frequency gradients, sensitive to the extra oscillators) by up to 17 % and 1 %.
- **The resonators against a float64 recurrence** (read): forward −86 dB re peak; gradients within 1e-4 to 1e-3. The
  chunked PyTorch recurrence is as far from float64 (−85 dB): the fused bank differs from it at −74 to −86 dB.
- **The score**: on one saved render, the new code's terms equal the old code's to 7e-9 and its gradient to 3e-7
  (relative norm); the validation path's total unchanged. (Across processes the score's gradient varies by ~1e-3: the
  render's `index_add` sums in a varying order, and the L1 terms flip sign where render and reference nearly meet.)
- **Tests**: `tests/test_cuda.py` (6 tests: fused against reference with octave and per-partial curves, history,
  re-strikes, dampers and pedal; the activity rule; block rendering; the resonators against float64); all 122 pass.
- **A 5-minute training run** (`runs/scratch/speed_check/`, smoke 2's command with `--minutes 5`; read): 372 steps in
  5 minutes (4.5 of them in steps), 0.72 s of compute per step on random batches, validation, audio dumps and
  checkpoints as before, no batch out of memory, peak 16.6 GB. Smoke run 2 took 30 minutes for 332 steps. (A first
  attempt met the bug of section 4.)

## 4. Memory, and a bug found on the way

The first check run trained normally to step 236; at step 237 a batch with 498 notes ran out of memory, and every batch
after it did too, small ones included. `train.py`'s handler cleared `out`, `loss`, `logs` and others but not `parts`
(the score's terms), so the failed step's graph stayed alive. Fixed (every name that can hold the graph is cleared).
Phase 6's run probably met the same: it needed a restart after running out of memory at step ~3,190 (mine).

Peak memory on the densest batch of 400 drawn excerpts (312-385 notes per excerpt, read): 12.0 GB, held by the
residual (4.1 GB: attention and GRUs over notes × control frames) and the noise bank (4.3 GB: its per-note event
envelopes, `[batch, notes, samples]`); the strings now hold 1.3 GB. A batch of ~500 notes per excerpt still exceeds a
0.8 memory fraction; it is now skipped (one batch in the first attempt's 237, none in the second run's 372).

## 5. What is left (not built)

At 1.0 s per step (read; mine where it says so):
- the noise bank: its per-note envelopes are built for every sample of the window and contracted with einsum; per key
  or as event trains through a filter they would cost far less memory and time (it is also half the memory peak);
- the residual's MIDI summary GRU steps through 3,000 frames per render (12 s of history at 200 Hz); at the 20 ms
  control rate it would be 750 (a change to the network);
- validation every 100 steps renders 64 excerpts four times (two draws, with and without the residual): ~8 % of a
  run's wall clock before, shrinking with the renders.

## 6. The coupled strings (2026-10-03)

The coupled model (`string_model=coupled`, `docs/physics_revamp.md`) has its own kernels: the bus bank (six outputs
per oscillator, each with a sine and a cosine amplitude, and the glide) and the longitudinal force (a scan over
512-sample segments in three passes). Its step without the score is 0.74 s against the mode model's 0.53 s at the
same 70 dB activity threshold (~0.91 against ~0.70 s with the score, +30 %), peak memory 14.6 against 11.6 GB; the
normal-mode decomposition is shared by a render's three `modes` calls and by a step's two renders. Numbers, checks and
what is left: `docs/physics_revamp.md` 11.
