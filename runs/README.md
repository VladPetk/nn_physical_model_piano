# Runs

Outputs of training, evaluation and measurement. Only this index is in git; everything else is local.

## Layout

```
runs/<experiment>/<run>/
    best.pt, last.pt     checkpoints (best on validation, latest)
    config.json          the run's configuration
    log.jsonl            every log line, with time and numbers
    train.log            the console, if the run was launched with a redirect
    audio/               renders written during training
    eval/                scripts/evaluate.py: report.md, eval.log, long renders
```

Scripts that chain several runs, and results that compare them, sit in the experiment folder. Launch a run with
its log inside its own folder:

```bash
mkdir -p runs/exp1/train && python -m pianonn.train ... --out runs/exp1/train > runs/exp1/train/train.log 2>&1
```

## Contents

| folder | what | reported in |
|---|---|---|
| `measurements/` | taken from the recordings, inputs to training: `mined_2018.json` (per-key B and stretch from isolated notes, `scripts/mine_notes.py`), `phantoms_2018.*` (`scripts/measure_phantoms.py`), `prepare_2018.log` | `docs/trial_2018.md`, `docs/plan_round2.md` (A8) |
| `measurements/bench_2018.json` | the note bench: isolated notes, note-offs, free decays and re-strike runs of 2018, stratified, split by piece (`pianonn/measures.py`) | `docs/tone_measures.md` |
| `measurements/prepare_all_years.log` | converting the other nine years into `data/maestro24k` (2026-09-30); `index.json` now lists all 1276 pieces, so pass `--years 2018` to scripts whose default is every year (`pianonn.train`, `mine_notes.py`, `measure_phantoms.py`, `overfit_excerpt.py`) | |
| `measurements/isolated_notes_all_years.md` | how many isolated notes each MAESTRO year has, by register and clear time | `docs/tone_measures.md`, section 2 |
| `round1_trial/` | the first MAESTRO 2018 trial, old loss (MR-STFT) | `docs/trial_2018.md` |
| `round1_trial/overfit/` | go/no-go: one excerpt, growing parameter sets | `docs/plan_phase0_1.md` |
| `round1_trial/main/` | the 2 h run and its continuation to 220 min (`*_2h` = the 2 h checkpoint); `eval_stage1`, `eval_2h`, `eval_cont`, and `eval_r2` (re-evaluated on the round-2 scale) | `docs/trial_2018.md`, `docs/round2_results.md` |
| `round1_trial/aborted/` | the first launch, stopped | `docs/trial_2018.md` |
| `round1_trial/mel/`, `ident/`, `velcurve/` | 40-minute follow-ups: log-mel term, identifiability (frozen frequencies), per-year velocity curve | `docs/trial_2018.md`, section 10 |
| `round1_trial/post_run*.sh` | the scripts that chained those runs | |
| `round2/` | review 4's re-fit with the round-2 loss | `docs/round2_results.md` |
| `round2/loss_bias_gate.md` | phase A's loss-bias gate on the trial checkpoint | `docs/plan_round2.md` (A1) |
| `round2/stage1/` | shared stage 1, 30 min (`stage1_end.pt` is where the branches start) | |
| `round2/a_residual/`, `b_control/`, `c_gan/` | the three 20-minute stage-2 branches | |
| `round2/compare.md`, `level_diag.md` | paired comparison over noise seeds; level by velocity and in attacks | |
| `round2/listen/`, `ab/` | listening material: 2 × 20 s, and 8 × 12 s from soft to loud (FLAC, `manifest.json`) | |
| `round2/attack.md`, `attack.json` | attacks of isolated notes, recording vs model (`scripts/measure_attack.py`) | |
| `round2/ab_symp/` | the same 8 excerpts: the control's physics without and with the sympathetic bank (untrained) | |
| `round2/bench_v1/` | the note bench's first pass (N6 attack, old N10; superseded) | `docs/tone_measures.md`, section 9 |
| `round2/bench/` | the note bench's second pass on the round-2 models and the trial model (`scripts/note_bench.py`): knock, sharpness, room, image, re-strike. Its N9 (on the table's partials) and N10 sections are superseded: N9 by the own-partials version (docs, section 6), N10 by `bench_release` | `docs/tone_measures.md`, sections 6 and 10 |
| `round2/bench_release/` | N10 rebuilt (partial tracks, note-offs in any texture), `--parts releases` | `docs/tone_measures.md`, section 10 |
| `round2/glide_extra/` | N3 glide (R2–R4) and N12 extra peaks (pedal-up R3–R7) on the bench, the control model as rendered and with its sympathetic bank on (`scripts/measure_glide_extra.py`) | `docs/tone_measures.md`, section 11 |
| `survey/after_silence/` | N12 on single notes after 1 s of silence, recordings of every year (`scripts/survey_after_silence.py`) | `docs/tone_measures.md`, section 11 |
| `round2/chain.sh`, `compare.sh` | the scripts that ran phase B | |
| `phase3/step0/bench/` | the note bench (notes and E4/P3 on 24 excerpts) on the round-2 control: as it was (`control`), with per-key B (`cb`), and with per-key B and the bridge-end comb sign (`cbc`) | `docs/tone_measures.md`, section 12 |
| `phase3/step0/attack_waveforms/` | string delay after the first arrival, R1–R3 (`scripts/attack_waveforms.py`), the control with and without the bridge-end comb; `waves.svg` | `docs/tone_measures.md`, section 12.3 |
| `phase3/step1/hall/` | the hall's T60 and band levels set from the calibration group's free decays (`scripts/set_hall.py`) on the step-0 model with the body's Q capped at 50; `model.pt` is the step-1 checkpoint | `docs/tone_measures.md`, 12.4 |
| `phase3/step1/bench/` | the bench (notes, releases, free decays, 24 music excerpts): step 0 (`s0`), + Q cap (`q`), + Q cap and hall (`s1`) | `docs/tone_measures.md`, 12.4 |
| `phase3/step1/extra_s0/`, `extra_s1/` | N12 extra peaks (recurring body modes) on step 0 and step 1 (`scripts/measure_glide_extra.py --parts extra --no-symp`) | `docs/tone_measures.md`, 12.4 |
| `phase3/step2/velmap/` | note fit (`scripts/fit_notes.py --fit velmap`) on the calibration notes from step 1: the velocity map, per-key level and brightness, cells where the recording or the model stands over its background; `model.pt` is the step-2a checkpoint | `docs/tone_measures.md`, 12.5 |
| `phase3/step2/nomap/` | the same without the velocity map (the control) | `docs/tone_measures.md`, 12.5 |
| `phase3/step2/knock/` | note fit `--fit knock` from `velmap/`: the attack's existing parts (knock noise, thump, knock impulse) on the attack window re the early window and between the partials | `docs/tone_measures.md`, 12.5 |
| `phase3/step2/knock_2k5/` | the same with the knock noise's spectrum above 2.5 kHz kept (`--knock-max-hz 2500`); `model.pt` is the step-2 checkpoint | `docs/tone_measures.md`, 12.5 |
| `phase3/step2/bench/` | the bench (notes, releases, free decays, 24 music excerpts): step 1 (`s1`), velocity map (`vm`), attack refit (`kn`) | `docs/tone_measures.md`, 12.5 |
| `phase3/step2/bench_knock/` | the bench's notes: velocity map (`vm`), attack refit (`kn`), attack refit with the top bands kept (`k25`) | `docs/tone_measures.md`, 12.5 |
| `phase3/step2/recording_selected/` | the first step-2 fits and bench, with cells selected on the recording alone (biased by up to −7 dB near the background; superseded) | `docs/tone_measures.md`, 12.5 |
| `phase3/onset_term/check/` | the proposed onset-aligned attack term on 64 validation segments (`scripts/onset_term_check.py`) on the attack refit `step2/knock`: tolerance, parameter scans against the training loss's terms, bias gate (before the pooled variant) | `docs/tone_measures.md`, 12.6 |
| `phase3/onset_term/check_k25/` | the same on the step-2 checkpoint, with the pooled variant | `docs/tone_measures.md`, 12.6 |
| `phase3/onset_term/pooled/` | `pianonn.losses.OnsetLoss` as built (pooled over the batch) on 128 validation segments: tolerance, scans (64), the bias gate for batches of 1-8 segments, the attack levels' spread; `windows.npz` holds every window's powers (`--from` recomputes the report) | `docs/tone_measures.md`, 12.6 |
| `phase3/onset_term/rel_k25/`, `rel_s3/` | the onset term's check with the relative form and the running pool on 512 validation segments: the step-2 model and the step-3 model (per-strike variation on); `windows.npz` holds the early window's powers too | `docs/tone_measures.md`, 12.8 |
| `phase3/step3/calib_k25/` | the bench's calibration notes (1003) on the step-2 model: the base for the spreads | `docs/tone_measures.md`, 12.7 |
| `phase3/step3/probes/`, `probes2/` | the same notes with one per-strike dimension switched on at a time (`probes2`: brightness, decay and tilt keeping the level; supersedes `probes`' brightness and decay) | `docs/tone_measures.md`, 12.7 |
| `phase3/step3/spread/` | `scripts/strike_spread.py` on those: per-strike and per-key spreads, the probes' sensitivities, the solved spreads (`solved.json`); `model.pt` is the step-3 checkpoint (the step-2 weights with the spreads in its config) | `docs/tone_measures.md`, 12.7 |
| `phase3/step3/bench/` | the bench's evaluation notes and 24 music excerpts: the step-2 model (`k25`) and with the spreads (`s3`); `spread_eval/`, `spread_eval_s3/`: their per-strike spreads | `docs/tone_measures.md`, 12.7 |
| `phase3/step4/train/` | step 4: one training run on music from the step-3 checkpoint, 450 min, 17,840 steps (`chain.sh`: resumes from `last.pt` after a crash, with a watchdog for a hung process; then `post.sh`); `last.pt` is the step-4 checkpoint (`best.pt`: step 11,500), `audio/` (dumps every 1000 steps, the per-strike variation on), `eval/` (`scripts/evaluate.py` on `last.pt`) | `docs/tone_measures.md`, 12.9 |
| `phase3/step4/val_curve.py`, `val_curve.svg`, `moved.py` | the run's validation loss and learning-rate factor against step; what the run moved, per parameter, from the step-3 checkpoint | `docs/tone_measures.md`, 12.9 |
| `phase3/step4/lr_checks/` | the learning-rate checks before it (150–180 steps from the step-3 model: 1e-3 without and with warm-up and decay, 3e-4 constant), `swap.py` (which parameters make the validation rise), `split.py` (validation and training segments), `notes.md` | `docs/tone_measures.md`, 12.9 |
| `phase3/step4/listen/`, `listen_long/`, `listen_45s/` | listening, each with an `index.html` player page (`scripts/listen_page.py`): 8 × 12 s test excerpts soft to loud (FLAC) and the demo, 2 × 20 s (WAV with `<i>_ab.wav`): the recording, the round-2 control, the step-3 model, step 4 (physics, and with the residual); 3 × 45 s (FLAC): the recording, the round-2 control, step 4; the per-strike variation on | `docs/tone_measures.md`, 12.9 |
| `phase3/step4/compare.md`, `bench/` | distances on 96 test excerpts over 2 noise seeds (the variation off), and the bench's evaluation notes and 24 music excerpts (the variation on) | `docs/tone_measures.md`, 12.9 |
| `residual/ceiling/` | the residual's ceiling (`scripts/residual_ceiling.py`): the step-4 model frozen, its context net replaced by free per-note and/or per-frame outputs in the net's language and bounds, fitted per test excerpt (24, 150 steps) and scored on unseen noise seeds against the physics and the trained net; `outputs.npz` holds the fitted outputs with the notes; `analyse.py` → `outputs.md` (what the per-note outputs are made of), `constant.py` → `constant.md` (every note given their medians) | `docs/tone_measures.md`, 13.1 |
| `residual/aware/` | the physics-aware residual (`pianonn/residual.py`, config `residual_kind=aware`): 30 min on the frozen step-4 model, no budget (`chain.sh`); `train/` (checkpoints, log), `compare.md` (96 test excerpts, 2 seeds, variation off: physics, the GRU residual, the aware one), `compare_train.md` (the same on 96 excerpts of the training pieces) | `docs/tone_measures.md`, 13.2 |
| `residual/memorise/` | memorisation test: the aware residual (from `aware/train/last.pt`) fitted to the first 8 test excerpts alone, 400 steps (`scripts/residual_ceiling.py --variants aware --max-batches 1`), scored as the ceiling | `docs/tone_measures.md`, 13.3 |
| `scratch/` | smoke tests; safe to delete | |
