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
| `scratch/` | smoke tests; safe to delete | |
