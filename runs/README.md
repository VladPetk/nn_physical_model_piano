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

Cleaned up on 2026-10-03: the runs of rounds 1-2, phases 3-5, the residual's and the composite's smoke runs and the
GAN runs after `loss_compare` were deleted; their results are in the docs (`docs/tone_measures.md`, review 6,
`docs/composite_score.md`) and the index rows in this file's git history. Throwaway runs go in `scratch/` at the repo
root, not here.


| folder | what | reported in |
|---|---|---|
| `measurements/` | taken from the recordings, inputs to training: `mined_2018.json` (per-key B and stretch from isolated notes, `scripts/mine_notes.py`), `phantoms_2018.*` (`scripts/measure_phantoms.py`), `prepare_2018.log` | `docs/tone_measures.md`; first in `docs/trial_2018.md` (removed, see below) |
| `measurements/bench_2018.json` | the note bench: isolated notes, note-offs, free decays and re-strike runs of 2018, stratified, split by piece (`pianonn/measures.py`) | `docs/tone_measures.md` |
| `measurements/prepare_all_years.log` | converting the other nine years into `data/maestro24k` (2026-09-30); `index.json` now lists all 1276 pieces, so pass `--years 2018` to scripts whose default is every year (`pianonn.train`, `mine_notes.py`, `measure_phantoms.py`, `overfit_excerpt.py`) | |
| `measurements/isolated_notes_all_years.md` | how many isolated notes each MAESTRO year has, by register and clear time | `docs/tone_measures.md`, section 2 |
| `phase6/envelope/` | N13, the whole note's envelope (`scripts/note_envelope.py --validate`): 2,200 notes of 2018 in any texture, each partial's fade at 0.1-2.5 s, recording against phase 5, plus phase 5 with the decay of partials 13+ doubled and with partial 2's aftersound +6 dB | `docs/tone_measures.md`, 17.3 |
| `phase6/env_fit/`, `env_fit2/` | envelope fit (`scripts/fit_envelope.py`): the strings' decay and the aftersound's level on N13's fades of the training pieces' notes, held-out notes of validation and test; `env_fit/` 400 steps from phase 5, `env_fit2/` 1,000 steps from it at lr 1e-2 | `docs/tone_measures.md`, 17.4 |
| `phase6/train_run/` | one 310-min run on music from `env_fit2/` (`chain.sh`; `launch.log` names the start model; one restart after running out of GPU memory at step ~3,190) with the envelope term in the loss (`--env-weight 0.5`, the decays free) and the aware residual (fresh) from 55 %; `post.sh`: listening (`listen_long` with the envelope fit alone too), test excerpts against phase 5, N13, the bench, the R3 profile | `docs/tone_measures.md`, 17.5-17.6 |
| `score_check/run1/`, `score_check/gates/` | `scripts/score_check.py` on phase 6, 48 validation excerpts: each term's floor (the model against a varied draw of itself), its gap to the recording, five known changes and a knock-gain sweep against a take with timing scatter; `gates/` compares the partial view's loudness gates (no sweep) | review 6, step 3 |
| `score_check/pooled/`, `score_check/composite/`, `score_check/exposed_p1/`, `_p2/`, `_p4/`, `score_check/sweeps/` | the pooled partial view (energy summed per partial group and note age before the log: 5.3x the per-read sensitivity to partials 5-8 +3 dB); the first composite (its between terms still had a reading-order bug, fixed before `exposed_*`); exposure-weighted pooling at powers 1, 2, 4 (no sweep); two-way sweeps of the decay rates and the aftersound (per-excerpt pooling) | review 6, step 3 |
| `score_check/across/`, `both_varied/`, `energy/`, `energy_seed3/`, `varoff_seed3/` | the composite with its pooled terms pooled across the 48 excerpts, best settings with errors (from `energy_seed3/` on): reference with variation off (`across/`, `varoff_seed3/`: every term, old or new, pulls the aftersound ~4 dB low, z 6-13); both varied (`both_varied/`); energy scores, d(ref, take) - d(ref, ref')/2 (`energy*/`: the pulls shrink, aftersound -2.4 +- 0.9 dB); `*_seed3`: another take | review 6, step 3 |
| `gan_check/check1/` | `scripts/gan_check.py` on the GAN as built (round 2's critic, train.py's settings), from `phase6/env_fit2`: (1) recordings against recordings with a known shelf above 1 kHz, the shelf the generator's only parameter: from +3 dB it ends at -2.2 dB, from -3 dB at -1.0 dB, the critic's loss at chance throughout (fails); (2) the critic trained 600 steps against the model's renders: held-out AUC 0.53 (64 + 64), its adversarial push uncorrelated with the band error (+0.05; the paired feature matching +0.64); (3) at weight 0.1 its gradient on the residual is 0.1 % of the composite's | the owner, 2026-10-02 |
| `gan_check/wide1/` | the same check on the wide critic (`--critic wide --reach residual --fm-weight 0`): the shelf ends -0.70 +- 0.17 dB from +3 and -0.78 +- 0.24 from -3 (means over steps 200-600: -1.57, -0.91; still swinging); held-out AUC 1.000 against `env_fit2`'s renders, but its push uncorrelated with the band error (-0.03): it tells them apart by something other than the octave balance; at weight 0.1 its gradient on the residual is 7.7 % of the composite's (cosine +0.26); a step 0.74 s without the GAN, 1.09 s with the texture view, 1.07 s through the output (+2 GB) | the owner, 2026-10-02 |
| `gan_check/views1/` | `scripts/critic_views.py`: a fresh wide critic per restricted view of both sides (400 steps), renders of `loss_compare/B_stage1` against unpaired recordings, held-out AUC on 64 validation pairs: full 1.000, loudness equalised 1.000, < 500 Hz 0.720, 0.5-2 kHz 0.980, 2-6 kHz 0.995, > 6 kHz 1.000, mid 1.000, side 1.000 (the lowest critic loss, 0.29), side loudness equalised 1.000: no single artifact; every band from 500 Hz up and the side signal each give the renders away | the owner, 2026-10-03 |
| `gan_check/mono_r1_*/` | `scripts/gan_check.py` on the mono wide critic (`--mono`) at R1 0, 1, 10, from `loss_compare/B_stage1`: the octave-gain known answer, the critic against the renders, the share on the residual | the owner, 2026-10-03 |
| `loss_compare/` | the loss comparison (`chain.sh`): the old loss against the composite, each from `phase6/env_fit2`, 120 min of stage 1 then 90 min with the residual (`A_stage1/`, `B_stage1/`, `A_old/`, `B_comp/`), the envelope term off; `C_gan/` (the composite with the GAN, B's steps) only if `C_go` exists, once a critic passes `gan_check`; stopped at A_stage1 step 4,090 for the owner's GPU and resumed; A_stage1 then stopped by hand at step 17,000 (76 min, held-out plateau), and every later leg stops early (`--patience 8 --es-window 4 --min-delta 0.002`), C at most 135 min; `post.sh`: listening (`listen/`, the blind page in `samples/loss_compare_blind/` with its key in `listen/blind_key.*`), both scores (`eval*.md`), the old distances (`compare.md`), N13 (`envelope/`), the bench (`bench/`) | the owner, 2026-10-02 |
