# Phase 6 listening (2026-10-02)

The recording, phase 5 (`runs/phase5/train_run/train/last.pt`) and phase 6 (`runs/phase6/train_run/train/last.pt`),
without and with the residual, the per-strike variation on (`docs/tone_measures.md` 17). Phase 6 fits the strings'
decay on the whole note's envelope (each partial's fade to 2.5 s, measured in music) and keeps that term in the
training loss: the upper partials' excess ring after 0.5 s is about halved.

- [`listen/`](listen/): the same 8 test excerpts of 12 s as phase 5's (`<i>_recording`, `<i>_phase5`, `<i>_phase6`,
  `<i>_phase6_residual`), and the demo.
- [`listen_long/`](listen_long/): the same 2 excerpts of 20 s; `<i>_ab.wav` plays the recording, phase 5, the envelope
  fit alone (before training), phase 6 and phase 6 with the residual in turn. Excerpt 0 holds note 66, the D3 that
  rang too long: at 12.5 s in each clip, 12.5 / 33.0 / 53.5 / 74.0 / 94.5 s in `0_ab.wav` (`0_notes.md`, `0_roll.png`).

`index.html` in each folder is a player page (open it locally after cloning).
