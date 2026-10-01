# Phase 5 listening (2026-10-01)

The recording, phase 4 (`runs/phase4/train_run/train/last.pt`) and phase 5 (`runs/phase5/train_run/train/last.pt`),
both without the residual and with the per-strike variation on (`docs/tone_measures.md` 16). Phase 5 adds the
soundboard's ring-up (a note builds up over 10-35 ms instead of switching on), a less resonant body at 125-250 Hz, the
attack and the strings' decays fitted on isolated notes, a per-strike, per-partial variation of the decays, and an
onset jitter of ~6 ms per strike.

- [`listen/`](listen/): 8 test excerpts of 12 s, soft to loud (`<i>_recording`, `<i>_phase4`, `<i>_phase5`), and the
  demo. Excerpt details in [`listen/README.md`](listen/README.md).
- [`listen_long/`](listen_long/): the same 2 excerpts of 20 s as phase 4's; `<i>_ab.wav` plays recording, phase 4,
  phase 5 in turn (excerpt 0 holds the low notes under the high texture).
- [`r3_notes/`](r3_notes/): the same 12 isolated tenor notes as `samples/r3_notes/` (phase 4), recording against
  phase 5; `ab.wav` plays each pair in turn.

`index.html` in each folder is a player page (open it locally after cloning).
