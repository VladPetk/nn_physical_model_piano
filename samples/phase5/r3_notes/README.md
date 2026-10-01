# Isolated tenor notes (R3, MIDI 47–59, mf–f): recording and model

Model: `phase5=runs/phase5/train_run/train/last.pt:physics` (phase 4, per-strike variation on), rendering each note in its MIDI context. Each clip runs from 0.3 s before the note to the end of its clear second. The model's level is matched to the recording's over the first 0.5 s, so the comparison is of timbre, attack and decay, not level. `ab.wav` plays each pair in turn (recording, then model); `NN_recording.wav` and `NN_model.wav` are the single notes. Measurements: `docs/tone_measures.md` 15.

- 00. MIDI 50, velocity 50, held: `MIDI-Unprocessed_Recital1-3_MID--AUDIO_03_R1_2018_wav--2` at 678.8 s (train); model level matched by +0.6 dB
- 01. MIDI 47, velocity 51, held: `MIDI-Unprocessed_Recital5-7_MID--AUDIO_06_R1_2018_wav--1` at 837.8 s (train); model level matched by +3.2 dB
- 02. MIDI 59, velocity 51, held: `MIDI-Unprocessed_Recital16_MID--AUDIO_16_R1_2018_wav--3` at 180.8 s (test); model level matched by +2.5 dB
- 03. MIDI 53, velocity 57, held: `MIDI-Unprocessed_Recital17-19_MID--AUDIO_17_R1_2018_wav--3` at 395.7 s (validation); model level matched by +0.1 dB
- 04. MIDI 50, velocity 59, pedal down: `MIDI-Unprocessed_Recital1-3_MID--AUDIO_03_R1_2018_wav--1` at 680.2 s (train); model level matched by +4.2 dB
- 05. MIDI 57, velocity 60, pedal down: `MIDI-Unprocessed_Recital17-19_MID--AUDIO_18_R1_2018_wav--1` at 587.6 s (train); model level matched by +2.5 dB
- 06. MIDI 50, velocity 62, held: `MIDI-Unprocessed_Recital1-3_MID--AUDIO_03_R1_2018_wav--4` at 233.5 s (train); model level matched by +3.2 dB
- 07. MIDI 51, velocity 63, held: `MIDI-Unprocessed_Recital13-15_MID--AUDIO_13_R1_2018_wav--3` at 1153.1 s (train); model level matched by +3.8 dB
- 08. MIDI 50, velocity 68, held: `MIDI-Unprocessed_Recital1-3_MID--AUDIO_03_R1_2018_wav--4` at 235.1 s (train); model level matched by +2.9 dB
- 09. MIDI 59, velocity 71, held: `MIDI-Unprocessed_Recital8_MID--AUDIO_08_R1_2018_wav--3` at 344.8 s (train); model level matched by +3.1 dB
- 10. MIDI 49, velocity 76, held: `MIDI-Unprocessed_Recital8_MID--AUDIO_08_R1_2018_wav--4` at 1275.8 s (train); model level matched by +0.8 dB
- 11. MIDI 59, velocity 93, pedal down: `MIDI-Unprocessed_Recital9-11_MID--AUDIO_10_R1_2018_wav--1` at 843.1 s (train); model level matched by +5.1 dB
