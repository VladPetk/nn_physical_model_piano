# Training losses in neural MIDI-to-audio (literature, 2026-10-03)

The question: what loss do published neural MIDI-to-audio systems train on, and what do they do about one MIDI file
having many possible recordings (timing scatter, phase, note-to-note variation)? Five papers, read in full text by a
literature agent (the PDF text, not the abstracts; the agent's first web summaries were unreliable and discarded).
**Read** = in the paper, with its section; **inferred** = the agent's or my reading. Not read: the JAES 2023 version
of DDSP-Piano (paywalled); the weight of DDSP-Piano's log term was lost in the text extraction.

## 1. The five papers

| paper | output | training loss (read) | one MIDI, many recordings (read) |
|---|---|---|---|
| Hawthorne et al., MAESTRO / Wave2Midi2Wave (ICLR 2019, arXiv 1810.12247) | 16-bit waveform, WaveNet on an onset piano roll | the WaveNet likelihood only (mixture of logistics), §6 | a likelihood over samples; MIDI aligned to audio by a CQT search plus DTW (§3.1); a year one-hot against timbral shifts (§6) |
| Renault, Mignot, Roebel, DDSP-Piano (DAFx 2022) | 16 kHz harmonic-plus-noise synth with inharmonicity, detuning, a GRU context net, a reverb IR per environment | multi-resolution STFT: L1 on linear plus weighted L1 on log magnitudes, FFT 2048-64 (§4.4, eq. 12-14); 0.01 L1 on the reverb IR; 0.1 on the inharmonicity's deviation in a second phase | not addressed: deterministic given MIDI, pedals and an environment ID |
| Hawthorne et al., Spectrogram Diffusion (ISMIR 2022, arXiv 2206.05408) | 128-bin log-mel (16 kHz, 20 ms hop), T5 encoder-decoder with a diffusion decoder, then a vocoder | diffusion noise prediction, weighted L1 (§3.2, eq. 1), classifier-free guidance; vocoder: multi-scale spectral plus adversarial plus feature matching (§3.3) | called "underspecified" (§5.2); MSE gave blurry spectrograms, mixtures of Gaussians were unstable (§3.1); segment-to-segment timbre shifts fixed by conditioning on the previous segment |
| Wu et al., MIDI-DDSP (ICLR 2022, arXiv 2112.09312; violin, URMP 3.75 h) | DDSP parameters via a three-level hierarchy | synthesis: cross-entropy on f0 + multi-resolution spectral L1 (lin and log) + LSGAN + feature matching (§3.3, eq. 2, app. B.4); expression: MSE (§3.4) | six per-note expression values **extracted from the recording** (volume, its fluctuation, peak position, vibrato, brightness, attack noise); the synthesiser learns the audio given them, the expression generator predicts them from the notes; the GAN against "over-smoothing" (app. B.5, a qualitative figure only) |
| Dong et al., Deep Performer (ICASSP 2022, arXiv 2202.06034) | log-mel from a score, then HiFi-GAN v2 | MSE on onset and duration (§2.1); MSE on log-mel (§2.2); the vocoder's loss not stated | timing from a separate alignment model (MAESTRO: the MIDI timing as given); a performer ID (on MAESTRO, the year); the MSE output smoother than the baseline, a GAN left as future work (§4.4) |

## 2. Results and limits that bear on the loss (read)

- **MAESTRO:** listening test (640 pairwise ratings): the recordings not significantly different from WaveNet on
  ground-truth or transcribed MIDI. The training likelihood does not reflect the conditioning's quality (teacher
  forcing), so the listening test is the evaluation. Timbral shifts on long outputs (fixed by the year one-hot), a
  "sonic crash" at the start (trimmed).
- **DDSP-Piano:** MOS with 52 listeners; Pianoteq rated best, above the real recordings; the model above a neural
  baseline (Tacotron-2 + NSF), on par with Fluidsynth by pianists. **Dropping the context network (note interactions)
  made no significant difference.** Without the L1 on the reverb IR, the reverb absorbed the notes' sustain.
- **Spectrogram Diffusion:** no listening test; FAD and transcription F1; inconsistent loudness and artifacts; the
  vocoder caps the quality.
- **MIDI-DDSP:** preferred over the baselines in a listening test (960 ratings); the baseline without the expression
  level is incoherent within notes.
- **Deep Performer:** better than a HiFi-GAN piano-roll baseline on piano in a 15-listener test; no loss ablation.
- No paper ablates its loss against the one-to-many problem.

## 3. What they do about one take against a prediction (inferred, the agent and mine)

- Deterministic models use spectral or regression losses and accept smoothing; two papers say so (MIDI-DDSP, Deep
  Performer).
- Only the likelihood (WaveNet) and diffusion model the distribution of recordings. Both need a model that samples
  audio or spectrograms.
- Variation is absorbed by conditioning rather than modelled: a recording-environment, year or performer ID, or (in
  MIDI-DDSP) continuous per-note values read from the recording.
- Timing is pre-aligned or predicted; magnitude-only losses ignore phase.
- MIDI-DDSP's split is the one that maps onto this project's problem: the per-take part is explained by values read
  from the take, so the audio comparison is against an explained take, and the context predictor learns those values
  by regression on them, not through the audio (review 6, 5.2 "explain the take, then compare").
- DDSP-Piano, the closest system to this one, saw what `docs/composite_score.md` 11-12 saw: a context network trained
  against single takes with a spectral loss adds nothing significant.
