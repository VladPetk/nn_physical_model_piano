from dataclasses import asdict, dataclass, fields

# MAESTRO v3 recording years. Each year is a different competition setup
# (piano, hall, microphones), so it is the natural "instrument/room" condition.
MAESTRO_YEARS = [2004, 2006, 2008, 2009, 2011, 2013, 2014, 2015, 2017, 2018]


def year_to_condition(year: int) -> int:
    return MAESTRO_YEARS.index(year) if year in MAESTRO_YEARS else len(MAESTRO_YEARS)


@dataclass
class PianoConfig:
    sample_rate: int = 24000
    hop: int = 120  # control rate = 200 Hz
    channels: int = 2  # microphone channels: a spaced pair must not be summed (review 3, F2)
    n_partials: int = 96  # transverse partials per string (A0 reaches ~5 kHz with inharmonicity)
    n_modes: int = 3  # coupled-string modes per partial (1 prompt + 2 aftersound)
    n_phantoms: int = 16  # phantom-partial pairs per series (2f_j and f_j + f_j+1), 0 = off
    n_conditions: int = 16
    symp_partials: int = 4  # partials per key in the sympathetic resonator bank
    # the bank's responding keys (MIDI, inclusive), its partials' ceiling (0: up to 0.45 of its rate) and decimation
    # (resonators at sample_rate / symp_decimate); the defaults are the full bank of rounds 1-3
    symp_lo_midi: int = 21
    symp_hi_midi: int = 108
    symp_max_hz: float = 0.0
    symp_decimate: int = 1
    body_seconds: float = 0.3  # learnable soundboard/case FIR (bridge force -> pressure), one per channel
    hall_seconds: float = 2.5  # parametric per-year hall tail
    # the stereo image (docs/physics_revamp.md 14): both microphones hear one board (the left body FIR, a gain per octave
    # band and microphone), so the direct sound arrives in phase; the hall tails a diffuse field seen by microphones
    # hall_mic_d m apart (coherence sinc(2 f d / c)); off / 0: two independent bodies and tails (the earlier models)
    shared_board: bool = False
    hall_mic_d: float = 0.0
    noise_bands: int = 32
    ctx_hidden: int = 128
    synth_chunk: int = 4096  # samples per oscillator-bank chunk (activity is decided per chunk)
    bank_elements: int = 40_000_000  # notes x oscillators x samples per oscillator-bank call (memory ~ 30 B each)
    rec_chunk: int = 256  # chunk length for the parallel linear recurrence
    init_gain_db: float = -34.0
    activity_db: float = 70.0  # a note is skipped in a chunk once all its partials are this far below its peak
    use_noise: bool = True
    use_impulse: bool = True  # deterministic knock: the hammer's force pulse straight into the body
    use_sympathetic: bool = False  # off by default: ~2x per training step (review 3); switch on in later stages
    use_context: bool = True
    use_room: bool = True  # body + hall; off = dry bridge-force signal
    use_floor: bool = True  # stationary microphone/hall noise floor per condition (review 3, F1)
    # the attack's noises: "noise" (rounds 1-3: knock noise and key-bottom thump sharing one spectrum and a step-onset
    # envelope per key) or "parts" (knock noise, thump and string-borne precursor, each with an envelope per band and
    # register and a spectrum capped at the frequency below; synth.NoiseBank, docs/tone_measures.md 14)
    attack_model: str = "noise"
    knock_max_hz: float = 2500.0
    thump_max_hz: float = 2000.0
    precursor_max_hz: float = 5000.0
    # strike comb as the force at the bridge end: sin(n pi (1 - x0)) = (-1)^(n+1) sin(n pi x0) with x0 measured from
    # the agraffe. Same partial levels, but the first transverse pulse reaches the bridge after (1 - x0) T/2 instead
    # of x0 T/2 (off: the round-1/2 models, whose comb is the agraffe-end force)
    bridge_end_comb: bool = False
    # cap on the Q of the body FIR's ringing (0 = off): per octave band the FIR decays after its direct arrival at
    # least as fast as a mode of Q = body_q_max at the band centre, with each band's energy kept. The round-2 bodies
    # grew modes 2.9 Hz wide (1158, 612, 2740 Hz) that ring after every note (docs/tone_measures.md 11.3); 50 is the
    # loss factor 0.02 the body was initialised with
    body_q_max: object = 0.0  # one value, or one per band of room.Q_BANDS (a list or "/"-separated string)
    # the soundboard's ring-up (docs/tone_measures.md 16): a partial reaching the board takes 10-35 ms to build up in the
    # recordings, the same at every velocity (a linear stage after the strings: a resonant board, Q ~20-50), where the
    # fitted body FIRs pass it within a few ms. Each octave band of the body (Q_BANDS, 31.25 Hz-8 kHz) is convolved with
    # a decaying noise kernel of this time constant (ms; one value, or one per band as a list or "/"-separated
    # string; 0 = that band as it is). The kernel's resonant fine structure is divided out at the strings' partial
    # frequencies (their per-key levels are fitted already), so it adds the ring-up and leaves the partials' levels
    body_ring_ms: object = 0.0
    # per-strike variation (docs/tone_measures.md 12.7): at equal key and velocity the piano's notes vary more than the
    # model's, from strike to strike rather than key to key (N11). Each field is the sd of a zero-median normal offset
    # drawn per note (clipped at 2.5 sd), either one value for every key or one per register R2..R6 (a list, or a
    # "/"-separated string; knots at MIDI 37.5, 53, 65.5, 77.5, 86, flat beyond). 0 = off
    strike_level_db: object = 0.0  # the note's level: strings, knock impulse and knock noise together
    # brightness and decay keep the note's energy over 0-0.3 s (the level has its own dimension)
    strike_log_fc: object = 0.0  # brightness: log of the contact time's inverse (the knock impulse's width too)
    strike_knock_db: object = 0.0  # the attack's noises: knock noise, key-bottom thump and knock impulse
    strike_log_decay: object = 0.0  # log of every decay rate of the note
    strike_decay_tilt: object = 0.0  # log decay rate per octave re 1 kHz: the high partials' early decay
    strike_onset_ms: object = 0.0  # the sound's onset re the MIDI note-on
    # per strike and per partial (docs/tone_measures.md 16, item 3): at equal key the piano's partials decay differently
    # from strike to strike, partial by partial, and neither the partial's number, its frequency nor the key predicts
    # it (15). How the hammer meets the unison's strings and their polarisations changes with every blow:
    strike_partial_decay: object = 0.0  # sd of the log of each partial's prompt decay rate (its early energy kept)
    strike_after: object = 0.0  # sd of a random part of each partial's aftersound amplitudes, re their key's value
    # the strings (docs/physics_revamp.md): "modes" (an in-phase mode and two aftersound modes per partial, independent;
    # every checkpoint before the revamp) or "coupled" (pianonn.coupled: the unison's strings in both polarisations
    # coupled through the bridge admittance, two radiation buses, the longitudinal force, the knock's resonances and the
    # pitch glide)
    string_model: str = "modes"
    strike_evenness: object = 0.0  # coupled: sd of the log evenness of each string's blow, drawn per strike
    prune_db: float = 50.0  # coupled: a mode is not rendered when its energy is this far under its partial's strongest
    # interaction tables (pianonn/interactions.py, docs/physics_revamp.md 7): decay, damper, re-strike, spectrum and the
    # unison's evenness against velocity, decay against the pedal's lift; each starts at zero and holds zero mean over
    # the factor it adds
    interactions: bool = False
    # the learned residual: "gru" (ContextNet: MIDI only, per-note corrections fixed at the onset) or "aware"
    # (residual.AwareResidual: sees each note's expected energy per octave group of partials from the physics, lets
    # the sounding notes attend to each other, and gives each note a gain curve per group over time)
    residual_kind: str = "gru"
    res_groups: int = 8  # octave groups of partials, centred 62.5 Hz ... 8 kHz
    res_control: int = 4  # frames per control step of the aware residual (4 x hop = 20 ms)
    res_dim: int = 96
    res_heads: int = 4
    res_layers: int = 2
    res_curve_db: float = 12.0  # bound of the per-group gain curves
    # the aware residual's wider outputs (review 6, 8.2; 0 / 16 = as before): a gain curve per partial for this many
    # partials (the rest share the last) instead of per octave group; the per-frame noise path in this many bands (the
    # noise bank has ``noise_bands``); a noise path per note in this many bands, re the note's own expected energy (it
    # follows the note's decay and dampers); this many random inputs per note, drawn afresh at every render (the
    # residual can vary from strike to strike, which an energy score can train)
    res_curve_partials: int = 0
    res_noise_bands: int = 16
    res_note_noise: int = 0
    res_latent: int = 0
    # the aware residual's GRU summary of the MIDI history (12 s) as an input; off: its outputs depend on the sounding
    # notes, their physics and the pedals only (docs/physics_revamp.md 12: can it recognise training passages?)
    res_history: bool = True
    checkpoint: bool = True

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        """Build from a (possibly older) saved dict, ignoring keys this version no longer has."""
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})
