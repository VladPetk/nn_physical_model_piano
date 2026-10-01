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
    body_seconds: float = 0.3  # learnable soundboard/case FIR (bridge force -> pressure), one per channel
    hall_seconds: float = 2.5  # parametric per-year hall tail
    noise_bands: int = 32
    ctx_hidden: int = 128
    synth_chunk: int = 4096  # samples per oscillator-bank chunk (activity is decided per chunk)
    bank_elements: int = 40_000_000  # notes x oscillators x samples per oscillator-bank call (memory ~ 30 B each)
    rec_chunk: int = 256  # chunk length for the parallel linear recurrence
    init_gain_db: float = -34.0
    activity_db: float = 90.0  # a note is skipped in a chunk once all its partials are this far below its peak
    use_noise: bool = True
    use_impulse: bool = True  # deterministic knock: the hammer's force pulse straight into the body
    use_sympathetic: bool = False  # off by default: ~2x per training step (review 3); switch on in later stages
    use_context: bool = True
    use_room: bool = True  # body + hall; off = dry bridge-force signal
    use_floor: bool = True  # stationary microphone/hall noise floor per condition (review 3, F1)
    # strike comb as the force at the bridge end: sin(n pi (1 - x0)) = (-1)^(n+1) sin(n pi x0) with x0 measured from
    # the agraffe. Same partial levels, but the first transverse pulse reaches the bridge after (1 - x0) T/2 instead
    # of x0 T/2 (off: the round-1/2 models, whose comb is the agraffe-end force)
    bridge_end_comb: bool = False
    # cap on the Q of the body FIR's ringing (0 = off): per octave band the FIR decays after its direct arrival at
    # least as fast as a mode of Q = body_q_max at the band centre, with each band's energy kept. The round-2 bodies
    # grew modes 2.9 Hz wide (1158, 612, 2740 Hz) that ring after every note (docs/tone_measures.md 11.3); 50 is the
    # loss factor 0.02 the body was initialised with
    body_q_max: float = 0.0
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
    checkpoint: bool = True

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        """Build from a (possibly older) saved dict, ignoring keys this version no longer has."""
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})
