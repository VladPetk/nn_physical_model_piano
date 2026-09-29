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
    checkpoint: bool = True

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        """Build from a (possibly older) saved dict, ignoring keys this version no longer has."""
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})
