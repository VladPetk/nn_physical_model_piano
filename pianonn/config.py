from dataclasses import asdict, dataclass

# MAESTRO v3 recording years. Each year is a different competition setup
# (piano, hall, microphones), so it is the natural "instrument/room" condition.
MAESTRO_YEARS = [2004, 2006, 2008, 2009, 2011, 2013, 2014, 2015, 2017, 2018]


def year_to_condition(year: int) -> int:
    return MAESTRO_YEARS.index(year) if year in MAESTRO_YEARS else len(MAESTRO_YEARS)


@dataclass
class PianoConfig:
    sample_rate: int = 24000
    hop: int = 120  # control rate = 200 Hz
    n_partials: int = 64  # transverse partials per string
    n_modes: int = 3  # coupled-string modes per partial (1 prompt + 2 aftersound)
    n_conditions: int = 16
    symp_partials: int = 4  # partials per key in the sympathetic resonator bank
    ir_seconds: float = 1.0  # soundboard + room + mic impulse response
    noise_fft: int = 512
    noise_bands: int = 32
    ctx_hidden: int = 128
    synth_chunk: int = 1024  # samples per checkpointed oscillator-bank chunk
    rec_chunk: int = 256  # chunk length for the parallel linear recurrence
    init_gain_db: float = -34.0
    use_noise: bool = True
    use_sympathetic: bool = True
    use_context: bool = True
    checkpoint: bool = True

    def to_dict(self):
        return asdict(self)
