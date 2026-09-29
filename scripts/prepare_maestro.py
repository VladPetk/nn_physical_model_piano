"""Convert a MAESTRO v3 download into mono FLAC at the model rate + cached MIDI.

    python scripts/prepare_maestro.py /path/to/maestro-v3.0.0 data/maestro24k --sr 24000
"""

import argparse
import csv
import json
import os
from functools import partial
from multiprocessing import Pool

from pianonn.data import prepare_piece


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("maestro_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--sr", type=int, default=24000)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--limit", type=int, help="only the first N pieces (for quick experiments)")
    args = ap.parse_args()

    csv_path = next(os.path.join(args.maestro_dir, f) for f in os.listdir(args.maestro_dir) if f.endswith(".csv"))
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))[: args.limit]
    os.makedirs(os.path.join(args.out_dir, "audio"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "midi"), exist_ok=True)
    with Pool(args.workers) as pool:
        index = pool.map(partial(prepare_piece, args.maestro_dir, out_dir=args.out_dir, sr=args.sr), rows)
    with open(os.path.join(args.out_dir, "index.json"), "w") as f:
        json.dump(index, f, indent=1)
    print(f"prepared {len(index)} pieces, {sum(p['duration'] for p in index) / 3600:.1f} h")


if __name__ == "__main__":
    main()
