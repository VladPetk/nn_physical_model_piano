"""Convert a MAESTRO v3 download into stereo FLAC at the model rate + cached MIDI.

    python scripts/prepare_maestro.py data/maestro-v3.0.0.zip data/maestro24k --years 2018
    python scripts/prepare_maestro.py /path/to/maestro-v3.0.0 data/maestro24k --sr 24000

The source is the extracted directory or the zip itself (members are read without unpacking
the 100 GB archive). Re-running adds pieces to an existing output directory and skips the ones
already converted; ``index.json`` always lists every piece present.
"""

import argparse
import csv
import io
import json
import os
import zipfile
from functools import partial
from multiprocessing import Pool

from pianonn.data import prepare_piece


def read_rows(source):
    if os.path.isdir(source):
        path = next(os.path.join(source, f) for f in os.listdir(source) if f.endswith(".csv"))
        with open(path, encoding="utf-8") as f:
            return list(csv.DictReader(f))
    with zipfile.ZipFile(source) as z:
        name = next(n for n in z.namelist() if n.endswith(".csv"))
        return list(csv.DictReader(io.TextIOWrapper(z.open(name), encoding="utf-8")))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", help="extracted maestro-v3.0.0 directory or maestro-v3.0.0.zip")
    ap.add_argument("out_dir")
    ap.add_argument("--sr", type=int, default=24000)
    ap.add_argument("--channels", type=int, default=2, choices=(1, 2))
    ap.add_argument("--years", type=int, nargs="*", help="only these competition years")
    ap.add_argument("--splits", nargs="*", default=["train", "validation", "test"])
    ap.add_argument("--workers", type=int, default=min(10, os.cpu_count()))
    ap.add_argument("--limit", type=int, help="only the first N selected pieces (for quick experiments)")
    args = ap.parse_args()

    rows = [r for r in read_rows(args.source)
            if (not args.years or int(r["year"]) in args.years) and r["split"] in args.splits][: args.limit]
    os.makedirs(os.path.join(args.out_dir, "audio"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "midi"), exist_ok=True)
    index_path = os.path.join(args.out_dir, "index.json")
    index = {}
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = {p["id"]: p for p in json.load(f)}
    work = partial(prepare_piece, args.source, out_dir=args.out_dir, sr=args.sr, channels=args.channels)
    with Pool(args.workers) as pool:
        for i, info in enumerate(pool.imap_unordered(work, rows), 1):
            index[info["id"]] = info
            print(f"[{i}/{len(rows)}] {info['year']} {info['split']:10s} {info['duration']:7.1f}s {info['id']}", flush=True)
    with open(index_path, "w") as f:
        json.dump(sorted(index.values(), key=lambda p: p["id"]), f, indent=1)
    print(f"index has {len(index)} pieces, {sum(p['duration'] for p in index.values()) / 3600:.1f} h")


if __name__ == "__main__":
    main()
