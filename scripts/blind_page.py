"""A blind listening page from a folder written by ``scripts/ab_render.py``: per excerpt the chosen versions under
shuffled letters (a new order for every excerpt), the recording named as the reference. The key goes to a separate
file, with a page that reveals it.

    python scripts/blind_page.py runs/loss_compare/listen --sources phase6 A_old B_comp C_gan \\
        --out samples/loss_compare_blind --key runs/loss_compare/listen/blind_key

Writes ``<out>/index.html`` and the renamed copies (``<i>_<letter>.<ext>``, ``demo_<letter>.<ext>``), and
``<key>.json`` and ``<key>.html`` (the same page with the names). Nothing is normalised: level is part of what is judged.
"""

import argparse
import html
import json
import os
import random
import shutil

STYLE = """
:root { --bg: #fbfbf9; --fg: #1d1d1b; --muted: #6b6b66; --line: #e2e2dc; }
@media (prefers-color-scheme: dark) { :root { --bg: #1b1b1a; --fg: #e9e9e4; --muted: #a0a09a; --line: #34342f; } }
body { background: var(--bg); color: var(--fg); font: 15px/1.45 system-ui, sans-serif; margin: 0 auto;
       max-width: 980px; padding: 16px; }
h1 { font-size: 1.4em; } h2 { font-size: 1.05em; margin: 0 0 4px; overflow-wrap: anywhere; }
p { color: var(--muted); margin: 0 0 8px; }
section { border-top: 1px solid var(--line); padding: 14px 0; }
.v { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 14px; margin: 4px 0; }
.l { min-width: 220px; } audio { width: min(100%, 520px); }
"""


def render(title, note, sections, src_dir):
    rows = []
    for head, sub, players in sections:
        cells = "".join(f'<div class="v"><div class="l">{html.escape(lab)}</div>'
                        f'<audio controls preload="none" src="{html.escape(src_dir + path)}"></audio></div>'
                        for lab, path in players)
        rows.append(f"<section><h2>{html.escape(head)}</h2><p>{html.escape(sub)}</p>{cells}</section>")
    return (f'<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1"><title>{html.escape(title)}</title>'
            f"<style>{STYLE}</style></head><body><h1>{html.escape(title)}</h1><p>{html.escape(note)}</p>"
            + "".join(rows) + "</body></html>\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--sources", nargs="+", required=True, help="the versions to hide (labels of ab_render's --model)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--key", required=True, help="path without extension for the key's .json and .html")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    with open(os.path.join(args.folder, "manifest.json"), encoding="utf-8") as f:
        m = json.load(f)
    ext = m["format"]
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.key)), exist_ok=True)
    rng = random.Random(args.seed)
    letters = "ABCDEFGH"
    key, blind, named = {}, [], []
    items = [(str(e["index"]), f"{e['index']}. {e['piece']} at {e['start_s']:.0f} s",
              f"mean velocity {e['velocity']}, pedal {100 * e['pedal']:.0f} %, {e['notes']} notes") for e in m["excerpts"]]
    if any(os.path.exists(os.path.join(args.folder, f"demo_{s}.{ext}")) for s in args.sources):
        items.append(("demo", "Demo score (samples/demo.mid)", "no recording"))
    for idx, head, sub in items:
        srcs = [s for s in args.sources if os.path.exists(os.path.join(args.folder, f"{idx}_{s}.{ext}"))]
        order = srcs[:]
        rng.shuffle(order)
        key[idx] = {letters[j]: s for j, s in enumerate(order)}
        players_b, players_n = [], []
        rec = f"{idx}_recording.{ext}"
        if os.path.exists(os.path.join(args.folder, rec)):
            shutil.copyfile(os.path.join(args.folder, rec), os.path.join(args.out, rec))
            players_b.append(("recording (reference)", rec))
            players_n.append(("recording (reference)", rec))
        for j, s in enumerate(order):
            name = f"{idx}_{letters[j]}.{ext}"
            shutil.copyfile(os.path.join(args.folder, f"{idx}_{s}.{ext}"), os.path.join(args.out, name))
            players_b.append((letters[j], name))
            players_n.append((f"{letters[j]} = {s}", name))
        blind.append((head, sub, players_b))
        named.append((head, sub, players_n))
    note = "The letters are shuffled anew for every excerpt. Note your ranking per excerpt before opening the key."
    with open(os.path.join(args.out, "index.html"), "w", encoding="utf-8") as f:
        f.write(render("Blind listening", note, blind, ""))
    with open(args.key + ".json", "w", encoding="utf-8") as f:
        json.dump({"folder": args.folder, "sources": args.sources, "seed": args.seed, "key": key}, f, indent=1)
    rel = os.path.relpath(args.out, os.path.dirname(os.path.abspath(args.key))).replace(os.sep, "/") + "/"
    with open(args.key + ".html", "w", encoding="utf-8") as f:
        f.write(render("Blind listening: the key", "The same letters as the blind page.", named, rel))
    print(f"{len(items)} items, {len(args.sources)} versions each; page {args.out}/index.html, key {args.key}.json")


if __name__ == "__main__":
    main()
