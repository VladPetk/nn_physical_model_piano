"""A local HTML page for a listening folder written by ``scripts/ab_render.py``: every excerpt with its versions side
by side, one player each, and the levels from ``manifest.json``. Open the page in a browser from disk.

    python scripts/listen_page.py runs/phase3/step4/listen runs/phase3/step4/listen_45s

Nothing is copied or normalised: the page points at the audio files next to it.
"""

import argparse
import html
import json
import os


def page(folder):
    with open(os.path.join(folder, "manifest.json"), encoding="utf-8") as f:
        m = json.load(f)
    ext, order, models = m["format"], m["order"], m.get("models", {})

    def label(src):
        if src == "recording":
            return "recording"
        info = models.get(src, {})
        return f"{src} ({'with' if info.get('residual') else 'without'} the residual)"

    rows = []
    for e in m["excerpts"]:
        cells = []
        for src in order:
            path = f"{e['index']}_{src}.{ext}"
            if not os.path.exists(os.path.join(folder, path)):
                continue
            lvl = e.get("level_db", {}).get(src)
            cells.append(f'<div class="v"><div class="l">{html.escape(label(src))}'
                         f'{f" · {lvl:+.1f} dB" if lvl is not None else ""}</div>'
                         f'<audio controls preload="none" src="{html.escape(path)}"></audio></div>')
        rows.append(f'<section><h2>{e["index"]}. {html.escape(e["piece"])} at {e["start_s"]:.0f} s</h2>'
                    f'<p>mean velocity {e["velocity"]}, pedal {100 * e["pedal"]:.0f} %, {e["notes"]} notes</p>'
                    + "".join(cells) + "</section>")
    demos = [s for s in order if os.path.exists(os.path.join(folder, f"demo_{s}.{ext}"))]
    if demos:
        rows.append("<section><h2>Demo score (samples/demo.mid)</h2>" + "".join(
            f'<div class="v"><div class="l">{html.escape(label(s))}</div>'
            f'<audio controls preload="none" src="demo_{s}.{ext}"></audio></div>' for s in demos) + "</section>")
    ckpts = "".join(f"<li><b>{html.escape(k)}</b>: <code>{html.escape(v['checkpoint'])}</code>, "
                    f"{'with' if v.get('residual') else 'without'} the residual</li>" for k, v in models.items())
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Listening: {html.escape(os.path.basename(os.path.normpath(folder)))}</title>
<style>
:root {{ --bg: #fbfbf9; --fg: #1d1d1b; --muted: #6b6b66; --line: #e2e2dc; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg: #1b1b1a; --fg: #e9e9e4; --muted: #a0a09a; --line: #34342f; }} }}
body {{ background: var(--bg); color: var(--fg); font: 15px/1.45 system-ui, sans-serif; margin: 0 auto;
       max-width: 980px; padding: 16px; }}
h1 {{ font-size: 1.4em; }} h2 {{ font-size: 1.05em; margin: 0 0 4px; overflow-wrap: anywhere; }}
p {{ color: var(--muted); margin: 0 0 8px; }}
section {{ border-top: 1px solid var(--line); padding: 14px 0; }}
.v {{ display: flex; flex-wrap: wrap; align-items: center; gap: 8px 14px; margin: 4px 0; }}
.l {{ min-width: 260px; }} audio {{ width: min(100%, 520px); }}
</style></head><body>
<h1>Listening: {m['seconds']:g} s {html.escape(m['split'])} excerpts</h1>
<p>Levels are dB RMS after a 20 Hz high-pass, nothing normalised: level is part of what is judged.</p>
<ul>{ckpts}</ul>
{''.join(rows)}
</body></html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="+")
    for folder in ap.parse_args().folders:
        out = os.path.join(folder, "index.html")
        with open(out, "w", encoding="utf-8") as f:
            f.write(page(folder))
        print("wrote", out)


if __name__ == "__main__":
    main()
