"""Build featlens_explainer.srt from the per-scene caption logs written during rendering.

    python video/make_srt.py OUT.srt Scene:duration [Scene:duration ...]

Scenes are listed in playback order with their video durations (seconds), which become the
time offsets of each scene's captions in the joined video.
"""

import json
import sys
from pathlib import Path

LOG = Path(__file__).resolve().parent / "renders" / "captions"


def ts(t):
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def main():
    out, specs = Path(sys.argv[1]), sys.argv[2:]
    offset, n, lines = 0.0, 0, []
    for spec in specs:
        scene, dur = spec.rsplit(":", 1)
        for start, end, text in json.loads((LOG / f"{scene}.json").read_text()):
            n += 1
            lines += [str(n), f"{ts(offset + start)} --> {ts(offset + end)}", text, ""]
        offset += float(dur)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out} ({n} subtitles)")


if __name__ == "__main__":
    main()
