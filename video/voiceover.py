"""Kokoro-82M voice-over with synchronized subtitles for Manim (a stand-in for manim-voiceover).

    class MyScene(NarratedScene):
        def construct(self):
            with self.voiceover("Hello there.") as vo:
                self.play(Write(title), run_time=vo.duration * 0.5)
            # leaving the block waits out the rest of the clip

Each line is synthesized once with voice ``af_heart`` and cached by a hash of (voice, text) in
``video/assets/voice/`` together with Kokoro's per-word timestamps. Each line is split into
subtitle phrases timed from those timestamps and logged to ``video/renders/captions/<Scene>.json``;
``make_srt.py`` turns the logs into an .srt for the joined video, and ``render.sh`` burns it in
with ffmpeg/libass (drawing subtitles inside Manim ghosts during animations).
"""

from __future__ import annotations

import hashlib
import json
import re
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf
from manim import Scene

VOICE = "af_heart"
SAMPLE_RATE = 24_000
GAP = 0.35  # silence after each line, seconds
HERE = Path(__file__).resolve().parent
CACHE = HERE / "assets" / "voice"
CAPTION_LOG = HERE / "renders" / "captions"

SUB_MAX_CHARS = 52  # longest subtitle phrase

# Numbers are spelled out for the voice; subtitles show them as digits.
DISPLAY = {
    "seven hundred sixty-eight": "768",
    "seven-hundred-sixty-eight-dimensional": "768-dimensional",
    "one hundred ninety-six": "196",
    "two hundred twenty-four": "224",
    "three hundred eighty-four": "384",
    "one thousand twenty-four": "1024",
    "fourteen-by-fourteen": "14 × 14",
    "sixteen by sixteen": "16 × 16",
    "sixteen-pixel": "16-pixel",
    "torch hub": "torch.hub",
}

_pipeline = None


def _pipe():
    global _pipeline
    if _pipeline is None:
        from kokoro import KPipeline
        _pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
    return _pipeline


def synthesize(text: str, voice: str = VOICE):
    """Return (wav path, duration, words) for ``text``; words are [text, space, start, end]."""
    CACHE.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(f"{voice}|{text}".encode()).hexdigest()[:16]
    wav, meta = CACHE / f"{key}.wav", CACHE / f"{key}.json"
    if not (wav.exists() and meta.exists()):
        chunks, words, offset = [], [], 0.0
        for result in _pipe()(text, voice=voice):
            audio = np.asarray(result.audio)
            for t in result.tokens or []:
                if t.start_ts is not None and t.end_ts is not None:
                    words.append([t.text, t.whitespace, offset + t.start_ts, offset + t.end_ts])
            chunks.append(audio)
            offset += len(audio) / SAMPLE_RATE
        sf.write(wav, np.concatenate(chunks), SAMPLE_RATE)
        meta.write_text(json.dumps({"text": text, "words": words}))
    words = json.loads(meta.read_text())["words"]
    return wav, sf.info(str(wav)).duration, words


def _merge_display_phrases(words):
    """Merge spelled-out number phrases into single display tokens ("seven hundred ..." -> "768")."""
    out, i = [], 0
    phrases = sorted(((p.split(" "), d) for p, d in DISPLAY.items()), key=lambda x: -len(x[0]))
    while i < len(words):
        for parts, disp in phrases:
            seg = words[i:i + len(parts)]
            if len(seg) == len(parts) and [w[0].lower() for w in seg] == parts:
                out.append([disp, seg[-1][1], seg[0][2], seg[-1][3]])
                i += len(parts)
                break
        else:
            out.append(list(words[i]))
            i += 1
    return out


def caption_chunks(words, duration):
    """Group words into subtitle phrases -> [(start, end, text)], breaking at punctuation."""
    words = _merge_display_phrases(words)
    groups, cur = [], []
    for w in words:
        if not cur and groups and re.fullmatch(r"[.,:;!?]", w[0]):
            groups[-1].append(w)    # keep punctuation with the phrase it ends
            continue
        cur.append(w)
        text = "".join(x[0] + x[1] for x in cur).strip()
        punct = re.fullmatch(r"[.,:;!?]", w[0]) is not None
        if (punct and len(text) >= 22) or len(text) >= SUB_MAX_CHARS - 8:
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    chunks = []
    for k, g in enumerate(groups):
        text = "".join(x[0] + x[1] for x in g).strip()
        start = g[0][2]
        end = groups[k + 1][0][2] if k + 1 < len(groups) else min(duration, g[-1][3] + 0.4)
        chunks.append((start, end, text))
    return chunks


class NarratedScene(Scene):
    def setup(self):
        super().setup()
        self.captions = []  # [(start, end, text)] in scene time, for the .srt

    def content(self):
        """Everything currently on screen (kept as the one place to filter overlays)."""
        return list(self.mobjects)

    @contextmanager
    def voiceover(self, text: str):
        path, duration, words = synthesize(text)
        chunks = caption_chunks(words, duration)
        start = self.renderer.time
        self.add_sound(str(path))

        try:
            yield SimpleNamespace(duration=duration, text=text)
            remaining = duration + GAP - (self.renderer.time - start)
            if remaining > 0:
                self.wait(remaining)
        finally:
            self.captions += [(start + s, start + e, txt) for s, e, txt in chunks]
            CAPTION_LOG.mkdir(parents=True, exist_ok=True)
            (CAPTION_LOG / f"{type(self).__name__}.json").write_text(
                json.dumps(self.captions, indent=1))
