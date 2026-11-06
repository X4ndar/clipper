import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from clipper.config import Config
from clipper.transcript import Transcript, Word, segment_sentences

ROOT = Path(__file__).resolve().parents[1]


def make_transcript(text_by_sentence, gap=0.4, wps=3.0, speaker="A", score=0.95):
    words, t = [], 1.0
    for sent in text_by_sentence:
        for tok in sent.split():
            d = max(0.15, len(tok) / (wps * 5))
            words.append(Word(word=tok, start=round(t, 3), end=round(t + d, 3), speaker=speaker, score=score))
            t += d + 0.05
        t += gap
    tr = Transcript(language="en", duration=t + 1, words=words, sentences=segment_sentences(words))
    return tr


@pytest.fixture
def cfg(tmp_path):
    return Config(home=str(tmp_path / "home"))


@pytest.fixture(scope="session")
def sample_video(tmp_path_factory):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    filters = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True).stdout
    if " flite " not in filters or " ass " not in filters:
        pytest.skip("ffmpeg lacks flite or libass")
    d = tmp_path_factory.mktemp("sample")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_sample.py"), str(d / "podcast")], check=True)
    return d / "podcast.mp4"
