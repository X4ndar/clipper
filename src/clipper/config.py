"""Configuration.

Precedence, highest first:
  1. explicit CLI flags (and --set key=value overrides)
  2. settings saved from the web UI ($CLIPPER_HOME/settings.json)
  3. config file (--config path, else $CLIPPER_HOME/config.yaml, else ./config.yaml)
  4. built-in defaults (the Pydantic models below)

Secrets never live here: API keys come from the environment or a .env file.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

ContentType = Literal["podcast", "interview", "talking_head", "tutorial", "other"]


class TranscribeConfig(BaseModel):
    backend: Literal["assemblyai", "fixture"] = "assemblyai"
    # Always sent explicitly (plural field). The singular `speech_model` is deprecated.
    speech_models: list[str] = ["universal-3-5-pro"]
    region: Literal["us", "eu"] = "us"
    # auto: on for podcast/interview or --speakers > 1, off otherwise.
    speaker_labels: Literal["auto", "on", "off"] = "auto"
    price_per_hour_usd: float = 0.21
    diarization_price_per_hour_usd: float = 0.02
    credit_usd: float = 50.0
    credit_reserve_usd: float = 1.0
    poll_interval_s: float = 5.0
    timeout_s: float = 3600.0
    # fixture backend: path to a word-level JSON. Default: <video>.words.json next to the video.
    fixture_path: Optional[str] = None


class QualityConfig(BaseModel):
    low_confidence_threshold: float = 0.5
    max_low_confidence_share: float = 0.15
    # share of words inside an immediately repeated n-gram run (a Whisper-style loop)
    max_repetition_share: float = 0.05
    repetition_ngram: int = 4
    max_words_per_second: float = 6.5
    block_on_fail: bool = True


class LLMConfig(BaseModel):
    backend: Literal["anthropic", "fake"] = "anthropic"
    candidate_model: str = "claude-haiku-5-5"
    judge_model: str = "claude-sonnet-5-5"
    # Current models reject custom temperature; effort is the lever instead.
    candidate_effort: Literal["low", "medium", "high"] = "low"
    judge_effort: Literal["low", "medium", "high"] = "medium"
    candidate_prompt: str = "candidates_v1"
    judge_prompt: str = "judge_v1"
    max_tokens: int = 16000
    # Server-side refusal fallback for the judge model ("default" or null to disable).
    judge_fallbacks: Optional[Literal["default"]] = "default"
    chunk_token_limit: int = 60000
    chunk_overlap_tokens: int = 2000
    candidate_multiplier: float = 3.0
    context_seconds: float = 30.0
    max_boundary_shift_sentences: int = 5
    max_retries: int = 2
    judge_concurrency: int = 4
    # Per-job LLM budget = max(min_budget_usd, max_cost_usd_per_hour * source hours).
    max_cost_usd_per_hour: float = 0.75
    min_budget_usd: float = 0.25
    cache_responses: bool = True
    # USD per million tokens: [input, output]. Verify against the live pricing page.
    prices_per_mtok: dict[str, list[float]] = Field(
        default_factory=lambda: {
            "claude-haiku-5-5": [0.10, 0.50],
            "claude-sonnet-5-5": [2.00, 10.00],
            "claude-opus-5-5": [4.00, 20.00],
        }
    )


class SelectionConfig(BaseModel):
    max_clips: int = 8
    min_duration: float = 20.0
    max_duration: float = 60.0
    extended_max_duration: float = 90.0
    hard_cap_duration: float = 180.0
    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "hook": 30,
            "peak": 25,
            "standalone_sense": 20,
            "speaker_energy": 15,
            "quotability": 10,
        }
    )
    gate_min: int = 6  # standalone_sense and payoff below this are rejected outright
    min_editorial: float = 6.0
    extended_bonus_required: float = 1.0  # clips over max_duration need min_editorial + this
    min_confidence: float = 0.6  # below this the clip goes to manual review
    dedupe_iou: float = 0.5
    dedupe_shorter_overlap: float = 0.6
    dedupe_text_jaccard: float = 0.6


class SnapConfig(BaseModel):
    pre_roll_ms: float = 80
    tail_ms: float = 150
    search_ms: float = 150
    frame_ms: float = 10
    fade_ms: float = 25
    leading_fillers: list[str] = Field(
        default_factory=lambda: ["um", "uh", "erm", "hmm", "mm", "uhm", "ah", "eh"]
    )


class RenderConfig(BaseModel):
    width: int = 1080
    height: int = 1920
    # face: follow faces (needs the `reframe` extra; falls back to center with a warning). center: fixed crop.
    reframe: Literal["face", "center"] = "face"
    video_codec: str = "libx264"
    preset: str = "veryfast"
    crf: int = 20
    audio_bitrate: str = "160k"
    threads: int = 0


class ReframeConfig(BaseModel):
    sample_fps: float = 4.0  # face detection samples per second
    analysis_width: int = 1280  # frames are downscaled to this width for detection
    min_confidence: float = 0.5
    min_face_rel_size: float = 0.5  # ignore faces narrower than this share of the largest one
    dead_zone: float = 0.15  # share of the crop width the subject may drift before the crop moves
    smooth_s: float = 1.0  # pan smoothing window
    cut_threshold: float = 0.12  # mean frame difference that marks a shot cut
    min_face_in_frame: float = 0.8  # below this share of frames, flag the clip for review
    model_url: str = (
        "https://storage.googleapis.com/mediapipe-models/face_detector/"
        "blaze_face_full_range/float16/1/blaze_face_full_range.tflite"
    )
    model_sha256: str = "3698b18f063835bc609069ef052228fbe86d9c9a6dc8dcb7c7c2d69aed2b181b"
    model_path: Optional[str] = None  # use a local copy instead of downloading


class CaptionsConfig(BaseModel):
    enabled: bool = True
    mode: Literal["word", "karaoke", "plain"] = "word"
    font: str = "Noto Sans"
    rtl_font: str = "Noto Sans Arabic"
    fonts_dir: Optional[str] = None  # default: bundled assets/fonts (+ $CLIPPER_HOME/fonts)
    size: int = 80
    bold: bool = True
    uppercase: bool = False
    primary_color: str = "#FFFFFF"
    highlight_color: str = "#FFD400"
    outline_color: str = "#000000"
    outline: float = 6.0
    shadow: float = 0.0
    box: bool = False
    box_color: str = "#000000A0"
    max_chars_per_line: int = 22
    max_lines: int = 2
    # Distance from the bottom edge, in a 1920-tall frame. Keeps text above the Shorts UI.
    margin_v: int = 640
    pop: bool = True
    max_page_gap_s: float = 0.6


class HookConfig(BaseModel):
    enabled: bool = True
    duration_s: float = 3.0
    max_words: int = 8
    size: int = 88
    primary_color: str = "#FFFFFF"
    # Opaque by default: semi-transparent per-line boxes overlap and look patchy.
    box_color: str = "#000000"
    box_padding: float = 14.0
    margin_v: int = 320  # from the top edge
    max_chars_per_line: int = 18


class Config(BaseModel):
    home: str = ""
    transcribe: TranscribeConfig = Field(default_factory=TranscribeConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    snap: SnapConfig = Field(default_factory=SnapConfig)
    render: RenderConfig = Field(default_factory=RenderConfig)
    reframe: ReframeConfig = Field(default_factory=ReframeConfig)
    captions: CaptionsConfig = Field(default_factory=CaptionsConfig)
    hook: HookConfig = Field(default_factory=HookConfig)

    @field_validator("home")
    @classmethod
    def _home(cls, v: str) -> str:
        return v or str(default_home())

    @property
    def home_path(self) -> Path:
        return Path(self.home).expanduser()


def default_home() -> Path:
    return Path(os.environ.get("CLIPPER_HOME", "~/.clipper")).expanduser()


def package_root() -> Path:
    return Path(__file__).resolve().parent


def bundled_fonts_dir() -> Path:
    # Source checkout: <repo>/assets/fonts. Installed wheel: clipper/assets/fonts (if packaged).
    for cand in (package_root() / "assets" / "fonts", package_root().parents[1] / "assets" / "fonts"):
        if cand.is_dir():
            return cand
    return package_root().parents[1] / "assets" / "fonts"


def deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def parse_set_overrides(items: list[str]) -> dict:
    """Turn ["llm.judge_model=claude-opus-5-5", "captions.size=90"] into a nested dict.

    Values are parsed as YAML scalars, so numbers, booleans and null work.
    """
    out: dict = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        key, raw = item.split("=", 1)
        value = yaml.safe_load(raw) if raw != "" else ""
        node = out
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return out


def load_env(home: Path) -> None:
    # Do not override variables that are already set.
    load_dotenv(Path.cwd() / ".env", override=False)
    load_dotenv(home / ".env", override=False)


def settings_path(home: Path) -> Path:
    return home / "settings.json"


def load_layers(config_path: Optional[str] = None, home: Optional[Path] = None) -> list[tuple[str, dict]]:
    home = home or default_home()
    layers: list[tuple[str, dict]] = []
    candidates = [Path(config_path)] if config_path else [home / "config.yaml", Path.cwd() / "config.yaml"]
    for p in candidates:
        if p.is_file():
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            if not isinstance(data, dict):
                raise ValueError(f"{p}: config must be a mapping")
            layers.append((str(p), data))
            break
    else:
        if config_path:
            raise FileNotFoundError(config_path)
    sp = settings_path(home)
    if sp.is_file():
        layers.append((str(sp), json.loads(sp.read_text(encoding="utf-8")) or {}))
    return layers


def load_config(
    config_path: Optional[str] = None,
    cli_overrides: Optional[dict] = None,
    home: Optional[Path] = None,
) -> Config:
    home = home or default_home()
    merged: dict[str, Any] = {}
    for _, data in load_layers(config_path, home):
        merged = deep_merge(merged, data)
    if cli_overrides:
        merged = deep_merge(merged, cli_overrides)
    merged.setdefault("home", str(home))
    cfg = Config.model_validate(merged)
    load_env(cfg.home_path)
    return cfg


SECRET_KEYS = ("ASSEMBLYAI_API_KEY", "ANTHROPIC_API_KEY")


def export_settings(cfg: Config, include_secrets: bool = False) -> dict:
    data = cfg.model_dump(exclude={"home"})
    if include_secrets:
        data["_secrets"] = {k: os.environ.get(k) for k in SECRET_KEYS if os.environ.get(k)}
    return data
