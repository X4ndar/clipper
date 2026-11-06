"""Regenerate config.example.yaml from the built-in defaults. Run after changing config.py."""

from pathlib import Path

import yaml

from clipper.config import Config

HEADER = """# clipper config example. Copy to $CLIPPER_HOME/config.yaml (default ~/.clipper/config.yaml)
# or pass --config path. Every key is optional; these are the built-in defaults.
# Regenerate with: uv run python scripts/gen_config_example.py
#
# Precedence (highest first): CLI flags and --set key=value, then $CLIPPER_HOME/settings.json
# (written by the web UI / `clipper config import`), then this file, then built-in defaults.
#
# API keys never go here. Put ASSEMBLYAI_API_KEY and ANTHROPIC_API_KEY in .env.
#
# Notes:
# - transcribe.speech_models is always sent explicitly (plural field).
# - transcribe.speaker_labels: auto = on for podcast/interview or --speakers > 1.
# - llm.*_effort replaces temperature: current Claude models reject custom sampling settings.
# - llm budget per job = max(min_budget_usd, max_cost_usd_per_hour * source hours).
# - llm.prices_per_mtok and transcribe.*price* drive cost logs and guards only.
#   Verify them against the providers' pricing pages.
# - selection.weights: speaker_energy has no signal until audio features land (M4);
#   missing signals are skipped and the rest renormalized.
# - captions.margin_v / hook.margin_v are pixels in a 1080x1920 frame.

"""


def render() -> str:
    data = Config(home="/unused").model_dump(exclude={"home"})
    return HEADER + yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "config.example.yaml"
    out.write_text(render(), encoding="utf-8")
    print(out)
