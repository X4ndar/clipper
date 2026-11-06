import json

import pytest

from clipper.config import export_settings, load_config, parse_set_overrides


def test_set_parsing():
    d = parse_set_overrides(["llm.judge_effort=high", "captions.size=90", "hook.enabled=false", "x.y=null"])
    assert d == {"llm": {"judge_effort": "high"}, "captions": {"size": 90}, "hook": {"enabled": False},
                 "x": {"y": None}}
    with pytest.raises(ValueError):
        parse_set_overrides(["nokey"])


def test_precedence_cli_over_ui_over_file_over_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    assert load_config(home=home).captions.size == 80  # default
    (home / "config.yaml").write_text("captions:\n  size: 70\n  font: FileFont\nllm:\n  judge_effort: low\n")
    c = load_config(home=home)
    assert c.captions.size == 70 and c.captions.font == "FileFont"
    (home / "settings.json").write_text(json.dumps({"captions": {"size": 60}}))
    c = load_config(home=home)
    assert c.captions.size == 60 and c.captions.font == "FileFont"  # UI settings over file, merged deeply
    c = load_config(cli_overrides={"captions": {"size": 50}}, home=home)
    assert c.captions.size == 50 and c.llm.judge_effort == "low"


def test_explicit_config_path_must_exist(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(str(tmp_path / "missing.yaml"), home=tmp_path)


def test_export_hides_keys_unless_asked(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    c = load_config(home=tmp_path)
    assert "_secrets" not in export_settings(c)
    assert "sk-test" not in json.dumps(export_settings(c))
    assert export_settings(c, include_secrets=True)["_secrets"]["ANTHROPIC_API_KEY"] == "sk-test"


def test_bad_value_rejected(tmp_path):
    with pytest.raises(Exception):
        load_config(cli_overrides={"llm": {"judge_effort": "extreme"}}, home=tmp_path)


def test_config_example_matches_defaults(tmp_path):
    import yaml
    from pathlib import Path

    from clipper.config import Config

    root = Path(__file__).resolve().parents[1]
    data = yaml.safe_load((root / "config.example.yaml").read_text())
    assert data == Config(home="/x").model_dump(exclude={"home"}), "run scripts/gen_config_example.py"
