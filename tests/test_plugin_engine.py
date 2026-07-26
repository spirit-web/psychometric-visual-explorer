import json
from pathlib import Path

from core.plugin_engine import default_validity_evidence, load_all_plugins, load_plugin, match_plugin

PLUGINS_DIR = Path(__file__).resolve().parent.parent / "plugins"


def test_load_all_plugins_loads_the_three_bundled_tests():
    plugins = load_all_plugins(PLUGINS_DIR)
    assert set(plugins) == {"gad7", "phq9", "ipip_bigfive"}
    assert len(plugins["gad7"].items) == 7
    assert len(plugins["phq9"].items) == 9
    assert len(plugins["ipip_bigfive"].items) == 50


def test_load_plugin_returns_none_for_invalid_json(tmp_path):
    bad_file = tmp_path / "broken.json"
    bad_file.write_text("{not valid json", encoding="utf-8")
    assert load_plugin(bad_file) is None


def test_load_plugin_returns_none_for_schema_violation(tmp_path):
    bad_file = tmp_path / "incomplete.json"
    bad_file.write_text(json.dumps({"plugin_id": "x"}), encoding="utf-8")
    assert load_plugin(bad_file) is None


def test_match_plugin_identifies_gad7_from_its_own_item_ids():
    plugins = load_all_plugins(PLUGINS_DIR)
    columns = [item.id for item in plugins["gad7"].items] + ["age", "gender"]
    result = match_plugin(columns, plugins)
    assert result is not None
    questionnaire, mapping = result
    assert questionnaire.plugin_id == "gad7"
    assert len(mapping) == 7


def test_match_plugin_returns_none_for_unrelated_columns():
    plugins = load_all_plugins(PLUGINS_DIR)
    result = match_plugin(["foo", "bar", "baz"], plugins)
    assert result is None


def test_default_validity_evidence_covers_all_three_bundled_tests():
    for plugin_id in ("gad7", "phq9", "ipip_bigfive"):
        evidence = default_validity_evidence(plugin_id)
        assert set(evidence.keys()) == {"response_processes", "consequences"}
        for status, summary in evidence.values():
            assert status in {"none", "limited", "moderate", "strong"}
            assert len(summary) > 20


def test_default_validity_evidence_empty_for_custom_test():
    assert default_validity_evidence("some_custom_test") == {}
