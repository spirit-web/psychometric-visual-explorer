from utils import i18n


def test_t_returns_swedish_by_default(monkeypatch):
    state = {}
    monkeypatch.setattr(i18n.st, "session_state", state)
    assert i18n.t("Importera Tester") == "Importera Tester"


def test_t_returns_english_when_language_set(monkeypatch):
    state = {"pve_language": "en"}
    monkeypatch.setattr(i18n.st, "session_state", state)
    assert i18n.t("Importera Tester") == "Import Tests"


def test_t_falls_back_to_key_for_unknown_string(monkeypatch):
    state = {"pve_language": "en"}
    monkeypatch.setattr(i18n.st, "session_state", state)
    assert i18n.t("Some Untranslated Label") == "Some Untranslated Label"
