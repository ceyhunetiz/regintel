"""Schema sanity checks for tests/eval_cases.yaml — catches a malformed
case (missing question, unknown mode, duplicate id) before it silently
breaks scripts/eval.py rather than at eval-run time.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

CASES_PATH = Path(__file__).resolve().parent / "eval_cases.yaml"


def _load_cases() -> list[dict]:
    return yaml.safe_load(CASES_PATH.read_text(encoding="utf-8"))["cases"]


def test_eval_cases_cover_all_20_ids():
    ids = {c["id"] for c in _load_cases()}
    expected = {f"Q{n}" for n in range(1, 13)} | {f"S{n}" for n in range(1, 9)}
    assert ids == expected


def test_eval_case_ids_are_unique():
    ids = [c["id"] for c in _load_cases()]
    assert len(ids) == len(set(ids))


def test_eval_cases_have_required_fields():
    for c in _load_cases():
        assert c["mode"] in ("ask", "compare"), c["id"]
        assert c["language"] in ("en", "tr"), c["id"]
        assert c["question"].strip(), c["id"]
        if c["mode"] == "compare":
            assert c.get("reg_a") and c.get("reg_b"), c["id"]
        assert isinstance(c.get("failure_mode", []), list), c["id"]
        assert isinstance(c.get("checks") or {}, dict), c["id"]


def test_eval_case_forbid_regex_patterns_compile():
    import re
    for c in _load_cases():
        checks = c.get("checks") or {}
        for pattern in checks.get("forbid_regex", []):
            re.compile(pattern)  # raises re.error if malformed
        if checks.get("require_regex"):
            re.compile(checks["require_regex"])
