"""Automated eval harness for the 20-case set in
Reponses/Questions_for_Rag.pdf, encoded machine-runnably in
tests/eval_cases.yaml (see that file's header for the check schema and
what is/isn't reliably machine-checkable about free-text legal answers).

Runs the full pipeline for each case, grades what can be graded
automatically, dumps per-case pass/fail + full answers to
data/eval_runs/<timestamp>.md, and prints a scorecard.

Two grading tiers:
  - Retrieval checks (must_retrieve_articles) run against
    RagPipeline._ask_prompt / _compare_prompt directly — the same
    internals ask()/compare() use, including jurisdiction routing and
    scenario decomposition — so they exercise the real retrieval path
    and need no LLM at all.
  - Answer checks (forbid_regex / require_regex / require_no_citation /
    citations_from_only) need a real generated answer, so they only run
    when a real LLM is in use; with EchoLLM they're skipped, not failed.

Usage:
    python scripts/eval.py                  # full run against Ollama
    python scripts/eval.py --llm echo       # retrieval-only, CI-safe, no model
    python scripts/eval.py --case Q3        # single case
    python scripts/eval.py --case S1 --case S2
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from regintel import config
from regintel.generation.llm import EchoLLM
from regintel.generation.rag import RagPipeline

CASES_PATH = Path(__file__).resolve().parent.parent / "tests" / "eval_cases.yaml"
RUNS_DIR = config.DATA_DIR / "eval_runs"


@dataclass
class CaseResult:
    id: str
    mode: str
    language: str
    failure_mode: list[str]
    status: str  # "PASS" | "FAIL" | "SKIP"
    reasons: list[str] = field(default_factory=list)
    answer: str = ""
    sources: list[str] = field(default_factory=list)
    expected: str = ""
    fail_condition: str = ""


def _retrieved_set(pipe: RagPipeline, case: dict, top_k: int = 8):
    """The (regulation, article) pairs AND (regulation, document_label)
    pairs the real pipeline path would hand to the LLM for this case —
    via _ask_prompt/_compare_prompt directly, so jurisdiction routing and
    scenario decomposition are exercised exactly as ask()/compare() would
    use them. Document-label pairs let a case require "the Board decision
    was retrieved" separately from "statute Article N was retrieved" —
    for several cases the documented ground truth is the Board decision,
    not the statute article, once that decision is actually indexed.
    """
    if case["mode"] == "compare":
        results, _ = pipe._compare_prompt(
            case["question"], case["reg_a"], case["reg_b"], top_k_each=top_k)
    else:
        results, _ = pipe._ask_prompt(case["question"], case.get("regulation"), top_k)
    article_pairs = {(r.metadata["regulation"], str(r.metadata["article_number"]))
                     for r in results}
    doc_pairs = {(r.metadata["regulation"], r.metadata["document_label"])
                for r in results if r.metadata.get("document_label")}
    return article_pairs, doc_pairs, results


def grade_case(pipe: RagPipeline, case: dict, indexed: set[str],
               use_llm: bool) -> CaseResult:
    checks = case.get("checks") or {}
    reasons: list[str] = []

    regs_needed = {case["regulation"]} if case.get("regulation") else set()
    if case["mode"] == "compare":
        regs_needed = {case["reg_a"], case["reg_b"]}
    regs_needed |= {m["regulation"] for m in checks.get("must_retrieve_articles", [])}
    regs_needed |= {m["regulation"] for m in checks.get("must_retrieve_documents", [])}
    regs_needed -= {None}
    if regs_needed - indexed:
        return CaseResult(id=case["id"], mode=case["mode"], language=case["language"],
                          failure_mode=case.get("failure_mode", []), status="SKIP",
                          reasons=[f"not indexed on this machine: {regs_needed - indexed}"],
                          expected=case.get("expected", ""),
                          fail_condition=case.get("fail_condition", ""))

    must = checks.get("must_retrieve_articles", [])
    must_docs = checks.get("must_retrieve_documents", [])
    if must or must_docs:
        retrieved_articles, retrieved_docs, _ = _retrieved_set(pipe, case)
        for m in must:
            pair = (m["regulation"], str(m["article"]))
            if pair not in retrieved_articles:
                reasons.append(f"retrieval miss: {m['regulation']} Art {m['article']}")
        for m in must_docs:
            pair = (m["regulation"], m["document_label"])
            if pair not in retrieved_docs:
                reasons.append(f"retrieval miss: {m['regulation']} {m['document_label']}")

    answer, sources_display = "", []
    checked_anything = bool(must or must_docs)

    if use_llm:
        if case["mode"] == "compare":
            resp = pipe.compare(case["question"], case["reg_a"], case["reg_b"])
        else:
            resp = pipe.ask(case["question"], regulation=case.get("regulation"))
        answer = resp.answer
        sources_display = [s.citation for s in resp.sources]
        cited_regs = {s.metadata["regulation"] for s in resp.sources}

        for pattern in checks.get("forbid_regex", []):
            checked_anything = True
            if re.search(pattern, answer):
                reasons.append(f"forbidden pattern matched: {pattern!r}")

        req = checks.get("require_regex")
        if req:
            checked_anything = True
            if not re.search(req, answer):
                reasons.append(f"required pattern missing: {req!r}")

        if checks.get("require_no_citation"):
            checked_anything = True
            if resp.sources:
                reasons.append(f"expected no citation footer, got {sources_display}")

        allowed = checks.get("citations_from_only")
        if allowed:
            checked_anything = True
            bad = cited_regs - set(allowed)
            if bad:
                reasons.append(f"citations from disallowed instrument(s): {sorted(bad)}")

    if not checked_anything:
        status = "SKIP"
        if not reasons:
            reasons = ["no machine-checkable assertions for this case"
                      + ("" if use_llm else " under --llm echo")]
    else:
        status = "FAIL" if reasons else "PASS"

    return CaseResult(id=case["id"], mode=case["mode"], language=case["language"],
                      failure_mode=case.get("failure_mode", []), status=status,
                      reasons=reasons, answer=answer, sources=sources_display,
                      expected=case.get("expected", ""),
                      fail_condition=case.get("fail_condition", ""))


def _write_report(results: list[CaseResult], use_llm: bool) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out = RUNS_DIR / f"{datetime.now():%Y%m%d-%H%M}.md"
    counts = {"PASS": 0, "FAIL": 0, "SKIP": 0}
    for r in results:
        counts[r.status] += 1

    lines = [
        f"# RegIntel eval run — {datetime.now():%Y-%m-%d %H:%M}",
        "",
        f"LLM: {'live model' if use_llm else 'EchoLLM (retrieval checks only)'}",
        f"Scorecard: **{counts['PASS']} pass / {counts['FAIL']} fail / "
        f"{counts['SKIP']} skip** of {len(results)} run",
        "",
        "| Case | Mode | Lang | Failure mode | Status |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        fm = ", ".join(r.failure_mode) or "-"
        lines.append(f"| {r.id} | {r.mode} | {r.language} | {fm} | {r.status} |")

    lines.append("")
    for r in results:
        lines.append(f"## {r.id} — {r.status}")
        lines.append("")
        if r.reasons:
            lines.append("**Reasons:**")
            for reason in r.reasons:
                lines.append(f"- {reason}")
            lines.append("")
        if r.expected:
            lines.append(f"**Expected:** {r.expected}")
            lines.append("")
        if r.fail_condition:
            lines.append(f"**Fail condition:** {r.fail_condition}")
            lines.append("")
        if r.answer:
            lines.append("**Answer:**")
            lines.append("")
            lines.append(r.answer)
            lines.append("")
        if r.sources:
            lines.append("**Cited sources:**")
            for s in r.sources:
                lines.append(f"- {s}")
            lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", choices=["auto", "echo"], default="auto",
                    help="'echo' forces EchoLLM (retrieval checks only, "
                    "no model needed, CI-safe); 'auto' uses Ollama if "
                    "available, else falls back to EchoLLM automatically.")
    ap.add_argument("--case", action="append", dest="cases",
                    help="run only this case id (repeatable)")
    ap.add_argument("--cases", dest="cases_path", default=str(CASES_PATH),
                    help="path to a case file (default: tests/eval_cases.yaml). "
                         "Lets a separate set — e.g. tests/eval_cases_v4.yaml — "
                         "run without disturbing the original 20-case set.")
    args = ap.parse_args()

    cases_path = Path(args.cases_path)
    if not cases_path.exists():
        print(f"No such case file: {cases_path}")
        raise SystemExit(1)
    cases = yaml.safe_load(cases_path.read_text(encoding="utf-8"))["cases"]
    if args.cases:
        wanted = set(args.cases)
        cases = [c for c in cases if c["id"] in wanted]

    llm = EchoLLM() if args.llm == "echo" else None
    pipe = RagPipeline(llm=llm)
    use_llm = not isinstance(pipe.llm, EchoLLM)
    if not use_llm:
        print("Using EchoLLM — grading retrieval checks only "
              "(answer-text checks need a live model).\n")

    indexed = set(pipe.store.regulations())
    print(f"Indexed regulations: {sorted(indexed) or '(none)'}\n")
    print(f"{'ID':<6}{'STATUS':<7}reasons")
    print("-" * 70)

    results = []
    for case in cases:
        result = grade_case(pipe, case, indexed, use_llm)
        results.append(result)
        reason_str = "; ".join(result.reasons)[:100]
        print(f"{result.id:<6}{result.status:<7}{reason_str}")

    counts = {"PASS": 0, "FAIL": 0, "SKIP": 0}
    for r in results:
        counts[r.status] += 1
    print("\n" + "=" * 40)
    print(f"{counts['PASS']} pass / {counts['FAIL']} fail / {counts['SKIP']} skip "
          f"of {len(results)} run")

    out = _write_report(results, use_llm)
    print(f"\nFull report (answers, sources, reasons): {out}")


if __name__ == "__main__":
    main()
