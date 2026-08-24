"""Citation-integrity / jurisdiction-integrity regression harness.

Complements scripts/evaluate.py (which measures retrieval Hit@k/MRR only).
This harness checks answer-level behaviour: did retrieval reach the right
article, does every [n] marker resolve, are citations confined to the
right instrument, and — for false-premise / existence questions — does
the answer say roughly the right thing (best-effort keyword check; exact
wording isn't reliably assertable against a local LLM's free-text output,
so treat these as advisory, not as strict as the retrieval/citation checks).

Every case runs --runs times (default 3). Divergence in which articles
got cited across runs is reported as a failure — for an audit-facing
tool, run-to-run stability matters as much as any single run being right.

Usage:
    python scripts/eval_regression.py
    python scripts/eval_regression.py --runs 1        # fast smoke check
    python scripts/eval_regression.py --case p0-1     # single case
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from regintel.generation.rag import RagPipeline

CASES_PATH = Path(__file__).resolve().parent.parent / "tests" / "regression_set.json"


def run_once(pipe: RagPipeline, case: dict):
    """Returns (answer, cited_pairs, retrieved_pairs) for one run.
    cited_pairs / retrieved_pairs are sets of (regulation, article_number).
    """
    if case["mode"] == "compare":
        resp = pipe.compare(case["question"], case["reg_a"], case["reg_b"])
        query = pipe._retrieval_query(case["question"])
        retrieved = (pipe.store.search(query, top_k=8, regulation=case["reg_a"]) +
                    pipe.store.search(query, top_k=8, regulation=case["reg_b"]))
    else:
        reg = case.get("regulation")
        resp = pipe.ask(case["question"], regulation=reg, top_k=case.get("top_k", 8))
        query = pipe._retrieval_query(case["question"])
        retrieved = pipe.store.search(query, top_k=case.get("top_k", 8), regulation=reg)

    cited_pairs = {(s.metadata["regulation"], s.metadata["article_number"])
                  for s in resp.sources}
    retrieved_pairs = {(r.metadata["regulation"], r.metadata["article_number"])
                       for r in retrieved}
    return resp, cited_pairs, retrieved_pairs


def check_run(case: dict, resp, cited_pairs: set, retrieved_pairs: set) -> list[str]:
    """Returns a list of failure reasons for one run (empty = pass)."""
    fails = []
    answer_lower = resp.answer.lower()

    for req in case.get("must_retrieve_articles", []):
        if (req["regulation"], req["article"]) not in retrieved_pairs:
            fails.append(f"retrieval miss: {req['regulation']} Art {req['article']}")

    if case.get("zero_citations") and cited_pairs:
        fails.append(f"expected zero citations, got {sorted(cited_pairs)}")

    allowed = case.get("citations_from_only")
    if allowed:
        bad = {reg for reg, _ in cited_pairs if reg not in allowed}
        if bad:
            fails.append(f"citations from disallowed instrument(s): {bad}")

    contains_any = case.get("answer_contains_any")
    if contains_any and not any(kw.lower() in answer_lower for kw in contains_any):
        fails.append(f"answer missing any of expected phrases: {contains_any}")

    for kw in case.get("answer_not_contains_any", []):
        if kw.lower() in answer_lower:
            fails.append(f"answer contains forbidden phrase: {kw!r}")

    # Universal check: every [n] marker in the answer must resolve to a
    # cited source (cited_sources() should already guarantee this by
    # construction — this is a regression trip-wire, not a new check).
    markers = {int(m) for m in re.findall(r"\[(\d+)\]", resp.answer)}
    if markers and len(markers) > len(resp.cited_indices):
        fails.append(f"more markers in text ({len(markers)}) than cited sources "
                     f"({len(resp.cited_indices)}) — dangling reference leaked through")

    return fails


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--case", help="run only this case id")
    args = ap.parse_args()

    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))["cases"]
    if args.case:
        cases = [c for c in cases if c["id"] == args.case]

    pipe = RagPipeline()
    indexed = set(pipe.store.regulations())

    overall_pass = 0
    print(f"Running {len(cases)} cases x {args.runs} repeat(s)\n")

    for case in cases:
        regs_needed = {r["regulation"] for r in case.get("must_retrieve_articles", [])}
        regs_needed |= {case.get("regulation")} if case.get("regulation") else set()
        regs_needed |= {case.get("reg_a"), case.get("reg_b")} if case["mode"] == "compare" else set()
        if regs_needed - indexed - {None}:
            print(f"{case['id']:<10} SKIP — {regs_needed - indexed} not indexed")
            continue

        run_fails = []
        run_cited_sets = []
        for run_i in range(args.runs):
            resp, cited_pairs, retrieved_pairs = run_once(pipe, case)
            fails = check_run(case, resp, cited_pairs, retrieved_pairs)
            run_fails.append(fails)
            run_cited_sets.append(frozenset(cited_pairs))

        diverged = len(set(run_cited_sets)) > 1
        any_fail = any(run_fails) or diverged
        status = "FAIL" if any_fail else "PASS"
        if not any_fail:
            overall_pass += 1

        print(f"{case['id']:<10} {status}  [{case['category']}]")
        print(f"           Q: {case['question'][:80]}")
        for i, fails in enumerate(run_fails):
            if fails:
                print(f"           run {i+1}: " + "; ".join(fails))
        if diverged:
            print(f"           DIVERGED across runs — cited sets differ: "
                  f"{[sorted(s) for s in run_cited_sets]}")
        print()

    print("=" * 60)
    print(f"{overall_pass}/{len(cases)} cases passed all {args.runs} run(s)")


if __name__ == "__main__":
    main()
