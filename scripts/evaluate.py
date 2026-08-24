"""Evaluate RegIntel retrieval accuracy against the gold set.

The primary metric is retrieval quality, because it is objective: for each
question we know which article actually contains the answer, so we can check
whether the system retrieved it. This does not need an LLM judge and is fully
reproducible.

Metrics reported:
  Hit@k  : fraction of questions where a correct article was in the top-k
           retrieved chunks (the answer was retrievable).
  MRR    : Mean Reciprocal Rank — 1/rank of the first correct article.
           Rewards ranking the right article higher.
  Also broken down by regulation and by language (EN vs TR), which shows
  how well cross-language query rewriting works.

Usage:
  python scripts/evaluate.py                 # uses tests/eval_set.json, top_k=6
  python scripts/evaluate.py --top-k 10
  python scripts/evaluate.py --answers       # also generate answers (needs Ollama)
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from regintel import config
from regintel.generation.rag import RagPipeline

EVAL_SET = Path(__file__).resolve().parent.parent / "tests" / "eval_set.json"
REPORT_DIR = config.DATA_DIR / "eval"


def article_hit_rank(results, expected: list[str]) -> int | None:
    """Rank (1-based) of the first retrieved chunk whose article is expected."""
    for i, r in enumerate(results, 1):
        if str(r.metadata.get("article_number")) in expected:
            return i
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=config.DEFAULT_TOP_K)
    ap.add_argument("--answers", action="store_true",
                    help="also generate answers with the LLM (slower)")
    args = ap.parse_args()

    data = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    items = data["items"]
    pipe = RagPipeline()
    indexed = set(pipe.store.regulations())

    rows = []
    by_reg = defaultdict(lambda: [0, 0])   # reg -> [hits, total]
    by_lang = defaultdict(lambda: [0, 0])  # lang -> [hits, total]
    rr_sum = 0.0
    evaluated = 0

    print(f"Evaluating {len(items)} questions at top_k={args.top_k}\n")
    print(f"{'ID':<9}{'reg':<6}{'lang':<5}{'hit':<5}{'rank':<5}question")
    print("-" * 78)

    for it in items:
        if it["regulation"] not in indexed:
            continue  # skip regs that aren't ingested on this machine
        evaluated += 1

        query = pipe._retrieval_query(it["question"])
        results = pipe.store.search(query, top_k=args.top_k,
                                    regulation=it["regulation"])
        rank = article_hit_rank(results, it["expected_articles"])
        hit = rank is not None
        rr_sum += (1.0 / rank) if hit else 0.0

        by_reg[it["regulation"]][1] += 1
        by_reg[it["regulation"]][0] += int(hit)
        by_lang[it["language"]][1] += 1
        by_lang[it["language"]][0] += int(hit)

        retrieved_articles = [str(r.metadata.get("article_number"))
                              for r in results]
        row = {
            "id": it["id"], "regulation": it["regulation"],
            "language": it["language"], "difficulty": it["difficulty"],
            "question": it["question"],
            "expected_articles": it["expected_articles"],
            "retrieved_articles": retrieved_articles,
            "hit": hit, "first_hit_rank": rank,
        }
        if args.answers:
            row["answer"] = pipe.ask(it["question"],
                                     regulation=it["regulation"]).answer

        rows.append(row)
        print(f"{it['id']:<9}{it['regulation']:<6}{it['language']:<5}"
              f"{'YES' if hit else 'no':<5}{str(rank or '-'):<5}"
              f"{it['question'][:40]}")

    if not evaluated:
        print("No evaluable questions — none of the gold regulations are "
              "indexed. Run scripts/ingest.py first.")
        return

    hit_total = sum(v[0] for v in by_reg.values())
    print("\n" + "=" * 40)
    print(f"Overall Hit@{args.top_k}: {hit_total}/{evaluated} "
          f"= {hit_total / evaluated:.1%}")
    print(f"MRR:              {rr_sum / evaluated:.3f}")
    print("\nBy regulation:")
    for reg, (h, t) in sorted(by_reg.items()):
        print(f"  {reg:<6} {h}/{t} = {h / t:.0%}")
    print("\nBy language (tests cross-language query rewriting):")
    for lang, (h, t) in sorted(by_lang.items()):
        print(f"  {lang:<4} {h}/{t} = {h / t:.0%}")

    # Save machine-readable report
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / f"eval-{datetime.now():%Y%m%d-%H%M}.json"
    out.write_text(json.dumps({
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "top_k": args.top_k,
        "overall_hit_rate": hit_total / evaluated,
        "mrr": rr_sum / evaluated,
        "by_regulation": {k: {"hits": v[0], "total": v[1]}
                          for k, v in by_reg.items()},
        "by_language": {k: {"hits": v[0], "total": v[1]}
                        for k, v in by_lang.items()},
        "results": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nDetailed report saved to {out}")
    print("Open it to inspect which articles were retrieved for each miss.")


if __name__ == "__main__":
    main()
