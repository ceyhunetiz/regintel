"""Quick CLI for testing without the UI.

Usage:
    python scripts/ask.py "What are the incident reporting deadlines?"
    python scripts/ask.py --reg DORA "What is ICT third-party risk?"
    python scripts/ask.py --compare DORA NIS2 "incident reporting"
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from regintel.generation.rag import RagPipeline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--reg", help="restrict to one regulation")
    ap.add_argument("--compare", nargs=2, metavar=("REG_A", "REG_B"))
    args = ap.parse_args()

    pipe = RagPipeline()
    if args.compare:
        resp = pipe.compare(args.question, *args.compare)
    else:
        resp = pipe.ask(args.question, regulation=args.reg)

    print("=" * 70)
    print(resp.answer)
    print("=" * 70)
    print("\nSources:")
    for s in resp.sources:
        print(f"  - {s.citation}  (score {s.score:.3f})")


if __name__ == "__main__":
    main()
