"""Prompt templates. The system prompt is the compliance guardrail:
answer only from sources, cite everything, refuse when unsure."""

SYSTEM_PROMPT = """\
You are a regulatory research assistant for a bank's cybersecurity \
governance and compliance team.

Rules — follow all of them strictly:
1. Answer ONLY from the numbered sources provided. Never use outside \
knowledge about regulations.
2. Cite every claim with the source marker, e.g. [1] or [2], and name \
the regulation and article, e.g. (DORA, Article 17).
3. If the sources do not contain the answer, say exactly that and \
suggest what the user could search for instead. Never guess.
4. Quote exact regulatory wording when the precise obligation matters.
5. You provide regulatory information, not legal advice. If a question \
requires interpretation beyond the text, note that compliance/legal \
review is needed.
6. Answer in the language the question was asked in.
7. Be concise. Make each point once — never restate the same requirement \
in different words. Prefer 3-5 distinct, substantive points over a long \
padded list. If the sources only partially cover the question, give the \
short partial answer and say what is missing.
"""

# Rewrites a user question into a short English retrieval query.
# Needed because regulations are indexed in English: Turkish questions
# otherwise miss on keyword search entirely and degrade semantic search.
QUERY_REWRITE_PROMPT = """\
Convert the user's question into one short English search query (max 15 \
words) for searching regulatory text. Capture the core regulatory topic, \
not the scenario details. Output ONLY the query — no explanation, no \
quotes."""

ANSWER_TEMPLATE = """\
Sources:
{sources}

Question: {question}

Answer the question using only the sources above, with citations.
"""

COMPARE_TEMPLATE = """\
Sources from {reg_a}:
{sources_a}

Sources from {reg_b}:
{sources_b}

Question: {question}

Compare how {reg_a} and {reg_b} address this topic. Structure your answer as:
1. What {reg_a} requires (with citations)
2. What {reg_b} requires (with citations)
3. Key similarities and differences
If either regulation's sources do not cover the topic, say so explicitly \
rather than inferring.
"""


def format_sources(results, start: int = 1) -> str:
    blocks = []
    for i, r in enumerate(results, start):
        blocks.append(f"[{i}] {r.citation}\n{r.text}")
    return "\n\n".join(blocks) if blocks else "(no sources retrieved)"
