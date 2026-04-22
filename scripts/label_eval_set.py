"""Build an LLM-labeled eval set for cross-encoder reranker benchmarking.

Produces tests/fixtures/rerank_eval.json:
  - 100 authored queries across 4 buckets (nl / entity / ticker / time)
  - Up to 40 candidates per query via BM25 over the repo's data/news.db
  - Each (query, item) labeled 0/1 by an LLM relevance judge

Run from the spike/rerank-mxbai-eval worktree:
    python3 scripts/label_eval_set.py

Env: reads the parent repo's .env so ANTHROPIC_API_KEY / LLM_MODEL etc. resolve
the same way they do for the app. This script only writes tests/fixtures/
and is read-only against data/news.db.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import re
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Load parent .env into os.environ BEFORE litellm / settings import — mirrors
# how the main app resolves config via pydantic-settings.
PARENT_ENV = Path("/mnt/c/ZhengyuanCui/Projects/News_retrieval_agents/.env")
if PARENT_ENV.exists():
    for line in PARENT_ENV.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v

import litellm  # noqa: E402

litellm.suppress_debug_info = True

DB_PATH = "/mnt/c/ZhengyuanCui/Projects/News_retrieval_agents/data/news.db"
OUT_PATH = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "rerank_eval.json"

# --- Labeler config --------------------------------------------------------
# Use Haiku for cost: ~$1/MTok input, negligible for ~4000 labels batched 10/call.
# Anthropic key comes from parent .env (ANTHROPIC_API_KEY).
LABEL_MODEL = os.environ.get("SPIKE_LABEL_MODEL", "anthropic/claude-haiku-4-5-20251001")
LABEL_API_KEY = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("LLM_API_KEY") or ""

LABEL_BATCH_SIZE = 10          # items per LLM call
LABEL_CONCURRENCY = 5          # per task spec
CANDIDATES_PER_QUERY = 40
BUDGET_HARD_STOP_USD = 2.0     # task spec
# Rough cost accounting for Haiku 4.5 (USD per M tokens, conservative guess).
COST_IN_PER_MTOK = 1.0
COST_OUT_PER_MTOK = 5.0


QUERIES: list[tuple[str, str]] = [
    # ── Bucket: nl (40 natural-language questions) ──────────────────────────
    ("What did OpenAI announce about GPT-5?", "nl"),
    ("How is Anthropic improving Claude's coding ability?", "nl"),
    ("What are the latest developments in AI safety research?", "nl"),
    ("Why did Nvidia's stock price rise recently?", "nl"),
    ("How are Federal Reserve rate decisions affecting markets?", "nl"),
    ("What is happening with the war in Ukraine?", "nl"),
    ("How is inflation affecting the US economy?", "nl"),
    ("What are researchers saying about mechanistic interpretability?", "nl"),
    ("Why are tech stocks selling off?", "nl"),
    ("How does retrieval-augmented generation work?", "nl"),
    ("What did Google DeepMind publish recently on reasoning models?", "nl"),
    ("How is the Chinese economy performing?", "nl"),
    ("What are the risks of superintelligent AI according to researchers?", "nl"),
    ("Why is Apple's iPhone demand slowing in China?", "nl"),
    ("How is Tesla handling production issues?", "nl"),
    ("What is happening with Elon Musk and Twitter/X?", "nl"),
    ("Why are oil prices rising?", "nl"),
    ("What are experts saying about a possible recession?", "nl"),
    ("How are AI chips affecting semiconductor supply?", "nl"),
    ("What is the status of US-China trade relations?", "nl"),
    ("Why did Boeing face another 737 MAX incident?", "nl"),
    ("How is climate change affecting global agriculture?", "nl"),
    ("What are the latest breakthroughs in large language models?", "nl"),
    ("Why did the cryptocurrency market crash?", "nl"),
    ("How are startups raising money in this funding environment?", "nl"),
    ("What did Andrej Karpathy say about training neural networks?", "nl"),
    ("Why is Meta investing so heavily in AI infrastructure?", "nl"),
    ("What is the current state of autonomous vehicles?", "nl"),
    ("How are layoffs reshaping the tech industry?", "nl"),
    ("What is Sam Altman's latest plan for OpenAI?", "nl"),
    ("Which AI models lead on coding benchmarks?", "nl"),
    ("What did the European Union decide about AI regulation?", "nl"),
    ("How is quantum computing progressing commercially?", "nl"),
    ("What are the implications of the latest arXiv paper on mixture of experts?", "nl"),
    ("Why did Microsoft invest in OpenAI?", "nl"),
    ("How are US banks performing this quarter?", "nl"),
    ("What does Yann LeCun think about current LLMs?", "nl"),
    ("Why is Amazon's cloud business slowing?", "nl"),
    ("What is happening with SpaceX's Starship program?", "nl"),
    ("How is generative AI changing software development?", "nl"),

    # ── Bucket: entity (30 entity / topic queries) ──────────────────────────
    ("OpenAI", "entity"),
    ("Anthropic Claude", "entity"),
    ("Google DeepMind", "entity"),
    ("Meta AI research", "entity"),
    ("Mistral AI", "entity"),
    ("Hugging Face", "entity"),
    ("Sam Altman", "entity"),
    ("Dario Amodei", "entity"),
    ("Andrej Karpathy", "entity"),
    ("Yann LeCun", "entity"),
    ("Lilian Weng", "entity"),
    ("Neel Nanda interpretability", "entity"),
    ("Federal Reserve", "entity"),
    ("European Central Bank", "entity"),
    ("Boston Dynamics", "entity"),
    ("Waymo robotaxi", "entity"),
    ("Figure humanoid robot", "entity"),
    ("Runway generative video", "entity"),
    ("Stability AI", "entity"),
    ("Cohere enterprise AI", "entity"),
    ("arXiv machine learning", "entity"),
    ("reinforcement learning from human feedback", "entity"),
    ("retrieval-augmented generation", "entity"),
    ("mechanistic interpretability", "entity"),
    ("AI alignment", "entity"),
    ("mixture of experts models", "entity"),
    ("multimodal foundation models", "entity"),
    ("open source LLM", "entity"),
    ("diffusion models", "entity"),
    ("self-driving cars", "entity"),

    # ── Bucket: ticker (20 ticker / proper-noun queries) ────────────────────
    ("NVDA earnings", "ticker"),
    ("TSLA stock", "ticker"),
    ("AAPL iPhone sales", "ticker"),
    ("MSFT cloud revenue", "ticker"),
    ("GOOGL advertising", "ticker"),
    ("AMZN AWS growth", "ticker"),
    ("META Reality Labs", "ticker"),
    ("AMD MI300 chip", "ticker"),
    ("INTC foundry", "ticker"),
    ("NFLX subscribers", "ticker"),
    ("BRK.B Warren Buffett", "ticker"),
    ("JPM Jamie Dimon", "ticker"),
    ("GS Goldman Sachs", "ticker"),
    ("BA Boeing 737 MAX", "ticker"),
    ("COIN Coinbase crypto", "ticker"),
    ("PLTR Palantir", "ticker"),
    ("SNOW Snowflake", "ticker"),
    ("ORCL Oracle cloud", "ticker"),
    ("UBER rideshare", "ticker"),
    ("DIS Disney streaming", "ticker"),

    # ── Bucket: time (10 time-scoped queries) ───────────────────────────────
    ("AI news this week", "time"),
    ("latest stock market moves today", "time"),
    ("earnings reports this quarter", "time"),
    ("tech news yesterday", "time"),
    ("AI research papers published last month", "time"),
    ("breaking news in markets today", "time"),
    ("what happened in AI last week", "time"),
    ("Federal Reserve announcement today", "time"),
    ("crypto market this week", "time"),
    ("major tech announcements this month", "time"),
]

assert len(QUERIES) == 100, f"Expected 100 queries, got {len(QUERIES)}"
assert sum(1 for _, b in QUERIES if b == "nl") == 40
assert sum(1 for _, b in QUERIES if b == "entity") == 30
assert sum(1 for _, b in QUERIES if b == "ticker") == 20
assert sum(1 for _, b in QUERIES if b == "time") == 10


# --- Candidate retrieval (BM25 via SQLite FTS5, read-only) -----------------
_FTS_TOKEN = re.compile(r"[A-Za-z0-9]+")


def fts_query(q: str) -> str:
    """Build a safe FTS5 MATCH expression from an arbitrary user query.

    Tokenizes on alphanumerics and joins terms with OR so we get recall.
    Each token is quoted to defang FTS operators.
    """
    toks = [t for t in _FTS_TOKEN.findall(q) if len(t) > 1]
    if not toks:
        return '""'
    return " OR ".join(f'"{t}"' for t in toks)


def bm25_candidates(conn: sqlite3.Connection, query: str, limit: int) -> list[dict]:
    match_q = fts_query(query)
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT n.id, n.title, n.summary, n.content, n.source, n.published_at,
                   bm25(news_items_fts) AS rank
            FROM news_items_fts f
            JOIN news_items n ON n.id = f.id
            WHERE f.news_items_fts MATCH ?
              AND n.is_duplicate = 0
            ORDER BY rank
            LIMIT ?
            """,
            (match_q, limit * 2),
        )
        rows = cur.fetchall()
    except sqlite3.OperationalError as e:
        print(f"  FTS error for {query!r}: {e}", file=sys.stderr)
        return []

    seen_titles: set[str] = set()
    out: list[dict] = []
    for rid, title, summary, content, source, published_at, _rank in rows:
        if not title:
            continue
        key = title.lower().strip()[:80]
        if key in seen_titles:
            continue
        seen_titles.add(key)
        body = summary or (content or "")[:500]
        out.append(
            {
                "item_id": rid,
                "title": title,
                "summary": body,
                "source": source,
                "published_at": published_at,
            }
        )
        if len(out) >= limit:
            break
    return out


# --- LLM labeling ----------------------------------------------------------
LABEL_PROMPT = """You are a relevance judge for a news-retrieval system. Given a user query and a list of news items (title + summary), output ONE line per item with a single digit: 1 if the item is clearly relevant to the query, 0 otherwise.

Output format: {n} lines, each line is just "0" or "1", in the same order as the items. No item numbers, no explanations, no extra text.

Query: {query}

Items:
{items_block}"""


def format_items_block(items: list[dict]) -> str:
    lines = []
    for i, it in enumerate(items, 1):
        title = it["title"].strip().replace("\n", " ")[:200]
        summary = (it.get("summary") or "").strip().replace("\n", " ")[:400]
        lines.append(f"[{i}] Title: {title}\n    Summary: {summary}")
    return "\n".join(lines)


def parse_labels(raw: str, n: int) -> list[int]:
    """Extract exactly n 0/1 labels from model output. Missing/malformed → 0."""
    digits = re.findall(r"\b[01]\b", raw)
    if len(digits) >= n:
        return [int(d) for d in digits[:n]]
    # Fallback: pad with zeros
    return [int(d) for d in digits] + [0] * (n - len(digits))


# Running cost tally (approximate)
_cost_lock = asyncio.Lock()
_cost_usd = 0.0
_tokens_in = 0
_tokens_out = 0


async def label_batch(
    sem: asyncio.Semaphore,
    query: str,
    items: list[dict],
    attempt: int = 0,
) -> list[int]:
    global _cost_usd, _tokens_in, _tokens_out
    prompt = LABEL_PROMPT.format(
        query=query,
        n=len(items),
        items_block=format_items_block(items),
    )

    async with sem:
        try:
            resp = await litellm.acompletion(
                model=LABEL_MODEL,
                max_tokens=100,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
                **({"api_key": LABEL_API_KEY} if LABEL_API_KEY else {}),
            )
        except Exception as e:
            if attempt == 0:
                await asyncio.sleep(1.0)
                return await label_batch(sem, query, items, attempt=1)
            print(f"  LABEL FAIL ({type(e).__name__}): {query[:40]!r} → all 0", file=sys.stderr)
            return [0] * len(items)

    raw = (resp.choices[0].message.content or "").strip()
    labels = parse_labels(raw, len(items))

    # Rough cost tracking.
    try:
        ti = getattr(resp.usage, "prompt_tokens", 0) or 0
        to = getattr(resp.usage, "completion_tokens", 0) or 0
    except Exception:
        ti, to = 0, 0
    async with _cost_lock:
        _tokens_in += ti
        _tokens_out += to
        _cost_usd += (ti / 1_000_000) * COST_IN_PER_MTOK + (to / 1_000_000) * COST_OUT_PER_MTOK

    return labels


async def label_query(
    sem: asyncio.Semaphore,
    query: str,
    candidates: list[dict],
) -> list[int]:
    """Label all candidates for a query by batching LABEL_BATCH_SIZE at a time."""
    all_labels: list[int] = []
    for i in range(0, len(candidates), LABEL_BATCH_SIZE):
        batch = candidates[i : i + LABEL_BATCH_SIZE]
        if _cost_usd > BUDGET_HARD_STOP_USD:
            all_labels.extend([0] * len(batch))
            continue
        labs = await label_batch(sem, query, batch)
        all_labels.extend(labs)
    return all_labels


# --- Driver ---------------------------------------------------------------
async def main() -> None:
    if not LABEL_API_KEY:
        print("ABORT: no ANTHROPIC_API_KEY / LLM_API_KEY in env", file=sys.stderr)
        sys.exit(2)

    print(f"labeler model: {LABEL_MODEL}")
    print(f"db: {DB_PATH}")
    print(f"out: {OUT_PATH}")

    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)

    # Step 1: gather candidates for every query (sync, fast).
    queries_data: list[dict] = []
    missing = 0
    rng = random.Random(17)
    for q, bucket in QUERIES:
        cands = bm25_candidates(conn, q, CANDIDATES_PER_QUERY)
        if len(cands) < CANDIDATES_PER_QUERY:
            # Pad from a larger random sample if BM25 didn't yield enough, so
            # every query has exactly 40 candidates (mix of relevant + noise).
            needed = CANDIDATES_PER_QUERY - len(cands)
            cur = conn.cursor()
            cur.execute(
                """
                SELECT id, title, summary, content, source, published_at
                FROM news_items
                WHERE is_duplicate = 0 AND title IS NOT NULL
                ORDER BY RANDOM()
                LIMIT ?
                """,
                (needed * 3,),
            )
            seen = {c["item_id"] for c in cands}
            for rid, title, summary, content, source, published_at in cur.fetchall():
                if rid in seen or not title:
                    continue
                seen.add(rid)
                cands.append(
                    {
                        "item_id": rid,
                        "title": title,
                        "summary": summary or (content or "")[:500],
                        "source": source,
                        "published_at": published_at,
                    }
                )
                if len(cands) >= CANDIDATES_PER_QUERY:
                    break
            if len(cands) < CANDIDATES_PER_QUERY:
                missing += 1
        queries_data.append({"query": q, "bucket": bucket, "candidates": cands[:CANDIDATES_PER_QUERY]})

    total_pairs = sum(len(qd["candidates"]) for qd in queries_data)
    print(f"queries: {len(queries_data)}, total (q,item) pairs: {total_pairs}, queries short on candidates (padded): {missing}")

    # Step 2: LLM label all pairs.
    sem = asyncio.Semaphore(LABEL_CONCURRENCY)
    t0 = time.perf_counter()

    async def process_one(qd):
        labs = await label_query(sem, qd["query"], qd["candidates"])
        for c, lab in zip(qd["candidates"], labs):
            c["label"] = int(lab)
        return qd

    done = 0
    tasks = [asyncio.create_task(process_one(qd)) for qd in queries_data]
    for fut in asyncio.as_completed(tasks):
        await fut
        done += 1
        if done % 10 == 0 or done == len(tasks):
            elapsed = time.perf_counter() - t0
            print(
                f"  [{done}/{len(tasks)}] elapsed={elapsed:.1f}s, cost~${_cost_usd:.3f}, "
                f"tokens in={_tokens_in} out={_tokens_out}"
            )

    # Step 3: compute positive rate and write out.
    labels_flat = [c["label"] for qd in queries_data for c in qd["candidates"]]
    pos_rate = sum(labels_flat) / max(1, len(labels_flat))
    print(f"label positive rate: {pos_rate:.3f} ({sum(labels_flat)}/{len(labels_flat)})")
    if pos_rate < 0.05:
        print("ABORT: positive rate below 5% floor", file=sys.stderr)
        sys.exit(3)

    out = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "llm_model": LABEL_MODEL,
            "n_queries": len(queries_data),
            "candidates_per_query": CANDIDATES_PER_QUERY,
            "corpus_source": "data/news.db",
            "corpus_size": 6571,
            "label_positive_rate": round(pos_rate, 4),
            "label_count_positive": int(sum(labels_flat)),
            "label_count_total": len(labels_flat),
            "queries_short_on_bm25_candidates_padded": missing,
            "approx_cost_usd": round(_cost_usd, 4),
            "tokens_in": _tokens_in,
            "tokens_out": _tokens_out,
            "notes": (
                "Queries authored by spike operator. Candidates retrieved via "
                "SQLite FTS5 BM25 (title+content) from parent data/news.db. "
                "Items with insufficient BM25 hits padded with random items "
                "from the corpus; such padded items will mostly be labeled 0, "
                "which is desirable (adds negatives). "
                "Labeler: single-call-per-batch, 10 items/call, temperature=0. "
                "Parse failures on an API error are retried once, then default "
                "to label=0 for the whole batch."
            ),
        },
        "queries": queries_data,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"wrote {OUT_PATH} ({OUT_PATH.stat().st_size} bytes)")


if __name__ == "__main__":
    asyncio.run(main())
