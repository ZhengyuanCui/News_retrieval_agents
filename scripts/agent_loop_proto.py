"""Bounded agentic Q&A prototype for Spike B (#10).

Compares the current single-shot QA path against bounded `balanced` and `deep`
research loops:
  current   -> one retrieval + one synthesis call
  balanced  -> bounded planner/retrieve loop with modest search budget
  deep      -> same loop with a larger search budget

This environment uses local SQLite BM25 retrieval only. The script does not
call `init_db()` and avoids the async repository/session path entirely.
Results are written to `scripts/agent_loop_results.json` and a markdown
comparison table is printed to stdout.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import mean
from typing import Any

from news_agent.config import settings
from news_agent.models import NewsItem

RESULTS_OUT = Path(__file__).resolve().parent / "agent_loop_results.json"
DB_PATH = Path(__file__).resolve().parents[1] / "data" / "news.db"
DEFAULT_HOURS = 168
DEFAULT_SEARCH_LIMIT = 20
LLM_TIMEOUT_SECONDS = 5
LOCAL_RETRIEVAL_LABEL = "BM25-only local SQLite evidence"
_LLM_UNAVAILABLE = False

DEFAULT_QUESTIONS = [
    "What changed this week in OpenAI's model roadmap?",
    "How is Anthropic positioning Claude against OpenAI right now?",
    "What is new in AI chip competition between Nvidia, AMD, and hyperscalers?",
    "What are the biggest recent AI infrastructure cost stories?",
    "How are regulators responding to frontier-model safety concerns?",
    "What happened recently with open-weight model releases?",
    "How are Microsoft, Google, and Amazon differentiating their AI platforms?",
    "What is the latest on AI agent products and research loops?",
    "How are publishers and media companies reacting to generative AI deals?",
    "What changed in the last week around robotics plus foundation models?",
]

PLANNER_PROMPT = """\
You are designing a bounded research loop for a news QA system.

Question: "{question}"
Mode: {mode}
Iteration: {iteration}/{iterations}
Already tried queries: {tried_queries}
Current evidence titles:
{evidence_titles}

Return ONLY valid JSON with:
- "next_queries": array of up to {branch_limit} focused follow-up searches
- "stop": boolean
- "reason": short string

Rules:
- Use distinct searches that cover different facets or missing evidence.
- Prefer fewer queries unless the question is clearly multi-faceted.
- Avoid repeating searches already tried.
- If the current evidence is already sufficient, set "stop": true.
"""

JUDGE_PROMPT = """\
You are grading answers to a news question.

Question: "{question}"

Score each answer from 1 to 5 for usefulness, factual grounding, and coverage.
Prefer answers that are direct, well-supported, and avoid speculation.

Return ONLY valid JSON with this shape:
{{
  "scores": {{
    "current": {{"score": 1, "reason": "..."}},
    "balanced": {{"score": 1, "reason": "..."}},
    "deep": {{"score": 1, "reason": "..."}}
  }},
  "winner": "current|balanced|deep|tie",
  "summary": "..."
}}

Answers:
CURRENT:
{current_answer}

BALANCED:
{balanced_answer}

DEEP:
{deep_answer}
"""

QA_PROMPT = """\
You are a knowledgeable analyst. A user has asked the following question:

"{question}"

Use ONLY the news articles below to answer. Do not invent facts.

Write ONLY:
Line 1: A direct one-sentence answer to the question (max 25 words).
Lines 2 onward: 3–5 bullet points, each on its own line, starting with a bold source label like \
"**Reuters:**" followed by 1–2 sentences from that article that support or nuance the answer.
End with one sentence summarising the overall picture if the evidence is mixed.

If the articles do not contain enough information to answer, say so plainly in line 1.

News articles ({n} total):
{items_text}
"""


@dataclass(frozen=True)
class LoopMode:
    name: str
    iterations: int
    branch_limit: int
    search_limit: int
    final_limit: int


@dataclass(frozen=True)
class RetrievalBackend:
    label: str
    db_path: Path


MODES = {
    "balanced": LoopMode("balanced", iterations=3, branch_limit=2, search_limit=12, final_limit=24),
    "deep": LoopMode("deep", iterations=6, branch_limit=3, search_limit=14, final_limit=36),
}


def _litellm_module() -> Any:
    import litellm

    return litellm


def _main_model() -> str:
    if settings.claude_model:
        return f"anthropic/{settings.claude_model}"
    return settings.llm_model


def _main_api_key() -> str | None:
    return settings.llm_api_key or settings.anthropic_api_key or None


def _key_for_model(model: str, explicit_key: str = "") -> str | None:
    if explicit_key:
        return explicit_key
    m = model.lower()
    if m.startswith("anthropic/"):
        return settings.anthropic_api_key or settings.llm_api_key or None
    if m.startswith("gemini/") or m.startswith("google/"):
        return settings.gemini_api_key or None
    if m.startswith("openai/"):
        return settings.openai_api_key or None
    return settings.llm_api_key or None


def _analysis_key() -> str | None:
    return _key_for_model(settings.analysis_model, settings.analysis_api_key)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions-file", type=Path, default=None)
    parser.add_argument("--hours", type=float, default=DEFAULT_HOURS)
    parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT)
    parser.add_argument("--output", type=Path, default=RESULTS_OUT)
    parser.add_argument("--live-llm", action="store_true", help="Use live LLM calls instead of local fallbacks")
    return parser.parse_args()


def _messages_cost(model: str, messages: list[dict], completion_text: str) -> float:
    litellm = _litellm_module()
    try:
        return float(
            litellm.completion_cost(
                model=model,
                messages=messages,
                completion=completion_text,
                call_type="acompletion",
            )
        )
    except Exception:
        return 0.0


def _parse_json_object(text: str) -> dict:
    raw = (text or "").strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = next((part for part in parts if "{" in part), raw).strip()
        if raw.startswith("json"):
            raw = raw[4:].strip()
    if raw and raw[0] != "{":
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            raw = raw[start:end + 1]
    return json.loads(raw)


def _parse_datetime(value: str | None) -> datetime:
    if not value:
        return datetime.now(UTC).replace(tzinfo=None)
    raw = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return datetime.now(UTC).replace(tzinfo=None)
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone().replace(tzinfo=None)
    return parsed


def _fallback_plan(question: str) -> dict[str, Any]:
    words = [w.strip(" ?.,:;!").lower() for w in question.split()]
    candidates = []
    if len(words) >= 3:
        candidates.append(" ".join(words[:3]))
    if len(words) >= 6:
        candidates.append(" ".join(words[-3:]))
    deduped = []
    for query in candidates:
        if query and query not in deduped:
            deduped.append(query)
    return {"next_queries": deduped[:2], "stop": False, "reason": "local fallback planner"}


def _fallback_answer(question: str, items: list[NewsItem]) -> str:
    if not items:
        return "No relevant news articles found to answer this question."
    lines = [
        f"BM25-only local prototype answer for: {question}",
        "",
        f"Evidence reviewed: {len(items)} articles.",
    ]
    for item in items[: min(5, len(items))]:
        summary = (item.summary or item.content or "").replace("\n", " ").strip()
        if len(summary) > 180:
            summary = summary[:177].rstrip() + "..."
        lines.append(
            f"- {item.title} ({item.source}, {item.published_at.strftime('%b %d')}): {summary}"
        )
    return "\n".join(lines)


def _fallback_judge(variants: dict[str, dict]) -> dict[str, Any]:
    def score_variant(name: str) -> tuple[int, str]:
        data = variants[name]
        sources = len(data["items"])
        queries = len(data["queries"])
        score = min(5, max(1, 1 + min(3, sources // 4) + min(1, max(0, queries - 1))))
        return score, f"Local heuristic judge using BM25 evidence breadth ({sources} sources, {queries} queries)."

    scores = {}
    ranked: list[tuple[int, int, str]] = []
    for name in ("current", "balanced", "deep"):
        score, reason = score_variant(name)
        scores[name] = {"score": score, "reason": reason}
        ranked.append((score, len(variants[name]["items"]), name))
    ranked.sort(reverse=True)
    winner = "tie" if len(ranked) > 1 and ranked[0][:2] == ranked[1][:2] else ranked[0][2]
    return {
        "scores": scores,
        "winner": winner,
        "summary": "BM25-only local heuristic judge used because model judging was unavailable or timed out.",
    }


def _combine_cost_status(current: str, new: str) -> str:
    if current == new:
        return current
    if "partial" in {current, new}:
        return "partial"
    if "measured" in {current, new} and "fallback" in {current, new}:
        return "partial"
    if current == "not_needed":
        return new
    if new == "not_needed":
        return current
    return new


async def _call_json(
    model: str,
    api_key: str | None,
    prompt: str,
    *,
    fallback_payload: dict[str, Any] | None = None,
) -> tuple[dict, float, str]:
    global _LLM_UNAVAILABLE
    if _LLM_UNAVAILABLE or not settings.__dict__.get("_agent_loop_live_llm", False):
        return fallback_payload or _fallback_plan(prompt), 0.0, "fallback"
    litellm = _litellm_module()
    base_messages = [{"role": "user", "content": prompt}]
    for attempt in range(2):
        messages = list(base_messages)
        if attempt:
            messages.append(
                {
                    "role": "user",
                    "content": "Your previous reply was not valid JSON. Return only one valid JSON object.",
                }
            )
        try:
            response = await asyncio.wait_for(
                litellm.acompletion(
                    model=model,
                    messages=messages,
                    max_tokens=600,
                    temperature=0,
                    response_format={"type": "json_object"},
                    **({"api_key": api_key} if api_key else {}),
                ),
                timeout=LLM_TIMEOUT_SECONDS,
            )
        except Exception:
            _LLM_UNAVAILABLE = True
            break
        content = response.choices[0].message.content or ""
        try:
            return _parse_json_object(content), _messages_cost(model, messages, content), "measured"
        except json.JSONDecodeError:
            continue
    return fallback_payload or _fallback_plan(prompt), 0.0, "fallback"


async def _call_text(
    model: str,
    api_key: str | None,
    prompt: str,
    *,
    max_tokens: int = 900,
) -> tuple[str | None, float, str]:
    global _LLM_UNAVAILABLE
    if _LLM_UNAVAILABLE or not settings.__dict__.get("_agent_loop_live_llm", False):
        return None, 0.0, "fallback"
    litellm = _litellm_module()
    messages = [{"role": "user", "content": prompt}]
    try:
        response = await asyncio.wait_for(
            litellm.acompletion(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.2,
                **({"api_key": api_key} if api_key else {}),
            ),
            timeout=LLM_TIMEOUT_SECONDS,
        )
    except Exception:
        _LLM_UNAVAILABLE = True
        return None, 0.0, "fallback"
    content = response.choices[0].message.content or ""
    return content, _messages_cost(model, messages, content), "measured"


def _round_cost(cost: float) -> float:
    return round(cost, 6)


def _format_cost(cost: float, status: str) -> str:
    suffix = "" if status == "measured" else f" [{status}]"
    return f"${cost:.4f}{suffix}"


def _average_cost(results: list[dict], name: str) -> float:
    return mean(row["variants"][name]["cost_usd"] for row in results)


def _items_text(items: list[NewsItem]) -> str:
    return "\n\n".join(
        (
            f"[{idx}] {item.title}\n"
            f"Source: {item.source} | {item.published_at.strftime('%b %d') if item.published_at else ''}\n"
            f"{item.summary or (item.content or '')[:300]}"
        )
        for idx, item in enumerate(items, start=1)
    )


def _dedupe_items(items: list[NewsItem]) -> list[NewsItem]:
    seen_ids: set[str] = set()
    deduped: list[NewsItem] = []
    for item in items:
        if item.id in seen_ids:
            continue
        seen_ids.add(item.id)
        deduped.append(item)
    return deduped


def _fts_escape(query: str) -> str:
    words = re.findall(r"\w+", query)
    return " ".join(f'"{w}"' for w in words) if words else '""'


async def _retrieve_items(
    backend: RetrievalBackend,
    query: str,
    *,
    hours: float,
    limit: int,
) -> list[NewsItem]:
    if not backend.db_path.exists():
        return []
    cutoff = datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=hours)
    conn = sqlite3.connect(str(backend.db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT ni.id, ni.source, ni.topic, ni.title, ni.url, ni.content, ni.author,
                   ni.published_at, ni.raw_score, ni.fetched_at, ni.summary, ni.tags,
                   ni.sentiment, ni.relevance_score, ni.key_entities, ni.is_duplicate,
                   ni.duplicate_of, ni.cluster_id, ni.language
            FROM news_items_fts fts
            JOIN news_items ni ON ni.id = fts.id
            WHERE news_items_fts MATCH ?
              AND ni.published_at >= ?
              AND COALESCE(ni.is_duplicate, 0) = 0
            ORDER BY bm25(news_items_fts), ni.published_at DESC
            LIMIT ?
            """,
            (_fts_escape(query), cutoff.isoformat(sep=" "), limit * 2),
        ).fetchall()
    finally:
        conn.close()

    items: list[NewsItem] = []
    for row in rows:
        items.append(
            NewsItem(
                source=row["source"],
                topic=row["topic"],
                title=row["title"],
                url=row["url"],
                content=row["content"] or "",
                author=row["author"],
                published_at=_parse_datetime(row["published_at"]),
                raw_score=row["raw_score"] or 0.0,
                fetched_at=_parse_datetime(row["fetched_at"]),
                summary=row["summary"],
                tags=json.loads(row["tags"]) if row["tags"] else [],
                sentiment=row["sentiment"],
                relevance_score=row["relevance_score"],
                key_entities=json.loads(row["key_entities"]) if row["key_entities"] else [],
                is_duplicate=bool(row["is_duplicate"]),
                duplicate_of=row["duplicate_of"],
                cluster_id=row["cluster_id"],
                language=row["language"] or "en",
            )
        )
    return items[:limit]


async def _single_shot_answer(
    backend: RetrievalBackend,
    question: str,
    hours: float,
    limit: int,
) -> tuple[dict, float, str]:
    items = await _retrieve_items(backend, question, hours=max(hours, DEFAULT_HOURS), limit=limit)
    top_items = items[:20]
    if not top_items:
        return {
            "answer": "No relevant news articles found to answer this question.",
            "items": [],
            "queries": [question],
        }, 0.0, "not_needed"
    prompt = QA_PROMPT.format(question=question, n=len(top_items), items_text=_items_text(top_items))
    answer, cost, cost_status = await _call_text(_main_model(), _main_api_key(), prompt, max_tokens=800)
    return {"answer": answer or _fallback_answer(question, top_items), "items": top_items, "queries": [question]}, cost, cost_status


async def _loop_answer(
    backend: RetrievalBackend,
    question: str,
    hours: float,
    mode: LoopMode,
) -> tuple[dict, float, str]:
    planner_model = settings.analysis_model
    planner_key = _analysis_key()
    total_cost = 0.0
    cost_status = "not_needed"
    tried_queries = [question]
    collected = await _retrieve_items(
        backend,
        question,
        hours=max(hours, DEFAULT_HOURS),
        limit=mode.search_limit,
    )

    for iteration in range(1, mode.iterations + 1):
        evidence_titles = "\n".join(f"- {item.title}" for item in collected[:12]) or "- none yet"
        plan, step_cost, step_status = await _call_json(
            planner_model,
            planner_key,
            PLANNER_PROMPT.format(
                question=question,
                mode=mode.name,
                iteration=iteration,
                iterations=mode.iterations,
                tried_queries=json.dumps(tried_queries),
                evidence_titles=evidence_titles,
                branch_limit=mode.branch_limit,
            ),
            fallback_payload=_fallback_plan(question),
        )
        total_cost += step_cost
        cost_status = _combine_cost_status(cost_status, step_status)
        next_queries = [
            q.strip() for q in plan.get("next_queries", [])
            if isinstance(q, str) and q.strip() and q.strip() not in tried_queries
        ][:mode.branch_limit]
        if plan.get("stop") or not next_queries:
            break
        for subquery in next_queries:
            tried_queries.append(subquery)
            collected.extend(await _retrieve_items(
                backend,
                subquery,
                hours=max(hours, DEFAULT_HOURS),
                limit=mode.search_limit,
            ))
        collected = _dedupe_items(collected)

    final_items = collected[:mode.final_limit]
    if not final_items:
        return {
            "answer": "No relevant news articles found to answer this question.",
            "items": [],
            "queries": tried_queries,
        }, total_cost, cost_status

    prompt = QA_PROMPT.format(question=question, n=len(final_items), items_text=_items_text(final_items))
    answer, answer_cost, answer_status = await _call_text(_main_model(), _main_api_key(), prompt, max_tokens=1000)
    total_cost += answer_cost
    cost_status = _combine_cost_status(cost_status, answer_status)
    return {"answer": answer or _fallback_answer(question, final_items), "items": final_items, "queries": tried_queries}, total_cost, cost_status


async def evaluate_question(backend: RetrievalBackend, question: str, hours: float, limit: int) -> dict:
    current, current_cost, current_cost_status = await _single_shot_answer(backend, question, hours, limit)
    balanced, balanced_cost, balanced_cost_status = await _loop_answer(backend, question, hours, MODES["balanced"])
    deep, deep_cost, deep_cost_status = await _loop_answer(backend, question, hours, MODES["deep"])
    variants = {"current": current, "balanced": balanced, "deep": deep}

    judge_payload, judge_cost, judge_cost_status = await _call_json(
        settings.analysis_model,
        _analysis_key(),
        JUDGE_PROMPT.format(
            question=question,
            current_answer=current["answer"],
            balanced_answer=balanced["answer"],
            deep_answer=deep["answer"],
        ),
        fallback_payload={},
    )
    if set(judge_payload.get("scores", {}).keys()) != {"current", "balanced", "deep"}:
        judge_payload = _fallback_judge(variants)
        judge_cost = 0.0
        judge_cost_status = "fallback"

    return {
        "question": question,
        "evidence_mode": backend.label,
        "variants": {
            "current": {
                "queries": current["queries"],
                "sources": len(current["items"]),
                "cost_usd": _round_cost(current_cost),
                "cost_status": current_cost_status,
                "answer": current["answer"],
                **judge_payload["scores"]["current"],
            },
            "balanced": {
                "queries": balanced["queries"],
                "sources": len(balanced["items"]),
                "cost_usd": _round_cost(balanced_cost),
                "cost_status": balanced_cost_status,
                "answer": balanced["answer"],
                **judge_payload["scores"]["balanced"],
            },
            "deep": {
                "queries": deep["queries"],
                "sources": len(deep["items"]),
                "cost_usd": _round_cost(deep_cost),
                "cost_status": deep_cost_status,
                "answer": deep["answer"],
                **judge_payload["scores"]["deep"],
            },
        },
        "judge_cost_usd": _round_cost(judge_cost),
        "judge_cost_status": judge_cost_status,
        "winner": judge_payload["winner"],
        "judge_summary": judge_payload["summary"],
    }


def summarize(results: list[dict]) -> dict:
    wins = defaultdict(int)
    for row in results:
        wins[row["winner"]] += 1

    def avg_score(name: str) -> float:
        return mean(row["variants"][name]["score"] for row in results)

    summary = {
        "wins": dict(wins),
        "avg_scores": {name: round(avg_score(name), 3) for name in ("current", "balanced", "deep")},
        "avg_cost_usd": {name: _round_cost(_average_cost(results, name)) for name in ("current", "balanced", "deep")},
        "cost_status": {
            name: (
                "partial"
                if any(row["variants"][name]["cost_status"] == "partial" for row in results)
                else "fallback"
                if any(row["variants"][name]["cost_status"] == "fallback" for row in results)
                else "not_needed"
                if all(row["variants"][name]["cost_status"] == "not_needed" for row in results)
                else "measured"
            )
            for name in ("current", "balanced", "deep")
        },
        "evidence_mode": LOCAL_RETRIEVAL_LABEL,
        "llm_mode": "live" if settings.__dict__.get("_agent_loop_live_llm", False) else "fallback",
    }
    balanced_wins = wins.get("balanced", 0)
    balanced_cost = summary["avg_cost_usd"]["balanced"]
    if summary["cost_status"]["balanced"] != "measured":
        summary["recommendation"] = "quality-only result; rerun with --live-llm for cost-based decision"
    elif balanced_wins >= 6 and balanced_cost <= 0.05:
        summary["recommendation"] = "ship balanced (BM25-only evidence in this environment)"
    elif balanced_wins < 4 or balanced_cost > 0.15:
        summary["recommendation"] = "kill full loop; consider only query expansion (BM25-only evidence in this environment)"
    else:
        summary["recommendation"] = "ship only after tighter budgets or one-depth simplification (BM25-only evidence in this environment)"
    return summary


def markdown_table(results: list[dict], summary: dict) -> str:
    lines = [
        f"Evidence mode: {summary['evidence_mode']}",
        "",
        "| Question | Current | Balanced | Deep | Winner |",
        "|---|---:|---:|---:|---|",
    ]
    for row in results:
        lines.append(
            f"| {row['question']} | "
            f"{row['variants']['current']['score']}/5 ({_format_cost(row['variants']['current']['cost_usd'], row['variants']['current']['cost_status'])}) | "
            f"{row['variants']['balanced']['score']}/5 ({_format_cost(row['variants']['balanced']['cost_usd'], row['variants']['balanced']['cost_status'])}) | "
            f"{row['variants']['deep']['score']}/5 ({_format_cost(row['variants']['deep']['cost_usd'], row['variants']['deep']['cost_status'])}) | "
            f"{row['winner']} |"
        )
    lines.extend(
        [
            "",
            f"LLM mode: {summary['llm_mode']}",
            f"Average scores: current={summary['avg_scores']['current']}, balanced={summary['avg_scores']['balanced']}, deep={summary['avg_scores']['deep']}",
            f"Average cost/question: current={_format_cost(summary['avg_cost_usd']['current'], summary['cost_status']['current'])}, balanced={_format_cost(summary['avg_cost_usd']['balanced'], summary['cost_status']['balanced'])}, deep={_format_cost(summary['avg_cost_usd']['deep'], summary['cost_status']['deep'])}",
            f"Recommendation: {summary['recommendation']}",
        ]
    )
    return "\n".join(lines)


def load_questions(path: Path | None) -> list[str]:
    if path is None:
        return DEFAULT_QUESTIONS
    data = json.loads(path.read_text())
    if not isinstance(data, list) or not all(isinstance(q, str) for q in data):
        raise ValueError("questions file must be a JSON array of strings")
    return data


async def main() -> None:
    args = parse_args()
    setattr(settings, "_agent_loop_live_llm", bool(args.live_llm))
    questions = load_questions(args.questions_file)
    backend = RetrievalBackend(label=LOCAL_RETRIEVAL_LABEL, db_path=DB_PATH)
    rows = []
    for index, question in enumerate(questions, start=1):
        print(f"[{index}/{len(questions)}] {question}", flush=True)
        rows.append(await evaluate_question(backend, question, args.hours, args.limit))

    summary = summarize(rows)
    payload = {
        "questions": questions,
        "summary": summary,
        "results": rows,
        "environment_notes": [
            "Local SQLite BM25 retrieval only; async DB/session path is skipped in this environment.",
            "Semantic/vector retrieval is intentionally disabled because external model downloads are unavailable here.",
            "Planner, answer, and judge calls are bounded with timeouts and local fallbacks.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(markdown_table(rows, summary))
    print(f"\nWrote results to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
