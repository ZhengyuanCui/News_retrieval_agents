from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime

import litellm

from news_agent.config import settings
from news_agent.models import NewsItem

logger = logging.getLogger(__name__)

# Suppress litellm's verbose output
litellm.suppress_debug_info = True

ITEM_ANALYSIS_PROMPT = """\
You are a news analyst. Analyze the following {n} news items about "{topic}" and for each one provide:
1. A 2-3 sentence summary focusing on the key facts and implications
2. A relevance score (1-10) for the topic "{topic}" (10 = extremely relevant)
3. Key entities mentioned (companies, people, technologies, tickers)
4. Sentiment: "positive", "negative", or "neutral"
5. Relevant tags (3-5 short keywords)

Return ONLY a valid JSON array with exactly {n} objects, one per item, with these fields:
- "summary": string
- "relevance_score": number (1-10)
- "key_entities": array of strings
- "sentiment": "positive" | "negative" | "neutral"
- "tags": array of strings

News items:
{items_json}
"""

_QUESTION_STARTERS = (
    "what", "why", "how", "will", "would", "could", "should",
    "do ", "does", "did", "is ", "are ", "can ", "which", "who ",
    "when", "where", "might", "shall",
)

def is_question(query: str) -> bool:
    """Return True if the query looks like a natural-language question."""
    q = query.strip().lower()
    return (
        q.endswith("?")
        or any(q.startswith(s) for s in _QUESTION_STARTERS)
        or len(q.split()) >= 7        # long free-text → treat as question
    )


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

DIGEST_PROMPT = """\
You are an expert analyst writing a concise daily briefing for a professional audience.

Below are the top {n} {topic_label} news items from today.

Write ONLY the following — no preamble, no JSON, no extra text:
Line 1: one sentence headline (max 20 words) capturing the single biggest story.
Lines 2 onward: 5–8 bullet points, each on its own line, starting with a bold label like "**Adobe:**" or "**Markets:**" followed by 1–2 sentences covering a distinct development.

Example format:
Markets tumble as Fed signals further rate hikes amid persistent inflation.
**Fed Policy:** The Federal Reserve indicated another 25bps hike is likely, citing sticky core inflation above 3%.
**Tech Selloff:** Nasdaq fell 2.3% as growth stocks bore the brunt of rising yield expectations.

News items:
{items_text}
"""


def _api_key() -> str | None:
    """Return the API key for the main digest/Q&A model."""
    return settings.llm_api_key or settings.anthropic_api_key or None


def _key_for_model(model: str, explicit_key: str = "") -> str | None:
    """Return the API key to use for a given model string."""
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


def _analysis_api_key() -> str | None:
    """Return the API key for the primary analysis model."""
    return _key_for_model(settings.analysis_model, settings.analysis_api_key)


class _ModelSlot:
    """One model endpoint with its own RPM-based rate limiter.

    Multiple batches assigned to the same slot queue through its lock,
    each waiting the minimum interval before making an API call.
    This enforces the model's RPM cap regardless of how many batches
    are running concurrently.
    """
    def __init__(self, model: str, api_key: str | None, rpm: int) -> None:
        self.model = model
        self.api_key = api_key
        self.rpm = rpm
        self._interval = 60.0 / max(1, rpm)   # min seconds between calls
        self._last: float = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until this slot can accept another request within its RPM limit."""
        async with self._lock:
            loop = asyncio.get_event_loop()
            now = loop.time()
            wait = self._last + self._interval - now
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = loop.time()

    def __repr__(self) -> str:
        return f"<ModelSlot {self.model} {self.rpm}RPM>"


def _build_weighted_pool() -> list[_ModelSlot]:
    """Build a weighted list of _ModelSlot objects for batch assignment.

    Each model appears proportionally to its RPM relative to the lowest-RPM model,
    so that faster/higher-capacity models receive more batches per round.

    Example — RPMs [10, 50]:
      min=10 → model_A gets weight 1, model_B gets weight 5
      pool = [A, B, B, B, B, B]
      batch_i → pool[i % 6]: every 6 batches, A gets 1 and B gets 5.

    All slots for the same (model, key) share a single _ModelSlot instance so their
    rate-limiter lock is shared — preventing bursts even when many batches pile up.
    """
    # Collect (model, key, rpm) triples
    if settings.analysis_models:
        entries = []
        for i, model in enumerate(settings.analysis_models):
            key = settings.analysis_api_keys[i] if i < len(settings.analysis_api_keys) else ""
            rpm = settings.analysis_rpms[i] if i < len(settings.analysis_rpms) else 10
            entries.append((model, _key_for_model(model, key), rpm))
    else:
        rpm = settings.analysis_rpms[0] if settings.analysis_rpms else 10
        entries = [(settings.analysis_model, _analysis_api_key(), rpm)]

    if not entries:
        return []

    # One shared slot per unique (model, key) combination
    slot_map: dict[tuple[str, str], _ModelSlot] = {}
    for model, key, rpm in entries:
        k = (model, key or "")
        if k not in slot_map:
            slot_map[k] = _ModelSlot(model, key, rpm)

    # Weighted list: each model repeated proportionally to its RPM
    min_rpm = min(rpm for _, _, rpm in entries)
    weighted: list[_ModelSlot] = []
    for model, key, rpm in entries:
        slot = slot_map[(model, key or "")]
        count = max(1, round(rpm / min_rpm))
        weighted.extend([slot] * count)

    logger.info(
        "Analysis model pool: %s",
        ", ".join(f"{s.model}({s.rpm}RPM)" for s in dict.fromkeys(weighted)),
    )
    return weighted


def _model() -> str:
    """Return the litellm model string, falling back to deprecated claude_model if set."""
    if settings.claude_model:
        # Backward compat: bare model name like "claude-sonnet-4-6"
        return f"anthropic/{settings.claude_model}"
    return settings.llm_model


class LLMAnalyzer:
    """Provider-agnostic news analyzer powered by any LLM via litellm."""

    def _build_items_json(self, items: list[NewsItem]) -> str:
        return json.dumps(
            [{"index": i, "title": item.title, "content": item.content[:500], "source": item.source}
             for i, item in enumerate(items)],
            ensure_ascii=False,
        )

    async def analyze_batch(self, items: list[NewsItem], topic: str) -> list[NewsItem]:
        """
        Analyze a batch of NewsItems using the configured LLM.
        Fills in summary, relevance_score, key_entities, sentiment, and tags.
        Skips items that already have a summary (cached).

        Batches are processed concurrently (up to analysis_concurrency at a time)
        rather than sequentially, cutting wall-clock time by ~5×.
        """
        to_analyze = [item for item in items if not item.summary]
        if not to_analyze:
            return items

        topic_label = topic.title()
        pool = _build_weighted_pool()
        sem = asyncio.Semaphore(settings.analysis_concurrency)

        batches = [
            to_analyze[i: i + settings.batch_size]
            for i in range(0, len(to_analyze), settings.batch_size)
        ]

        async def _process_batch(batch: list[NewsItem], batch_idx: int) -> list[NewsItem]:
            prompt = ITEM_ANALYSIS_PROMPT.format(
                n=len(batch),
                topic=topic_label,
                items_json=self._build_items_json(batch),
            )
            messages = [{"role": "user", "content": prompt}]

            async with sem:
                # On rate-limit, rotate to the next model rather than retrying
                # the same exhausted quota.  Try every slot in the pool once.
                for attempt in range(len(pool)):
                    slot = pool[(batch_idx + attempt) % len(pool)]
                    await slot.acquire()  # enforce per-model RPM interval
                    try:
                        response = await litellm.acompletion(
                            model=slot.model,
                            max_tokens=4096,
                            messages=messages,
                            **({"api_key": slot.api_key} if slot.api_key else {}),
                        )
                        raw = response.choices[0].message.content.strip()

                        if "```" in raw:
                            raw = raw.split("```")[1]
                            if raw.startswith("json"):
                                raw = raw[4:]

                        analyses = json.loads(raw)
                        if not isinstance(analyses, list):
                            raise ValueError("Expected JSON array")

                        for item, analysis in zip(batch, analyses):
                            item.summary = analysis.get("summary")
                            item.relevance_score = float(analysis.get("relevance_score", 5.0))
                            item.key_entities = analysis.get("key_entities", [])
                            item.sentiment = analysis.get("sentiment", "neutral")
                            item.tags = analysis.get("tags", [])
                        return batch

                    except litellm.RateLimitError:
                        logger.warning(
                            "Rate limit on %s (attempt %d/%d) — trying next model",
                            slot.model, attempt + 1, len(pool),
                        )
                        continue
                    except json.JSONDecodeError as e:
                        logger.error("LLM returned invalid JSON: %s", e)
                        return batch
                    except litellm.APIError as e:
                        logger.error("LLM API error: %s", e)
                        return batch
                    except Exception as e:
                        logger.error("Unexpected analyzer error: %s", e)
                        return batch

                logger.error("All %d models rate-limited for batch %d — skipping", len(pool), batch_idx)
                return batch

        batch_results = await asyncio.gather(*[_process_batch(b, i) for i, b in enumerate(batches)])
        results = [item for batch in batch_results for item in batch]

        result_map = {item.id: item for item in results}
        return [result_map.get(item.id, item) for item in items]

    async def generate_digest(self, items: list[NewsItem], topic: str) -> str:
        """Generate a prose narrative digest for the given topic."""
        candidates = [i for i in items if i.topic == topic and not i.is_duplicate]
        scored = [i for i in candidates if i.relevance_score and i.relevance_score >= 5]
        pool = scored if scored else candidates
        pool.sort(key=lambda x: x.relevance_score or x.raw_score or 0, reverse=True)
        top_items = pool[:20]

        if not top_items:
            return f"No {topic} news items found for this period."

        topic_label = topic.title()
        items_text = "\n\n".join(
            f"[{i+1}] {item.title}\nSource: {item.source}\n{item.summary or item.content[:300]}"
            for i, item in enumerate(top_items)
        )
        model = _model()
        api_key = _api_key()

        for attempt in range(3):
            try:
                response = await litellm.acompletion(
                    model=model,
                    max_tokens=1500,
                    messages=[{"role": "user", "content": DIGEST_PROMPT.format(
                        n=len(top_items), topic_label=topic_label, items_text=items_text,
                    )}],
                    **({"api_key": api_key} if api_key else {}),
                )
                return response.choices[0].message.content.strip()
            except litellm.RateLimitError:
                wait = 30 * (attempt + 1)
                logger.warning("LLM rate limit on digest '%s' — waiting %ds", topic, wait)
                await asyncio.sleep(wait)
            except Exception as e:
                logger.error("Digest generation failed for topic '%s': %s", topic, e)
                return f"Digest generation failed: {e}"
        return "Digest generation failed: rate limit retries exhausted."

    async def generate_digest_stream(self, items: list[NewsItem], topic: str):
        """Stream digest generation, yielding text chunks as the LLM writes them."""
        candidates = [i for i in items if i.topic == topic and not i.is_duplicate]
        scored = [i for i in candidates if i.relevance_score and i.relevance_score >= 5]
        pool = scored if scored else candidates
        pool.sort(key=lambda x: x.relevance_score or x.raw_score or 0, reverse=True)
        top_items = pool[:20]

        if not top_items:
            yield f"No recent {topic} news found."
            return

        topic_label = topic.title()
        items_text = "\n\n".join(
            f"[{i+1}] {item.title}\nSource: {item.source}\n{item.summary or item.content[:300]}"
            for i, item in enumerate(top_items)
        )
        prompt = DIGEST_PROMPT.format(n=len(top_items), topic_label=topic_label, items_text=items_text)
        model = _model()
        api_key = _api_key()

        for attempt in range(3):
            try:
                response = await litellm.acompletion(
                    model=model,
                    max_tokens=1500,
                    messages=[{"role": "user", "content": prompt}],
                    stream=True,
                    **({"api_key": api_key} if api_key else {}),
                )
                async for chunk in response:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        yield delta
                return
            except litellm.RateLimitError:
                wait = 30 * (attempt + 1)
                logger.warning("LLM rate limit on digest stream '%s' — waiting %ds (attempt %d/3)", topic, wait, attempt + 1)
                await asyncio.sleep(wait)
            except Exception as e:
                logger.error("Digest stream failed for '%s': %s", topic, e)
                raise

    async def answer_question_stream(self, question: str, items: list[NewsItem]):
        """Stream an LLM answer to a natural-language question grounded in news articles."""
        pool = [i for i in items if not i.is_duplicate]
        pool.sort(key=lambda x: x.relevance_score or x.raw_score or 0, reverse=True)
        top_items = pool[:20]

        if not top_items:
            yield "No relevant news articles found to answer this question."
            return

        items_text = "\n\n".join(
            f"[{i+1}] {item.title}\nSource: {item.source} | {item.published_at.strftime('%b %d') if item.published_at else ''}\n"
            f"{item.summary or item.content[:300]}"
            for i, item in enumerate(top_items)
        )
        prompt = QA_PROMPT.format(question=question, n=len(top_items), items_text=items_text)
        model = _model()
        api_key = _api_key()

        for attempt in range(3):
            try:
                response = await litellm.acompletion(
                    model=model,
                    max_tokens=800,
                    messages=[{"role": "user", "content": prompt}],
                    stream=True,
                    **({"api_key": api_key} if api_key else {}),
                )
                async for chunk in response:
                    delta = chunk.choices[0].delta
                    text = getattr(delta, "content", None) or ""
                    if text:
                        yield text
                return
            except litellm.RateLimitError:
                wait = 30 * (attempt + 1)
                logger.warning("LLM rate limit on Q&A — waiting %ds", wait)
                await asyncio.sleep(wait)
            except Exception as e:
                logger.error("Q&A stream failed: %s", e)
                raise
