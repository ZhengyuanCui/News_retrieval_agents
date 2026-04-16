from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime

import anthropic

from news_agent.config import settings
from news_agent.models import NewsItem

logger = logging.getLogger(__name__)

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


class ClaudeAnalyzer:
    def __init__(self) -> None:
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self.async_client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        self.model = settings.claude_model

    def _build_items_json(self, items: list[NewsItem]) -> str:
        return json.dumps(
            [{"index": i, "title": item.title, "content": item.content[:500], "source": item.source}
             for i, item in enumerate(items)],
            ensure_ascii=False,
        )

    async def analyze_batch(self, items: list[NewsItem], topic: str) -> list[NewsItem]:
        """
        Analyze a batch of NewsItems using Claude.
        Fills in summary, relevance_score, key_entities, sentiment, and tags.
        Skips items that already have a summary (cached).
        """
        to_analyze = [item for item in items if not item.summary]
        if not to_analyze:
            return items

        topic_label = topic.title()
        results = []

        # Process in batches of BATCH_SIZE
        for i in range(0, len(to_analyze), settings.batch_size):
            if i > 0:
                await asyncio.sleep(2)  # avoid bursting through token-per-minute limit
            batch = to_analyze[i: i + settings.batch_size]
            for attempt in range(3):
                try:
                    prompt = ITEM_ANALYSIS_PROMPT.format(
                        n=len(batch),
                        topic=topic_label,
                        items_json=self._build_items_json(batch),
                    )
                    response = await self.async_client.messages.create(
                        model=self.model,
                        max_tokens=4096,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    raw = response.content[0].text.strip()

                    # Extract JSON array (handle markdown code blocks)
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
                        results.append(item)
                    break  # success

                except anthropic.RateLimitError:
                    wait = 30 * (attempt + 1)
                    logger.warning("Claude rate limit hit on batch %d — waiting %ds", i, wait)
                    await asyncio.sleep(wait)
                    if attempt == 2:
                        results.extend(batch)
                except json.JSONDecodeError as e:
                    logger.error("Claude returned invalid JSON for batch %d: %s", i, e)
                    results.extend(batch)
                    break
                except anthropic.APIError as e:
                    logger.error("Claude API error for batch %d: %s", i, e)
                    results.extend(batch)
                    break
                except Exception as e:
                    logger.error("Unexpected analyzer error for batch %d: %s", i, e)
                    results.extend(batch)
                    break

        # Merge back: items that already had summaries stay unchanged
        analyzed_ids = {item.id for item in results}
        final = []
        for item in items:
            if item.id in analyzed_ids:
                final.append(next(r for r in results if r.id == item.id))
            else:
                final.append(item)
        return final

    async def generate_digest(self, items: list[NewsItem], topic: str) -> str:
        """Generate a prose narrative digest for the given topic."""
        candidates = [i for i in items if i.topic == topic and not i.is_duplicate]
        # Prefer items that Claude has scored; fall back to raw engagement score
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

        for attempt in range(3):
            try:
                response = await self.async_client.messages.create(
                    model=self.model,
                    max_tokens=1500,
                    messages=[{"role": "user", "content": DIGEST_PROMPT.format(
                        n=len(top_items), topic_label=topic_label, items_text=items_text,
                    )}],
                )
                return response.content[0].text.strip()
            except anthropic.RateLimitError:
                wait = 30 * (attempt + 1)
                logger.warning("Claude rate limit on digest '%s' — waiting %ds", topic, wait)
                await asyncio.sleep(wait)
            except Exception as e:
                logger.error("Digest generation failed for topic '%s': %s", topic, e)
                return f"Digest generation failed: {e}"
        return "Digest generation failed: rate limit retries exhausted."

    async def generate_digest_stream(self, items: list[NewsItem], topic: str):
        """Stream digest generation, yielding raw text chunks as Claude writes them."""
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

        async with self.async_client.messages.stream(
            model=self.model,
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            async for text in stream.text_stream:
                yield text
