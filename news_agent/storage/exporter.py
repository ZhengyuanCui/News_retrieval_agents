from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from news_agent.models import NewsItem


class Exporter:
    def __init__(self, export_dir: str = "data/exports") -> None:
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)

    def export_json(self, items: list[NewsItem], date: str | None = None) -> Path:
        date = date or datetime.utcnow().strftime("%Y-%m-%d")
        path = self.export_dir / f"{date}.json"
        data = [item.model_dump(mode="json") for item in items]
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        return path

    def export_markdown(
        self,
        items: list[NewsItem],
        digest_ai: str | None = None,
        digest_stocks: str | None = None,
        date: str | None = None,
    ) -> Path:
        date = date or datetime.utcnow().strftime("%Y-%m-%d")
        path = self.export_dir / f"{date}.md"
        lines = [f"# News Digest — {date}\n"]

        if digest_ai:
            lines += ["## AI Development Summary\n", digest_ai, "\n"]
        if digest_stocks:
            lines += ["## Stock Market Summary\n", digest_stocks, "\n"]

        ai_items = [i for i in items if i.topic == "ai" and not i.is_duplicate]
        stock_items = [i for i in items if i.topic == "stocks" and not i.is_duplicate]

        if ai_items:
            lines.append("## AI Development News\n")
            for item in sorted(ai_items, key=lambda x: x.relevance_score or 0, reverse=True):
                lines.append(f"### [{item.title}]({item.url})")
                lines.append(f"*Source: {item.source} | {item.published_at.strftime('%Y-%m-%d %H:%M UTC')}*\n")
                if item.summary:
                    lines.append(item.summary)
                if item.tags:
                    lines.append(f"\n**Tags:** {', '.join(item.tags)}")
                if item.sentiment:
                    lines.append(f"**Sentiment:** {item.sentiment}")
                lines.append("")

        if stock_items:
            lines.append("## Stock Market News\n")
            for item in sorted(stock_items, key=lambda x: x.relevance_score or 0, reverse=True):
                lines.append(f"### [{item.title}]({item.url})")
                lines.append(f"*Source: {item.source} | {item.published_at.strftime('%Y-%m-%d %H:%M UTC')}*\n")
                if item.summary:
                    lines.append(item.summary)
                if item.tags:
                    lines.append(f"\n**Tags:** {', '.join(item.tags)}")
                if item.sentiment:
                    lines.append(f"**Sentiment:** {item.sentiment}")
                lines.append("")

        path.write_text("\n".join(lines), encoding="utf-8")
        return path
