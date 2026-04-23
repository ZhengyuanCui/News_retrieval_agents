from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


@pytest.fixture
async def isolated_db():
    yield


def _load_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "agent_loop_proto.py"
    spec = importlib.util.spec_from_file_location("agent_loop_proto_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_fallback_mode_marks_cost_unknown():
    mod = _load_module()
    setattr(mod.settings, "_agent_loop_live_llm", False)

    results = [
        {
            "winner": "tie",
            "variants": {
                "current": {"score": 1, "cost_usd": 0.0, "cost_status": "fallback"},
                "balanced": {"score": 4, "cost_usd": 0.0, "cost_status": "fallback"},
                "deep": {"score": 4, "cost_usd": 0.0, "cost_status": "fallback"},
            },
        }
    ]

    summary = mod.summarize(results)
    table = mod.markdown_table(
        [
            {
                "question": "Q",
                "winner": "tie",
                "variants": {
                    "current": {"score": 1, "cost_usd": 0.0, "cost_status": "fallback"},
                    "balanced": {"score": 4, "cost_usd": 0.0, "cost_status": "fallback"},
                    "deep": {"score": 4, "cost_usd": 0.0, "cost_status": "fallback"},
                },
            }
        ],
        summary,
    )

    assert summary["avg_cost_usd"]["balanced"] == 0.0
    assert summary["cost_status"]["balanced"] == "fallback"
    assert summary["recommendation"] == "quality-only result; rerun with --live-llm for cost-based decision"
    assert "[fallback]" in table


def test_no_evidence_fallback_marks_current_cost_unknown():
    mod = _load_module()
    setattr(mod.settings, "_agent_loop_live_llm", False)

    result = mod.asyncio.run(
        mod._single_shot_answer(
            mod.RetrievalBackend(label=mod.LOCAL_RETRIEVAL_LABEL, db_path=Path("/tmp/does-not-exist.db")),
            "Q",
            24,
            10,
        )
    )

    _, cost, cost_status = result
    assert cost == 0.0
    assert cost_status == "not_needed"
