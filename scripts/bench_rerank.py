"""Offline cross-encoder reranker benchmark.

Compares:
    cross-encoder/ms-marco-MiniLM-L-6-v2   (current production)
    mixedbread-ai/mxbai-rerank-xsmall-v1   (candidate)

Reads the labeled eval set at tests/fixtures/rerank_eval.json, runs each
model over every query's 40 candidates, reports NDCG@10, MRR@10, latency
(median / P95 / P99, split cold-vs-warm), and per-bucket NDCG@10.

Writes detailed results to scripts/rerank_bench_results.json and prints a
markdown comparison table to stdout.

Run from the spike/rerank-mxbai-eval worktree:
    python3 scripts/bench_rerank.py
"""
from __future__ import annotations

import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import ndcg_score

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "rerank_eval.json"
RESULTS_OUT = Path(__file__).resolve().parent / "rerank_bench_results.json"

MODELS = [
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "mixedbread-ai/mxbai-rerank-xsmall-v1",
]

TOP_K = 10


def ndcg_at_k(labels_in_original_order: list[int], scores_in_original_order: list[float], k: int = TOP_K) -> float:
    """NDCG@k for a single query.

    We pass y_true (binary relevance) and y_score (reranker scores) both in
    the ORIGINAL candidate order to sklearn.ndcg_score — it internally ranks
    by y_score and computes gain against y_true. This is the documented usage
    and avoids the double-sort bug of passing labels-in-reranked-order.

    All-zero y_true → sklearn returns 0.0 (documented behavior). We return 0.0
    in that case too, so skip-queries contribute equally to both models.
    """
    if sum(labels_in_original_order) == 0:
        return 0.0
    y_true = np.asarray([labels_in_original_order], dtype=float)
    y_score = np.asarray([scores_in_original_order], dtype=float)
    return float(ndcg_score(y_true, y_score, k=k))


def mrr_at_k(labels_in_reranked_order: list[int], k: int = TOP_K) -> float:
    """Reciprocal rank of the first relevant item within the top-k, else 0."""
    for rank, lab in enumerate(labels_in_reranked_order[:k], start=1):
        if lab == 1:
            return 1.0 / rank
    return 0.0


# --- Self-test for ndcg_score usage on a known answer ---------------------
def _sanity_ndcg() -> None:
    """Pin sklearn.ndcg_score behavior on a toy example so we catch any drift.

    Candidates A,B,C,D,E with labels [1,0,1,0,0].
    Reranker scores [0.9, 0.1, 0.8, 0.2, 0.0] → reranked order A, C, B, D, E.
    Ideal order: A, C, B, D, E as well.  Expected NDCG@3 ≈ 1.0.
    With scores [0.1, 0.9, 0.0, 0.2, 0.8] → reranked order B, E, D, A, C.
    Top-3 gains = [0,0,0] (A and C fall outside top-3), DCG=0, NDCG=0.0.
    """
    labels = [1, 0, 1, 0, 0]
    good = [0.9, 0.1, 0.8, 0.2, 0.0]
    bad =  [0.1, 0.9, 0.0, 0.2, 0.8]
    g = ndcg_at_k(labels, good, k=3)
    b = ndcg_at_k(labels, bad, k=3)
    assert abs(g - 1.0) < 1e-9, f"expected NDCG@3 = 1.0, got {g}"
    assert abs(b - 0.0) < 1e-9, f"expected NDCG@3 = 0.0, got {b}"


def load_fixture() -> dict:
    data = json.loads(FIXTURE.read_text())
    # Drop queries where no positive label exists — NDCG is undefined / always 0
    # for them and would dilute the mean identically for both models. Report
    # them in meta so the result table is honest about n_eval.
    kept = []
    dropped = []
    for q in data["queries"]:
        if any(c["label"] == 1 for c in q["candidates"]):
            kept.append(q)
        else:
            dropped.append(q["query"])
    data["_kept"] = kept
    data["_dropped"] = dropped
    return data


def bench_model(model_name: str, queries: list[dict]) -> dict:
    print(f"\n═══ {model_name} ═══")
    print("  loading model...")
    t_load = time.perf_counter()
    from sentence_transformers import CrossEncoder
    model = CrossEncoder(model_name)
    load_time_s = time.perf_counter() - t_load
    print(f"  loaded in {load_time_s:.2f}s")

    per_query: list[dict] = []
    latencies_ms: list[float] = []

    for qi, q in enumerate(queries):
        text = q["query"]
        cands = q["candidates"]
        pairs = [
            (text, f"{(c['title'] or '').strip()} {(c['summary'] or '').strip()}")
            for c in cands
        ]
        labels = [int(c["label"]) for c in cands]

        t0 = time.perf_counter()
        raw_scores = model.predict(pairs, show_progress_bar=False)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(dt_ms)

        # Cast to plain floats for JSON serialization.
        scores = [float(s) for s in raw_scores]

        # Reranked label order for MRR@10.
        order = sorted(range(len(cands)), key=lambda i: scores[i], reverse=True)
        labels_reranked = [labels[i] for i in order]

        ndcg = ndcg_at_k(labels, scores, k=TOP_K)
        mrr = mrr_at_k(labels_reranked, k=TOP_K)

        per_query.append(
            {
                "query": text,
                "bucket": q["bucket"],
                "ndcg@10": ndcg,
                "mrr@10": mrr,
                "latency_ms": dt_ms,
                "n_positive": sum(labels),
            }
        )

        if (qi + 1) % 20 == 0 or qi + 1 == len(queries):
            print(f"  [{qi+1}/{len(queries)}] latest latency={dt_ms:.1f}ms")

    # Aggregate
    ndcg_vals = [r["ndcg@10"] for r in per_query]
    mrr_vals = [r["mrr@10"] for r in per_query]

    def pct(vs: list[float], p: float) -> float:
        """Linear interp percentile — stdlib has this in 3.10+ as quantiles but
        we want a single percentile value on a ~99-element list without fiddling
        with n-tile indexing."""
        if not vs:
            return 0.0
        xs = sorted(vs)
        k = (len(xs) - 1) * (p / 100.0)
        f = int(k)
        c = min(f + 1, len(xs) - 1)
        if f == c:
            return xs[f]
        return xs[f] + (xs[c] - xs[f]) * (k - f)

    # Cold vs warm latency split: first call per model is the cold one.
    cold_ms = latencies_ms[0] if latencies_ms else 0.0
    warm = latencies_ms[1:]

    per_bucket: dict[str, list[float]] = defaultdict(list)
    for r in per_query:
        per_bucket[r["bucket"]].append(r["ndcg@10"])
    per_bucket_mean = {b: statistics.mean(vs) for b, vs in per_bucket.items()}

    return {
        "model": model_name,
        "load_time_s": load_time_s,
        "n_queries": len(per_query),
        "mean_ndcg@10": statistics.mean(ndcg_vals) if ndcg_vals else 0.0,
        "mean_mrr@10": statistics.mean(mrr_vals) if mrr_vals else 0.0,
        "latency_ms": {
            "cold": cold_ms,
            "warm_n": len(warm),
            "warm_median": statistics.median(warm) if warm else 0.0,
            "warm_p95": pct(warm, 95.0),
            "warm_p99": pct(warm, 99.0),
            "warm_mean": statistics.mean(warm) if warm else 0.0,
        },
        "per_bucket_mean_ndcg@10": per_bucket_mean,
        "per_query": per_query,
    }


def format_table(results: list[dict]) -> str:
    def fmt(v, d=3):
        return f"{v:.{d}f}"
    lines = [
        "| Model | NDCG@10 | MRR@10 | P95 latency (ms) | Cold (ms) |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['model'].split('/')[-1]} | {fmt(r['mean_ndcg@10'])} | {fmt(r['mean_mrr@10'])} "
            f"| {fmt(r['latency_ms']['warm_p95'], 1)} | {fmt(r['latency_ms']['cold'], 1)} |"
        )
    return "\n".join(lines)


def format_bucket_table(results: list[dict], buckets_n: dict[str, int]) -> str:
    r0, r1 = results[0], results[1]
    lines = [
        "| Bucket | ms-marco | mxbai | Δ |",
        "|---|---|---|---|",
    ]
    for b in ["nl", "entity", "ticker", "time"]:
        v0 = r0["per_bucket_mean_ndcg@10"].get(b, 0.0)
        v1 = r1["per_bucket_mean_ndcg@10"].get(b, 0.0)
        n = buckets_n.get(b, 0)
        lines.append(f"| {b} (n={n}) | {v0:.3f} | {v1:.3f} | {v1 - v0:+.3f} |")
    return "\n".join(lines)


def verdict(results: list[dict]) -> tuple[str, str]:
    msmarco = next(r for r in results if "ms-marco" in r["model"])
    mxbai = next(r for r in results if "mxbai" in r["model"])
    d_ndcg = mxbai["mean_ndcg@10"] - msmarco["mean_ndcg@10"]
    p95 = mxbai["latency_ms"]["warm_p95"]

    # Kill criteria first (stricter)
    if d_ndcg < 0 or p95 > 600:
        why = []
        if d_ndcg < 0:
            why.append(f"mxbai NDCG@10 regresses by {-d_ndcg:.3f}")
        if p95 > 600:
            why.append(f"mxbai P95 {p95:.0f}ms > 600ms ceiling")
        return "KILL", "; ".join(why)
    # Ship criteria
    if d_ndcg >= 0.03 and p95 <= 400:
        return "SHIP", f"mxbai NDCG@10 gains +{d_ndcg:.3f} (≥0.03) and P95 {p95:.0f}ms (≤400ms)"
    # Otherwise ambiguous
    bits = []
    if d_ndcg < 0.03:
        bits.append(f"NDCG@10 gain only +{d_ndcg:.3f} (<0.03 ship threshold)")
    if 400 < p95 <= 600:
        bits.append(f"P95 {p95:.0f}ms in ambiguous band (400-600ms)")
    return "AMBIGUOUS", "; ".join(bits) if bits else "thresholds not met in either direction"


def main() -> None:
    _sanity_ndcg()
    print("sanity: sklearn.ndcg_score behaves as expected ✓")

    data = load_fixture()
    queries = data["_kept"]
    print(f"loaded fixture: {len(data['queries'])} queries, {len(queries)} with ≥1 positive (eval set)")
    print(f"  dropped (all-zero labels): {data['_dropped']}")

    buckets_n = defaultdict(int)
    for q in queries:
        buckets_n[q["bucket"]] += 1

    results: list[dict] = []
    for model_name in MODELS:
        res = bench_model(model_name, queries)
        results.append(res)

    out = {
        "fixture_meta": data["meta"],
        "eval_n_queries": len(queries),
        "dropped_queries": data["_dropped"],
        "buckets_n": dict(buckets_n),
        "results": results,
    }
    RESULTS_OUT.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {RESULTS_OUT} ({RESULTS_OUT.stat().st_size} bytes)")

    # Print comparison
    print("\n═══ Results (warm, n={}) ═══".format(results[0]["latency_ms"]["warm_n"]))
    print(format_table(results))
    print("\n═══ Per-bucket NDCG@10 ═══")
    print(format_bucket_table(results, dict(buckets_n)))

    v, why = verdict(results)
    print(f"\nVerdict: {v} — {why}")


if __name__ == "__main__":
    main()
