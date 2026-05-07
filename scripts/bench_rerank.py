"""Offline cross-encoder reranker benchmark for cross_encoder_top_k tuning.

Compares the production reranker, cross-encoder/ms-marco-MiniLM-L-6-v2,
with two different rerank slice sizes:
    top_k=20   (legacy behavior)
    top_k=10   (#14 candidate default)

Reads the labeled eval set at tests/fixtures/rerank_eval.json, reranks only
the configured top-k slice for each query, leaves the remaining candidates in
their original order, and reports NDCG@5, NDCG@10, MRR@10, and latency.

Writes detailed results to scripts/rerank_bench_results.json and prints a
markdown comparison table to stdout.
"""
from __future__ import annotations

import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "rerank_eval.json"
RESULTS_OUT = Path(__file__).resolve().parent / "rerank_bench_results.json"

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_TOP_KS = [20, 10]
METRIC_K_NDCG = (5, 10)
METRIC_K_MRR = 10


def ndcg_at_k(labels_in_ranked_order: list[int], k: int) -> float:
    """NDCG@k for an already-ranked result list."""
    if sum(labels_in_ranked_order) == 0:
        return 0.0
    gains = labels_in_ranked_order[:k]
    ideal = sorted(labels_in_ranked_order, reverse=True)[:k]

    def dcg(vs: list[int]) -> float:
        total = 0.0
        for rank, rel in enumerate(vs, start=1):
            total += rel / math.log2(rank + 1)
        return total

    actual = dcg(gains)
    best = dcg(ideal)
    return actual / best if best else 0.0


def mrr_at_k(labels_in_ranked_order: list[int], k: int = METRIC_K_MRR) -> float:
    """Reciprocal rank of the first relevant item within the top-k, else 0."""
    for rank, lab in enumerate(labels_in_ranked_order[:k], start=1):
        if lab == 1:
            return 1.0 / rank
    return 0.0


# --- Self-test for ndcg_score usage on a known answer ---------------------
def _sanity_ndcg() -> None:
    """Pin NDCG behavior on already-ranked lists so metric drift is obvious."""
    good = [1, 1, 0, 0, 0]
    bad = [0, 0, 0, 1, 1]
    g = ndcg_at_k(good, k=3)
    b = ndcg_at_k(bad, k=3)
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


def bench_top_k(model_name: str, rerank_top_k: int, queries: list[dict]) -> dict:
    print(f"\n═══ {model_name} / top_k={rerank_top_k} ═══")
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
        rerank_slice = cands[:rerank_top_k]
        untouched_tail = cands[rerank_top_k:]
        pairs = [
            (text, f"{(c['title'] or '').strip()} {(c['summary'] or '').strip()}")
            for c in rerank_slice
        ]
        labels_rerank_slice = [int(c["label"]) for c in rerank_slice]
        labels_untouched_tail = [int(c["label"]) for c in untouched_tail]

        t0 = time.perf_counter()
        raw_scores = model.predict(pairs, show_progress_bar=False)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(dt_ms)

        scores = [float(s) for s in raw_scores]
        order = sorted(range(len(rerank_slice)), key=lambda i: scores[i], reverse=True)
        labels_ranked = [labels_rerank_slice[i] for i in order] + labels_untouched_tail

        per_query.append(
            {
                "query": text,
                "bucket": q["bucket"],
                "rerank_top_k": rerank_top_k,
                "ndcg@5": ndcg_at_k(labels_ranked, k=5),
                "ndcg@10": ndcg_at_k(labels_ranked, k=10),
                "mrr@10": mrr_at_k(labels_ranked, k=10),
                "latency_ms": dt_ms,
                "n_positive": sum(labels_ranked),
            }
        )

        if (qi + 1) % 20 == 0 or qi + 1 == len(queries):
            print(f"  [{qi+1}/{len(queries)}] latest latency={dt_ms:.1f}ms")

    # Aggregate
    ndcg5_vals = [r["ndcg@5"] for r in per_query]
    ndcg10_vals = [r["ndcg@10"] for r in per_query]
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
        "rerank_top_k": rerank_top_k,
        "load_time_s": load_time_s,
        "n_queries": len(per_query),
        "mean_ndcg@5": statistics.mean(ndcg5_vals) if ndcg5_vals else 0.0,
        "mean_ndcg@10": statistics.mean(ndcg10_vals) if ndcg10_vals else 0.0,
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
        "| Variant | NDCG@5 | NDCG@10 | MRR@10 | P95 latency (ms) | Cold (ms) |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| top_k={r['rerank_top_k']} | {fmt(r['mean_ndcg@5'])} | {fmt(r['mean_ndcg@10'])} | {fmt(r['mean_mrr@10'])} "
            f"| {fmt(r['latency_ms']['warm_p95'], 1)} | {fmt(r['latency_ms']['cold'], 1)} |"
        )
    return "\n".join(lines)


def format_bucket_table(results: list[dict], buckets_n: dict[str, int]) -> str:
    r0, r1 = results[0], results[1]
    lines = [
        f"| Bucket | top_k={r0['rerank_top_k']} | top_k={r1['rerank_top_k']} | Δ |",
        "|---|---|---|---|",
    ]
    for b in ["nl", "entity", "ticker", "time"]:
        v0 = r0["per_bucket_mean_ndcg@10"].get(b, 0.0)
        v1 = r1["per_bucket_mean_ndcg@10"].get(b, 0.0)
        n = buckets_n.get(b, 0)
        lines.append(f"| {b} (n={n}) | {v0:.3f} | {v1:.3f} | {v1 - v0:+.3f} |")
    return "\n".join(lines)


def verdict(results: list[dict]) -> tuple[str, str]:
    baseline = next(r for r in results if r["rerank_top_k"] == 20)
    candidate = next(r for r in results if r["rerank_top_k"] == 10)
    ndcg5_drop = baseline["mean_ndcg@5"] - candidate["mean_ndcg@5"]
    p95_drop_ratio = 1.0 - (
        candidate["latency_ms"]["warm_p95"] / baseline["latency_ms"]["warm_p95"]
        if baseline["latency_ms"]["warm_p95"] > 0
        else 0.0
    )
    if ndcg5_drop <= 0.01 and p95_drop_ratio >= 0.30:
        return "SHIP", (
            f"NDCG@5 drop {ndcg5_drop:.3f} (<= 0.01) and P95 latency improvement "
            f"{p95_drop_ratio * 100:.1f}% (>= 30%)"
        )
    reasons = []
    if ndcg5_drop > 0.01:
        reasons.append(f"NDCG@5 drop {ndcg5_drop:.3f} exceeds 0.01")
    if p95_drop_ratio < 0.30:
        reasons.append(f"P95 latency improvement only {p95_drop_ratio * 100:.1f}%")
    return "REVERT", "; ".join(reasons)


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
    for rerank_top_k in RERANK_TOP_KS:
        res = bench_top_k(MODEL_NAME, rerank_top_k, queries)
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
