# Spike B Report: Bounded Agentic Research Loop

## Scope

Issue: [#10](https://github.com/ZhengyuanCui/News_retrieval_agents/issues/10)  
Title: `[Spike B] Bounded agentic research loop prototype`

This spike evaluates whether the repo should ship a Perplexica-style research
loop for news Q&A with user-visible `current`, `balanced`, and `deep` modes.

Prototype under test:
- `scripts/agent_loop_proto.py`

Primary artifact:
- `scripts/agent_loop_results.json`

## Evaluation Criteria

Issue `#10` defines the decision thresholds:

- Ship criterion:
  - `balanced` improves subjective quality on `>= 6/10` questions
  - and `cost/question <= $0.05`
- Kill criterion:
  - `balanced` wins `< 4/10`
  - or `cost > $0.15`
  - in that case, kill the full loop and keep only a single-shot query
    expansion step

## Environment Used

The final run in this branch was intentionally bounded to the environment that
was actually available during implementation:

- Retrieval: `BM25-only local SQLite evidence`
- Database path: local `data/news.db`
- No `init_db()` call inside the spike script
- No semantic embedding retrieval in this run

Why this matters:

- the normal semantic path depends on the sentence-transformer model
  `all-MiniLM-L6-v2`
- in the default sandboxed environment, outbound model download failed on
  Hugging Face DNS/network resolution
- because of that, the spike was scoped to local BM25 retrieval rather than
  the full hybrid retrieval pipeline

This means the result below is valid for the bounded local prototype that
actually ran, but it is not a full evaluation of the production-intended
hybrid retrieval stack.

## Experiments Performed

### 1. Initial prototype runs

The first version of the script was built to compare:

- `current`: one retrieval + one synthesis step
- `balanced`: bounded planner/retrieve/synthesize loop
- `deep`: larger bounded planner/retrieve/synthesize loop

These early runs established the control flow and artifact shape but were not
yet decision-quality because of two problems:

- the environment could not support the intended semantic retrieval path
- the script could run in fallback mode and still emit misleading `$0.0000`
  costs

### 2. Cost-tracking investigation

An additional investigation was run after a zero-cost artifact was observed.

Finding:

- the cost helper itself was not broken
- the real cause was that the script defaults to fallback mode unless
  `--live-llm` is passed
- fallback mode substitutes local planner/answer/judge heuristics, so those
  runs do not produce valid live-LLM cost evidence

Fixes made:

- `cost_usd` now stays numerically factual
- `cost_status` was added so the artifact can distinguish:
  - `measured`
  - `fallback`
  - `partial`
  - `not_needed`
- fallback-substituted work is no longer presented as a trustworthy zero-cost
  evaluation

### 3. Retrieval debugging and fixes

The first BM25-only run produced `0` sources for all 10 questions. That turned
out to be two separate problems:

- the clean Spike B worktree had no local `data/news.db`, so the prototype was
  pointing at a non-existent DB path and returning `[]`
- the FTS query builder was too strict because it turned the full natural
  language question into an all-terms match

Fixes made:

- the prototype now points directly at the main workspace DB
- the BM25 query builder now drops filler words and searches on content tokens
  instead of requiring the full question text

Focused verification added:

- `tests/scripts/test_agent_loop_proto.py`
- `4 passed`

### 4. Live-run reliability fixes

Before the final run, the prototype was tightened further:

- the global "one failure flips all later calls to fallback" behavior was
  removed
- LLM timeout was raised from `5s` to `30s`
- the clean worktree was given the repo `.env` so the live run actually had
  provider credentials available

These changes were necessary to get a true end-to-end live LLM run instead of
another fallback-heavy artifact.

### 5. Final decision run

Final command used:

```bash
/home/zhengyuancui/news-agent-venv/bin/python scripts/agent_loop_proto.py --live-llm
```

This run completed all 10 questions and wrote the final artifact to
`scripts/agent_loop_results.json`.

## Final Results

### Summary

- Evidence mode: `BM25-only local SQLite evidence`
- LLM mode: `live`
- Average score:
  - `current = 3.9`
  - `balanced = 4.2`
  - `deep = 3.7`
- Average cost/question:
  - `current = $0.0085 [measured]`
  - `balanced = $0.0123 [measured]`
  - `deep = $0.0190 [measured]`
- Wins:
  - `current = 3`
  - `balanced = 3`
  - `deep = 2`
  - `tie = 2`
- Script recommendation:
  - `kill full loop; consider only query expansion (BM25-only evidence in this environment)`

### Per-question results

| # | Question (short) | Current | Balanced | Deep | Winner |
|---|---|---:|---:|---:|---|
| 1 | OpenAI roadmap | 4 | 4 | 3 | current |
| 2 | Anthropic vs OpenAI | 4 | 3 | 3 | current |
| 3 | AI chip competition | 3 | 4 | 4 | deep |
| 4 | AI infra costs | 2 | 4 | 4 | balanced |
| 5 | Regulation / safety | 5 | 5 | 5 | deep |
| 6 | Open-weight releases | 4 | 4 | 3 | tie |
| 7 | Big-tech AI platforms | 4 | 3 | 3 | current |
| 8 | Agents / research loops | 4 | 5 | 4 | balanced |
| 9 | Publishers / AI deals | 5 | 5 | 5 | tie |
| 10 | Robotics + foundation models | 4 | 5 | 3 | balanced |

### Interpretation

The prototype still missed the ship threshold:

- `balanced` did **not** improve quality on `>= 6/10`
- it won `3/10`

It also still triggered the kill threshold:

- `balanced wins < 4/10`

Cost was still not the failure mode:

- `balanced` cost/question was about `$0.0123`, far below the issue's cost cap
- `deep` cost/question was about `$0.0190`

So the decision did not fail on cost. It failed on the issue's explicit
"balanced wins" threshold.

## Why The Scores Were So Flat

The dominant limiting factor in the final run was not missing evidence anymore;
the DB/path bug was fixed and the prototype now retrieves articles. The more
important pattern is:

- `balanced` improved the average score relative to `current`
- but the gains were not decisive often enough to satisfy the issue's ship bar
- `deep` cost more and did not outperform `balanced`

So the result is not "the loop never helps." It is narrower:

- in this local BM25-only environment, `balanced` can help some questions
- but not consistently enough to justify shipping the full multi-depth loop
  under the issue's threshold

## Decision

Based on the issue's own thresholds and the final live run:

- do **not** ship the full `balanced/deep` bounded research loop
- if anything from Spike B is kept, keep only the narrower idea:
  - a single-shot query expansion step

## Caveats

- This was not a full hybrid retrieval evaluation.
- The final evidence is restricted to BM25 over local SQLite.
- Because semantic retrieval was not part of the final run, the result should
  be interpreted as:
  - "do not ship the full loop based on the evidence available here"
  - not as a universal statement that the loop could never help under a
    stronger retrieval stack

## Files Produced By This Work

- `scripts/agent_loop_proto.py`
- `scripts/agent_loop_results.json`
- `tests/scripts/test_agent_loop_proto.py`
- `scripts/agent_loop_report.md`
