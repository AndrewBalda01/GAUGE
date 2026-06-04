# Synthetic Data Generation Pipeline

> **Portfolio role:** the data engineering project. The hard part of synthetic data is not generating it — any LLM can produce text. The hard part is guaranteeing **quality**, controlling **diversity**, and measuring both. This pipeline does all three, and feeds its output directly into Project 01's eval suite, closing the loop.

---

## The problem

Evaluation datasets go stale. Real annotated data is expensive, slow to produce, and often unavailable for new domains. Synthetic data offers a path forward — but naive generation produces:

1. **Mode collapse**: the LLM gravitates to 5-6 patterns regardless of how many examples you request.
2. **Near-duplicates**: slightly rephased versions of the same question fill the dataset without adding coverage.
3. **Low quality**: the LLM produces boilerplate, disclaimers, or factually incorrect answers.

This pipeline addresses all three with a staged quality pipeline and measures each stage with concrete numbers.

---

## Pipeline stages

```
Spec & Plan
  ↓
Generate (LLM, structured prompt + seed diversifiers)
  ↓
Validate (structural checks)
  ↓
Dedup (exact SHA-256 + semantic MinHash)
  ↓
Filter (heuristic rules + optional LLM judge)
  ↓
Diversity analysis (TF-IDF + k-means)
  ↓
Output (eval-harness JSONL + report + charts)
```

---

## Stage 1: The plan (anti-mode-collapse)

The plan defines *exactly* what the pipeline should produce: how many examples per (topic, subtopic, difficulty) combination. Without a plan, the LLM over-represents easy topics and neglects edge cases.

```python
TopicSlot("contract", "breach", DifficultyLevel.ANALYTICAL, count=2,
          edge_cases=["partial performance", "anticipatory repudiation"])
```

The `legal_qa_expansion` plan has 24 slots across contract, tort, criminal, IP, privacy, company, property, and procedural law — producing 63 targeted examples.

---

## Stage 2: Seed diversifiers (anti-mode-collapse)

Each example is generated from a unique (persona × scenario × framing) seed. Without seeds, the LLM produces the same ~10 surface patterns.

**Example seed:**
```
Persona:  "a compliance officer at a fintech company"
Scenario: "following a regulatory investigation"
Framing:  "ask about an edge case or exception to a general rule"
```

This forces the prompt: *"a compliance officer, following a regulatory investigation, asks about an edge case in GDPR consent"* — a very different generation than *"what is GDPR consent?"*

The seed bank contains 12 personas × 12 scenarios × 10 framings = 1,440 unique combinations, sampled without replacement per generation run.

---

## Stage 3: Validate

Fast structural checks before any expensive API calls:

| Check | Threshold | Failure example |
|-------|-----------|-----------------|
| Question length | ≥ 5 words | `"Why?"` |
| Answer length | ≥ 15 words | `"It is a legal rule."` |
| Answer repeats question | first 30 chars | Answer starts with the question verbatim |
| Terminal punctuation | `.!?)` | Answer ends mid-sentence |
| Question too long | ≤ 800 chars | Malformed generation |

---

## Stage 4: Deduplication

### Exact dedup

SHA-256 hash of the normalised question. Catches identical copies from retry logic or reused seeds.

### Semantic near-dedup (MinHash)

The hard case: *"What is consideration in contract law and why does it matter?"* vs *"What is consideration in contract law and why is it important?"* — different strings, same information.

**Implementation:** bag of word trigrams → MinHash signature (128 hashes) → pairwise Jaccard estimate. Pairs with Jaccard ≥ 0.75 are considered near-duplicates; the second is discarded.

This is the technique used in large-scale dataset cleaning (C4, RedPajama). Implementing it from scratch — rather than calling `sklearn.feature_extraction` — is an explicit demonstration of understanding the algorithm, not just the library.

**Complexity:** O(n²) pairwise comparison. Acceptable for dataset sizes < ~10,000. For larger datasets, switch to LSH bucketing (same MinHash signatures, O(n) average lookup).

---

## Stage 5: Quality filter

### Heuristic (no API cost)

| Rule | Reason |
|------|--------|
| Boilerplate detection | LLMs often produce "As an AI…" or "Please consult a lawyer" |
| Repetition score | Trigram repetition ratio > 0.35 → poor answer |
| Excessive questions | Answer containing > 3 question marks → confused generation |
| Disproportionate length | Answer > 10× question length → off-topic generation |

### LLM judge (optional, uses Project 01's rubric)

When `--use-judge` is passed, survivors of the heuristic stage are scored by the same judge model as Project 01. Examples scoring below `--judge-min-score` (default 3/5) are rejected.

This reuses the judge calibrated in Project 01 — the quality threshold is not arbitrary.

---

## Stage 6: Diversity analysis

TF-IDF vectors (from scratch, no sklearn) + k-means (from scratch, k-means++ init) cluster the final examples into 8 clusters. Two metrics:

- **Coverage score**: fraction of clusters with ≥ 1 example. 1.0 = all topics represented.
- **Balance score**: 1 − normalised std of cluster sizes. 1.0 = perfectly even distribution.

The before/after comparison (post-dedup vs post-filter) is the visual proof that the pipeline improves diversity, not just reduces volume.

---

## Example pipeline run

```
Plan: legal_qa_expansion  (63 total examples)
  contract/formation [basic] × 4
  contract/formation [applied] × 3
  ... (24 slots)

Step 1/5 Generating…
  contract/breach → 2/2
  tort/negligence → 4/4
  ...
  Generated 63 raw examples

Step 2/5 Validating…
  passed=61  rejected=2  (e.g. 'answer too short (11 words < 15)')

Step 3/5 Deduplicating…
  Dedup: 61 in → 58 kept  (exact=1, near=2)

Step 4/5 Filtering…
  Filter: 58 in → 55 kept  (heuristic=3, judge=0)

Step 5/5 Writing output…
  Written 55 examples → output/legal_qa_synth.v1.jsonl

Done.  55 examples ready for eval-harness.
Diversity: 55 examples, 8 clusters, coverage=1.00, balance=0.71
```

**From 63 raw → 55 final: overall discard rate 12.7%**

---

## Example funnel chart

```
Generated    ████████████████████████████████████  63
Post-validate ███████████████████████████████████  61  (-3.2%)
Post-dedup   ████████████████████████████████████  58  (-4.9%)
Post-filter  █████████████████████████████████████ 55  (-5.2%)
```

> **Actual chart (funnel.png) generated in output/ after each run.**

---

## Output format

The output JSONL is directly compatible with `eval-harness/datasets/`:

```jsonl
{"id": "syn-con-3a7f2b", "input": "In the context of a fintech compliance review, what constitutes a valid exclusion clause under UCTA 1977?", "expected": "An exclusion clause is valid under UCTA 1977 if it satisfies the reasonableness test...", "metrics": ["judge"], "tags": ["contract", "terms", "applied"]}
```

To add synthetic examples to the eval suite:

```bash
cat output/legal_qa_synth.v1.jsonl >> ../eval-harness/datasets/legal_qa.v1.jsonl
python ../eval-harness/datasets/make_manifest.py  # regenerates SHA-256
```

---

## Quick start

```bash
cd synthetic-data
pip install -e ".[dev]"
cp .env.example .env     # set ANTHROPIC_API_KEY for real generation

# Dry run (no API calls — verifies the pipeline end-to-end)
python pipeline.py dry-run --output output/dry_run.jsonl

# Real run (requires API key)
python pipeline.py run \
  --model claude-haiku-4-5-20251001 \
  --concurrency 5 \
  --output output/legal_qa_synth.v1.jsonl

# Real run with judge filtering (slower, higher quality)
python pipeline.py run \
  --model claude-haiku-4-5-20251001 \
  --use-judge \
  --judge-min-score 3 \
  --output output/legal_qa_synth_judged.v1.jsonl

# Run tests (all 47, no API calls)
python -m pytest tests/ -v
```

---

## Extending to a new domain

1. Define a new `GenerationPlan` in `spec/plan.py` with appropriate `TopicSlot`s.
2. Run the pipeline. No other code changes required.

The pipeline is domain-agnostic: the plan carries the domain knowledge, the pipeline carries the engineering.

---

## Limits

- **Bias from the generator model**: the synthetic examples reflect the biases and knowledge gaps of the generating LLM. If the model has inaccurate legal knowledge, the dataset propagates those errors. Always validate a sample manually.
- **Near-dedup false positives**: MinHash at threshold 0.75 may remove legitimately distinct examples that happen to share many trigrams (e.g., two questions about "consideration" that ask genuinely different things). Lower thresholds increase recall but reduce precision.
- **No adversarial coverage**: the pipeline does not generate adversarial examples (jailbreaks, prompt injections, edge cases designed to fail). Those require a different generation strategy.
- **Cost of judge filtering**: running the LLM judge on 60 examples costs ~$0.003 with Haiku. At scale (10,000 examples), budget accordingly.
- **Mode collapse residual**: the seeding strategy reduces but does not eliminate mode collapse. If the LLM strongly prefers certain patterns, the diversity score will reveal this — but the pipeline does not automatically fix it.

---

## Connections to other projects

- **→ 01 Eval Harness**: output JSONL drops directly into `datasets/`. The manifest is regenerated with `make_manifest.py`. New test cases immediately available to all eval runs.
- **← 02 Local Inference**: generation can run on a local model (`engine.py`) instead of the cloud API by pointing the generator at the local endpoint. Cost: $0.00 per token. Tradeoff: lower generation quality.
- **→ 03 Observability**: a large generation run (e.g. 1,000 examples) is a good workload to trace — you can watch cost accumulate in real time in the observability dashboard.
