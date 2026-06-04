"""
Diversity analysis: cluster examples using TF-IDF vectors + k-means.

Why this matters:
  The classic failure mode is a dataset that *looks* large but actually
  covers only 3-4 semantic clusters — "mode collapsed" despite superficial
  surface variety.  Clustering exposes this visually (the before/after
  chart is the portfolio's money shot).

Dependencies: pure Python + optional matplotlib. No sklearn, no torch.
We implement a minimal k-means over TF-IDF vectors entirely from scratch
so the portfolio shows we understand the algorithm, not just the API.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass, field

from spec.schema import SyntheticExample


# ---------------------------------------------------------------------------
# TF-IDF vectoriser (minimal, no external deps)
# ---------------------------------------------------------------------------

_STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "is", "are", "was", "were", "be", "been", "being",
    "it", "its", "this", "that", "which", "what", "how", "when", "where",
    "as", "by", "from", "into", "through", "during", "has", "have",
    "had", "not", "no", "so", "if", "then", "than", "also", "i", "you",
}


def _tokenize(text: str) -> list[str]:
    import re
    return [
        w for w in re.sub(r"[^a-z ]", " ", text.lower()).split()
        if w not in _STOP_WORDS and len(w) > 2
    ]


def _build_vocabulary(docs: list[list[str]], min_df: int = 2) -> list[str]:
    counter: Counter = Counter()
    for doc in docs:
        counter.update(set(doc))
    return [w for w, c in counter.items() if c >= min_df]


def _tfidf(docs: list[list[str]], vocab: list[str]) -> list[list[float]]:
    n = len(docs)
    vocab_idx = {w: i for i, w in enumerate(vocab)}
    idf = {}
    for w in vocab:
        df = sum(1 for doc in docs if w in set(doc))
        idf[w] = math.log((n + 1) / (df + 1)) + 1.0

    vectors = []
    for doc in docs:
        tf: Counter = Counter(doc)
        total = max(len(doc), 1)
        vec = [0.0] * len(vocab)
        for w, cnt in tf.items():
            if w in vocab_idx:
                vec[vocab_idx[w]] = (cnt / total) * idf[w]
        # L2 normalise
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        vectors.append([v / norm for v in vec])
    return vectors


# ---------------------------------------------------------------------------
# Cosine similarity + k-means
# ---------------------------------------------------------------------------

def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    # already normalised → cosine = dot product
    return dot


def _kmeans(
    vectors: list[list[float]],
    k: int,
    max_iter: int = 30,
    seed: int = 42,
) -> list[int]:
    """Returns cluster assignment for each vector."""
    rng = random.Random(seed)
    n = len(vectors)
    if n == 0 or k >= n:
        return list(range(n))

    # k-means++ initialisation
    centroids = [vectors[rng.randint(0, n - 1)]]
    for _ in range(k - 1):
        dists = [min(1 - _cosine(v, c) for c in centroids) for v in vectors]
        total = sum(dists) or 1.0
        probs = [d / total for d in dists]
        cum = 0.0
        r = rng.random()
        chosen = n - 1
        for i, p in enumerate(probs):
            cum += p
            if r <= cum:
                chosen = i
                break
        centroids.append(vectors[chosen])

    labels = [0] * n
    for _ in range(max_iter):
        # Assign
        new_labels = [
            max(range(k), key=lambda c: _cosine(v, centroids[c]))
            for v in vectors
        ]
        if new_labels == labels:
            break
        labels = new_labels
        # Update centroids
        for c in range(k):
            members = [vectors[i] for i, lbl in enumerate(labels) if lbl == c]
            if not members:
                continue
            d = len(members[0])
            mean = [sum(m[j] for m in members) / len(members) for j in range(d)]
            norm = math.sqrt(sum(x * x for x in mean)) or 1.0
            centroids[c] = [x / norm for x in mean]

    return labels


# ---------------------------------------------------------------------------
# Diversity metrics
# ---------------------------------------------------------------------------

@dataclass
class DiversityReport:
    n_examples: int
    n_clusters: int
    cluster_sizes: list[int] = field(default_factory=list)
    coverage_score: float = 0.0      # fraction of clusters with ≥1 example
    balance_score: float = 0.0       # 1 - normalised std of cluster sizes (higher = better)
    labels: list[int] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"Diversity: {self.n_examples} examples, {self.n_clusters} clusters, "
            f"coverage={self.coverage_score:.2f}, balance={self.balance_score:.2f}"
        )


def analyse_diversity(
    examples: list[SyntheticExample],
    n_clusters: int = 8,
) -> DiversityReport:
    if not examples:
        return DiversityReport(0, n_clusters, [], 0.0, 0.0, [])

    texts = [ex.input + " " + ex.expected for ex in examples]
    docs  = [_tokenize(t) for t in texts]
    vocab = _build_vocabulary(docs)

    if not vocab:
        return DiversityReport(len(examples), n_clusters, [], 0.0, 0.0, [])

    vectors = _tfidf(docs, vocab)
    k = min(n_clusters, len(examples))
    labels = _kmeans(vectors, k=k)

    sizes = [labels.count(c) for c in range(k)]
    non_empty = sum(1 for s in sizes if s > 0)
    coverage = non_empty / k if k > 0 else 0.0

    import statistics
    if len(sizes) > 1:
        mean_s = statistics.mean(sizes)
        std_s  = statistics.stdev(sizes)
        balance = max(0.0, 1.0 - std_s / (mean_s or 1.0))
    else:
        balance = 1.0

    return DiversityReport(
        n_examples=len(examples),
        n_clusters=k,
        cluster_sizes=sizes,
        coverage_score=round(coverage, 3),
        balance_score=round(balance, 3),
        labels=labels,
    )
