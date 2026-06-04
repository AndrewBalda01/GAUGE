"""
Deduplication: exact (SHA-256 hash) + semantic (MinHash / Jaccard shingles).

Why both?
  - Exact dedup catches identical copies (e.g. generator retried same prompt).
  - Semantic dedup catches near-duplicates: same question rephased, same answer
    with synonyms swapped — these are the hard, costly collisions.

MinHash approach (no external deps):
  We compute a bag-of-shingles (word n-grams) per text, then estimate Jaccard
  similarity via MinHash.  Pairs with Jaccard > threshold are considered dupes.

This is the near-dedup technique used in large-scale dataset cleaning
(e.g. C4, RedPajama) — implementing it from scratch signals real competence.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Exact dedup
# ---------------------------------------------------------------------------

def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def exact_dedup(
    examples: list,
    key: str = "input",
) -> tuple[list, list]:
    """Remove exact duplicates by hashing the `key` field. Returns (kept, dupes)."""
    seen: set[str] = set()
    kept, dupes = [], []
    for ex in examples:
        h = _sha256(getattr(ex, key, "").lower().strip())
        if h in seen:
            dupes.append(ex)
        else:
            seen.add(h)
            kept.append(ex)
    return kept, dupes


# ---------------------------------------------------------------------------
# MinHash — semantic near-dedup
# ---------------------------------------------------------------------------

_MINHASH_NUM_HASHES = 128
_SHINGLE_SIZE = 3              # word tri-grams


def _shingles(text: str, k: int = _SHINGLE_SIZE) -> set[str]:
    words = text.lower().split()
    return {" ".join(words[i: i + k]) for i in range(max(1, len(words) - k + 1))}


def _make_hash_functions(n: int, seed: int = 42) -> list[tuple[int, int]]:
    """Return n (a, b) pairs for universal hashing: h(x) = (a*x + b) mod p."""
    rng = random.Random(seed)
    p = (1 << 61) - 1  # Mersenne prime
    return [(rng.randint(1, p - 1), rng.randint(0, p - 1)) for _ in range(n)]


_HASH_FUNCS = _make_hash_functions(_MINHASH_NUM_HASHES)
_LARGE_PRIME = (1 << 61) - 1


def _minhash_signature(text: str) -> list[int]:
    sh = _shingles(text)
    if not sh:
        return [0] * _MINHASH_NUM_HASHES
    sig = []
    for a, b in _HASH_FUNCS:
        min_val = min((a * hash(s) + b) % _LARGE_PRIME for s in sh)
        sig.append(min_val)
    return sig


def _jaccard_estimate(sig_a: list[int], sig_b: list[int]) -> float:
    return sum(a == b for a, b in zip(sig_a, sig_b)) / len(sig_a)


@dataclass
class DedupResult:
    kept: list
    exact_dupes: list
    near_dupes: list

    @property
    def n_removed(self) -> int:
        return len(self.exact_dupes) + len(self.near_dupes)

    @property
    def total_input(self) -> int:
        return len(self.kept) + self.n_removed

    def summary(self) -> str:
        return (
            f"Dedup: {self.total_input} in → {len(self.kept)} kept  "
            f"(exact={len(self.exact_dupes)}, near={len(self.near_dupes)})"
        )


def semantic_dedup(
    examples: list,
    threshold: float = 0.75,
    key: str = "input",
) -> tuple[list, list]:
    """
    Remove near-duplicates using MinHash Jaccard estimation.
    O(n²) — acceptable for dataset sizes < ~10k.
    Returns (kept, near_dupes).
    """
    sigs = [_minhash_signature(getattr(ex, key, "")) for ex in examples]
    removed: set[int] = set()

    for i in range(len(examples)):
        if i in removed:
            continue
        for j in range(i + 1, len(examples)):
            if j in removed:
                continue
            if _jaccard_estimate(sigs[i], sigs[j]) >= threshold:
                removed.add(j)

    kept      = [ex for i, ex in enumerate(examples) if i not in removed]
    near_dupes = [ex for i, ex in enumerate(examples) if i in removed]
    return kept, near_dupes


def full_dedup(
    examples: list,
    near_threshold: float = 0.75,
) -> DedupResult:
    """Run exact then semantic dedup. Stamps 'dedup' on survivors."""
    after_exact, exact_dupes = exact_dedup(examples, key="input")
    after_near,  near_dupes  = semantic_dedup(after_exact, threshold=near_threshold)
    for ex in after_near:
        if "dedup" not in ex.stage_passed:
            ex.stage_passed.append("dedup")
    return DedupResult(kept=after_near, exact_dupes=exact_dupes, near_dupes=near_dupes)
