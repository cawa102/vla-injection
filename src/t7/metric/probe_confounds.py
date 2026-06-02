"""L1 confound-control scaffolding (Codex review #2 #11, playbook §4b-(I)).

The pre-registered controls that must pass before an L1 "internal-rep" win/lose
is reportable — so the result is not a task prior, a suffix lexical fingerprint,
target leakage, or a GCG-family artifact. These are the **model-free** primitives
the M2 analysis runs (held-out tasks/suffix-seeds/target-specs disjointness is
enforced separately by :func:`t7.eval.splits.assert_disjoint`):

* :func:`shuffle_labels` — the label-shuffle control input: a deterministic,
  multiset-preserving permutation so the probe can be retrained on permuted
  labels and shown to collapse to chance.
* :func:`probe_auc` — the shared AUC comparator (reuses ``eval.metrics.roc_auc``)
  every control reports on (honest probe vs label-shuffled vs an L0 lexical
  baseline scored on the same rollouts).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from t7.eval.metrics import roc_auc
from t7.metric.probe_internal import ActivationFeatures, InternalProbe


def shuffle_labels(labels: Sequence[int], *, seed: int) -> list[int]:
    """Return a deterministic permutation of ``labels`` (multiset preserved).

    The label-shuffle control: training on the result destroys the label↔feature
    association while keeping class balance, so a probe that still separates is
    fitting an artifact. Does not mutate the input.

    Args:
        labels: the 0/1 labels to permute.
        seed: RNG seed; a fixed seed is reproducible, different seeds give
            different permutations.
    """
    rng = np.random.default_rng(seed)
    arr = np.asarray(labels)
    return rng.permutation(arr).tolist()


def probe_auc(
    probe: InternalProbe,
    features: Sequence[ActivationFeatures],
    labels: Sequence[int],
) -> float:
    """ROC-AUC of ``probe`` over labelled ``features`` (benign=0, injected=1).

    Splits the probe's per-feature scores by true label and defers to the shared
    ``eval.metrics.roc_auc`` so every layer/control reports AUC identically (DRY).

    Raises:
        ValueError: if ``features``/``labels`` misalign or both classes are not
            present (AUC is undefined for a single class).
    """
    if len(features) != len(labels):
        raise ValueError(
            f"features and labels must align (got {len(features)} vs {len(labels)})"
        )
    y = np.asarray(labels, dtype=int)
    if set(np.unique(y).tolist()) != {0, 1}:
        raise ValueError("need both benign (0) and injected (1) labels to compute AUC")
    scores = np.array([probe.score(f).value for f in features])
    _, _, auc = roc_auc(scores[y == 0], scores[y == 1])
    return auc
