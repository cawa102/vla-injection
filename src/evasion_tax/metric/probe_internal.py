"""L1 internal-representation probe — the activation-delta arm (playbook §4b-(I)).

The **L1** layer of the Embodiment-Evasion-Tax instrument (L0 input < L1
internal-probe < L2 action-monitor). Primary form = an **activation-delta linear
probe** (Task-Drift lineage, ``2406.00799``): logistic regression over the
hidden-state delta across the injection point in OpenVLA's transformer trunk,
trained benign-vs-injected. It emits one ``Score`` per rollout and calibrates
through the **same** ``calibrate`` every layer reuses (plan invariant #4), so L0
/ L1 / L2 are compared on one footing.

Structure (Dependency Inversion, mirroring the perplexity baseline's seam):

* :class:`ActivationExtractor` — the swappable activations backend.
* :class:`SyntheticActivationExtractor` — deterministic, model-free stand-in for
  local tests (no OpenVLA); fabricates a class-correlated feature exactly as the
  perplexity mock correlates symbol density with injection.
* :class:`RealActivationExtractor` — the OpenVLA forward-pass backend, a GPU-only
  stub.
* :class:`InternalProbe` — the fitted logistic-regression probe.

Scope note: the **attention-map MLP** ablation (AlignSentinel lineage,
``2602.13597``; differentiate from Concept-Dictionary ``2602.01834`` / IGAR
``2603.06001``) is pre-registered but deferred to M2 on the GPU — it extends the
feature record + adds an MLP head then, and is intentionally **not** built here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from sklearn.linear_model import LogisticRegression

from evasion_tax.records import Rollout, Score
from evasion_tax.repro import stable_seed

# Fixed seed for the synthetic class-signal direction: stable across processes
# and runs (reproducibility — never the salted built-in hash()).
_DIRECTION_SEED = 20260603


@dataclass(frozen=True)
class ActivationFeatures:
    """Decision-step features for one rollout (immutable; plan invariant #6).

    Attributes:
        activation_delta: hidden-state delta across the injection point — the
            primary probe feature. Coerced to a float tuple; must be non-empty
            and finite.
        window_end: the step index the resulting score is anchored to. An
            instruction-channel decision is made up-front, so it defaults to 0
            (as the perplexity filter does).
    """

    activation_delta: tuple[float, ...]
    window_end: int = 0

    def __post_init__(self) -> None:
        try:
            coerced = tuple(float(x) for x in self.activation_delta)
        except (TypeError, ValueError) as exc:
            raise ValueError("activation_delta must be a sequence of numbers") from exc
        if len(coerced) == 0:
            raise ValueError("activation_delta must be non-empty")
        if not all(np.isfinite(coerced)):
            raise ValueError("activation_delta must be finite")
        object.__setattr__(self, "activation_delta", coerced)


@runtime_checkable
class ActivationExtractor(Protocol):
    """Seam mapping a rollout to its decision-step :class:`ActivationFeatures`.

    The deployable L1 backend (:class:`RealActivationExtractor`) runs an OpenVLA
    forward pass on the GPU node; the probe depends only on this abstraction,
    never on the model API.
    """

    def extract(self, rollout: Rollout) -> ActivationFeatures:
        """Return the decision-step activation features for ``rollout``."""
        ...


def _signal_direction(dim: int) -> np.ndarray:
    """A fixed unit vector the synthetic injected-class shift points along."""
    d = np.random.default_rng(_DIRECTION_SEED).standard_normal(dim)
    return d / np.linalg.norm(d)


class SyntheticActivationExtractor:
    """Deterministic, model-free activation features for local tests (no OpenVLA).

    A test stand-in only — like :class:`~evasion_tax.baselines.perplexity.MockPerplexityScorer`,
    it *fabricates* a class-correlated signal so the probe's mechanics are
    exercisable on the local dev host. Per-rollout noise is seeded from
    ``(run_id, task_id, seed)`` (process-stable), so a given rollout always yields
    the same features; an injected rollout (``steps[0].attacked``) is shifted by
    ``signal`` along a fixed direction. ``signal = 0`` makes the classes
    identical (the confound floor). The real backend replaces this on the GPU.

    Args:
        dim: feature dimensionality (>= 1).
        signal: injected-class mean shift along the fixed direction (>= 0).
    """

    def __init__(self, *, dim: int = 16, signal: float = 2.0) -> None:
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        if signal < 0:
            raise ValueError(f"signal must be >= 0, got {signal}")
        self._dim = dim
        self._signal = float(signal)
        self._direction = _signal_direction(dim)

    def extract(self, rollout: Rollout) -> ActivationFeatures:
        if len(rollout.steps) == 0:
            raise ValueError("cannot extract features from an empty rollout")
        step = rollout.steps[0]
        rng = np.random.default_rng(stable_seed(step.run_id, step.task_id, step.seed))
        vec = rng.standard_normal(self._dim)
        if step.attacked:
            vec = vec + self._signal * self._direction
        return ActivationFeatures(activation_delta=tuple(map(float, vec)), window_end=0)


class RealActivationExtractor:
    """The OpenVLA forward-pass backend — deferred to the GPU node (raises here).

    On the granted A100/H100 this returns the hidden-state delta across the
    injection point at the decision step; it requires the model and so is
    unavailable on the local host.
    """

    def extract(self, rollout: Rollout) -> ActivationFeatures:
        raise NotImplementedError(
            "GPU: the real OpenVLA activation extractor requires the model and is "
            "not available on a local dev host without CUDA; use "
            "SyntheticActivationExtractor for tests."
        )


def _feature_matrix(features: Sequence[ActivationFeatures], *, n_features: int) -> np.ndarray:
    """Stack features into an ``(n, n_features)`` array, validating each width."""
    rows = []
    for i, f in enumerate(features):
        if len(f.activation_delta) != n_features:
            raise ValueError(
                f"feature {i} has dim {len(f.activation_delta)}, expected {n_features}"
            )
        rows.append(f.activation_delta)
    return np.asarray(rows, dtype=float)


@dataclass(frozen=True)
class InternalProbe:
    """A fitted activation-delta logistic-regression probe (higher = more injected).

    Construct via :meth:`fit`; the dataclass holds the fitted estimator and the
    expected feature width so a mis-shaped feature at score time fails loudly.
    """

    model: LogisticRegression
    n_features: int
    seed: int = 0

    @classmethod
    def fit(
        cls,
        features: Sequence[ActivationFeatures],
        labels: Sequence[int],
        *,
        seed: int = 0,
        C: float = 1.0,
    ) -> InternalProbe:
        """Fit the probe on labelled features (0 = benign, 1 = injected).

        Args:
            features: per-rollout :class:`ActivationFeatures`.
            labels: matching 0/1 labels; both classes must be present.
            seed: estimator ``random_state`` (reproducibility).
            C: inverse L2-regularisation strength.

        Raises:
            ValueError: if ``features``/``labels`` differ in length, fewer than
                both classes are present, or feature widths are inconsistent.
        """
        if len(features) != len(labels):
            raise ValueError(
                f"features and labels must align (got {len(features)} vs {len(labels)})"
            )
        if len(features) == 0:
            raise ValueError("need at least one feature to fit the probe")
        y = np.asarray(labels, dtype=int)
        if set(np.unique(y).tolist()) != {0, 1}:
            raise ValueError("need both benign (0) and injected (1) labels to fit the probe")
        n_features = len(features[0].activation_delta)
        x = _feature_matrix(features, n_features=n_features)
        # `random_state` is inert under the default `lbfgs` solver (deterministic);
        # `seed` is recorded on the probe for provenance and stays meaningful for
        # any future variant whose solver/head consumes `random_state`.
        model = LogisticRegression(C=C, max_iter=1000, random_state=seed).fit(x, y)
        return cls(model=model, n_features=n_features, seed=seed)

    def score(self, features: ActivationFeatures) -> Score:
        """Score one rollout's features → injection probability in ``[0, 1]``."""
        if len(features.activation_delta) != self.n_features:
            raise ValueError(
                f"feature dim {len(features.activation_delta)} != probe dim {self.n_features}"
            )
        x = np.asarray(features.activation_delta, dtype=float).reshape(1, -1)
        p = float(self.model.predict_proba(x)[0, 1])
        return Score(value=float(np.clip(p, 0.0, 1.0)), window_end=features.window_end)

    def score_rollout(
        self, rollout: Rollout, extractor: ActivationExtractor
    ) -> list[Score]:
        """One decision score for ``rollout`` (a single-element list, like L0).

        Extracts the decision-step features via ``extractor`` and scores them, so
        the result feeds the **same** ``calibrate`` / ``rollout_fires`` as every
        other layer (invariant #4).
        """
        return [self.score(extractor.extract(rollout))]
