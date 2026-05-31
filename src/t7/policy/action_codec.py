"""OpenVLA action codec: discrete action token ids -> continuous 7-DoF (Task 3).

This is the one offline-buildable "policy" piece: de-tokenise OpenVLA's discrete
action tokens and un-normalise them to continuous actions. On GB10 the model's
own ``predict_action`` performs this; we re-implement the *decode* direction so
that RoboGCG target *tokens* can be mapped to a continuous action region
(``TargetActionSpec``, decision D2) without loading the 7B model.

**Provenance (invariant #8 — formula verified from source, not memory).** Source:
``openvla/openvla`` @ commit ``c8f03f48af692657d3060c19588038c7220e9af9``:

* ``prismatic/vla/action_tokenizer.py`` (ll. 31-32, 65-68):
  ``bins = np.linspace(-1, 1, n_bins)``; ``bin_centers = (bins[:-1]+bins[1:])/2``;
  ``discretized = vocab_size - token_id``; ``idx = clip(discretized - 1, 0,
  len(bin_centers)-1)``; ``norm = bin_centers[idx]``.
* ``prismatic/extern/hf/modeling_prismatic.py`` (ll. 522-534): the identical
  decode, then ``np.where(mask, 0.5*(norm+1)*(q99-q01)+q01, norm)``.

Two faithfulness points that differ from a naive reading:

* Decode depends on ``vocab_size`` — the OpenVLA HF de-tokenisation size
  (``config.text_config.vocab_size - config.pad_to_multiple_of``). It is a
  required construction argument; the real checkpoint value is recorded as
  provenance, never hardcoded here.
* The gripper dim is **not** a hardcoded passthrough: un-normalisation is
  ``mask``-driven. Dimensions whose ``mask`` is ``False`` pass the normalised
  value through unchanged; LIBERO's stats happen to set only the final
  (gripper) dim to ``False``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ActionCodec:
    """Decode OpenVLA discrete action tokens to continuous actions.

    Immutable (coding-style + plan invariant #6). Build from fetched dataset
    statistics with :meth:`from_stats`, or directly from per-dim quantiles.

    Attributes:
        q01: Per-dim 1st-percentile action statistic (length = action dim).
        q99: Per-dim 99th-percentile action statistic.
        mask: Per-dim un-normalisation mask; ``False`` dims pass through.
        vocab_size: OpenVLA de-tokenisation vocab size (see module docstring).
        n_bins: Number of uniform action bins (OpenVLA default 256).
    """

    q01: tuple[float, ...]
    q99: tuple[float, ...]
    mask: tuple[bool, ...]
    vocab_size: int
    n_bins: int = 256

    def __post_init__(self) -> None:
        # Coerce to immutable tuples of the right element types (boundary check).
        object.__setattr__(self, "q01", tuple(float(x) for x in self.q01))
        object.__setattr__(self, "q99", tuple(float(x) for x in self.q99))
        object.__setattr__(self, "mask", tuple(bool(x) for x in self.mask))

        if not (len(self.q01) == len(self.q99) == len(self.mask)):
            raise ValueError(
                "q01, q99 and mask must have the same length (got "
                f"q01={len(self.q01)}, q99={len(self.q99)}, mask={len(self.mask)})"
            )
        if len(self.q01) == 0:
            raise ValueError("action statistics must be non-empty")
        for lo, hi in zip(self.q01, self.q99, strict=True):
            if lo > hi:
                raise ValueError(f"q01 must be <= q99 per dim, got q01={lo} > q99={hi}")
        if self.n_bins < 2:
            raise ValueError(f"n_bins must be >= 2, got {self.n_bins}")
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")

    @classmethod
    def from_stats(
        cls,
        stats: Mapping[str, object],
        unnorm_key: str,
        *,
        vocab_size: int,
        n_bins: int = 256,
    ) -> ActionCodec:
        """Build a codec from a fetched ``dataset_statistics`` mapping.

        Mirrors OpenVLA's ``get_action_stats``: reads
        ``stats[unnorm_key]["action"]`` for ``q01``, ``q99`` and optional
        ``mask`` (default all-True, matching the source).

        Args:
            stats: The ``dataset_statistics.json`` mapping (key -> per-dataset block).
            unnorm_key: Which dataset's statistics to use (e.g. ``libero_spatial_no_noops``).
            vocab_size: OpenVLA de-tokenisation vocab size.
            n_bins: Number of action bins.

        Raises:
            KeyError: If ``unnorm_key`` is absent (message lists available keys).
            ValueError: If the selected block lacks ``action`` q01/q99.
        """
        if unnorm_key not in stats:
            raise KeyError(
                f"unnorm_key {unnorm_key!r} not in dataset statistics; "
                f"available: {sorted(stats.keys())}"
            )
        block = stats[unnorm_key]
        if not isinstance(block, Mapping) or "action" not in block:
            raise ValueError(f"stats[{unnorm_key!r}] has no 'action' statistics block")
        action = block["action"]
        if not isinstance(action, Mapping) or "q01" not in action or "q99" not in action:
            raise ValueError(f"action stats for {unnorm_key!r} must contain 'q01' and 'q99'")

        q01 = tuple(float(x) for x in action["q01"])
        q99 = tuple(float(x) for x in action["q99"])
        raw_mask = action.get("mask", [True] * len(q01))
        mask = tuple(bool(x) for x in raw_mask)
        return cls(q01=q01, q99=q99, mask=mask, vocab_size=vocab_size, n_bins=n_bins)

    @property
    def action_dim(self) -> int:
        """Dimensionality of the action space (= number of quantile entries)."""
        return len(self.q01)

    @property
    def bins(self) -> np.ndarray:
        """The ``n_bins`` uniform bin edges over ``[-1, 1]``."""
        return np.linspace(-1.0, 1.0, self.n_bins)

    @property
    def bin_centers(self) -> np.ndarray:
        """The ``n_bins - 1`` bin centres (what de-tokenised tokens map to)."""
        edges = self.bins
        return (edges[:-1] + edges[1:]) / 2.0

    def token_to_bin(self, token_id: int) -> int:
        """Map a single action token id to a bin-centre index in ``[0, n_bins-2]``.

        Implements the verified offset + documented off-by-one clip:
        ``idx = clip((vocab_size - token_id) - 1, 0, len(bin_centers)-1)``.
        """
        discretized = self.vocab_size - int(token_id)
        return int(np.clip(discretized - 1, 0, self.bin_centers.shape[0] - 1))

    def bin_to_norm(self, bin_index: int) -> float:
        """Return the normalised value (bin centre, in ``[-1, 1]``) for a bin index."""
        return float(self.bin_centers[bin_index])

    def unnormalize(self, norm: np.ndarray) -> np.ndarray:
        """Un-normalise per-dim normalised actions using q01/q99 under ``mask``.

        ``actions = where(mask, 0.5*(norm+1)*(q99-q01)+q01, norm)`` — masked dims
        are rescaled to the dataset range; unmasked dims pass through.
        """
        q01 = np.asarray(self.q01, dtype=float)
        q99 = np.asarray(self.q99, dtype=float)
        mask = np.asarray(self.mask, dtype=bool)
        norm = np.asarray(norm, dtype=float)
        return np.where(mask, 0.5 * (norm + 1.0) * (q99 - q01) + q01, norm)

    def decode(self, token_ids: Sequence[int]) -> np.ndarray:
        """Decode a full action's token ids to a continuous action vector.

        Args:
            token_ids: Exactly ``action_dim`` token ids (one per action dim).

        Returns:
            A length-``action_dim`` float array of continuous actions.

        Raises:
            ValueError: If ``len(token_ids) != action_dim``.
        """
        ids = list(token_ids)
        if len(ids) != self.action_dim:
            raise ValueError(
                f"expected {self.action_dim} token ids (one per action dim), got {len(ids)}"
            )
        norm = np.array([self.bin_to_norm(self.token_to_bin(t)) for t in ids], dtype=float)
        return self.unnormalize(norm)
