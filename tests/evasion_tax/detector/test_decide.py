"""Tests for the single-step / per-rollout decision logic (Task 6).

Covers: strict ``value > tau`` firing with the boundary (``value == tau`` → not
hold), causal first-exceedance scanning in ``rollout_fires`` (never looks ahead),
the no-fire sentinel ``step == -1``, and ``detection_latency`` semantics
(0 at onset, positive when later, ``None`` when never fired or fires before onset).
"""

from evasion_tax.detector.decide import decide, detection_latency, rollout_fires
from evasion_tax.records import Decision, Score


def scores(values, *, start=0) -> list[Score]:
    """Build an ordered list of Score objects with window_end = start, start+1, ..."""
    return [Score(value=v, window_end=start + i) for i, v in enumerate(values)]


# --------------------------------------------------------------------------- #
# decide (single step)                                                         #
# --------------------------------------------------------------------------- #


def test_decide_fires_when_value_strictly_exceeds_tau():
    d = decide(Score(value=0.8, window_end=3), tau=0.5)
    assert d == Decision(hold=True, step=3)


def test_decide_does_not_fire_when_value_below_tau():
    d = decide(Score(value=0.2, window_end=3), tau=0.5)
    assert d == Decision(hold=False, step=3)


def test_decide_boundary_value_equal_tau_does_not_fire():
    # Strict >: value == tau must NOT hold.
    d = decide(Score(value=0.5, window_end=7), tau=0.5)
    assert d.hold is False
    assert d.step == 7


def test_decide_matches_value_gt_tau_exactly_over_a_grid():
    tau = 0.42
    for v in [0.0, 0.41, 0.42, 0.43, 0.99, 1.0]:
        d = decide(Score(value=v, window_end=0), tau=tau)
        assert d.hold == (v > tau)


# --------------------------------------------------------------------------- #
# rollout_fires (causal first-exceedance)                                     #
# --------------------------------------------------------------------------- #


def test_rollout_fires_returns_first_exceedance_step():
    # Exceeds at index 0 (window_end 10) and again later; must return the FIRST.
    s = scores([0.1, 0.2, 0.9, 0.95, 0.99], start=10)
    d = rollout_fires(s, threshold=0.5)
    assert d.hold is True
    assert d.step == 12  # window_end of the first value (0.9) above tau


def test_rollout_fires_is_causal_first_not_max():
    # A later, larger spike must not override the first exceedance.
    s = scores([0.0, 0.6, 0.99], start=0)
    d = rollout_fires(s, threshold=0.5)
    assert d.step == 1  # the 0.6 step, not the 0.99 step


def test_rollout_fires_no_exceedance_returns_hold_false_step_minus_one():
    s = scores([0.0, 0.1, 0.49, 0.5], start=0)  # 0.5 == tau → not a fire
    d = rollout_fires(s, threshold=0.5)
    assert d == Decision(hold=False, step=-1)


def test_rollout_fires_empty_rollout_does_not_fire():
    d = rollout_fires([], threshold=0.5)
    assert d == Decision(hold=False, step=-1)


def test_rollout_fires_does_not_mutate_input():
    s = scores([0.1, 0.9], start=0)
    snapshot = list(s)
    rollout_fires(s, threshold=0.5)
    assert s == snapshot


# --------------------------------------------------------------------------- #
# detection_latency                                                            #
# --------------------------------------------------------------------------- #


def test_detection_latency_zero_when_fire_at_onset():
    d = Decision(hold=True, step=20)
    assert detection_latency(d, attack_onset_step=20) == 0


def test_detection_latency_positive_when_fire_after_onset():
    d = Decision(hold=True, step=25)
    assert detection_latency(d, attack_onset_step=20) == 5


def test_detection_latency_none_when_never_fired():
    d = Decision(hold=False, step=-1)
    assert detection_latency(d, attack_onset_step=20) is None


def test_detection_latency_none_when_fire_before_onset():
    # A fire strictly before attack onset is not a valid detection of THIS attack.
    d = Decision(hold=True, step=15)
    assert detection_latency(d, attack_onset_step=20) is None
