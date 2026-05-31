"""Policy-side helpers (Task 3).

Currently exposes the OpenVLA action codec (discrete token ids -> continuous
7-DoF). Model inference itself is deferred to GB10; this is the model-free decode
path, with its formula verified against the OpenVLA source (see
``action_codec`` module docstring for provenance).
"""

from t7.policy.action_codec import ActionCodec

__all__ = ["ActionCodec"]
