"""Policy-side helpers (Task 3).

Currently exposes the OpenVLA action codec (discrete token ids -> continuous
7-DoF). Model inference itself is deferred to the GPU node; this is the model-free decode
path, with its formula verified against the OpenVLA source (see
``action_codec`` module docstring for provenance).
"""

from evasion_tax.policy.action_codec import ActionCodec

__all__ = ["ActionCodec"]
