"""Policy-side helpers (Task 3).

Exposes the OpenVLA action codec (discrete token ids -> continuous 7-DoF, the
model-free decode path verified against the OpenVLA source — see ``action_codec``
module docstring for provenance) and :func:`validate_action_vector`, the
model-free "valid action vector" gate the GPU load smoke / rollout body assert
against. Model inference itself is deferred to the GPU node.
"""

from evasion_tax.policy.action_check import validate_action_vector
from evasion_tax.policy.action_codec import ActionCodec

__all__ = ["ActionCodec", "validate_action_vector"]
