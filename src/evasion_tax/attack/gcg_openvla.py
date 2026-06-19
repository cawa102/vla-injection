"""OpenVLA loss/gradient seam for the GCG core (step-6 Task 2) — GPU-guarded.

Implements the :class:`~evasion_tax.attack.gcg.LossGradientFn` Protocol against the
**frozen bf16** OpenVLA-7B, reusing the verified step-5.5 machinery
(``scripts/smoke_openvla_gradient.py``): the token-embedding forward hook whose
``delta.grad`` *is* ``d(loss)/d(inputs_embeds)``, and OpenVLA's built-in CE loss to
a teacher-forced target action-token span. ``token_gradient`` projects that
input-embedding gradient through the embedding matrix to the ``[L, vocab]`` one-hot
gradient GCG needs; ``loss_of`` batches candidate suffixes through the frozen
forward (no grad).

**Split (D6-2).** The *pure* pieces — :func:`project_onehot_grad` and
:func:`suffix_span_in_ids` — are model-free and unit-tested off-GPU. The real
forward/gradient (``OpenVlaGcgTarget`` + the D6-9 faithfulness gate) is GPU-only:
**torch / transformers / PIL are imported inside the guarded methods, never at
module top**, so this module stays importable on a CUDA-free host exactly like the
smoke scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def project_onehot_grad(
    grad_embeds_suffix: np.ndarray, embedding_matrix: np.ndarray
) -> np.ndarray:
    """Project the input-embedding gradient to the one-hot token gradient.

    For suffix position ``i``, the linearised loss change of swapping token
    ``i → v`` is ``g[i, v] = (d loss / d e_i) · W[v, :]`` (GCG / nanoGCG). With the
    per-position embedding gradient ``grad_embeds_suffix`` (``[L, d]``, the
    step-5.5 ``delta.grad`` rows at the suffix span) and the token-embedding matrix
    ``W`` (``[V, d]``), this is ``grad_embeds_suffix @ W.T`` → ``[L, V]``.

    Args:
        grad_embeds_suffix: ``[L, d]`` input-embedding gradient at the suffix span.
        embedding_matrix: ``[V, d]`` token-embedding matrix ``W``.

    Returns:
        ``[L, V]`` one-hot token gradient — the array the model-free core consumes.
    """
    return np.asarray(grad_embeds_suffix) @ np.asarray(embedding_matrix).T


def suffix_span_in_ids(prompt_ids: np.ndarray, suffix_len: int) -> slice:
    """Locate the adversarial suffix as the trailing ``suffix_len`` tokens.

    The seam builds the prompt head ``prefix + instruction + suffix`` and appends
    the fixed template tail (``?\\nOut:``) and the teacher-forced target span
    *separately*, so within the head the GCG suffix is the **trailing run** of
    ``suffix_len`` tokens. The on-box ``decode_span`` faithfulness gate (D6-9)
    confirms this slice decodes to the intended adversarial text against the real
    processor — the placement detail RoboGCG leaves to ``[VERIFY on the box]``.

    Args:
        prompt_ids: 1-D token ids of the prompt head (instruction + suffix).
        suffix_len: Number of trailing suffix tokens.

    Returns:
        A ``slice`` selecting the suffix token positions.

    Raises:
        ValueError: If ``suffix_len`` is not in ``1 .. len(prompt_ids)``.
    """
    n = int(np.asarray(prompt_ids).shape[-1])
    if not (1 <= suffix_len <= n):
        raise ValueError(
            f"suffix_len must be in 1..{n} (len(prompt_ids)), got {suffix_len}"
        )
    return slice(n - suffix_len, n)


# OpenVLA HF model-card prompt template (RoboGCG `model_utils.py` confirms the same
# string; the GCG suffix is spliced after the instruction text — Task 0 notes B).
_PROMPT_PREFIX = "In: What action should the robot take to {instruction}"
_PROMPT_TAIL = "?\nOut:"
_LABEL_IGNORE = -100  # HF CE ignore index (matches smoke_openvla_gradient.py).


@dataclass(frozen=True)
class FaithfulnessReport:
    """Result of the D6-9 GPU seam-faithfulness gate.

    The projected one-hot gradient is only unit-tested on fake arrays; on the box
    we check it predicts the *measured* one-token-swap loss delta (finite
    difference), and that the suffix span decodes to the intended adversarial text.

    Attributes:
        n_samples: Number of (position, token) swaps sampled.
        sign_agreement: Fraction of swaps where the projected gradient's predicted
            loss-change **sign** matched the measured finite-difference delta.
        decoded_suffix: The suffix span decoded under the real processor.
        passed: ``sign_agreement`` is above chance (the seam is faithful, not merely
            plausible — span, dtype, tied embeddings, sign convention all hold).
    """

    n_samples: int
    sign_agreement: float
    decoded_suffix: str
    passed: bool


class OpenVlaGcgTarget:
    """GPU-only :class:`~evasion_tax.attack.gcg.LossGradientFn` for frozen bf16 OpenVLA-7B.

    Reuses the verified step-5.5 mechanism (``scripts/smoke_openvla_gradient.py``):
    weights frozen, a forward hook on the token-embedding module installs a zero
    ``delta`` leaf so ``delta.grad`` == ``d(loss)/d(inputs_embeds)``, and OpenVLA's
    built-in CE loss to a teacher-forced target action-token span is the objective.

    **Construction (exact suffix span).** The text input ids are built by
    *concatenation* — ``prefix_ids ⊕ suffix_ids ⊕ tail_ids ⊕ target_ids`` — so the
    suffix occupies a known, exact span (``suffix_span_in_ids`` on the head); no
    re-tokenisation search. ``pixel_values`` come from the processor (image
    processing is independent of the text); OpenVLA fuses the image patches inside
    ``forward`` and masks their label positions internally (the 5.5 finding).

    **GPU-only / guarded.** ``torch`` / ``transformers`` / ``PIL`` are imported
    inside the methods, never at module top, so this module stays importable on a
    CUDA-free host. Every method here needs a loaded model + CUDA; the model-free
    core never calls them off-GPU.

    Args:
        model: A loaded, frozen bf16 OpenVLA-7B (``requires_grad_(False)``).
        processor: The matching ``AutoProcessor`` (its ``.tokenizer`` is reused).
        image: A ``PIL.Image`` observation.
        instruction: The benign instruction text the suffix is appended to.
        suffix_len: Number of adversarial suffix tokens.
        target_action_ids: The 7 target action token ids (teacher-forced span).
        device: A ``torch.device``.
        init_suffix_token_id: Initial suffix filler token id (default: a fixed
            benign token); the search overwrites these.
    """

    def __init__(
        self,
        model: Any,
        processor: Any,
        *,
        image: Any,
        instruction: str,
        suffix_len: int,
        target_action_ids: np.ndarray,
        device: Any,
        init_suffix_token_id: int | None = None,
    ) -> None:
        if suffix_len < 1:
            raise ValueError(f"suffix_len must be >= 1, got {suffix_len}")
        self._model = model
        self._processor = processor
        self._device = device
        self._tokenizer = processor.tokenizer
        self._vocab_size = int(self._tokenizer.vocab_size)
        self._suffix_len = int(suffix_len)
        self._target_action_ids = np.asarray(target_action_ids, dtype=np.int64)

        # Text token layout by concatenation (exact, search-free suffix span).
        prefix_ids = self._encode(_PROMPT_PREFIX.format(instruction=instruction), bos=True)
        tail_ids = self._encode(_PROMPT_TAIL, bos=False)
        if init_suffix_token_id is None:
            # A fixed benign filler; the search overwrites it. Encoding of " x".
            init_ids = self._encode(" x", bos=False)
            init_suffix_token_id = int(init_ids[-1])
        self._init_suffix = np.full(self._suffix_len, int(init_suffix_token_id), dtype=np.int64)

        head_ids = np.concatenate([prefix_ids, self._init_suffix])
        self._suffix_span = suffix_span_in_ids(head_ids, self._suffix_len)
        # Static (non-suffix) layout pieces, concatenated at call time around the suffix.
        self._prefix_ids = prefix_ids
        self._tail_ids = tail_ids
        # Prompt length (everything that is NOT the teacher-forced target span).
        self._prompt_len = len(prefix_ids) + self._suffix_len + len(tail_ids)

        # pixel_values: processor handles the image independently of the text.
        proc = processor(_PROMPT_PREFIX.format(instruction=instruction) + _PROMPT_TAIL, image)
        self._pixel_values = proc["pixel_values"].to(device)

    # -- pure-ish helpers (still off-GPU-safe: tokenizer is CPU) --------------- #

    def _encode(self, text: str, *, bos: bool) -> np.ndarray:
        ids = self._tokenizer(text, add_special_tokens=bos, return_tensors="np")["input_ids"][0]
        return np.asarray(ids, dtype=np.int64)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def suffix_span(self) -> slice:
        """The exact suffix token span within the prompt (head) ids."""
        return self._suffix_span

    def init_suffix_ids(self) -> np.ndarray:
        return self._init_suffix.copy()

    # -- GPU body (torch imported inside; never runs off-GPU) ------------------ #

    def _full_ids(self, suffix_ids: np.ndarray) -> np.ndarray:
        """Concatenate prefix ⊕ suffix ⊕ tail ⊕ target into one text id row."""
        return np.concatenate(
            [self._prefix_ids, np.asarray(suffix_ids, dtype=np.int64),
             self._tail_ids, self._target_action_ids]
        )

    def _labels(self, full_ids: np.ndarray) -> np.ndarray:
        """Mask everything but the teacher-forced target action span (CE objective)."""
        labels = np.full_like(full_ids, _LABEL_IGNORE)
        labels[self._prompt_len:] = full_ids[self._prompt_len:]
        return labels

    def token_gradient(self, suffix_ids: np.ndarray) -> np.ndarray:
        """``[L, vocab]`` one-hot token gradient at ``suffix_ids`` (GPU).

        Runs the 5.5 hooked backward, then projects the suffix-span input-embedding
        gradient through the token-embedding matrix (:func:`project_onehot_grad`).
        """
        import torch  # type: ignore[import-not-found]

        full_ids = self._full_ids(suffix_ids)
        input_ids = torch.tensor(full_ids[None, :], device=self._device, dtype=torch.long)
        attn = torch.ones_like(input_ids)
        labels = torch.tensor(
            self._labels(full_ids)[None, :], device=self._device, dtype=torch.long
        )

        captured: dict[str, Any] = {}

        def _embed_hook(_module, _args, output):  # type: ignore[no-untyped-def]
            if "delta" not in captured:
                captured["delta"] = torch.zeros_like(output, requires_grad=True)
            return output + captured["delta"]

        handle = self._model.get_input_embeddings().register_forward_hook(_embed_hook)
        try:
            self._model.zero_grad(set_to_none=True)
            out = self._model(
                input_ids=input_ids,
                attention_mask=attn,
                pixel_values=self._pixel_values,
                labels=labels,
            )
            out.loss.backward()
        finally:
            handle.remove()

        grad = captured["delta"].grad  # [1, n_text, hidden]
        grad_suffix = grad[0, self._suffix_span, :]  # [L, hidden]
        weight = self._model.get_input_embeddings().weight  # [V, hidden]
        # token_grad = grad_suffix @ W.T (project_onehot_grad), done on-GPU in fp32.
        token_grad = (grad_suffix.float() @ weight.float().t()).detach().cpu().numpy()
        return token_grad

    def _loss_single(self, suffix_ids: np.ndarray) -> float:
        """Reference per-candidate loss via OpenVLA's built-in CE (the 5.5 scalar)."""
        import torch  # type: ignore[import-not-found]

        full_ids = self._full_ids(suffix_ids)
        input_ids = torch.tensor(full_ids[None, :], device=self._device, dtype=torch.long)
        attn = torch.ones_like(input_ids)
        labels = torch.tensor(
            self._labels(full_ids)[None, :], device=self._device, dtype=torch.long
        )
        with torch.no_grad():
            out = self._model(
                input_ids=input_ids,
                attention_mask=attn,
                pixel_values=self._pixel_values,
                labels=labels,
            )
        return float(out.loss.detach().float())

    def loss_of(self, candidate_suffixes: np.ndarray) -> np.ndarray:
        """``[B, L]`` candidates → ``[B]`` CE losses (no grad, peak-VRAM tracked).

        MVP batched path = a loop of the 5.5-verified single forward; a tighter
        single-forward batched CE is a box optimisation gated by the D6-3 wiring
        check (``loss_of`` batch == per-candidate single — :meth:`batched_matches_single`).
        Resets/reads CUDA peak stats so Task 4 can record peak VRAM per call.
        """
        import torch  # type: ignore[import-not-found]

        cands = np.asarray(candidate_suffixes, dtype=np.int64)
        if cands.ndim != 2 or cands.shape[1] != self._suffix_len:
            raise ValueError(
                f"candidate_suffixes must be [B, {self._suffix_len}], got {cands.shape}"
            )
        torch.cuda.reset_peak_memory_stats(self._device)
        losses = np.array([self._loss_single(row) for row in cands], dtype=float)
        self._last_peak_bytes = int(torch.cuda.max_memory_allocated(self._device))
        return losses

    # -- D6-9 faithfulness gate (on the box, BEFORE the tiny run) -------------- #

    def decode_span(self, suffix_ids: np.ndarray | None = None) -> str:
        """Decode the suffix span to text under the real processor (span sanity)."""
        ids = self._init_suffix if suffix_ids is None else np.asarray(suffix_ids, dtype=np.int64)
        return self._tokenizer.decode(ids.tolist())

    def gradient_agrees_with_swaps(
        self, *, n_samples: int, rng: np.random.Generator
    ) -> FaithfulnessReport:
        """Check the projected gradient's sign predicts measured one-token-swap deltas.

        For ``n_samples`` random (position, token) swaps of the init suffix, compare
        ``sign(g[i, v_new] - g[i, v_old])`` (the linearised predicted loss change of
        the swap) with ``sign(loss(swapped) - loss(init))`` (finite difference). A
        faithful seam agrees well above chance; a wrong span / dtype / tied-embedding
        / sign slip does not (D6-9).
        """
        base = self._init_suffix
        grad = self.token_gradient(base)  # [L, V]
        base_loss = self._loss_single(base)
        agree = 0
        for _ in range(n_samples):
            pos = int(rng.integers(0, self._suffix_len))
            new_tok = int(rng.integers(0, self._vocab_size))
            predicted = float(grad[pos, new_tok] - grad[pos, int(base[pos])])
            swapped = base.copy()
            swapped[pos] = new_tok
            measured = self._loss_single(swapped) - base_loss
            if (predicted < 0) == (measured < 0):
                agree += 1
        sign_agreement = agree / n_samples if n_samples else 0.0
        return FaithfulnessReport(
            n_samples=n_samples,
            sign_agreement=sign_agreement,
            decoded_suffix=self.decode_span(base),
            passed=sign_agreement > 0.5,
        )

    def batched_matches_single(self, candidate_suffixes: np.ndarray, *, atol: float = 1e-3) -> bool:
        """Wiring gate (D6-3): batched ``loss_of`` equals per-candidate single eval."""
        batched = self.loss_of(candidate_suffixes)
        single = np.array(
            [self._loss_single(r) for r in np.asarray(candidate_suffixes)], dtype=float
        )
        return bool(np.allclose(batched, single, atol=atol))
