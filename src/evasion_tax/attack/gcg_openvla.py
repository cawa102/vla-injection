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

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np


def chunked_losses(
    full: np.ndarray,
    eval_batch: int | None,
    forward_fn: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Score ``[B, seq]`` rows in chunks of ``eval_batch``, concatenating per-chunk losses.

    ``eval_batch=None`` ⇒ **one** ``forward_fn`` call over all rows (the single-forward
    path, preserved bit-for-bit so the D8 ``sw=32`` path and its tests are unchanged). An
    int ⇒ ``ceil(B/eval_batch)`` chunks scored in order and concatenated. Because per-row
    CE is independent, the concatenated ``[B]`` equals the single-forward ``[B]`` — a pure
    reassociation — so the GPU :meth:`OpenVlaGcgTarget.loss_of` fits 24 GB at
    ``search_width=512`` by capping the per-forward batch (DE-7), peak VRAM becoming the
    **max single-chunk** peak. This helper is torch-free; ``forward_fn`` owns the GPU work.

    Args:
        full: ``[B, seq]`` candidate rows (already concatenated prompt+target ids).
        eval_batch: Chunk size; ``None`` for one forward over all rows.
        forward_fn: ``rows -> [n]`` per-row losses for one chunk (the GPU forward).

    Returns:
        ``[B]`` per-row losses, row order preserved.

    Raises:
        ValueError: If ``eval_batch`` is an int ``< 1`` (no silent zero-chunk).
    """
    rows = np.asarray(full)
    if eval_batch is None:
        return np.asarray(forward_fn(rows))
    if eval_batch < 1:
        raise ValueError(f"eval_batch must be >= 1 or None, got {eval_batch}")
    parts = [
        np.asarray(forward_fn(rows[start : start + eval_batch]))
        for start in range(0, rows.shape[0], eval_batch)
    ]
    return np.concatenate(parts)


def per_sequence_ce(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    ignore_index: int = -100,
) -> np.ndarray:
    """Per-sequence mean cross-entropy (the contract the true-batch ``loss_of`` mirrors).

    Causal-LM shift: predict ``labels[:, 1:]`` from ``logits[:, :-1]``; per-row mean CE.
    Mirrors ``CrossEntropyLoss(ignore_index=-100, reduction='mean')`` applied per
    sequence, so the batched path equals the per-candidate ``out.loss`` path. The GPU
    ``loss_of`` runs the torch equivalent on ``out.logits``.

    Args:
        logits: ``[B, T, V]`` next-token logits.
        labels: ``[B, T]`` target ids, ``ignore_index`` where masked.
        ignore_index: Label value excluded from the per-row mean.

    Returns:
        ``[B]`` mean CE over the non-ignored shifted positions of each row.
    """
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels)
    shift_logits = logits[:, :-1, :]  # [B, T-1, V]
    shift_labels = labels[:, 1:]  # [B, T-1]
    m = shift_logits.max(axis=-1, keepdims=True)
    log_probs = shift_logits - (m + np.log(np.exp(shift_logits - m).sum(axis=-1, keepdims=True)))
    valid = shift_labels != ignore_index  # [B, T-1]
    safe_labels = np.where(valid, shift_labels, 0)  # in-bounds gather index for masked rows
    true_lp = np.take_along_axis(log_probs, safe_labels[..., None], axis=-1)[..., 0]
    ce = np.where(valid, -true_lp, 0.0)  # [B, T-1]; ignored positions contribute nothing
    counts = valid.sum(axis=-1)  # [B] non-ignored positions per row
    totals = ce.sum(axis=-1)  # [B]
    # Per-row mean; a fully-ignored row (count 0) is documented as 0.0, never NaN.
    return np.where(counts > 0, totals / np.maximum(counts, 1), 0.0)


@dataclass(frozen=True)
class EquivalenceCheck:
    """Verdict of one ``[B]`` loss-vector comparison (DB-4 batched-vs-single / determinism).

    Codex (2026-06-19) flagged that absolute closeness is **necessary-not-sufficient**: GCG
    selects the ``argmin`` candidate, so the batched ``loss_of`` must agree with the
    per-candidate reference in **rank**, not only value. But strict ``argmin`` *index*
    equality is too brittle under the registered **bf16** precision: matmul reduction order
    varies with the batch dim, so when the top candidates are within bf16 noise (~0.1–0.3 CE)
    the index can flip while selecting an *equally good* candidate. The load-bearing invariant
    is therefore **selection regret** — the true (reference) loss of the candidate the compared
    path would pick, relative to the true best — which is zero/​tiny for a benign flip and large
    only for a real misranking. ``argmin_match`` (strict index equality) is kept as a diagnostic.

    Attributes:
        n: Batch size (length of the compared vectors).
        max_abs_diff: ``max |ref - cmp|`` across the batch.
        allclose: Whether the two vectors agree within ``atol``.
        argmin_match: Whether both vectors pick the same ``argmin`` index (diagnostic only).
        selection_regret: ``ref[argmin(cmp)] - min(ref)`` — how much worse, in the reference
            losses, is the candidate the compared path would select vs the true best (``>= 0``).
        regret_ok: Whether ``selection_regret <= regret_tol``.
    """

    n: int
    max_abs_diff: float
    allclose: bool
    argmin_match: bool
    selection_regret: float
    regret_ok: bool

    @property
    def passed(self) -> bool:
        """Absolute agreement (within ``atol``) **and** a tolerable selection regret."""
        return self.allclose and self.regret_ok


def equivalence_verdict(
    loss_ref: np.ndarray,
    loss_cmp: np.ndarray,
    *,
    atol: float = 1e-3,
    regret_tol: float = 1e-3,
) -> EquivalenceCheck:
    """Validate that ``loss_cmp`` (e.g. true-batch) agrees with ``loss_ref`` (per-candidate single).

    Two invariants (DB-4, refined on the 2026-06-22 bf16 measurement):

    - **absolute** — ``|ref - cmp|`` within ``atol``. Tight at ``B=1`` this proves the CE
      *formula* (same forward); a generous bound at ``B>1`` only guards against divergence,
      since bf16 batch-order noise makes exact agreement unattainable.
    - **selection regret** (load-bearing) — the candidate ``loss_cmp`` would select,
      ``argmin(loss_cmp)``, has a reference loss within ``regret_tol`` of the true best
      ``min(loss_ref)``. GCG acts on the ``argmin``, so *this* — not strict index equality —
      is what must hold: when the top candidates are within bf16 noise the index may flip while
      the selected candidate is equally good (≈ zero regret).

    ``argmin_match`` (strict index equality) is reported as a diagnostic. Ties are broken by
    :func:`numpy.argmin`'s first-occurrence rule. Reused for the determinism re-run (identical
    inputs ⇒ zero regret).

    Args:
        loss_ref: ``[B]`` reference losses (per-candidate single, or run 1) — the ground truth.
        loss_cmp: ``[B]`` losses of the path under test (batched, or run 2).
        atol: Absolute tolerance for ``allclose``.
        regret_tol: Maximum tolerated selection regret.

    Returns:
        An :class:`EquivalenceCheck`.

    Raises:
        ValueError: If the vectors are not 1-D, differ in shape, or are empty.
    """
    ref = np.asarray(loss_ref, dtype=float)
    cmp = np.asarray(loss_cmp, dtype=float)
    if ref.ndim != 1 or cmp.ndim != 1:
        raise ValueError(f"loss vectors must be 1-D, got {ref.ndim}-D and {cmp.ndim}-D")
    if ref.shape != cmp.shape:
        raise ValueError(f"loss vectors must match in shape, got {ref.shape} vs {cmp.shape}")
    if ref.shape[0] == 0:
        raise ValueError("loss vectors must be non-empty")
    cmp_choice = int(np.argmin(cmp))
    regret = float(ref[cmp_choice] - ref.min())
    return EquivalenceCheck(
        n=int(ref.shape[0]),
        max_abs_diff=float(np.max(np.abs(ref - cmp))),
        allclose=bool(np.allclose(ref, cmp, atol=atol)),
        argmin_match=int(np.argmin(ref)) == cmp_choice,
        selection_regret=regret,
        regret_ok=bool(regret <= regret_tol),
    )


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
    we check that its **ranking is loss-aligned** — the gradient-recommended
    (most-negative-grad) tokens reduce the measured CE loss and beat a random
    control — and that the suffix span decodes to the intended adversarial text.
    That is exactly the property GCG's top-k selection exploits; a wrong span /
    projection / tied-embedding / sign flip destroys the recommended-beats-random
    separation, so it still catches a broken seam. (A per-random-swap *sign* test is
    ~chance even for a correct gradient — the first-order model is over-optimistic,
    predicting most swaps help when only the recommended ones do; that is why GCG
    evaluates candidates instead of trusting the gradient. ``sign_agreement`` is kept
    as a reported diagnostic, **not** a pass/fail signal.)

    Attributes:
        n_samples: Number of (position, token) probes sampled.
        sign_agreement: Diagnostic — fraction of random swaps where the projected
            gradient's predicted loss-change sign matched the measured delta.
        recommended_mean_delta: Mean measured Δloss of gradient-recommended tokens
            (top-k most-negative grad). Faithful ⇒ clearly negative.
        random_mean_delta: Mean measured Δloss of random control tokens.
        decoded_suffix: The suffix span decoded under the real processor.
        passed: The recommended tokens reduce loss **and** beat the random control
            (the gradient ranking is loss-aligned — the seam is faithful, not merely
            plausible: span, dtype, tied embeddings, sign convention all hold).
    """

    n_samples: int
    sign_agreement: float
    recommended_mean_delta: float
    random_mean_delta: float
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
        eval_batch: Candidate-eval chunk size for :meth:`loss_of` (DE-7). ``None``
            (default) ⇒ one forward over all ``B`` candidates (the D8 ``sw=32`` path,
            unchanged); an int ⇒ forward ``eval_batch`` candidates at a time so a
            ``search_width=512`` attack fits 24 GB. Validated at use (:func:`chunked_losses`).
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
        eval_batch: int | None = None,
    ) -> None:
        if suffix_len < 1:
            raise ValueError(f"suffix_len must be >= 1, got {suffix_len}")
        self._eval_batch = eval_batch
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

        # pixel_values: processor handles the image independently of the text. Cast to
        # the model's (bf16) param dtype — the frozen vision encoder's Conv bias is bf16,
        # so a float32 image errors ("Input type (float) and bias type (c10::BFloat16)
        # should be the same"). Same idiom as smoke_openvla_gradient.py's bf16 cast.
        proc = processor(_PROMPT_PREFIX.format(instruction=instruction) + _PROMPT_TAIL, image)
        model_dtype = next(model.parameters()).dtype
        self._pixel_values = proc["pixel_values"].to(device=device, dtype=model_dtype)

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

    def _target_span_ce_torch(self, logits: Any) -> Any:
        """Torch per-row mean CE over the trailing target span (equiv. of :func:`per_sequence_ce`).

        OpenVLA fuses image patches into the sequence, so ``out.logits`` (length ``S``)
        includes the vision positions; the teacher-forced target tokens stay the **final**
        ``n_target`` positions (built ``prefix ⊕ suffix ⊕ tail ⊕ target``), independent of how
        many patches are inserted. So per-row CE is read from the ``n_target`` positions that
        *predict* the target tokens (``S-n_target-1 .. S-2``, predict-t-from-t-1) — exactly the
        non-ignored set OpenVLA's own ``out.loss`` uses, without needing the vision-token count.
        The batch-1 equality to :meth:`_loss_single` is the on-box ``batched_matches_single``
        gate (D6-3/DB-4).
        """
        import torch  # type: ignore[import-not-found]

        n_target = int(self._target_action_ids.shape[0])
        target = torch.tensor(self._target_action_ids, device=self._device, dtype=torch.long)
        pred_logits = logits[:, -n_target - 1 : -1, :].float()  # [B, n_target, V]
        log_probs = torch.log_softmax(pred_logits, dim=-1)
        idx = target.view(1, -1, 1).expand(log_probs.shape[0], -1, 1)  # [B, n_target, 1]
        true_lp = log_probs.gather(-1, idx).squeeze(-1)  # [B, n_target]
        return (-true_lp).mean(dim=-1)  # [B] mean CE over the target span

    def loss_of(self, candidate_suffixes: np.ndarray) -> np.ndarray:
        """``[B, L]`` candidates → ``[B]`` CE losses via no-grad forward(s) (DE-7 chunked).

        Candidates are fixed-length (``prefix ⊕ suffix(L) ⊕ tail ⊕ target``, identical length),
        so they stack to ``[B, seq]`` with no padding. With ``eval_batch=None`` they run through
        a **single** ``torch.no_grad()`` forward (the D8 ``sw=32`` path, unchanged); with an int
        they are forwarded ``eval_batch`` at a time and the per-chunk ``[b]`` losses concatenated
        (:func:`chunked_losses`), so a ``search_width=512`` attack fits 24 GB — peak VRAM becomes
        the max single-chunk peak (DE-7). Per-row CE is read from ``out.logits`` by
        :meth:`_target_span_ce_torch` — **not** ``out.loss``, which mean-reduces across the
        batch. **``labels`` is intentionally NOT passed:** the model would then compute its own
        (unused) full-vocab loss, materialising an fp32 ``shift_logits`` copy of ``[B, seq, V]``
        that OOMs at large ``B`` on a 24 GB card; ``out.logits`` is identical with or without
        ``labels``, so omitting it is a pure memory saving (the per-candidate :meth:`_loss_single`
        reference still uses ``out.loss`` at ``B=1`` and the :meth:`batched_matches_single`
        gate (D6-3/DB-4) confirms equality). Resets/reads CUDA peak so Task 4 records peak VRAM.
        """
        import torch  # type: ignore[import-not-found]

        cands = np.asarray(candidate_suffixes, dtype=np.int64)
        if cands.ndim != 2 or cands.shape[1] != self._suffix_len:
            raise ValueError(
                f"candidate_suffixes must be [B, {self._suffix_len}], got {cands.shape}"
            )
        full = np.stack([self._full_ids(row) for row in cands])  # [B, seq]

        def _forward_chunk(rows: np.ndarray) -> np.ndarray:
            input_ids = torch.tensor(rows, device=self._device, dtype=torch.long)
            attn = torch.ones_like(input_ids)
            # One observation broadcast across the chunk; forward only reads pixel_values (no
            # in-place write), so an expand+contiguous view is safe and non-aliasing.
            pixel_values = self._pixel_values.expand(
                input_ids.shape[0], *self._pixel_values.shape[1:]
            ).contiguous()
            with torch.no_grad():
                out = self._model(
                    input_ids=input_ids,
                    attention_mask=attn,
                    pixel_values=pixel_values,
                )
            losses = (
                self._target_span_ce_torch(out.logits).detach().float().cpu().numpy().astype(float)
            )
            if self._eval_batch is not None:
                # Return this chunk's reservation to the driver before the next chunk so the
                # near-ceiling sw=512 sweep never accumulates fragmented blocks (DE-7).
                torch.cuda.empty_cache()
            return losses

        torch.cuda.reset_peak_memory_stats(self._device)
        losses = chunked_losses(full, self._eval_batch, _forward_chunk)
        self._last_peak_bytes = int(torch.cuda.max_memory_allocated(self._device))
        return losses

    # -- D6-9 faithfulness gate (on the box, BEFORE the tiny run) -------------- #

    def decode_span(self, suffix_ids: np.ndarray | None = None) -> str:
        """Decode the suffix span to text under the real processor (span sanity)."""
        ids = self._init_suffix if suffix_ids is None else np.asarray(suffix_ids, dtype=np.int64)
        return self._tokenizer.decode(ids.tolist())

    def gradient_agrees_with_swaps(
        self, *, n_samples: int, rng: np.random.Generator, gate_top_k: int = 16
    ) -> FaithfulnessReport:
        """Check the projected gradient's **ranking** is loss-aligned (D6-9).

        For ``n_samples`` sampled positions, compare the measured loss change of a
        **gradient-recommended** token (drawn from the ``gate_top_k`` most-negative
        ``g[i, ·]`` — the pool GCG's top-k selection samples) against a **random
        control** token. A faithful seam's recommended tokens reduce the CE loss and
        beat the random control; a wrong span / dtype / tied-embedding / sign flip
        destroys that separation. The per-random-swap *sign* match is computed too,
        but only as a reported diagnostic — it is ~chance even for a correct gradient
        (first-order over-optimism), so it is not the gate (see ``FaithfulnessReport``).
        """
        base = self._init_suffix
        grad = self.token_gradient(base)  # [L, V]
        base_loss = self._loss_single(base)
        k = min(int(gate_top_k), self._vocab_size)
        rec_deltas: list[float] = []
        rnd_deltas: list[float] = []
        sign_agree = 0
        for _ in range(n_samples):
            pos = int(rng.integers(0, self._suffix_len))
            topk = np.argpartition(grad[pos], k - 1)[:k]  # k most-negative grads
            rec_tok = int(rng.choice(topk))
            rnd_tok = int(rng.integers(0, self._vocab_size))
            rec = base.copy()
            rec[pos] = rec_tok
            rnd = base.copy()
            rnd[pos] = rnd_tok
            d_rec = self._loss_single(rec) - base_loss
            d_rnd = self._loss_single(rnd) - base_loss
            rec_deltas.append(d_rec)
            rnd_deltas.append(d_rnd)
            # diagnostic only: first-order sign prediction vs measured (random swap).
            predicted = float(grad[pos, rnd_tok] - grad[pos, int(base[pos])])
            if (predicted < 0) == (d_rnd < 0):
                sign_agree += 1
        rec_mean = float(np.mean(rec_deltas)) if rec_deltas else 0.0
        rnd_mean = float(np.mean(rnd_deltas)) if rnd_deltas else 0.0
        sign_agreement = sign_agree / n_samples if n_samples else 0.0
        return FaithfulnessReport(
            n_samples=n_samples,
            sign_agreement=sign_agreement,
            recommended_mean_delta=rec_mean,
            random_mean_delta=rnd_mean,
            decoded_suffix=self.decode_span(base),
            passed=(rec_mean < 0.0 and rec_mean < rnd_mean),
        )

    def batched_matches_single(self, candidate_suffixes: np.ndarray, *, atol: float = 1e-3) -> bool:
        """Wiring gate (D6-3): batched ``loss_of`` equals per-candidate single eval."""
        batched = self.loss_of(candidate_suffixes)
        single = np.array(
            [self._loss_single(r) for r in np.asarray(candidate_suffixes)], dtype=float
        )
        return bool(np.allclose(batched, single, atol=atol))
