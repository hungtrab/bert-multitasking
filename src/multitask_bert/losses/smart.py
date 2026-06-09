"""SMART regularisation (Jiang et al., 2020).

Two ingredients:

1. **Smoothness adversarial regulariser** ``R_s(θ)``:
   for each input ``x``, find a perturbation ``δ`` (with ``‖δ‖ ≤ ε``) that
   maximises a divergence between ``f(x; θ)`` and ``f(x + δ; θ)``. This is
   solved by one step of projected gradient ascent. The divergence is

   * symmetric KL for classification logits, and
   * squared error for regression scores.

2. **Bregman proximal step** in :func:`SmartRegulariser.bregman_div`,
   intended to be added as ``μ * D_breg(θ, θ̃)`` to the loss when calling
   the optimiser. ``θ̃`` is the EMA of past ``θ``.

Important detail (lessons learned from v1): the adversarial pass must
disable dropout, otherwise the divergence between ``f(x)`` and
``f(x + δ)`` is dominated by random mask differences rather than
genuine smoothness violations. We handle this by toggling ``model.eval()``
around the inner loop.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable

import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------- helpers


def symmetric_kl(logits_p: torch.Tensor, logits_q: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    """Symmetric KL divergence between two logit tensors (classification)."""
    log_p = F.log_softmax(logits_p, dim=-1)
    log_q = F.log_softmax(logits_q, dim=-1)
    p = log_p.exp().clamp_min(eps)
    q = log_q.exp().clamp_min(eps)
    kl_pq = (p * (log_p - log_q)).sum(-1)
    kl_qp = (q * (log_q - log_p)).sum(-1)
    return (kl_pq + kl_qp).mean()


def squared_error(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Squared error between two regression score tensors."""
    return ((p - q) ** 2).mean()


@contextmanager
def _eval_mode(module: torch.nn.Module):
    was_training = module.training
    module.eval()
    try:
        yield
    finally:
        if was_training:
            module.train()


# ----------------------------------------------------------------------- regulariser


class SmartRegulariser:
    """Computes the SMART smoothness term for one forward pass.

    Usage::

        smart = SmartRegulariser(epsilon=1e-5, lambda_s=1.0)
        ...
        # 1) get clean logits
        logits = model.predict_sentiment(ids, mask)
        loss = ce(logits, y)
        # 2) add smoothness term
        loss = loss + smart(model, encoder=model.encoder,
                            input_ids=ids, attention_mask=mask,
                            forward_fn=lambda emb: head(model.encoder.forward_from_embeddings(emb, mask).cls),
                            clean_output=logits, divergence='kl')
    """

    def __init__(
        self,
        *,
        epsilon: float = 1e-5,
        lambda_s: float = 1.0,
        step_size: float = 1e-3,
        norm_p: int = 2,
    ):
        self.epsilon = float(epsilon)
        self.lambda_s = float(lambda_s)
        self.step_size = float(step_size)
        self.norm_p = norm_p

    def __call__(
        self,
        model: torch.nn.Module,
        *,
        encoder,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        forward_fn: Callable[[torch.Tensor], torch.Tensor],
        clean_output: torch.Tensor,
        divergence: str = "kl",
    ) -> torch.Tensor:
        """Return ``λ_s · R_s(θ)`` for the current batch.

        ``forward_fn(embeddings)`` should run the model from word embeddings
        (with the given attention mask in scope) and return logits / scores.
        """
        if divergence == "kl":
            div_fn = symmetric_kl
        elif divergence == "mse":
            div_fn = squared_error
        else:  # pragma: no cover - guard
            raise ValueError(f"Unknown divergence '{divergence}'")

        with _eval_mode(model):
            embeddings = encoder.embed_tokens(input_ids).detach()
            delta = torch.zeros_like(embeddings, requires_grad=True)

            # one PGD ascent step on the divergence
            adv_out = forward_fn(embeddings + delta)
            div = div_fn(adv_out, clean_output.detach())
            grad = torch.autograd.grad(div, delta, retain_graph=False, create_graph=False)[0]

            # project step into ε-ball under L_p
            with torch.no_grad():
                if self.norm_p == 2:
                    norm = grad.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                    delta_new = (self.step_size * grad / norm).clamp(-self.epsilon, self.epsilon)
                else:
                    delta_new = self.step_size * grad.sign() * self.epsilon

            adv_out = forward_fn(embeddings + delta_new)
            r_s = div_fn(adv_out, clean_output.detach())

        return self.lambda_s * r_s

    @staticmethod
    def bregman_div(
        logits_now: torch.Tensor,
        logits_ema: torch.Tensor,
        *,
        kind: str = "kl",
    ) -> torch.Tensor:
        """``D_Breg(θ, θ̃)`` — divergence between current and EMA-averaged logits."""
        if kind == "kl":
            return symmetric_kl(logits_now, logits_ema.detach())
        return squared_error(logits_now, logits_ema.detach())
