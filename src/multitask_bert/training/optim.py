"""Optimiser + LR-scheduler factories driven by config."""

from __future__ import annotations

import math
from typing import Iterable

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


_NO_DECAY = ("bias", "LayerNorm.weight", "layer_norm.weight")


def _decay_groups(parameters: Iterable[tuple[str, torch.nn.Parameter]], wd: float):
    decay, no_decay = [], []
    for name, p in parameters:
        if not p.requires_grad:
            continue
        if any(nd in name for nd in _NO_DECAY):
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Newton-Schulz orthogonalization used by Muon.

    Based on Keller Jordan's reference Muon implementation. It intentionally
    uses the quintic coefficients from that implementation and runs well in
    bfloat16 on CUDA.
    """
    if G.ndim < 2:
        raise ValueError("Muon orthogonalization requires at least a 2D tensor")

    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X


def _muon_update(
    grad: torch.Tensor,
    momentum: torch.Tensor,
    *,
    beta: float = 0.95,
    ns_steps: int = 5,
    nesterov: bool = True,
) -> torch.Tensor:
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp(momentum, beta) if nesterov else momentum
    original_shape = update.shape
    if update.ndim == 4:
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, update.size(-2) / update.size(-1)) ** 0.5
    return update.reshape(original_shape)


def _adam_update(
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    step: int,
    betas: tuple[float, float],
    eps: float,
) -> torch.Tensor:
    exp_avg.lerp_(grad, 1 - betas[0])
    exp_avg_sq.lerp_(grad.square(), 1 - betas[1])
    exp_avg_c = exp_avg / (1 - betas[0] ** step)
    exp_avg_sq_c = exp_avg_sq / (1 - betas[1] ** step)
    return exp_avg_c / (exp_avg_sq_c.sqrt() + eps)


class SingleDeviceMuonWithAuxAdam(torch.optim.Optimizer):
    """Muon for hidden matrices plus AdamW for all remaining parameters.

    Muon should not be applied to embeddings, classifier heads, biases, or
    LayerNorm gains. This optimizer uses explicit param groups with
    ``use_muon`` flags so the trainer can still treat it as one optimizer.
    """

    def __init__(self, param_groups):
        for group in param_groups:
            if "use_muon" not in group:
                raise ValueError("Muon param groups must specify use_muon")
            if group["use_muon"]:
                group["lr"] = group.get("lr", 0.002)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0.0)
                group["ns_steps"] = group.get("ns_steps", 5)
                group["nesterov"] = group.get("nesterov", True)
            else:
                group["lr"] = group.get("lr", 1e-5)
                group["betas"] = group.get("betas", (0.9, 0.999))
                group["eps"] = group.get("eps", 1e-8)
                group["weight_decay"] = group.get("weight_decay", 0.0)
        super().__init__(param_groups, {})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = _muon_update(
                        p.grad,
                        state["momentum_buffer"],
                        beta=float(group["momentum"]),
                        ns_steps=int(group["ns_steps"]),
                        nesterov=bool(group["nesterov"]),
                    )
                    p.mul_(1 - float(group["lr"]) * float(group["weight_decay"]))
                    p.add_(update, alpha=-float(group["lr"]))
            else:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = _adam_update(
                        p.grad,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        state["step"],
                        tuple(group["betas"]),
                        float(group["eps"]),
                    )
                    p.mul_(1 - float(group["lr"]) * float(group["weight_decay"]))
                    p.add_(update, alpha=-float(group["lr"]))
        return loss


def _is_muon_hidden_matrix(name: str, p: torch.nn.Parameter) -> bool:
    if p.ndim != 2:
        return False
    if not name.startswith("encoder."):
        return False
    lowered = name.lower()
    if any(x in lowered for x in ("embed", "embedding", "pooler", "layer_norm", "layernorm", "norm")):
        return False
    return "bert_layers" in name or "encoder.layer" in name


def _muon_groups(model: torch.nn.Module, cfg):
    muon, adam_decay, adam_no_decay = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if _is_muon_hidden_matrix(name, p):
            muon.append(p)
        elif any(nd in name for nd in _NO_DECAY):
            adam_no_decay.append(p)
        else:
            adam_decay.append(p)

    if not muon:
        raise ValueError("No Muon-compatible hidden matrices were found")

    return [
        {
            "params": sorted(muon, key=lambda x: x.size(), reverse=True),
            "lr": float(cfg.training.get("muon_learning_rate", 0.002)),
            "momentum": float(cfg.training.get("muon_momentum", 0.95)),
            "weight_decay": float(cfg.training.get("muon_weight_decay", 0.0)),
            "ns_steps": int(cfg.training.get("muon_ns_steps", 5)),
            "nesterov": bool(cfg.training.get("muon_nesterov", True)),
            "use_muon": True,
        },
        {
            "params": adam_decay,
            "lr": float(cfg.training.learning_rate),
            "betas": tuple(cfg.training.get("adam_betas", (0.9, 0.999))),
            "eps": float(cfg.training.get("adam_eps", 1e-8)),
            "weight_decay": float(cfg.training.weight_decay),
            "use_muon": False,
        },
        {
            "params": adam_no_decay,
            "lr": float(cfg.training.learning_rate),
            "betas": tuple(cfg.training.get("adam_betas", (0.9, 0.999))),
            "eps": float(cfg.training.get("adam_eps", 1e-8)),
            "weight_decay": 0.0,
            "use_muon": False,
        },
    ]


def build_optimizer(model: torch.nn.Module, cfg):
    if str(cfg.training.get("optimizer", "adamw")) == "muon_adamw":
        return SingleDeviceMuonWithAuxAdam(_muon_groups(model, cfg))
    groups = _decay_groups(model.named_parameters(), float(cfg.training.weight_decay))
    return AdamW(groups, lr=float(cfg.training.learning_rate))


def build_scheduler(optimizer, cfg, num_training_steps: int) -> LambdaLR:
    """Build the configured LR schedule."""
    schedule = str(cfg.training.get("lr_schedule", "constant"))
    if schedule == "constant":
        return LambdaLR(optimizer, lambda _: 1.0)

    warmup_ratio = float(cfg.training.get("warmup_ratio", 0.0))
    warmup = int(cfg.training.get("warmup_steps", 0))
    if warmup <= 0 and warmup_ratio > 0.0:
        warmup = int(num_training_steps * warmup_ratio)

    def lr_lambda(step: int) -> float:
        if warmup > 0 and step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, num_training_steps - warmup)
        if schedule == "linear":
            return max(0.0, 1.0 - progress)
        if schedule == "cosine":
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        if schedule == "constant_with_warmup":
            return 1.0
        raise ValueError(f"Unknown training.lr_schedule: {schedule}")

    return LambdaLR(optimizer, lr_lambda)
