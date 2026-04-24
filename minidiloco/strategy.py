import torch
from torch.optim import AdamW, SGD, Adam, RMSprop
from transformers import get_cosine_schedule_with_warmup

from abc import ABC, abstractmethod
from typing import override, Callable, Optional


# ─── optimizer factory helpers ────────────────────────────────────────────────

def make_adamw(lr=4e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1):
    """Return a factory that creates an AdamW optimizer."""
    def factory(params):
        return AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    factory.__name__ = f"AdamW(lr={lr})"
    return factory


def make_adam(lr=4e-4, betas=(0.9, 0.999), eps=1e-8):
    """Return a factory that creates an Adam optimizer."""
    def factory(params):
        return Adam(params, lr=lr, betas=betas, eps=eps)
    factory.__name__ = f"Adam(lr={lr})"
    return factory


def make_sgd(lr=0.7, momentum=0.9, nesterov=True):
    """Return a factory that creates an SGD optimizer (suitable for outer loop)."""
    def factory(params):
        return SGD(params, lr=lr, momentum=momentum, nesterov=nesterov)
    factory.__name__ = f"SGD(lr={lr},nesterov={nesterov})"
    return factory


def make_rmsprop(lr=1e-3, alpha=0.99, eps=1e-8):
    """Return a factory that creates an RMSprop optimizer."""
    def factory(params):
        return RMSprop(params, lr=lr, alpha=alpha, eps=eps)
    factory.__name__ = f"RMSprop(lr={lr})"
    return factory


# ─── base strategy ───────────────────────────────────────────────────────────

class Strategy(ABC):
    """Base class for distributed training strategies."""

    def _init_node(self, model, rank, world_size):
        self.model = model
        self.rank = rank
        self.world_size = world_size

    @abstractmethod
    def step(self, batch):
        pass


# ─── DiLoCo ──────────────────────────────────────────────────────────────────

class Diloco(Strategy):
    """DiLoCo distributed training strategy.

    Paper: https://arxiv.org/pdf/2311.08105

    Parameters
    ----------
    inner_optimizer_factory : callable
        A zero-argument factory (or one accepting *params*) that returns an
        ``torch.optim.Optimizer`` used for the inner loop.  Use the
        ``make_*`` helpers above, or provide your own callable.
        Default: AdamW with the hyper-parameters from Table 5 of the paper.

    outer_optimizer_factory : callable
        Same, but for the outer (synchronisation) loop.
        Default: Nesterov SGD with lr=0.7 from the paper.

    warmup_steps : int
        Number of warm-up steps for the cosine LR scheduler.

    H : int
        Number of inner steps between two outer synchronisations.
    """

    def __init__(
        self,
        inner_optimizer_factory: Optional[Callable] = None,
        outer_optimizer_factory: Optional[Callable] = None,
        warmup_steps: int = 1000,
        H: int = 500,
    ):
        self.inner_optimizer_factory = inner_optimizer_factory or make_adamw()
        self.outer_optimizer_factory = outer_optimizer_factory or make_sgd()
        self.warmup_steps = warmup_steps
        self.H = H

    @override
    def _init_node(self, model, rank, world_size, total_steps=100):
        """Instantiate optimizers and LR scheduler for this node."""
        super()._init_node(model, rank, world_size)

        local_params = list(model.parameters())
        global_params = [p.clone().detach().requires_grad_(True) for p in model.parameters()]

        self.inner_opt = self.inner_optimizer_factory(local_params)
        self.outer_opt = self.outer_optimizer_factory(global_params)

        self.scheduler = get_cosine_schedule_with_warmup(
            self.inner_opt,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps * self.H,
        )

        inner_name = getattr(self.inner_optimizer_factory, "__name__", type(self.inner_opt).__name__)
        outer_name = getattr(self.outer_optimizer_factory, "__name__", type(self.outer_opt).__name__)
        print(f"Rank {self.rank}: DiLoCo H={self.H}  inner={inner_name}  outer={outer_name}")

    def step(self, batch):
        """Perform one inner-loop step and return the scalar loss."""
        self.inner_opt.zero_grad()
        outputs = self.model(**batch)
        loss = outputs.loss
        loss.backward()
        self.inner_opt.step()
        self.scheduler.step()
        return loss.item()
