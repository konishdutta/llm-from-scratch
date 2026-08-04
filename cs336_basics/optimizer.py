from collections.abc import Callable, Iterable
import numpy as np
from typing import Optional
import torch
import math


class SGDOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)


    def step(self, closure: Optional[Callable]=None):
        # closure is an optional fn that recomputes the loss
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # get learning rate
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0) # get step number
                grad = p.grad.data # get gradient
                p.data -= lr / math.sqrt(t + 1) * grad # scale SGD loss by step number
                state["t"] = t + 1 # increment step number
        return loss


class AdamWOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if len(betas) != 2:
            raise ValueError(f"Expected 2 betas, got {len(betas)} betas")
        for beta in betas:
            if beta >= 1 or beta < 0:
                raise ValueError(f"Invalid beta: {beta}")
        if eps < 0:
            raise ValueError(f"Invalid eps: {eps}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight decay: {weight_decay}")
        defaults = {
                    "lr": lr,
                    "betas": betas,
                    "eps": eps,
                    "weight_decay": weight_decay,
                   }
        super().__init__(params, defaults)
    

    def step(self,  closure: Optional[Callable]=None):
        # closure is an optional fn that recomputes the loss
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # get learning rate
            beta1 = group["betas"][0]
            beta2 = group["betas"][1]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)
                t += 1
                adjusted_lr = lr * math.sqrt(1 - beta2 ** t)/(1 - beta1 ** t)
                p.data *= (1 - lr * weight_decay) # weight decay
                m = state.get("m", torch.zeros_like(p.grad))
                v = state.get("v", torch.zeros_like(p.grad))
                m = beta1 * m + (1 - beta1) * p.grad
                v = beta2 * v + (1 - beta2) * p.grad ** 2
                p.data -= (adjusted_lr * m / (v.sqrt() + eps))
                state["t"], state["m"], state["v"] = t, m, v

        return loss


def get_lr_cosine_schedule(t, max_lr, min_lr, t_warmup, t_anneal):
    if t < t_warmup:
        return max_lr * t / t_warmup
    elif t < t_anneal:
        cos_coeff = 0.5 + 0.5 * math.cos(math.pi * ((t - t_warmup)/(t_anneal - t_warmup)))
        return min_lr + cos_coeff * (max_lr - min_lr)
    else:
        return min_lr


def grad_clip(param_list: Iterable[torch.nn.Parameter], max_norm: float, eps=1e-8) -> None:
    for p in param_list:
        norm = torch.linalg.norm(p.grad)
        if norm > max_norm:
            scaling_factor = max_norm / (norm + eps)
            p.grad *= scaling_factor


class AuroraOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=1e-2, mu=0.95, beta=0.5, eps=1e-7):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {
            "lr": lr, 
            "mu": mu, 
            "eps": eps, 
            "beta": beta,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    def step(self,  closure: Optional[Callable]=None):
        # closure is an optional fn that recomputes the loss
        loss = None if closure is None else closure()
        with torch.no_grad():
            for group in self.param_groups:
                lr = group["lr"]
                eps = group["eps"]
                beta = group["beta"]
                mu = group["mu"]
                weight_decay = group["weight_decay"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    
                    state = self.state[p]
                    if "M" in state:
                        M = state["M"]
                    else:
                        M = torch.zeros_like(p.grad)

                    M = mu * M + (1 - mu) * p.grad
                    state["M"] = M
                    A = (1- mu) * p.grad + mu * M

                    m, n = p.size(-2), p.size(-1)
                    if m > n:
                        # row normalize when its a tall matrix
                        target = n / m
                        d = 1/ (torch.linalg.norm(A, dim=1, keepdim=True) + eps)
                        u = polar_cans12(d * A)
                        s = (u ** 2).sum(dim=1, keepdim=True)
                        d = d * (target / (s + eps)) ** beta
                        u = polar_cans12(d * A)
                        u *= (m / n) ** 0.5

                    else:
                        u = polar_cans12(A)

                    p *= (1 - lr * weight_decay)
                    p -= lr * u

        return loss

def polar_exact(X):
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    return U @ Vh

def polar_cans12(X):
    if X.ndim < 2:
        raise ValueError("polar operation requires a matrix")
    X = X.float()
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    coeffs = (
        (5.182503604966906,  -5.178098480082684),
        (2.586120737395915,  -0.6479542005271643),
        (2.567364126726186,  -0.6454968804392178),
        (2.520560084348265,  -0.6393528082067044),
        (2.410759275435182,  -0.6248683598710716),
        (2.1883348130094173, -0.5952022073798908),
        (1.8595760874873613, -0.5504490972723968),
        (1.589020160467417,  -0.5126569802066718),
        (1.5051653981684994, -0.5007377068751799),
        (1.5, -0.5),
        (1.5, -0.5),
        (1.5, -0.5),
    )

    for a, b in coeffs:
        A = X @ X.mT
        X = a * X + b * A @ X
    a, b = 1.5, -0.5

    if transposed:
        X = X.mT

    return X

### --- TESTING CODE --- ###

import pytest

@pytest.mark.parametrize(
    "m, n",
    [
        (1, 1),
        (1, 8),
        (8, 1),
        (4, 4),
        (8, 4),
        (4, 8),
        (64, 16),
        (16, 64),
    ]
)

def test_polar_cans12_matches_exact(m: int, n: int):
    torch.manual_seed(0)
    A = torch.randn(m, n, dtype=torch.float32)

    expected = polar_exact(A)
    actual = polar_cans12(A)

    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-3,
        atol=1e-3
    )

def test_polar_cans12_scale_invariance():
    torch.manual_seed(1)
    A = torch.randn(20, 8)
    P_small = polar_cans12(1e-3 * A)
    P_normal = polar_cans12(A)
    P_large = polar_cans12(1e3 * A)
    torch.testing.assert_close(
        P_normal,
        P_small,
        rtol=1e-3,
        atol=1e-3
    )
    torch.testing.assert_close(
        P_normal,
        P_large,
        rtol=1e-3,
        atol=1e-3
    )

def test_polar_cans12_rank_deficient_is_finite():
    A = torch.randn(8, 3)
    A[:, 2] = A[:, 1]  # Make columns linearly dependent.

    P = polar_cans12(A)

    assert torch.isfinite(P).all()
