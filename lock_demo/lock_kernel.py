from __future__ import annotations

from functools import partial

import torch
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import AdditiveKernel, Kernel, ProductKernel, ScaleKernel
from gpytorch.priors import GammaPrior, NormalPrior

from .blosum50 import get_blosum50_matrix
from .utils import _gpytorch_default_setting_closure, reshape_inputs


class LinearGlobalLOCKKernel(Kernel):
    """Linear LOCK kernel with a single global learnable exponent."""

    has_lengthscale = False

    def __init__(self, num_positions: int, **kwargs) -> None:
        """
        Initialize the kernel.

        Args:
            num_positions: Number of positions in the aligned sequence.
            **kwargs: Additional arguments passed to parent Kernel class.
        """
        super().__init__(**kwargs)
        _, blosum = get_blosum50_matrix(normalize=True)
        self.register_buffer("blosum", blosum)
        self.num_positions = num_positions
        self.register_parameter(
            name="log_global_exponent",
            parameter=torch.nn.Parameter(torch.tensor(0.0)),
        )
        self.register_prior(
            "global_exponent_prior",
            NormalPrior(0.0, 1.0),
            lambda m: m.log_global_exponent,
            partial(_gpytorch_default_setting_closure, param_name="log_global_exponent"),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag: bool = False, **kwargs) -> torch.Tensor:
        """
        Compute global linear kernel.

        Args:
            x1: First input tensor of shape (N, L*A) or (N, L, A).
            x2: Second input tensor of shape (M, L*A) or (M, L, A).
            diag: If True, return only diagonal elements (x1 == x2).
            **kwargs: Additional arguments.

        Returns:
            Kernel matrix of shape (N, M) or (N,) if diag=True.
        """
        x1 = reshape_inputs(x1, self.num_positions)
        x2 = reshape_inputs(x2, self.num_positions)
        if diag:
            return self.num_positions * torch.ones(x1.shape[0], dtype=x1.dtype, device=x1.device)
        global_exponent = self.log_global_exponent.exp().to(dtype=x1.dtype, device=x1.device)
        blosum = self.blosum.to(dtype=x1.dtype, device=x1.device)
        blosum = (global_exponent * blosum).exp()
        return torch.einsum("bla,aA,BlA->bB", x1, blosum, x2)


class NonlinearLocalLOCKKernel(Kernel):
    """Nonlinear LOCK kernel with local per-position exponents."""

    has_lengthscale = False

    def __init__(self, num_positions: int, **kwargs) -> None:
        """
        Initialize the kernel.

        Args:
            num_positions: Number of positions in the aligned sequence.
            **kwargs: Additional arguments passed to parent Kernel class.
        """
        super().__init__(**kwargs)
        _, blosum = get_blosum50_matrix(normalize=True)
        self.register_buffer("blosum", blosum)
        self.num_positions = num_positions
        self.register_parameter(
            name="log_global_exponent",
            parameter=torch.nn.Parameter(torch.tensor(0.0)),
        )
        self.register_parameter(
            name="log_local_exponents",
            parameter=torch.nn.Parameter(torch.zeros(num_positions)),
        )
        self.register_prior(
            "global_exponent_prior",
            NormalPrior(0.0, 1.0),
            lambda m: m.log_global_exponent,
            partial(_gpytorch_default_setting_closure, param_name="log_global_exponent"),
        )
        self.register_prior(
            "local_exponents_prior",
            NormalPrior(0.0, 0.5),
            lambda m: m.log_local_exponents,
            partial(_gpytorch_default_setting_closure, param_name="log_local_exponents"),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag: bool = False, **kwargs) -> torch.Tensor:
        """
        Compute local nonlinear kernel.

        Args:
            x1: First input tensor of shape (N, L*A) or (N, L, A).
            x2: Second input tensor of shape (M, L*A) or (M, L, A).
            diag: If True, return only diagonal elements (x1 == x2).
            **kwargs: Additional arguments.

        Returns:
            Kernel matrix of shape (N, M) or (N,) if diag=True.
        """
        x1 = reshape_inputs(x1, self.num_positions)
        x2 = reshape_inputs(x2, self.num_positions)
        if diag:
            return torch.ones(x1.shape[0], dtype=x1.dtype, device=x1.device)
        global_exponent = self.log_global_exponent.exp().to(dtype=x1.dtype, device=x1.device)
        local_exponents = self.log_local_exponents.exp().to(dtype=x1.dtype, device=x1.device)
        blosum = self.blosum.to(dtype=x1.dtype, device=x1.device)
        blosum = global_exponent * local_exponents[:, None, None] * blosum
        return torch.einsum("bla,BlA,laA->bB", x1, x2, blosum).exp()


def build_lock_kernel(num_positions: int) -> Kernel:
    """
    Build the full LOCK kernel.

    Note that it is generally recommended to include all sequence positions
    that may be variable at inference time, because restricing to only those
    positions that are variable in the training data means that other positions
    will effectively be ignored at inference time.

    Args:
        num_positions: Number of positions in the aligned sequence.

    Returns:
        Composite LOCK kernel.
    """
    linear1 = LinearGlobalLOCKKernel(num_positions=num_positions)
    linear2 = LinearGlobalLOCKKernel(num_positions=num_positions)
    nonlinear = NonlinearLocalLOCKKernel(num_positions=num_positions)
    return AdditiveKernel(
        ScaleKernel(
            ProductKernel(nonlinear, linear1),
            outputscale_prior=GammaPrior(2.0, 2.0),
            outputscale_constraint=GreaterThan(1e-4),
        ),
        ScaleKernel(
            linear2,
            outputscale_prior=GammaPrior(2.0, 2.0),
            outputscale_constraint=GreaterThan(1e-4),
        ),
    )
