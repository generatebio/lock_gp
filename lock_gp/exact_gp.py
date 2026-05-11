from __future__ import annotations

import gpytorch
import torch
from gpytorch.constraints import GreaterThan
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import Kernel
from gpytorch.means import ConstantMean
from gpytorch.priors import GammaPrior


class ExactGPModel(gpytorch.models.ExactGP):
    """
    Shared ExactGP implementation for all kernel types.

    Takes a pre-constructed kernel and provides standard GP functionality
    with ConstantMean, GaussianLikelihood, and BoTorch compatibility.
    """

    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor, kernel: Kernel) -> None:
        """
        Initialize the ExactGP model.

        Args:
            train_x: Training inputs of shape (N, D).
            train_y: Training targets of shape (N,).
            kernel: GPyTorch kernel instance.
        """
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_prior=GammaPrior(2.0, 2.0),
            noise_constraint=GreaterThan(1e-4),
        )
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = kernel

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        """
        Forward pass through the GP model.

        Args:
            x: Input tensor of shape (N, D).

        Returns:
            MultivariateNormal distribution over outputs.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def transform_inputs(self, X: torch.Tensor) -> torch.Tensor:
        """
        Pass-through transform to satisfy BoTorch fit utilities.

        Args:
            X: Input tensor.

        Returns:
            Input tensor X.
        """
        return X
