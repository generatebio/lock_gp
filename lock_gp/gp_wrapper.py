from __future__ import annotations

from collections.abc import Callable

import gpytorch
import torch
from botorch.fit import fit_gpytorch_mll
from gpytorch.kernels import Kernel

from .exact_gp import ExactGPModel
from .utils import standardize
import logging
import gc

logger = logging.getLogger(__name__)


class GPWrapper:
    """
    Lightweight GP wrapper with shared fit/predict logic.

    Args:
        kernel_factory: Callable that returns a Kernel for a given input tensor.
    """

    # Instance attributes set by fit method
    model: ExactGPModel
    y_mean: float
    y_std: float

    def __init__(self, kernel_factory: Callable[[torch.Tensor], Kernel]) -> None:
        """Initialize the GP wrapper."""
        self.kernel_factory = kernel_factory

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """
        Fit the GP. Standardizes targets internally.

        Args:
            x: Input tensor of shape (N, L, A).
            y: Target tensor of shape (N,).

        Raises:
            ValueError: If x or y have incorrect shapes.
        """
        if x.ndim != 3:
            raise ValueError("Expected x to have shape (N, L, A).")
        if x.shape[0] != y.shape[0]:
            raise ValueError("Mismatched x/y batch dimensions.")
        if any(hasattr(self, attr) for attr in ["model", "y_mean", "y_std"]):
            logger.warning("Model already fitted. Overwriting.")
            del self.model, self.y_mean, self.y_std

        train_y, y_mean, y_std = standardize(y)

        # Flatten x for GPyTorch
        n, seq_len, alphabet_size = x.shape
        x_flat = x.view(n, seq_len * alphabet_size)

        # Create kernel and model
        kernel = self.kernel_factory(x)
        model = ExactGPModel(train_x=x_flat, train_y=train_y, kernel=kernel)
        model = model.to(x.device, dtype=torch.float64)
        model.train()

        # Fit
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        self.model = model
        self.y_mean = y_mean
        self.y_std = y_std

        # Clean up to free memory
        del mll
        gc.collect()
        if x.is_cuda:
            torch.cuda.empty_cache()

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict mean and variance, unstandardizing to original scale.

        Args:
            x: Input tensor of shape (N, L, A).

        Returns:
            Tuple of (mean, variance) tensors, each of shape (N,).

        Raises:
            RuntimeError: If model is not fit yet.
            ValueError: If x has incorrect shape.
        """
        if (
            not hasattr(self, "model")
            or not hasattr(self, "y_mean")
            or not hasattr(self, "y_std")
        ):
            raise RuntimeError("Model is not fit yet.")
        if x.ndim != 3:
            raise ValueError("Expected x to have shape (N, L, A).")

        self.model.eval()

        with gpytorch.settings.fast_pred_var():
            n, seq_len, alphabet_size = x.shape
            x_flat = x.view(n, seq_len * alphabet_size)

            posterior = self.model.likelihood(self.model(x_flat))
            mean = posterior.mean * self.y_std + self.y_mean
            var = posterior.variance * (self.y_std**2)

            return mean, var
