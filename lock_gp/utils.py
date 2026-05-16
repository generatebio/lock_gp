from __future__ import annotations

import gpytorch
import torch


def reshape_inputs(x: torch.Tensor, num_positions: int) -> torch.Tensor:
    """
    Reshape input tensor from 2D (N, L*A) to 3D (N, L, A) if needed.

    Args:
        x: Input tensor of shape (N, L, A) or (N, L*A).
        num_positions: Number of positions L.

    Returns:
        Tensor of shape (N, L, A).

    Raises:
        ValueError: If x has incorrect number of dimensions or feature dimension
            is not divisible by num_positions.
    """
    if x.ndim == 3:
        return x
    if x.ndim != 2:
        raise ValueError("Expected x to have shape (N, L, A) or (N, L*A).")
    if x.shape[1] % num_positions != 0:
        raise ValueError("Input feature dimension is not divisible by num_positions.")
    alphabet_size = x.shape[1] // num_positions
    return x.view(x.shape[0], num_positions, alphabet_size)


def _gpytorch_default_setting_closure(
    module: gpytorch.Module,
    value: torch.Tensor,
    param_name: str,
) -> None:
    """Set a parameter value, handling the raw transform if needed.

    This is the inverse of the default closure. It sets the parameter
    by using the setter property which handles constraint transformations.

    Note: The signature is (module, value, param_name) because botorch calls
    setting_closure(module, sampled_value) and param_name is bound via partial.
    """
    # Get the raw parameter name (parameters are stored with "raw_" prefix when constrained)
    raw_param_name = f"raw_{param_name}"
    if hasattr(module, raw_param_name):
        # For constrained parameters, use the constraint's inverse transform
        param = getattr(module, raw_param_name)
        constraint = module.constraint_for_parameter_name(raw_param_name)
        if constraint is not None:
            value = constraint.inverse_transform(value)
    else:
        # For unconstrained parameters, set directly
        param = getattr(module, param_name)

    param.data.copy_(value)
