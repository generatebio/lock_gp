from __future__ import annotations

from collections.abc import Sequence

import torch
from gpytorch.kernels import Kernel

from lock_gp.blosum50 import get_blosum50_matrix
from lock_gp.utils import reshape_inputs


def _validate_alphabet_size(x1: torch.Tensor, x2: torch.Tensor, expected_size: int) -> None:
    if x1.shape[2] != expected_size or x2.shape[2] != expected_size:
        raise ValueError(
            f"Expected one-hot inputs to use alphabet size {expected_size}; "
            f"got {x1.shape[2]} and {x2.shape[2]}."
        )


class TanimotoKernel(Kernel):
    """
    Tanimoto kernel using BLOSUM50 encoding with clamped eigenvalues.

    Encodes amino acids using eigendecomposition of (raw) log space BLOSUM matrix.
    """

    has_lengthscale = False

    def __init__(self, num_positions: int, alphabet: Sequence[str]) -> None:
        """
        Initialize the kernel.

        Args:
            num_positions: Number of variable positions in the aligned sequence.
            alphabet: Alphabet used to one-hot encode inputs.
        """
        super().__init__()
        blosum_alphabet, blosum = get_blosum50_matrix(normalize=False)
        if list(alphabet) != blosum_alphabet:
            raise ValueError("Input alphabet must match the BLOSUM50 alphabet order.")
        eigenvalues, eigenvectors = torch.linalg.eigh(blosum)
        encoding = eigenvectors @ torch.diag(torch.sqrt(torch.clamp(eigenvalues, min=0.0)))
        self.register_buffer("encoding", encoding)
        self.num_positions = num_positions

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag: bool = False, **kwargs) -> torch.Tensor:
        """
        Compute Tanimoto kernel.

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
        _validate_alphabet_size(x1, x2, expected_size=self.encoding.shape[0])
        if diag:
            return torch.ones(x1.shape[0], dtype=x1.dtype, device=x1.device)

        e = self.encoding.to(dtype=x1.dtype, device=x1.device)
        x1_enc = torch.einsum("bla,aA->blA", x1, e)
        x2_enc = torch.einsum("bla,aA->blA", x2, e)
        dot = torch.einsum("blA,BlA->bB", x1_enc, x2_enc)
        n1 = torch.linalg.vector_norm(x1_enc, dim=[1, 2], ord=2) ** 2
        n2 = torch.linalg.vector_norm(x2_enc, dim=[1, 2], ord=2) ** 2
        n1 = n1[:, None]
        n2 = n2[None, :]
        return dot / (n1 + n2 - dot)
