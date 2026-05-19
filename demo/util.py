from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F


def encode_one_hot(
    sequences: list[str],
    alphabet: Sequence[str],
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    Convert sequences to one-hot tensor of shape (N, L, A).

    Args:
        sequences: List of equal-length sequences.
        alphabet: Ordered alphabet of valid tokens.
        dtype: Floating-point dtype of the returned one-hot tensor. Defaults
            to ``torch.float64``.

    Raises:
        ValueError: if any sequence contains a token outside the alphabet or
            if the sequences are not all the same length.
    """
    seq_length = len(sequences[0])
    for seq in sequences:
        if len(seq) != seq_length:
            raise ValueError("All sequences must have the same length.")
    alphabet_index = {tok: i for i, tok in enumerate(alphabet)}
    unknown = {tok for seq in sequences for tok in seq if tok not in alphabet_index}
    if unknown:
        raise ValueError(f"Unknown token(s) {sorted(unknown)} encountered.")

    indices = torch.tensor([[alphabet_index[tok] for tok in seq] for seq in sequences], dtype=torch.int64)
    return F.one_hot(indices, num_classes=len(alphabet)).to(dtype=dtype)
