from __future__ import annotations

import pytest
import torch

from lock_gp.blosum50 import get_blosum50_matrix
from lock_gp.lock_kernel import LinearGlobalLOCKKernel, NonlinearLocalLOCKKernel, build_lock_kernel
from lock_gp.tanimoto_kernel import TanimotoKernel


def test_tanimoto_kernel_rejects_mismatched_alphabet_order() -> None:
    alphabet, _ = get_blosum50_matrix()

    with pytest.raises(ValueError, match="BLOSUM50 alphabet order"):
        TanimotoKernel(num_positions=1, alphabet=list(reversed(alphabet)))


def test_lock_kernel_rejects_mismatched_alphabet_order() -> None:
    alphabet, _ = get_blosum50_matrix()

    with pytest.raises(ValueError, match="BLOSUM50 alphabet order"):
        build_lock_kernel(num_positions=1, alphabet=list(reversed(alphabet)))


def test_tanimoto_kernel_forward_rejects_mismatched_alphabet_size() -> None:
    alphabet, _ = get_blosum50_matrix()
    kernel = TanimotoKernel(num_positions=1, alphabet=alphabet)
    x = torch.zeros(2, 1, len(alphabet) - 1, dtype=torch.float64)

    with pytest.raises(ValueError, match="alphabet size"):
        kernel.forward(x, x)


@pytest.mark.parametrize("kernel_cls", [LinearGlobalLOCKKernel, NonlinearLocalLOCKKernel])
def test_lock_kernel_forward_rejects_mismatched_alphabet_size(
    kernel_cls: type[LinearGlobalLOCKKernel | NonlinearLocalLOCKKernel],
) -> None:
    alphabet, _ = get_blosum50_matrix()
    kernel = kernel_cls(num_positions=1, alphabet=alphabet)
    x = torch.zeros(2, 1, len(alphabet) - 1, dtype=torch.float64)

    with pytest.raises(ValueError, match="alphabet size"):
        kernel.forward(x, x)
