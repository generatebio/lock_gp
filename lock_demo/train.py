from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import LinearKernel, ScaleKernel
from gpytorch.priors import GammaPrior
from scipy.stats import pearsonr, spearmanr

from .blosum50 import get_blosum50_matrix
from .utils import encode_one_hot
from .gp_wrapper import GPWrapper
from .lock_kernel import build_lock_kernel
from .tanimoto_kernel import TanimotoKernel


def train_test_split(
    x: torch.Tensor, y: torch.Tensor, num_train: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministic train/test split."""
    n = x.shape[0]
    if num_train <= 0 or num_train >= n:
        raise ValueError("num_train must be in (0, N).")
    x_train = x[:num_train].to(device)
    y_train = y[:num_train].to(device)
    x_test = x[num_train:].to(device)
    y_test = y[num_train:].to(device)
    return x_train, y_train, x_test, y_test


def evaluate(y_true: np.ndarray, y_pred_mean: np.ndarray, y_pred_var: np.ndarray) -> dict[str, float]:
    """Compute Spearman, Pearson, MAE, and NLL."""
    spearman = spearmanr(y_true, y_pred_mean).statistic
    pearson = pearsonr(y_true, y_pred_mean).statistic
    mae = float(np.mean(np.abs(y_true - y_pred_mean)))
    nll = float(
        np.mean(
            0.5 * np.log(2 * np.pi * y_pred_var)
            + 0.5 * (y_true - y_pred_mean) ** 2 / y_pred_var
        )
    )

    return {"spearman": spearman, "pearson": pearson, "mae": mae, "nll": nll}


def main() -> None:
    """Fit Linear, LOCK, and Tanimoto GPs to CR6261-H1 dataset."""
    parser = argparse.ArgumentParser(description="LOCK GP Demo")
    parser.add_argument("--num-training", type=int, default=256, help="Number of training points.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = Path("data") / "cr6261_h1.csv"
    df = pd.read_csv(data_path)
    sequences = df["sequence"].astype(str).tolist()
    y_np = df["fitness"].to_numpy(dtype=np.float64)

    alphabet, _ = get_blosum50_matrix()
    x = encode_one_hot(sequences, alphabet).double()
    y = torch.tensor(y_np, dtype=torch.float64)

    x_train, y_train, x_test, y_test = train_test_split(
        x, y, num_train=args.num_training, device=device
    )

    print(f"Device: {device}     Train size: {len(y_train)}     Test size: {len(y_test)}")

    # Linear GP
    linear_gp = GPWrapper(
        kernel_factory=lambda x: ScaleKernel(
            LinearKernel(),
            outputscale_prior=GammaPrior(2.0, 2.0),
            outputscale_constraint=GreaterThan(1e-4),
        )
    )
    linear_gp.fit(x_train, y_train)
    linear_mean, linear_var = linear_gp.predict(x_test)
    linear_metrics = evaluate(y_test.cpu().numpy(), linear_mean.cpu().numpy(), linear_var.cpu().numpy())

    # LOCK GP
    lock_gp = GPWrapper(kernel_factory=lambda x: build_lock_kernel(num_positions=x.shape[1]))
    lock_gp.fit(x_train, y_train)
    lock_mean, lock_var = lock_gp.predict(x_test)
    lock_metrics = evaluate(y_test.cpu().numpy(), lock_mean.cpu().numpy(), lock_var.cpu().numpy())

    # Tanimoto GP
    tanimoto_gp = GPWrapper(
        kernel_factory=lambda x: ScaleKernel(
            TanimotoKernel(num_positions=x.shape[1]),
            outputscale_prior=GammaPrior(2.0, 2.0),
            outputscale_constraint=GreaterThan(1e-4),
        )
    )
    tanimoto_gp.fit(x_train, y_train)
    tanimoto_mean, tanimoto_var = tanimoto_gp.predict(x_test)
    tanimoto_metrics = evaluate(y_test.cpu().numpy(), tanimoto_mean.cpu().numpy(), tanimoto_var.cpu().numpy())

    for name, metrics in [("Linear GP", linear_metrics), ("Tanimoto GP", tanimoto_metrics), ("LOCK GP", lock_metrics)]:
        spaces = " " * max(0, 15 - len(name))
        print(
            f"{name}:{spaces}spearman: {metrics['spearman']:.3f}  |  pearson: {metrics['pearson']:.3f}  |  "
            f"mae: {metrics['mae']:.3f}  |  nll:  {metrics['nll']:.3f}"
        )

    """
    Expected output for default args:

    Device: cpu     Train size: 256     Test size: 1556
    Linear GP:      spearman: 0.924  |  pearson: 0.858  |  mae: 0.334  |  nll:  0.550
    Tanimoto GP:    spearman: 0.939  |  pearson: 0.959  |  mae: 0.162  |  nll:  -0.050
    LOCK GP:        spearman: 0.978  |  pearson: 0.983  |  mae: 0.093  |  nll:  -0.513
    """


if __name__ == "__main__":
    main()
