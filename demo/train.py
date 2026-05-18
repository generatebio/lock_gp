from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

from demo.util import encode_one_hot
from lock_gp.blosum50 import get_blosum50_matrix
from lock_gp.gp_wrapper import GPWrapper, LinearGP, LockGP, TanimotoGP


def train_test_split(
    x: torch.Tensor, y: torch.Tensor, num_train: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Deterministic train/test split: first `num_train` rows are train, rest are
    test. No shuffling is performed.

    Args:
        x: Input tensor of shape (N, ...).
        y: Target tensor of shape (N,) aligned with `x` along dim 0.
        num_train: Number of training points. Must satisfy 0 < num_train < N.
        device: Device to place the returned tensors on.

    Returns:
        `(x_train, y_train, x_test, y_test)` with `num_train` and
        `N - num_train` rows respectively. Dtypes match the inputs.

    Raises:
        ValueError: If `num_train` is not strictly between 0 and N.
    """
    n = x.shape[0]
    if num_train <= 0 or num_train >= n:
        raise ValueError("num_train must be in (0, N).")
    x_train = x[:num_train].to(device)
    y_train = y[:num_train].to(device)
    x_test = x[num_train:].to(device)
    y_test = y[num_train:].to(device)
    return x_train, y_train, x_test, y_test


def evaluate(y_true: torch.Tensor, y_pred_mean: torch.Tensor, y_pred_var: torch.Tensor) -> dict[str, float]:
    """
    Compute Spearman, Pearson, MAE, and mean Gaussian NLL for scalar
    regression predictions.

    Args:
        y_true: Ground-truth targets of shape (N,).
        y_pred_mean: Predicted means of shape (N,).
        y_pred_var: Predicted variances of shape (N,). Must be strictly
            positive.

    Returns:
        Dict with keys `"spearman"`, `"pearson"`, `"mae"`, `"nll"`.
    """
    y_true_np = y_true.detach().cpu().numpy()
    y_pred_mean_np = y_pred_mean.detach().cpu().numpy()
    y_pred_var_np = y_pred_var.detach().cpu().numpy()

    spearman = spearmanr(y_true_np, y_pred_mean_np).statistic
    pearson = pearsonr(y_true_np, y_pred_mean_np).statistic
    mae = float(np.mean(np.abs(y_true_np - y_pred_mean_np)))
    nll = float(
        np.mean(0.5 * np.log(2 * np.pi * y_pred_var_np) + 0.5 * (y_true_np - y_pred_mean_np) ** 2 / y_pred_var_np)
    )

    return {"spearman": spearman, "pearson": pearson, "mae": mae, "nll": nll}


def main() -> None:
    """Fit Linear, LOCK, and Tanimoto GPs to CR6261-H1 dataset."""
    parser = argparse.ArgumentParser(description="LOCK GP Demo")
    parser.add_argument("--train-size", type=int, default=256, help="Number of training points.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = Path(__file__).parent / "cr6261_h1.csv"
    df = pd.read_csv(data_path)
    sequences = df["sequence"].astype(str).tolist()
    y_np = df["fitness"].to_numpy(dtype=np.float64)

    alphabet, _ = get_blosum50_matrix()
    x = encode_one_hot(sequences, alphabet, dtype=torch.float64)
    y = torch.tensor(y_np, dtype=torch.float64)

    x_train, y_train, x_test, y_test = train_test_split(x, y, num_train=args.train_size, device=device)

    print(f"Device: {device}     Train size: {len(y_train)}     Test size: {len(y_test)}")

    gp_models: list[tuple[str, GPWrapper]] = [
        ("Linear GP", LinearGP()),
        ("Tanimoto GP", TanimotoGP(alphabet=alphabet)),
        ("LOCK GP", LockGP(alphabet=alphabet)),
    ]

    results: dict[str, dict[str, float]] = {}
    for name, gp in gp_models:
        gp.fit(x_train, y_train)
        mean, var = gp.predict(x_test)
        results[name] = evaluate(y_test, mean, var)

    for name, metrics in results.items():
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
