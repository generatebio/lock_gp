<h1 align="center">
  <img src="assets/blosum.png" width="48" alt="BLOSUM icon">
  Flexible Kernels for Protein Property Prediction
  <img src="assets/blosum.png" width="48" alt="BLOSUM icon">
</h1>

<p align="center">
  Martin Jankowiak ⋅ Yerdos Ordabayev ⋅ Rudraksh Tuwani ⋅ Henry Ward ⋅ Hunter Nisonoff ⋅ James McFarland ⋅ Gevorg Grigoryan
</p>

<p align="center">
  ICML 2026
</p>

Minimal, self-contained demo of LOCK GP on CR6261-H1 antibody fitness data.

## Setup

Python 3.10 or later is required. We recommend [`uv`](https://docs.astral.sh/uv/getting-started/installation/) for easy and reproducible installation.

```bash
git clone git@github.com:generatebio/lock_gp.git
cd lock_gp
uv python install
uv sync
```

This repo was tested with **PyTorch 2.6**.

## Data

- `demo/cr6261_h1.csv` with columns `sequence` and `fitness`.
- Alphabet is fixed to the canonical 20 amino acids plus gap token (`-`).

## How to run

```bash
uv run python -m demo.train --train-size 256
```

To train on custom data, pass a CSV with `sequence` and `fitness` columns:

```bash
uv run python -m demo.train --data path/to/data.csv --train-size 256
```

The script:
- Fits LOCK, Linear & Tanimoto GPs
- Reports Test Spearman, Pearson, MAE, and NLL for each GP variant

## Citation

```bibtex
@inproceedings{jankowiak2026flexible,
  title={Flexible Kernels for Protein Property Prediction},
  author={Jankowiak, Martin and Ordabayev, Yerdos and Tuwani, Rudraksh and Ward, Henry and Nisonoff, Hunter and McFarland, James and Grigoryan, Gevorg},
  booktitle={International conference on machine learning},
  year={2026},
  organization={PMLR}
}
```
