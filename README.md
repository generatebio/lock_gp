# LOCK GP Demo

Minimal, self-contained demo of LOCK GP on CR6261-H1 antibody fitness data.

## Setup

We require Python 3.10 or later.

```bash
cd lock_demo
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Data

- `data/cr6261_h1.csv` with columns `sequence` and `fitness`.
- Alphabet is fixed to the canonical 20 amino acids plus gap token (`-`).

## How to run

```bash
python -m lock_demo.train --num-training 256
```

The script:
- Fits LOCK, Linear & Tanimoto GPs
- Reports Test Spearman, Pearson, MAE, and NLL for each GP variant
