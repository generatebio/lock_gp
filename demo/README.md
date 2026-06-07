# LOCK GP Demo 

## Example data from Shanker et al.

- `cr6261_h1.csv` with columns `sequence` and `fitness`
- Alphabet is fixed to the canonical 20 amino acids plus gap token (`-`)
- Sequences are subset to the variable region

## How to run

```bash
uv run python -m demo.train --train-size 256
```

The script:
- Fits LOCK, Linear & Tanimoto GPs
- Reports Test Spearman, Pearson, MAE, and NLL for each GP variant

To train on custom data, pass a CSV with `sequence` and `fitness` columns:

```bash
uv run python -m demo.train --data path/to/data.csv --train-size 256
```

## Reference

Shanker, V. R., Bruun, T. U., Hie, B. L., & Kim, P. S. (2024). 
[Unsupervised evolution of protein and antibody complexes with a structure-informed language model.](https://www.science.org/doi/10.1126/science.adk8946)
Science, 385(6704), 46-53.

[zenodo link](https://zenodo.org/record/11260318/files/data.tar.gz)
