# data/

Put your **predictive-model datasets** here (CSV files). The CE tools
(`ce_features`, `ce_train`, `ce_design`, `ce_model`) look in this folder automatically.

How a dataset is located, in priority order:

1. an explicit `--csv path/to/file.csv` on the command line;
2. the `MOLVAE_CE_CSV` environment variable;
3. a single `.csv` file in this `data/` folder.

If this folder contains exactly one CSV, it is used with no extra flags. If it contains
several, the tools ask you to pick one with `--csv` or `MOLVAE_CE_CSV`.

## Expected columns (Coulombic-Efficiency dataset)

The default CE pipeline expects these column headers:

- `Additive_SMILES` — the molecule (SMILES)
- `CE_aver. (%)` — the measured average Coulombic Efficiency
- `CE_1 (%)`, `CE_2 (%)`, `CE_3 (%)` — the replicate measurements (used for a noise estimate)
- `LogMolarRatio` — the log molar ratio context
- `Zn_mole (mmol)`, `Additive_mole (%)` — formulation context

Datasets are not committed to the repository (CSV files are git-ignored on purpose), so
your data stays local. This README is the only tracked file in the folder.
