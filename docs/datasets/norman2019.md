# Norman2019 Data Notes

## Target File

- expected file name: `NormanWeissman2019_filtered.h5ad`
- default raw path: `data/raw/NormanWeissman2019_filtered.h5ad`

## Current Local Workflow

Download:

```bash
./scripts/download_norman2019.sh
```

Inspect schema:

```bash
./scripts/run_inspect_anndata.sh \
  --input-path data/raw/NormanWeissman2019_filtered.h5ad \
  --output-json data/interim/norman2019_schema.json
```

Build a small local demo bundle:

```bash
./scripts/run_preprocess_demo.sh \
  --input-path data/raw/NormanWeissman2019_filtered.h5ad \
  --output-dir data/processed/norman2019_demo_bundle \
  --hvg-top-genes 256 \
  --max-cells-per-perturbation 100
```

## Schema Resolution Rules

The preprocessing pipeline now resolves dataset fields automatically for the `scperturb_norman2019` preset.

Preferred perturbation columns:

1. `perturbation_canonical`
2. `perturbation_new`
3. `perturbation`

Preferred batch columns:

1. `batch`
2. `gemgroup`

Preferred context columns:

1. `celltype_new`
2. `celltype`
3. `cell_line`

## Canonicalization Rules

- control-like labels are normalized to `control`
- multi-gene perturbation strings are converted to a semicolon-delimited form
- `_`, `+`, `,`, `|`, and `;` are treated as multi-gene separators for filtering

This makes single-gene filtering deterministic and prevents Norman2019 combinatorial perturbations from leaking into the MVP path.
