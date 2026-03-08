# Raw Data

This directory stores raw external datasets that are not committed to git.

For the local-first MVP, the default target dataset is:
- `NormanWeissman2019_filtered.h5ad`

Download command:

```bash
./scripts/download_norman2019.sh
```

After download, inspect schema:

```bash
./scripts/run_inspect_anndata.sh \
  --input-path data/raw/NormanWeissman2019_filtered.h5ad \
  --output-json data/interim/norman2019_schema.json
```

Then preprocess into a demo bundle:

```bash
./scripts/run_preprocess_demo.sh \
  --input-path data/raw/NormanWeissman2019_filtered.h5ad \
  --output-dir data/processed/norman2019_demo_bundle \
  --hvg-top-genes 256 \
  --max-cells-per-perturbation 100
```

