# Raw Data

This directory stores raw external datasets that are not committed to git.

For the local-first MVP, the default target dataset is:
- `NormanWeissman2019_filtered.h5ad`

Download command:

```bash
./scripts/download_norman2019.sh
```

If `curl` is unstable on your network, retry with the alternate backend:

```bash
./scripts/download_norman2019.sh --backend wget
```

If you downloaded the file manually, place it at
`data/raw/NormanWeissman2019_filtered.h5ad` and verify it locally:

```bash
./scripts/download_norman2019.sh --verify-only
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

If the real dataset is temporarily unavailable, you can still build the offline
showcase with:

```bash
./scripts/run_generate_synthetic_showcase.sh
```

