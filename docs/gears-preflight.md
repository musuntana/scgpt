# GEARS baseline: strict split-export preflight

**Status: preflight tooling only. No GEARS model has been installed, trained, or evaluated by this feature.**

The official [GEARS](https://github.com/snap-stanford/GEARS) `PertData.prepare_split(split="custom", split_dict_path=...)` reads a Python pickle whose values are lists of **condition names**. Our bundle instead stores **perturbed-cell indices**. Use this converter only for the group-disjoint `unseen` protocol; a cell-level `seen` split cannot be expressed as GEARS condition lists.

## 1. Prepare a fresh leakage-aware bundle

Use the repository's existing Norman2019 pipeline on your local authorized/public `.h5ad`:

```bash
./scripts/run_preprocess_demo.sh \
  --input-path data/raw/NormanWeissman2019_filtered.h5ad \
  --output-dir data/processed/norman2019_gears_preflight_bundle
```

The converter **rejects** legacy bundles that lack the training-only gene-filter/HVG provenance introduced by PR #10. Its validation does not regenerate data or train a model.

## 2. Export a condition-level split and review provenance

```bash
uv run python scripts/prepare_gears_baseline.py \
  --bundle-dir data/processed/norman2019_gears_preflight_bundle \
  --output-dir artifacts/gears_preflight
```

Output:

| File | Purpose |
| --- | --- |
| `gears_custom_split.pkl` | Locally generated Python pickle for official `PertData.prepare_split(split="custom", split_dict_path=...)` |
| `gears_custom_split.json` | Human-reviewable equivalent, without pickle parsing |
| `gears_preflight.json` | SHA-256 of all three bundle files, sample/condition counts, split contract, and **unverified** compatibility gates |

The conversion maps a single-gene condition `GENE` to `GENE+ctrl`. Shared `ctrl` samples belong to the GEARS training condition list only. A gene/condition present in more than one partition, missing samples, duplicate sample indices, legacy preprocessing, ambiguous multi-gene labels, or an incomplete split **fails closed**. Existing export files are not overwritten.

When the original H5AD contains a verified `obs['cell_type']`, `var['gene_name']`, a resolvable perturbation column, and exact matching sample IDs, optionally add:

```bash
uv run python scripts/prepare_gears_baseline.py \
  --bundle-dir data/processed/norman2019_gears_preflight_bundle \
  --output-dir artifacts/gears_preflight_with_source \
  --source-h5ad data/raw/NormanWeissman2019_filtered.h5ad
```

This performs **metadata identity checks only**; it does not rewrite the H5AD or automatically synthesize unverified gene-symbol/cell-type annotations. If the upstream schema is different, it fails until you supply a validated mapping. Source H5AD is opened read-only and its SHA-256 is recorded.

## 3. Separate, optional GEARS runtime (not yet implemented here)

The official library supports the custom split as follows **after** constructing a compatible GEARS `PertData` object:

```python
pert_data.prepare_split(
    split="custom",
    split_dict_path="artifacts/gears_preflight/gears_custom_split.pkl",
)
pert_data.get_dataloader(batch_size=32, test_batch_size=128)
```

Use an isolated GEARS environment. Do not add `cell-gears` / PyTorch Geometric to the project's core `uv.lock` until compatibility is verified. GEARS' `PertData` may download ontology resources and **filter out perturbations absent from its GO graph**. Check after loading that every exported train/val/test condition is present in the built dataset and graph, with the exact intended sample membership. If conditions disappear, **stop** and record the list; do not silently calculate a partial benchmark.

### Mandatory unresolved gates

1. **Cell membership:** GEARS' condition-list split does not encode a subset of cell IDs. The existing bundle may have capped/QC-filtered cells. A compatible GEARS input AnnData must preserve the exact intended sample membership and controls, not simply load its built-in Norman dataset.
2. **Graph coverage:** All held-out perturbation genes must exist in GEARS' graph at inference time. The exporter cannot check this without GEARS resources.
3. **Expression target:** GEARS evaluates expression; this repository evaluates predicted deltas against a batch/context-aware control mean over selected genes. Both target space and gene mapping must be matched before direct score comparison.
4. **Train/val/test isolation:** Reconfirm GEARS' dataloaders and downstream graph construction use train-only information where required; do not assume that an exported condition list proves scientific leakage freedom.
5. **Real evidence:** Run an isolated GEARS smoke test and at least three seeds of protocol-matched Norman2019 evaluation before any scientific performance claims.

**Never interpret `PREFLIGHT_ONLY` as a trained baseline or a successful zero-shot experiment.** The JSON deliberately sets `comparison_eligible: false`. Only locally generated trusted pickles should be passed to GEARS, because its upstream custom split loader uses Python pickle.

Tracking: [Issue #12](https://github.com/musuntana/scgpt/issues/12). Architectural comparison: [perturbation-modeling-options.md](perturbation-modeling-options.md).
