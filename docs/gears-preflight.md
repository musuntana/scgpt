# GEARS baseline: source-data export and leakage gates

**Status: optional GEARS runtime implemented; real-data training and evaluation are not yet validated.**

The official [GEARS](https://github.com/snap-stanford/GEARS) `PertData.prepare_split(split="custom", split_dict_path=...)` reads a Python pickle whose values are lists of **condition names**. This repository stores **perturbed-cell indices**. The preflight converter only accepts the group-disjoint `unseen` protocol; a cell-level `seen` split cannot be expressed as GEARS condition lists.

## 1. Prepare a fresh leakage-aware bundle

Use the repository's existing Norman2019 pipeline on the local authorized/public `.h5ad`:

```bash
./scripts/run_preprocess_demo.sh \
  --input-path data/raw/NormanWeissman2019_filtered.h5ad \
  --output-dir data/processed/norman2019_gears_preflight_bundle
```

The converter rejects legacy bundles that lack the training-only gene-filter/HVG provenance. Its validation does not silently regenerate or overwrite an existing bundle.

## 2. Export a split contract and verify the source

For a contract-only export (no raw H5AD required):

```bash
uv run python scripts/prepare_gears_baseline.py \
  --bundle-dir data/processed/norman2019_gears_preflight_bundle \
  --output-dir artifacts/gears_preflight
```

For a **verified GEARS-schema AnnData input**, supply the same raw source H5AD used to create the bundle:

```bash
uv run python scripts/prepare_gears_baseline.py \
  --bundle-dir data/processed/norman2019_gears_preflight_bundle \
  --source-h5ad data/raw/NormanWeissman2019_filtered.h5ad \
  --output-dir artifacts/gears_preflight_source
```

If preprocessing used CLI overrides, pass the same values again, for example `--hvg-top-genes`, `--max-cells-per-perturbation`, `--perturbation-col`, `--control-label`, `--batch-col`, and `--context-cols`. The exporter re-runs the configured preprocessing and refuses to export unless sample IDs, gene order, perturbation indices, all split indices, control expression and target deltas reproduce the saved bundle.

The source H5AD is not rewritten. The exporter preserves exactly the processed perturbation-cell IDs and includes the same filtered control reference cells, maps single-gene conditions to `GENE+ctrl`, adds `condition` / `cell_type` / `gene_name`, and records each perturbation cell's unseen train/validation/test partition. It requires one cell context (GEARS documents no cross-cell-type transfer support) and a unique, verified `var['gene_name']` mapping.

Output files:

| File | Purpose |
| --- | --- |
| `gears_custom_split.pkl` | Locally generated condition-level split for GEARS' custom split API |
| `gears_custom_split.json` | Human-reviewable equivalent, without pickle parsing |
| `gears_preflight.json` | Source/bundle SHA-256 hashes, reproduction checks, split contract, and unresolved gates |
| `gears_input.h5ad` | Created only when `--source-h5ad` is supplied and exact reproduction succeeds; carries the processed expression matrix and split labels |

The exporter rejects overlapping perturbation groups, missing/duplicate samples, ambiguous multi-gene labels, legacy preprocessing, source/bundle mismatches, expression mismatches, and incompatible cell/gene metadata. It refuses to overwrite existing output artifacts.

## 3. Important: do not use GEARS' default DE preprocessing for this split

The upstream `PertData.new_data_process(..., skip_calc_de=False)` calls `get_DE_genes` and `get_dropout_non_zero_genes` on the full AnnData **before** the custom split is used to make loaders. Those steps calculate condition-level DE rankings and expression-derived gene masks using validation/test condition cells. That is an information-leakage risk for an unseen-perturbation benchmark.

The exported `gears_input.h5ad` deliberately has **no GEARS DE statistics**. It is an input-stage artifact, not permission to call the default upstream processing path. The next runtime integration must provide a reviewed train-only metadata path, keep the held-out conditions out of co-expression graph construction and model fitting, and report metrics over a declared gene universe without presenting train-derived variable genes as true per-condition DE genes.

GEARS' custom split loader expects condition names and its training code consumes DE metadata for auxiliary metrics. Do not run the standard `GEARS.train()` evaluation path on this input until the train-only adapter is implemented and audited; otherwise it may fail or report misleading DE-specific metrics.

## 4. Remaining acceptance gates

1. **Source-cell identity:** verified only when `source_preprocessing_reproduction_verified=true` in the manifest.
2. **GO graph coverage:** check that every train/validation/test perturbation gene is represented by the GEARS GO graph. Stop if any condition is filtered or unrepresentable.
3. **Train-only graph construction:** the co-expression graph must use only train conditions and controls.
4. **Leakage-safe GEARS metadata:** DE rankings/dropout masks must not use validation/test expression. If using a common train-derived feature subset for model selection, label it as such; do not call it test-condition DE.
5. **Target alignment:** this repository evaluates deltas against batch/context-aware control means. GEARS predicts expression; any subtraction and aggregation must use the exact declared control reference and the same selected gene order.
6. **Real evidence:** run in an isolated optional environment, capture checkpoint/config/data/split hashes, runtime, memory, per-perturbation results and at least three seeds before making performance claims.

**Stop condition:** Until all gates pass, `PREFLIGHT_ONLY` and `INPUT_ONLY_NO_MODEL_RUN` must remain non-comparable artifacts. Do not place their values next to Transformer/MLP/XGBoost results or claim zero-shot generalization.

Tracking: [Issue #12](https://github.com/musuntana/scgpt/issues/12). Architectural comparison: [perturbation-modeling-options.md](perturbation-modeling-options.md).


## 5. GO-graph coverage-only check (no model run)

Run this only after a source-H5AD export has produced `PREFLIGHT_ONLY` and `INPUT_ONLY_NO_MODEL_RUN` artifacts. Install the optional official GEARS/PyG stack in a separate environment; do not change the core `uv.lock`.

```bash
python scripts/check_gears_coverage.py \
  --preflight-dir artifacts/gears_preflight_source \
  --bundle-dir data/processed/norman2019_gears_preflight_bundle \
  --output-dir artifacts/gears_coverage \
  --gears-data-dir .cache/gears-go-coverage
```

This command validates the input/bundle hashes, split conditions, ordered gene IDs/symbols, and train-only metadata contract before loading the official `cell-gears` GO annotation. It invokes `PertData.set_pert_genes()` directly; it does **not** call `new_data_process`, create per-cell PyG graphs, initialize `GEARS`, build the weighted GO similarity edges, train, or generate model metrics. The annotation cache is isolated from the report directory and its SHA-256 is recorded in `gears_go_graph_coverage.json`.

Interpret the result narrowly: a passing report means all required perturbation names occur in the official GEARS GO-annotation node set. It does not prove that weighted GO edges are built successfully, that every condition survives GEARS processing, that target equivalence holds, or that the model performs well. Any missing target exits non-zero with `STOPPED_GO_GRAPH_COVERAGE`; fix the naming/annotation issue and rerun into a new output directory. Never manually override the report.

## 6. Optional train-only GEARS runtime

Once the source-H5AD export succeeds, the repository now includes an optional runner that imports the official GEARS model but does **not** call its standard `GEARS.train()` method. The official trainer's built-in post-training path expects DE/dropout metadata and reports DE-labelled metrics; those assumptions are unsuitable for this unseen split.

Install the official GEARS / PyTorch Geometric stack in a separate environment compatible with your machine. Do not add it to the core `uv.lock`. Then start with a one-epoch smoke run:

```bash
python scripts/run_gears_baseline.py \
  --preflight-dir artifacts/gears_preflight_source \
  --bundle-dir data/processed/norman2019_gears_preflight_bundle \
  --output-dir artifacts/gears_runs/seed42-smoke \
  --seed 42 --epochs 1 --batch-size 16 --device cpu
```

Use a new output directory for every run. On Apple Silicon, CPU is the conservative first smoke-test target; CUDA is available only in a compatible NVIDIA environment. GEARS downloads/uses its official GO annotation resources and may take time on the first run.

The runner:
- Requires the input H5AD hash and all three bundle-file hashes to match the preflight manifest.
- Calls upstream `new_data_process(..., skip_calc_de=True)` and supplies only an all-gene loss mask; it does not compute condition-level DE or dropout metadata.
- Checks that every perturbation target is present in the loaded GEARS perturbation/GO graph and that the official custom split retains the exact train/val/test condition lists. It stops before training if a condition disappears.
- Reuses GEARS' official model, graph construction, data loaders and loss function, but uses a small custom training loop and **all-gene validation MSE** for checkpoint selection. The test loader is not used for checkpoint selection.
- The official co-expression graph builder receives the exact custom training condition list (including shared controls), so held-out perturbation cells are excluded. The runner records the graph CSV SHA-256 and canonical training-condition digest in the run manifest. Sparse expression validation inspects stored values without densifying the full matrix.
- Evaluates held-out conditions in **absolute-expression space** and saves `results.json`, `run_manifest.json`, `checkpoint/model.pt`, and the official model config.
- Also emits `delta_metrics`: subtracts the mean of `bundle.control_expression` for each held-out perturbation's samples from its absolute-expression prediction, then compares against the mean of those samples' matched-control `target_delta`. This preserves batch/context-specific matched controls instead of assuming the global GEARS control mean is interchangeable with every matched control.
- Verifies the processed bundle gene-ID order, the input AnnData gene-symbol order, and the symbol order retained by GEARS. The run manifest records SHA-256 digests for the ordered gene identities, the global control cells supplied to GEARS prediction, and the matched-control vectors used for delta conversion.
- `target_equivalence_verified` remains `false` until real-data outputs are reviewed; passing unit tests and gene-order checks does not establish real-data target equivalence.
- Only `mse_per_perturbation` and `pearson_per_perturbation` in `delta_metrics` are marked compatible with the repository's corresponding aggregated delta metrics. `overall_mse` and `pearson_per_gene` are not claimed to be directly comparable because GEARS returns one aggregated prediction per perturbation.

This runtime remains diagnostic. The delta conversion and gene-order checks are implemented and unit-tested, but **no real Norman2019 GEARS run has yet validated the end-to-end result**. Review real per-perturbation outputs and run at least three seeds before making a model-performance claim. For multi-seed reports, use separate output directories and seeds; do not reuse a run directory. The existing `./scripts/run_multi_seed_report.sh --artifact-root artifacts` command also discovers nested GEARS run manifests and summarizes only matched-control delta metrics. It reports GEARS as diagnostic-only until real-data target equivalence is reviewed, at least three distinct seeds are present, and graph/provenance checks pass. Duplicate artifacts for the same seed are excluded from the mean/std aggregate and explicitly block performance-comparison eligibility until resolved; graph hashes are checked across all artifacts, including ignored duplicates.
