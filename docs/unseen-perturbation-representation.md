# Unseen perturbation representation: shared gene identity prototype

## Motivation

The default Transformer and MLP use a learned embedding row for each perturbation ID. Under the `unseen` group split, held-out perturbation rows receive no training updates. Metrics from those models therefore do not establish zero-shot generalization to perturbation identities absent from training.

This repository now has an optional experimental Transformer conditioning path:

```bash
uv run python scripts/train_transformer.py \
  --bundle-dir data/processed/norman2019_gears_preflight_bundle \
  --output-dir artifacts/shared-gene-identity/seed42 \
  --split-prefix unseen \
  --perturbation-conditioning shared_gene_identity \
  --seed 42
```

The default `--perturbation-conditioning categorical` remains unchanged.

## Representation

The shared-gene model uses one learned gene-identity embedding table for both:
1. the gene tokens used to predict each output gene; and
2. the target perturbation token added to the input sequence.

The target gene's embedding is therefore trained through its role as an output-gene token even when that gene's perturbation condition is held out. The model has no separate `perturbation_embedding` table indexed by perturbation condition.

This is a minimal controlled prototype, not a replacement for a biologically informed representation such as GEARS' graph-based features. It may have weaker transfer behavior than models using gene ontology or other curated features.

## Mapping and fail-closed behavior

The current processed bundle stores `gene_names` and `perturbation_names`. The prototype only maps a perturbation when its label exactly matches a saved `gene_names` entry. It does not infer aliases, strip prefixes, convert Ensembl IDs, or silently substitute an unknown token.

Before training, inspect mapping coverage without loading the large expression arrays:

```bash
uv run python scripts/inspect_gene_identity_coverage.py \
  --bundle-dir data/processed/norman2019_demo_bundle \
  --output-json artifacts/shared-gene-identity/gene-identity-coverage.json
```

The command reads only `metadata.json`, writes ordered-name SHA-256 digests, exact mapping coverage, unmapped labels, and ambiguous duplicate gene names. It exits with status 2 when coverage is incomplete or gene names are ambiguous, while still preserving the report.

The dataset exposes:
- `perturbation_gene_index`: exact matched gene index, or `-1` when unmapped;
- `num_mapped_perturbations`: number of perturbation labels that map exactly;
- `unmapped_perturbation_names`: labels that need an explicitly verified mapping.

Training refuses to start unless every perturbation maps into the saved gene universe. If coverage is incomplete, inspect the original AnnData's gene-symbol annotations and introduce a verified mapping in a later change; do not guess aliases. The inference path applies the same exact-match rule.

This coverage gate is consequential: the feature universe is selected by HVG preprocessing, so a perturbation target can be absent from the saved output genes even when it exists in the original source AnnData. The prototype is not intended to claim coverage for targets outside that universe.

## Evaluation protocol

Use a freshly regenerated bundle whose preprocessing diagnostics record training-only filtering/HVG selection. For the first run, use `--split-prefix unseen` and a unique output directory. Record the generated `training_protocol.json`, checkpoint, metrics, data hashes, and mapping coverage.

Before comparing with the categorical Transformer or GEARS:
1. verify all perturbation targets map exactly and the train/validation/test condition groups are disjoint;
2. verify checkpoint selection uses only the matching `unseen_val` split;
3. run multiple independent seeds on the same bundle and config;
4. inspect per-perturbation errors and compare only identical target definitions and gene universes;
5. report the mapping coverage and all failures alongside metrics.

Unit tests prove the shared embedding path, input validation, trainer routing, and inference mapping. They do **not** prove real-data target alignment, biological zero-shot performance, or superiority over GEARS. Those claims remain blocked until real Norman2019 runs are completed and reviewed.
