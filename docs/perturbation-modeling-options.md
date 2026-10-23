# Perturbation modeling options and GEARS integration decision

**Status:** GEARS is the first external baseline to evaluate; integration remains optional and is not yet validated on this repository's Norman2019 pipeline.

## The current limitation

The custom Transformer and MLP use learned categorical perturbation-ID embeddings. Under the unseen-group protocol, held-out IDs have embedding rows that receive no training updates. A seen-trained checkpoint contains training samples from every perturbation identity, so its score on the unseen-group indices is not a held-out-identity evaluation.

Protocol-matched metrics are now enforced throughout training, summaries, comparisons, and reports. This fixes metric attribution, but it does not give the current model a representation for an unseen perturbation.

## Candidates

| Candidate | Why consider it | Important constraints | Recommendation |
| --- | --- | --- | --- |
| Current Transformer / MLP | Lightweight, already integrated with this repository's delta-expression bundle and training/evaluation pipeline | Categorical IDs do not provide a trained representation for a held-out identity | Keep as the known-perturbation baseline; do not claim zero-shot performance |
| Official GEARS | Purpose-built graph-based model for predicting transcriptional outcomes of novel gene perturbations; MIT license; supports single- and multi-gene perturbation workflows | Requires PyTorch Geometric and its own data pipeline; the official README notes limits around cross-cell-type transfer and combinatorial perturbations; built-in split semantics may differ from this repository | **First optional baseline to benchmark**, in an isolated environment |
| Official scGPT perturbation fine-tuning | Reuses a pretrained single-cell foundation model and its gene vocabulary rather than creating a custom representation from scratch | Larger checkpoint and runtime requirements; gene vocabulary, pretrained checkpoint, task objective, and evaluation protocol must all be recorded; a tutorial is not itself evidence of zero-shot validity | Second candidate if GEARS is unsuitable or if a pretrained gene representation is required |
| Custom gene-symbol hashing / character embedding | Easy to prototype | Gene symbols are identifiers, not a biologically grounded similarity space; generalization from string shape alone is weak evidence | Do not implement as the primary solution |

## Why GEARS first

The project should reuse a published implementation before building another graph or gene-conditioned model. GEARS is close to the specific missing capability: it uses gene relationships and perturbation structure rather than requiring every perturbation identity to have a trained, independent embedding row.

Official references:

- GEARS repository: https://github.com/snap-stanford/GEARS
- GEARS paper: https://www.nature.com/articles/s41587-023-01905-6
- Official scGPT repository: https://github.com/bowang-lab/scGPT
- Official scGPT perturbation tutorial: https://scgpt.readthedocs.io/en/stable/tutorial_perturbation.html

The official GEARS repository is MIT-licensed and documents installation through `cell-gears` plus PyTorch Geometric. Its dependency set includes pinned versions of NumPy, pandas, Scanpy, and SciPy. Those constraints are close to the current project ranges but should not be assumed compatible with the existing lockfile or every Apple Silicon/PyTorch combination.

## Integration boundary

Do not add GEARS or PyTorch Geometric to the mandatory project dependencies in the first iteration.

1. Create a separate optional environment for GEARS, keeping the existing `uv.lock` and core environment unchanged.
2. Load the raw Norman2019 AnnData through GEARS' supported data path. Confirm the required `gene_name`, condition, and cell-context metadata are available and mapped correctly.
3. Map the repository's intended held-out perturbation groups to a GEARS-supported custom split. If the split cannot be reproduced exactly, label the run as a separate protocol and do not compare its headline score directly with current runs.
4. Align the prediction target before comparing metrics. This repository currently evaluates delta expression relative to control means over a selected gene set. Confirm whether GEARS outputs can be transformed to that exact target without leakage; otherwise report a separate, clearly named evaluation.
5. Write a small adapter that converts GEARS outputs into the repository's metric and provenance schema. Keep imports lazy so ordinary tests and users without GEARS can still use the core project.
6. Run a small smoke test first, then at least three seeds on real Norman2019 data. Record wall time, peak memory, package versions, data/split/config/checkpoint hashes, per-perturbation metrics, and failure cases.

## Acceptance gate

GEARS is a valid candidate only if all of the following are true:

- It can load the intended Norman2019 data without silently changing condition labels or gene identities.
- The held-out perturbation identities are absent from training and can still be represented at inference time.
- Train/validation/test partitions are disjoint and their membership is recorded.
- Evaluation target, selected genes, control handling, and metrics are comparable—or differences are explicitly documented.
- The optional environment is reproducible and does not destabilize the core dependency lock.
- Results are labeled as diagnostic until the exact protocol and real-data run are reviewed.

If GEARS cannot satisfy the split/target contract, stop the adapter work and evaluate official scGPT perturbation fine-tuning next. Do not write a new custom gene-graph model before these options have been tested.

Related tracking issue: [M3: benchmark the official GEARS baseline](https://github.com/musuntana/scgpt/issues/12).
