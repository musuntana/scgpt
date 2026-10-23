# Experiment provenance

Every run summary written by `build_run_summary()` includes an additive
`provenance` object (schema version `1`). Existing top-level summary fields
remain unchanged, so current readers can continue consuming the artifact.

## Recorded fields

- **Source:** Git commit and whether the working tree was clean or dirty. If
  Git metadata is unavailable, the status is recorded as unavailable instead
  of aborting summary generation.
- **Runtime:** Python, platform, NumPy, PyTorch, Scanpy, scikit-learn, and
  XGBoost versions when installed.
- **Experiment:** model type, seen/unseen split protocol, split seed, training
  seed, and an explicit seed override when supplied.
- **Inputs:** SHA-256 and byte size for the raw dataset when it exists; the
  processed bundle's `arrays.npz`, `splits.npz`, and `metadata.json`; and
  the data/model/train configuration files.
- **Output:** SHA-256 and byte size for the model checkpoint and
  `training_protocol.json` when present. This protocol record is also used to
  reject evaluation or summary requests that conflict with the trained split.
- **Timestamp:** UTC creation time for the provenance record.

File hashing reads in 1 MiB chunks, so it does not load a large dataset or
checkpoint into memory. Hashing an existing raw dataset still reads the whole
file and may add I/O time. Missing or unreadable optional files carry an
explicit status and do not prevent summary generation.

## Comparing two runs

Before comparing metrics, check that both summaries use compatible:

1. Raw dataset and processed-bundle hashes, including the saved split indices.
2. Data, model, and training configuration hashes.
3. Split protocol and split seed, plus the training seed.
4. Model/checkpoint identity, source revision, and runtime package versions.

A matching path is not enough: a file can change without its path changing.
Different bundle or configuration hashes mean the runs are not a direct
like-for-like comparison unless the change is intentional and documented.

A `dirty` working tree means the recorded commit does not fully identify the
source code used. The summary records this condition but does not embed a
patch of uncommitted source changes. Commit the code before producing a result
that needs to be reproducible.

A matching SHA-256 establishes byte identity, not scientific validity. It does
not prove that the dataset is biologically appropriate, that the evaluation
protocol is leakage-free, or that training is deterministic across hardware.
Use the hashes alongside the evaluation protocol and test evidence, not as a
replacement for them.
