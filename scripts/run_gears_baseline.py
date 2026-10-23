# ruff: noqa: I001
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import platform
import pickle
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pairing import load_processed_bundle
from src.integrations.gears_contract import build_gears_split_contract
from src.integrations.gears_runtime import (
    attach_train_only_loss_metadata,
    compute_absolute_expression_metrics,
    compute_control_relative_delta_metrics,
    validate_gears_gene_order,
    validate_gears_graph_coverage,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the optional official GEARS model with train-only metadata and "
            "custom all-gene evaluation. GEARS/PyG must be installed separately."
        )
    )
    parser.add_argument("--preflight-dir", required=True)
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--test-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-go-gnn-layers", type=int, default=1)
    parser.add_argument("--num-gene-gnn-layers", type=int, default=1)
    parser.add_argument("--decoder-hidden-size", type=int, default=16)
    parser.add_argument("--num-similar-genes-go-graph", type=int, default=20)
    parser.add_argument("--num-similar-genes-co-express-graph", type=int, default=20)
    parser.add_argument("--coexpress-threshold", type=float, default=0.4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def validate_runtime_config(
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    test_batch_size: int,
    learning_rate: float,
    weight_decay: float,
    hidden_size: int,
    num_go_gnn_layers: int,
    num_gene_gnn_layers: int,
    decoder_hidden_size: int,
    num_similar_genes_go_graph: int,
    num_similar_genes_co_express_graph: int,
    coexpress_threshold: float,
    device_choice: str,
) -> None:
    """Reject invalid run settings before loading data or initializing GEARS."""
    if not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32).")
    if epochs < 1:
        raise ValueError("epochs must be positive.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if test_batch_size < 1:
        raise ValueError("test_batch_size must be positive.")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive.")
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and non-negative.")
    for name, value in (
        ("hidden_size", hidden_size),
        ("decoder_hidden_size", decoder_hidden_size),
        ("num_go_gnn_layers", num_go_gnn_layers),
        ("num_gene_gnn_layers", num_gene_gnn_layers),
        ("num_similar_genes_go_graph", num_similar_genes_go_graph),
        ("num_similar_genes_co_express_graph", num_similar_genes_co_express_graph),
    ):
        if value < 1:
            raise ValueError(f"{name} must be positive.")
    if not math.isfinite(coexpress_threshold) or not 0 <= coexpress_threshold < 1:
        raise ValueError("coexpress_threshold must be finite and in [0, 1).")
    if device_choice not in {"auto", "cpu", "cuda"}:
        raise ValueError("device_choice must be one of: auto, cpu, cuda.")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


EXPECTED_BUNDLE_ARTIFACTS = ("arrays.npz", "splits.npz", "metadata.json")


def validate_bundle_artifact_hashes(
    bundle_dir: str | Path,
    artifact_hashes: Any,
) -> None:
    """Require and verify the exact processed-bundle artifact set from preflight."""
    expected_names = set(EXPECTED_BUNDLE_ARTIFACTS)
    if not isinstance(artifact_hashes, dict) or set(artifact_hashes) != expected_names:
        expected = ", ".join(EXPECTED_BUNDLE_ARTIFACTS)
        raise ValueError(
            "GEARS preflight bundle_artifacts_sha256 must contain exactly: "
            f"{expected}."
        )

    bundle_path = Path(bundle_dir)
    for filename in EXPECTED_BUNDLE_ARTIFACTS:
        expected_sha = artifact_hashes[filename]
        if not isinstance(expected_sha, str) or re.fullmatch(r"[0-9a-f]{64}", expected_sha) is None:
            raise ValueError(
                f"GEARS preflight bundle artifact hash is not a valid SHA-256 digest: {filename}"
            )
        artifact_path = bundle_path / filename
        if not artifact_path.is_file():
            raise FileNotFoundError(
                f"Required processed-bundle artifact is missing: {artifact_path}"
            )
        if file_sha256(artifact_path) != expected_sha:
            raise ValueError(f"Bundle artifact hash changed since preflight: {filename}")


def package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def evaluate_all_gene_mse(loader: Any, model: Any, torch: Any, device: str) -> float:
    model.eval()
    squared_error = 0.0
    element_count = 0
    with torch.no_grad():
        for batch in loader:
            batch.to(device)
            prediction = model(batch)
            target = batch.y
            if prediction.shape != target.shape:
                raise RuntimeError(
                    "GEARS validation prediction/target shape mismatch: "
                    f"{tuple(prediction.shape)} != {tuple(target.shape)}."
                )
            if not bool(torch.isfinite(prediction).all().item()) or not bool(
                torch.isfinite(target).all().item()
            ):
                raise RuntimeError(
                    "Non-finite GEARS validation prediction or target; "
                    "refusing to select a checkpoint."
                )
            batch_squared_error = float(torch.sum((prediction - target) ** 2).item())
            if not math.isfinite(batch_squared_error):
                raise RuntimeError(
                    "Non-finite GEARS validation squared error; "
                    "refusing to select a checkpoint."
                )
            squared_error += batch_squared_error
            element_count += int(target.numel())
    if element_count == 0:
        raise RuntimeError("Validation loader has no expression targets.")
    validation_mse = squared_error / element_count
    if not math.isfinite(validation_mse):
        raise RuntimeError(
            "Non-finite GEARS validation MSE; refusing to select a checkpoint."
        )
    return validation_mse


def train_with_all_gene_validation(
    gears_model: Any,
    train_loader: Any,
    val_loader: Any,
    *,
    torch: Any,
    loss_fct: Any,
    device: str,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
) -> dict[str, Any]:
    """Reuse GEARS' model/loss/loaders, but avoid DE-labelled built-in evaluation."""
    if epochs < 1:
        raise ValueError("epochs must be positive.")
    model = gears_model.model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    best_validation_mse = float("inf")
    best_state: dict[str, Any] | None = None
    history: list[dict[str, float | int]] = []

    for epoch in range(epochs):
        model.train()
        train_loss_total = 0.0
        train_batches = 0
        for batch in train_loader:
            batch.to(device)
            optimizer.zero_grad()
            prediction = model(batch)
            loss = loss_fct(
                prediction,
                batch.y,
                batch.pert,
                ctrl=gears_model.ctrl_expression,
                dict_filter=gears_model.dict_filter,
                direction_lambda=gears_model.config["direction_lambda"],
            )
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError(f"Non-finite GEARS training loss at epoch {epoch + 1}.")
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
            optimizer.step()
            train_loss_total += float(loss.detach().item())
            train_batches += 1
        if train_batches == 0:
            raise RuntimeError(
                "GEARS train loader produced no batches. Reduce batch size or verify the split."
            )
        scheduler.step()
        validation_mse = evaluate_all_gene_mse(val_loader, model, torch, device)
        mean_train_loss = train_loss_total / train_batches
        history.append(
            {
                "epoch": epoch + 1,
                "mean_train_objective": mean_train_loss,
                "validation_mse_absolute_expression": validation_mse,
            }
        )
        print(
            f"Epoch {epoch + 1}/{epochs}: objective={mean_train_loss:.6f}; "
            f"validation_all_gene_mse={validation_mse:.6f}"
        )
        if validation_mse < best_validation_mse:
            best_validation_mse = validation_mse
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("GEARS training completed without a valid checkpoint.")
    model.load_state_dict(best_state)
    gears_model.model = model
    gears_model.best_model = copy.deepcopy(model)
    return {
        "best_validation_mse_absolute_expression": best_validation_mse,
        "checkpoint_selection_metric": "all_gene_validation_mse_absolute_expression",
        "history": history,
        "official_gears_train_called": False,
        "DE_specific_metrics_used_for_selection": False,
    }


def run_gears_baseline(
    *,
    preflight_dir: str | Path,
    bundle_dir: str | Path,
    output_dir: str | Path,
    seed: int = 42,
    epochs: int = 20,
    batch_size: int = 32,
    test_batch_size: int = 128,
    learning_rate: float = 1e-3,
    weight_decay: float = 5e-4,
    hidden_size: int = 64,
    num_go_gnn_layers: int = 1,
    num_gene_gnn_layers: int = 1,
    decoder_hidden_size: int = 16,
    num_similar_genes_go_graph: int = 20,
    num_similar_genes_co_express_graph: int = 20,
    coexpress_threshold: float = 0.4,
    device_choice: str = "auto",
) -> dict[str, Any]:
    validate_runtime_config(
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        test_batch_size=test_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        hidden_size=hidden_size,
        num_go_gnn_layers=num_go_gnn_layers,
        num_gene_gnn_layers=num_gene_gnn_layers,
        decoder_hidden_size=decoder_hidden_size,
        num_similar_genes_go_graph=num_similar_genes_go_graph,
        num_similar_genes_co_express_graph=num_similar_genes_co_express_graph,
        coexpress_threshold=coexpress_threshold,
        device_choice=device_choice,
    )
    preflight = Path(preflight_dir)
    bundle_path = Path(bundle_dir)
    output = Path(output_dir)
    if output.exists() and not output.is_dir():
        raise NotADirectoryError(f"Output path exists and is not a directory: {output}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty output directory {output}; choose a new path."
        )
    manifest_path = preflight / "gears_preflight.json"
    input_path = preflight / "gears_input.h5ad"
    split_path = preflight / "gears_custom_split.pkl"
    split_json_path = preflight / "gears_custom_split.json"
    for required_path in (manifest_path, input_path, split_path, split_json_path):
        if not required_path.is_file():
            raise FileNotFoundError(f"Required GEARS preflight artifact is missing: {required_path}")

    with manifest_path.open(encoding="utf-8") as handle:
        preflight_manifest = json.load(handle)
    if (
        preflight_manifest.get("source_preprocessing_reproduction_verified") is not True
        or preflight_manifest.get("gears_input_h5ad_exported") is not True
        or preflight_manifest.get("gears_input_h5ad_sha256") != file_sha256(input_path)
    ):
        raise ValueError(
            "GEARS runtime requires an exactly reproduced source H5AD export with a matching hash."
        )
    with split_json_path.open(encoding="utf-8") as handle:
        split_json = json.load(handle)
    if set(split_json) != {"train", "val", "test"}:
        raise ValueError("Custom split JSON must contain exactly train, val and test.")
    if not split_json["train"] or not split_json["val"] or not split_json["test"]:
        raise ValueError("Custom GEARS split must have non-empty train, val and test conditions.")

    bundle = load_processed_bundle(bundle_path)
    bundle_metadata = bundle["metadata"]
    expected_split, _ = build_gears_split_contract(bundle)
    if split_json != expected_split:
        raise ValueError(
            "GEARS custom split JSON no longer matches the hashed processed bundle; "
            "refusing a possible split/protocol substitution."
        )
    validate_bundle_artifact_hashes(
        bundle_path, preflight_manifest.get("bundle_artifacts_sha256")
    )

    try:
        import anndata as ad
        import torch
        from gears import GEARS, PertData
        from gears.utils import loss_fct
    except ImportError as exc:
        raise RuntimeError(
            "The optional GEARS runtime is not installed. Install official cell-gears and "
            "a compatible PyTorch Geometric stack in an isolated environment; core uv.lock "
            "is intentionally unchanged."
        ) from exc

    output.mkdir(parents=True, exist_ok=True)
    input_hash = file_sha256(input_path)
    # Each seed has an isolated output/data directory; keep the logical dataset ID stable.
    dataset_name = f"scgpt_{input_hash[:10]}"
    data_path = output / "gears_data"
    data_path.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if device_choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    device = (
        "cuda"
        if device_choice == "auto" and torch.cuda.is_available()
        else "cpu"
        if device_choice == "auto"
        else device_choice
    )

    adata = ad.read_h5ad(input_path)
    if "gene_name" not in adata.var:
        raise ValueError("GEARS input H5AD is missing var['gene_name'].")
    input_gene_ids = adata.var_names.astype(str).tolist()
    input_gene_symbols = adata.var["gene_name"].astype(str).tolist()
    # Validate bundle identity before GEARS builds its optional data artifacts.
    validate_gears_gene_order(
        bundle_gene_ids=list(bundle_metadata["gene_names"]),
        input_gene_ids=input_gene_ids,
        input_gene_symbols=input_gene_symbols,
    )
    for partition in ("train", "val", "test"):
        observed_conditions = set(
            adata.obs.loc[
                adata.obs["scgpt_split"].astype(str) == partition, "condition"
            ].astype(str)
        )
        if observed_conditions != set(split_json[partition]):
            raise ValueError(
                f"GEARS input H5AD {partition} conditions do not match the validated split JSON."
            )
    metadata_report = attach_train_only_loss_metadata(adata)
    validated_split_path = output / "validated_gears_custom_split.pkl"
    with validated_split_path.open("wb") as handle:
        pickle.dump(split_json, handle, protocol=4)
    pert_data = PertData(data_path=str(data_path), default_pert_graph=False)
    started = time.perf_counter()
    # This must remain True: GEARS' default DE/dropout preprocessing sees held-out cells.
    pert_data.new_data_process(
        dataset_name=dataset_name,
        adata=adata,
        skip_calc_de=True,
    )
    gene_order_report = validate_gears_gene_order(
        bundle_gene_ids=list(bundle_metadata["gene_names"]),
        input_gene_ids=input_gene_ids,
        input_gene_symbols=input_gene_symbols,
        gears_gene_symbols=[str(name) for name in pert_data.gene_names],
    )
    if int(pert_data.ctrl_adata.n_obs) < 1:
        raise RuntimeError("GEARS has no control cells for prediction baseline.")
    control_reference_mean = np.asarray(
        pert_data.ctrl_adata.X.mean(axis=0),
        dtype=np.float64,
    ).reshape(-1)
    graph_coverage = validate_gears_graph_coverage(
        bundle_metadata["perturbation_names"],
        {str(name) for name in pert_data.pert_names},
    )
    if not graph_coverage["verified"]:
        report = {
            "status": "STOPPED_GO_GRAPH_COVERAGE",
            "input_h5ad_sha256": input_hash,
            "graph_coverage": graph_coverage,
            "comparison_eligible": False,
        }
        with (output / "go_graph_coverage_failure.json").open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        raise RuntimeError(
            "GEARS GO graph cannot represent all perturbations. See "
            f"{output / 'go_graph_coverage_failure.json'}; no training was started."
        )

    available_conditions = {str(name) for name in pert_data.dataset_processed}
    required_conditions = set(split_json["train"] + split_json["val"] + split_json["test"])
    missing_conditions = sorted(required_conditions - available_conditions)
    if missing_conditions:
        raise RuntimeError(
            "GEARS filtered required conditions from its processed dataset: "
            f"{missing_conditions}. No training was started."
        )

    pert_data.prepare_split(
        split="custom",
        seed=seed,
        split_dict_path=str(validated_split_path),
    )
    for partition in ("train", "val", "test"):
        if set(pert_data.set2conditions[partition]) != set(split_json[partition]):
            raise RuntimeError(
                f"GEARS changed custom {partition} conditions; refusing protocol drift."
            )

    train_graph_count = sum(
        len(pert_data.dataset_processed[condition])
        for condition in split_json["train"]
        if condition in pert_data.dataset_processed
    )
    if train_graph_count < 1:
        raise RuntimeError("GEARS produced no training cell graphs.")
    effective_batch_size = min(batch_size, train_graph_count)
    pert_data.get_dataloader(
        batch_size=effective_batch_size,
        test_batch_size=max(test_batch_size, 1),
    )
    # The official trainer performs DE-labelled test analyses. Remove the test loader
    # and use the custom all-gene evaluator below; test data never selects a checkpoint.
    pert_data.dataloader.pop("test_loader", None)

    model = GEARS(pert_data, device=device)
    model.model_initialize(
        hidden_size=hidden_size,
        num_go_gnn_layers=num_go_gnn_layers,
        num_gene_gnn_layers=num_gene_gnn_layers,
        decoder_hidden_size=decoder_hidden_size,
        num_similar_genes_go_graph=num_similar_genes_go_graph,
        num_similar_genes_co_express_graph=num_similar_genes_co_express_graph,
        coexpress_threshold=coexpress_threshold,
        uncertainty=False,
        direction_lambda=0.1,
    )
    # Official GEARS builds the co-expression graph through
    # get_coexpression_network_from_train using set2conditions["train"].
    # Capture the exact generated artifact and the split identity in the run record.
    coexpression_graphs = sorted(
        (data_path / dataset_name).glob("*_co_expression_network.csv")
    )
    if len(coexpression_graphs) != 1:
        raise RuntimeError(
            "Expected exactly one official GEARS co-expression graph artifact after "
            f"model initialization; found {len(coexpression_graphs)}."
        )
    coexpression_graph_path = coexpression_graphs[0]
    training_conditions_sha256 = hashlib.sha256(
        json.dumps(
            sorted(split_json["train"]),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    graph_provenance = {
        "coexpression_graph_method": "official_gears_get_coexpression_network_from_train",
        "coexpression_graph_scope": "custom training conditions only (including shared controls)",
        "coexpression_graph_train_only": True,
        "training_conditions_count_including_control": len(split_json["train"]),
        "training_conditions_sha256": training_conditions_sha256,
        "coexpression_graph_path": str(coexpression_graph_path.relative_to(output)),
        "coexpression_graph_sha256": file_sha256(coexpression_graph_path),
        "go_graph_source": "official GEARS GO prior graph; no expression-derived statistics",
    }
    training_report = train_with_all_gene_validation(
        model,
        pert_data.dataloader["train_loader"],
        pert_data.dataloader["val_loader"],
        torch=torch,
        loss_fct=loss_fct,
        device=device,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    checkpoint_dir = output / "checkpoint"
    model.save_model(str(checkpoint_dir))
    test_indices = bundle["splits"]["unseen_test"]
    test_perturbation_indices = sorted(
        set(bundle["perturbation_index"][test_indices].astype(int).tolist())
    )
    test_names = [bundle_metadata["perturbation_names"][index] for index in test_perturbation_indices]
    prediction_output = model.predict([[name] for name in test_names])
    predictions = {name: prediction_output[name] for name in test_names}
    metrics = compute_absolute_expression_metrics(
        predictions,
        bundle,
        split_name="unseen_test",
    )
    delta_metrics = compute_control_relative_delta_metrics(
        predictions,
        bundle,
        split_name="unseen_test",
    )
    control_reference_sha256 = hashlib.sha256(
        control_reference_mean.astype("<f8", copy=False).tobytes()
    ).hexdigest()
    matched_control_reference = np.asarray(
        bundle["control_expression"], dtype=np.float64
    )[test_indices]
    matched_control_reference_sha256 = hashlib.sha256(
        matched_control_reference.astype("<f8", copy=False).tobytes()
    ).hexdigest()
    target_alignment = {
        "status": "IMPLEMENTED_NOT_REAL_DATA_VALIDATED",
        "gene_order_verified": bool(gene_order_report["verified"]),
        "matched_control_delta_conversion_implemented": True,
        "real_data_target_equivalence_verified": False,
        "prediction_space": "absolute_expression",
        "prediction_conversion": (
            "subtract the mean bundle matched-control expression for each held-out "
            "perturbation group"
        ),
        "target_space": "matched_control_relative_delta_expression",
        "target_source": "bundle.target_delta",
        "gears_prediction_control_cell_count": int(pert_data.ctrl_adata.n_obs),
        "gears_prediction_control_reference_sha256": control_reference_sha256,
        "matched_control_sample_count": int(len(test_indices)),
        "matched_control_reference_sha256": matched_control_reference_sha256,
        "gene_order": gene_order_report,
        "comparable_metric_keys": delta_metrics["comparable_metric_keys"],
        "non_comparable_metric_keys": delta_metrics["non_comparable_metric_keys"],
        "comparison_eligible": False,
        "comparison_blockers": [
            "review real Norman2019 per-perturbation results",
            "run at least three seeds",
            "confirm protocol-matched comparison before reporting model superiority",
        ],
    }

    elapsed_seconds = time.perf_counter() - started
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "TRAINED_DIAGNOSTIC_DELTA_ALIGNED_NOT_MULTI_SEED_VALIDATED",
        "input_h5ad_sha256": input_hash,
        "preflight_manifest_sha256": file_sha256(manifest_path),
        "preflight_split_pickle_sha256": file_sha256(split_path),
        "validated_split_pickle_sha256": file_sha256(validated_split_path),
        "split_json_sha256": file_sha256(split_json_path),
        "source_h5ad_sha256": preflight_manifest.get("source_h5ad_sha256"),
        "bundle_artifacts_sha256": preflight_manifest["bundle_artifacts_sha256"],
        "seed": seed,
        "dataset_name": dataset_name,
        "device": device,
        "model_config": {
            "epochs": epochs,
            "batch_size": effective_batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "hidden_size": hidden_size,
            "num_go_gnn_layers": num_go_gnn_layers,
            "num_gene_gnn_layers": num_gene_gnn_layers,
            "decoder_hidden_size": decoder_hidden_size,
            "num_similar_genes_go_graph": num_similar_genes_go_graph,
            "num_similar_genes_co_express_graph": num_similar_genes_co_express_graph,
            "coexpress_threshold": coexpress_threshold,
        },
        "training_only_metadata": metadata_report,
        "graph_coverage": graph_coverage,
        "graph_provenance": graph_provenance,
        "training": training_report,
        "metrics": metrics,
        "delta_metrics": delta_metrics,
        "target_alignment": target_alignment,
        "gene_order": gene_order_report,
        "checkpoint_sha256": file_sha256(checkpoint_dir / "model.pt"),
        "config_pickle_sha256": file_sha256(checkpoint_dir / "config.pkl"),
        "runtime_seconds": elapsed_seconds,
        "runtime_versions": {
            "python": platform.python_version(),
            "torch": package_version("torch"),
            "torch-geometric": package_version("torch-geometric"),
            "cell-gears": package_version("cell-gears"),
        },
        "official_gears_train_called": False,
        "test_loader_used_for_checkpoint_selection": False,
        "de_statistics_computed": False,
        "target_equivalence_verified": False,
        "directly_comparable_to_repository_delta_metrics": False,
        "comparison_eligible": False,
        "next_gates": [
            "review GO graph and target-alignment evidence on real Norman2019 data",
            "review per-perturbation failure cases",
            "run at least three seeds before interpreting stability",
            "confirm protocol-matched comparison before reporting model superiority",
        ],
    }
    for filename in ("results.json", "run_manifest.json"):
        with (output / filename).open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True, ensure_ascii=False)
    print(
        f"GEARS training complete: {len(test_names)} held-out perturbations; "
        f"status={result['status']}; multi-seed comparison remains blocked."
    )
    return result


def main() -> None:
    args = parse_args()
    run_gears_baseline(
        preflight_dir=args.preflight_dir,
        bundle_dir=args.bundle_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_size=args.hidden_size,
        num_go_gnn_layers=args.num_go_gnn_layers,
        num_gene_gnn_layers=args.num_gene_gnn_layers,
        decoder_hidden_size=args.decoder_hidden_size,
        num_similar_genes_go_graph=args.num_similar_genes_go_graph,
        num_similar_genes_co_express_graph=args.num_similar_genes_co_express_graph,
        coexpress_threshold=args.coexpress_threshold,
        device_choice=args.device,
    )


if __name__ == "__main__":
    main()
