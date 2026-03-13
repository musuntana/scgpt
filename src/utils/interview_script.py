from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.project_snapshot import build_project_snapshot

SUPPORTED_PITCH_TRACKS = ("both", "ai4bio", "ml-engineering")


def build_interview_script(
    project_root: str | Path = ".",
    *,
    track: str = "both",
) -> dict[str, Any]:
    """Build a structured interview/demo speaking script from the project snapshot."""
    if track not in SUPPORTED_PITCH_TRACKS:
        raise ValueError(f"Unsupported pitch track: {track}")
    snapshot = build_project_snapshot(project_root)
    headline = snapshot["headline"]
    assets = snapshot["assets"]
    commands = snapshot["commands"]
    unseen_error = snapshot.get("transformer_error_highlights", {}).get("unseen_test", {})
    best_model = headline.get("best_real_unseen_model") or "best baseline"
    best_real_unseen = _format_metric(headline.get("best_real_unseen_pearson"))
    transformer_unseen = _format_metric(headline.get("transformer_unseen_pearson"))
    transformer_deg100 = _format_metric(
        headline.get("transformer_unseen_top100_deg_overlap")
    )
    transformer_multiseed_runs = headline.get("transformer_multiseed_num_runs")
    transformer_multiseed_unseen = _format_mean_std(
        headline.get("transformer_multiseed_unseen_pearson_mean"),
        headline.get("transformer_multiseed_unseen_pearson_std"),
    )
    transformer_multiseed_deg100 = _format_mean_std(
        headline.get("transformer_multiseed_unseen_top100_deg_mean"),
        headline.get("transformer_multiseed_unseen_top100_deg_std"),
    )
    transformer_multiseed_suffix = _build_multiseed_suffix(
        transformer_multiseed_runs,
        transformer_multiseed_unseen,
        transformer_multiseed_deg100,
    )
    unseen_error_story = _build_unseen_error_story(unseen_error)

    title = "PerturbScope-GPT interview script"
    live_demo_script = [
        f"Start with the repository snapshot: `{commands['snapshot']}`.",
        f"Open the model-comparison figure: `{assets['real_comparison_figure']['path']}`.",
        "Explain that unseen perturbation evaluation is the main generalization metric.",
        f"Point to the Transformer inference preview: `{assets['real_inference_figure']['path']}`.",
        f"Launch the app: `{commands['app']}`.",
        "If the app is using synthetic fallback artifacts, say that explicitly before discussing any numbers.",
        "Select one perturbation gene, show predicted vs observed delta, then open the target ranking table.",
    ]
    if unseen_error_story:
        live_demo_script.append(
            f"Open the saved error-analysis highlights and explain that {unseen_error_story}."
        )
    live_demo_script.append(
        "Close by emphasizing reproducibility: doctor, snapshot, showcase, CI, and local-first execution."
    )
    honest_limitations = [
        "The MVP is intentionally limited to one public dataset: Norman2019.",
        "It only supports single-gene perturbations, not combinatorial perturbations.",
        "The ranking is heuristic and explicitly does not treat attention as causal evidence.",
        "This is a local-first MVP, not a cloud-scale or multi-dataset training platform.",
        "Synthetic showcase artifacts are for offline engineering demos only and should not be presented as biological evidence.",
    ]
    if unseen_error_story:
        honest_limitations.append(
            f"Saved error analysis shows that {unseen_error_story}, so low-signal conditions need extra care when interpreting failures."
        )
    next_steps = [
        "Evaluate on an additional perturbation dataset once the single-dataset path is fully stable.",
        "Add stronger biologically informed conditioning or gene-feature priors while preserving local runnability.",
        "Benchmark inference latency and package the demo for even smoother interview walkthroughs.",
    ]

    track_scripts = {
        "ai4bio": {
            "label": "AI4Bio",
            "one_liner": (
                "PerturbScope-GPT is a local-first AI4Bio MVP for single-cell perturbation "
                "response prediction, built on Norman2019 with end-to-end preprocessing, "
                "model comparison, DEG-based ranking, and a Streamlit demo."
            ),
            "resume_bullets": [
                (
                    "Built a reproducible single-cell perturbation pipeline on Norman2019 "
                    "(10,500 cells, 512 HVGs, 105 single-gene conditions), including QC, "
                    "batch-aware control-mean pairing, delta-expression targets, and split export."
                ),
                (
                    f"Implemented Transformer, MLP, and XGBoost baselines with seen/unseen "
                    f"perturbation evaluation; best unseen Pearson = {best_real_unseen} "
                    f"({best_model}), while the Transformer reached top-100 DEG overlap = "
                    f"{transformer_deg100}{transformer_multiseed_suffix}."
                ),
                (
                    "Shipped an interview-ready local product surface with Streamlit, notebooks, "
                    "CI, mypy/ruff/pre-commit, and CLI doctor/snapshot/showcase tooling."
                ),
            ],
            "thirty_second_pitch": [
                "I built a local-first single-cell perturbation response prediction project called PerturbScope-GPT.",
                "The pipeline starts from AnnData preprocessing and ends with model comparison, DEG-based target ranking, and a Streamlit demo.",
                (
                    f"On real Norman2019 data, all three models exceed 0.82 Pearson on unseen perturbations; "
                    f"the best is {best_model} at {best_real_unseen}, and the Transformer gets "
                    f"{transformer_deg100} top-100 DEG overlap."
                ),
                "I focused on both modeling and engineering quality, so the repo is reproducible, testable, and demo-ready on a single machine.",
            ],
            "two_minute_walkthrough": [
                "Problem: predict perturbation-induced delta expression from a control profile and perturbation identity.",
                (
                    "Data engineering: Norman2019 single-gene subset, QC, normalize_total, log1p, "
                    "512 HVGs, sparse-until-export processing, and batch-aware control means."
                ),
                (
                    "Modeling: a minimal Transformer with gene embeddings, scalar value projection, "
                    "and additive perturbation embeddings, compared against MLP and XGBoost baselines."
                ),
                (
                    "Evaluation: seen and unseen perturbation splits, per-perturbation Pearson and MSE, "
                    "plus top-k DEG overlap to test whether predicted deltas recover biologically relevant genes."
                ),
                (
                    f"Results: {best_model} is the best unseen-Pearson baseline at {best_real_unseen}; "
                    f"the Transformer remains competitive at {transformer_unseen} and is strong on DEG recovery "
                    f"with top-100 overlap {transformer_deg100}{transformer_multiseed_suffix}."
                ),
                (
                    f"Failure analysis: {unseen_error_story}."
                    if unseen_error_story
                    else "Failure analysis is available through saved per-perturbation diagnostics."
                ),
                (
                    "Productization: Streamlit UI, notebooks, CI, type checking, pre-commit hooks, "
                    "and helper commands like doctor, snapshot, showcase, and pitch."
                ),
            ],
            "interviewer_qa": [
                {
                    "question": "Why keep the Transformer if XGBoost has the best unseen Pearson?",
                    "answer": (
                        "Because the project goal is not just leaderboard performance; it is to demonstrate "
                        "a modern token-based perturbation model, compare it against strong baselines, and show "
                        "that the Transformer remains competitive while performing well on DEG recovery."
                    ),
                },
                {
                    "question": "Why predict delta expression instead of perturbed expression directly?",
                    "answer": (
                        "Delta expression aligns with the perturbation question directly and reduces nuisance variation, "
                        "because the model learns the perturbation-induced shift relative to a matched control baseline."
                    ),
                },
                {
                    "question": "How do you avoid over-claiming biological interpretation?",
                    "answer": (
                        "I keep the claims narrow: the ranking is heuristic, attention is not treated as causal evidence, "
                        "and I report DEG overlap as a recovery metric rather than mechanistic proof."
                    ),
                },
            ],
        },
        "ml-engineering": {
            "label": "ML Engineering",
            "one_liner": (
                "PerturbScope-GPT is a production-style local ML project that turns a research dataset "
                "into a reproducible end-to-end system with preprocessing, training, evaluation, comparison, "
                "and an interactive demo."
            ),
            "resume_bullets": [
                (
                    "Designed a config-driven end-to-end ML pipeline with explicit artifact boundaries for "
                    "preprocessing, training, evaluation, and UI layers."
                ),
                (
                    f"Implemented multi-model benchmarking with Transformer, MLP, and XGBoost; "
                    f"best unseen Pearson = {best_real_unseen} ({best_model}) with reproducible run summaries "
                    f"and structured result exports{transformer_multiseed_suffix}."
                ),
                (
                    "Added engineering guardrails including tests, mypy, ruff, pre-commit, CI, and "
                    "CLI tooling for health checks, project snapshots, showcase flows, and interview scripts."
                ),
            ],
            "thirty_second_pitch": [
                "I built PerturbScope-GPT as an end-to-end ML system rather than just a training script.",
                "It has explicit pipeline stages, reproducible artifacts, multiple baselines, quality gates, and a live demo surface.",
                (
                    f"On real unseen data, the best model reaches {best_real_unseen} Pearson, while the "
                    f"Transformer still delivers strong biological recovery with {transformer_deg100} top-100 DEG overlap."
                ),
                "The main value is that the project is understandable, testable, and demoable on a single machine.",
            ],
            "two_minute_walkthrough": [
                "System design: split the codebase into data, models, training, evaluation, ranking, utilities, scripts, and app layers.",
                (
                    "Reproducibility: configs are centralized in YAML, dependencies are pinned with uv, "
                    "and every meaningful run writes structured metrics and summaries."
                ),
                (
                    "Modeling strategy: compare a token-based Transformer against lower-complexity baselines "
                    "to understand trade-offs instead of committing to one model blindly."
                ),
                (
                    "Evaluation design: separate seen and unseen perturbations, use per-perturbation metrics, "
                    "and add DEG overlap so the output quality is not judged only by MSE."
                ),
                (
                    f"Debuggability: saved error-analysis artifacts show that {unseen_error_story}."
                    if unseen_error_story
                    else "Debuggability: saved error-analysis artifacts make the worst perturbations explicit."
                ),
                (
                    "Operational polish: CI, lint/typecheck, notebook support, health checks, snapshot export, "
                    "showcase orchestration, and pitch generation all make the repo interview-ready."
                ),
                (
                    "Product surface: Streamlit app, comparison figures, and a local-first fallback path using "
                    "synthetic artifacts when raw data is unavailable."
                ),
            ],
            "interviewer_qa": [
                {
                    "question": "What makes this more than a research prototype?",
                    "answer": (
                        "The artifact boundaries, quality gates, CLI workflows, reproducible configs, and demo tooling. "
                        "You can validate, inspect, and present the system without digging through ad hoc notebooks."
                    ),
                },
                {
                    "question": "How did you balance model quality against engineering simplicity?",
                    "answer": (
                        "I intentionally kept a minimal Transformer and strong baselines, then prioritized reproducibility, "
                        "clear interfaces, and local runnability over adding fragile complexity."
                    ),
                },
                {
                    "question": "What would you improve for a production ML stack?",
                    "answer": (
                        "I would add stronger experiment tracking, artifact versioning, and deployment packaging, but I would keep "
                        "the current local-first path as the stable baseline and avoid scope drift."
                    ),
                },
            ],
        },
    }

    if track == "both":
        selected_tracks = track_scripts
    else:
        selected_tracks = {track: track_scripts[track]}

    return {
        "title": title,
        "project_root": snapshot["project_root"],
        "selected_track": track,
        "tracks": selected_tracks,
        "live_demo_script": live_demo_script,
        "honest_limitations": honest_limitations,
        "next_steps": next_steps,
    }


def format_interview_script(script: dict[str, Any]) -> str:
    """Render a human-readable interview script."""
    lines = [
        script["title"],
        f"Project root: {script['project_root']}",
        f"Selected track: {script['selected_track']}",
        "",
    ]
    for track_key, track_script in script["tracks"].items():
        lines.extend(
            [
                "",
                f"Track: {track_script['label']} ({track_key})",
                "One-liner:",
                f"  {track_script['one_liner']}",
                "",
                "Resume bullets:",
            ]
        )
        lines.extend([f"  - {bullet}" for bullet in track_script["resume_bullets"]])

        lines.extend(["", "30-second pitch:"])
        lines.extend(
            [
                f"  {index}. {item}"
                for index, item in enumerate(track_script["thirty_second_pitch"], start=1)
            ]
        )

        lines.extend(["", "2-minute walkthrough:"])
        lines.extend(
            [
                f"  {index}. {item}"
                for index, item in enumerate(track_script["two_minute_walkthrough"], start=1)
            ]
        )

        lines.extend(["", "Common interviewer questions:"])
        for item in track_script["interviewer_qa"]:
            lines.append(f"  Q: {item['question']}")
            lines.append(f"  A: {item['answer']}")

    lines.extend(["", "Live demo script:"])
    lines.extend(
        [f"  {index}. {item}" for index, item in enumerate(script["live_demo_script"], start=1)]
    )

    lines.extend(["", "Honest limitations:"])
    lines.extend([f"  - {item}" for item in script["honest_limitations"]])

    lines.extend(["", "Next steps:"])
    lines.extend([f"  - {item}" for item in script["next_steps"]])

    return "\n".join(lines)


def _build_multiseed_suffix(num_runs: Any, unseen_text: str, deg100_text: str) -> str:
    if num_runs is None:
        return ""
    return (
        f"; across {num_runs} real Transformer seeds, unseen Pearson = {unseen_text} "
        f"and top-100 DEG overlap = {deg100_text}"
    )


def _build_unseen_error_story(unseen_error: dict[str, Any]) -> str:
    if not unseen_error:
        return ""
    dominant_label = unseen_error.get("dominant_failure_mode_label") or "n/a"
    dominant_count = unseen_error.get("dominant_failure_mode_count")
    num_perturbations = unseen_error.get("num_perturbations")
    count_text = ""
    if dominant_count is not None and num_perturbations:
        count_text = f" ({dominant_count}/{num_perturbations})"
    worst_pearson = unseen_error.get("worst_pearson_perturbation") or "n/a"
    worst_mse = unseen_error.get("worst_mse_perturbation") or "n/a"
    return (
        f"unseen misses are dominated by {dominant_label}{count_text}, "
        f"with worst Pearson on {worst_pearson} and worst MSE on {worst_mse}"
    )


def write_interview_script_text(script: dict[str, Any], output_path: str | Path) -> Path:
    """Write the formatted interview script to disk as plain text."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(format_interview_script(script), encoding="utf-8")
    return destination


def _format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def _format_mean_std(mean: Any, std: Any) -> str:
    if mean is None:
        return "n/a"
    if std is None:
        return _format_metric(mean)
    return f"{float(mean):.4f} +/- {float(std):.4f}"
