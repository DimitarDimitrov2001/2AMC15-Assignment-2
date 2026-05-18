"""Compact overview tables for assignment experiment runs."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from experiments.runner import RunResult


OVERVIEW_FIELDS = [
    "group",
    "condition",
    "algorithm",
    "grid",
    "seed",
    "success_rate",
    "mean_undiscounted_return",
    "mean_episode_length",
    "policy_difference_from_optimal",
    "train_stop",
    "stopped_early",
    "training_time_s",
]

AGGREGATED_OVERVIEW_FIELDS = [
    "group",
    "condition",
    "algorithm",
    "grid",
    "n_seeds",
    "success_rate_mean",
    "success_rate_std",
    "mean_undiscounted_return_mean",
    "mean_undiscounted_return_std",
    "mean_episode_length_mean",
    "mean_episode_length_std",
    "policy_difference_from_optimal_mean",
    "policy_difference_from_optimal_std",
    "train_stop_mean",
    "train_stop_std",
    "training_time_s_mean",
    "training_time_s_std",
]

AGGREGATED_METRICS = [
    "success_rate",
    "mean_undiscounted_return",
    "mean_episode_length",
    "policy_difference_from_optimal",
    "train_stop",
    "training_time_s",
]


def _metadata(result: RunResult) -> dict[str, Any]:
    if not result.history:
        return {}
    metadata = result.history.get("metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def _train_stop(result: RunResult) -> Any:
    metadata = _metadata(result)
    if "stop_episode" in metadata:
        return metadata["stop_episode"]
    if "iterations" in metadata:
        return metadata["iterations"]
    return ""


def _stopped_early(result: RunResult) -> Any:
    metadata = _metadata(result)
    if "stopped_early" in metadata:
        return metadata["stopped_early"]
    return ""


def _fmt(value: Any) -> str:
    if value in {"", None}:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _to_float(value: Any) -> float | None:
    if value in {"", None}:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed):
        return None
    return parsed


def _mean(values: list[float]) -> float | str:
    return round(mean(values), 6) if values else ""


def _std(values: list[float]) -> float | str:
    return round(stdev(values), 6) if len(values) > 1 else (0.0 if values else "")


def build_overview_rows(results: list[RunResult]) -> list[dict[str, Any]]:
    """Return one compact display row per run."""
    rows: list[dict[str, Any]] = []
    for result in results:
        row = result.row
        rows.append(
            {
                "group": row["setup_group"],
                "condition": row["condition"],
                "algorithm": row["algorithm"],
                "grid": row["grid"],
                "seed": row["seed"],
                "success_rate": row["success_rate"],
                "mean_undiscounted_return": row["mean_undiscounted_return"],
                "mean_episode_length": row["mean_episode_length"],
                "policy_difference_from_optimal": row["policy_difference_from_optimal"],
                "train_stop": _train_stop(result),
                "stopped_early": _stopped_early(result),
                "training_time_s": row["training_time_s"],
            }
        )
    return rows


def build_aggregated_overview_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate overview rows over seeds for each group/condition/algorithm/grid."""
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["group"], row["condition"], row["algorithm"], row["grid"])
        grouped[key].append(row)

    aggregated: list[dict[str, Any]] = []
    for (group, condition, algorithm, grid), group_rows in grouped.items():
        out_row: dict[str, Any] = {
            "group": group,
            "condition": condition,
            "algorithm": algorithm,
            "grid": grid,
            "n_seeds": len({row["seed"] for row in group_rows}),
        }
        for metric in AGGREGATED_METRICS:
            values = [
                parsed
                for row in group_rows
                if (parsed := _to_float(row.get(metric))) is not None
            ]
            out_row[f"{metric}_mean"] = _mean(values)
            out_row[f"{metric}_std"] = _std(values)
        aggregated.append(out_row)
    return aggregated


def format_markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    """Format overview rows as a GitHub-flavoured Markdown table."""
    if not rows:
        return "No experiment results.\n"

    display_rows = [
        {field: _fmt(row.get(field, "")) for field in fields}
        for row in rows
    ]
    widths = {
        field: max(len(field), *(len(row[field]) for row in display_rows))
        for field in fields
    }
    header = "| " + " | ".join(field.ljust(widths[field]) for field in fields) + " |"
    separator = "| " + " | ".join("-" * widths[field] for field in fields) + " |"
    body = [
        "| " + " | ".join(row[field].ljust(widths[field]) for field in fields) + " |"
        for row in display_rows
    ]
    return "\n".join([header, separator, *body]) + "\n"


def save_overview(rows: list[dict[str, Any]], out_dir: Path) -> None:
    """Save overview rows as CSV and Markdown next to the master results CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "overview.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OVERVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "overview.md").write_text(
        format_markdown_table(rows, OVERVIEW_FIELDS),
        encoding="utf-8",
    )


def save_aggregated_overview(rows: list[dict[str, Any]], out_dir: Path) -> None:
    """Save seed-aggregated overview rows as CSV and Markdown."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "aggregated_overview.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=AGGREGATED_OVERVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "aggregated_overview.md").write_text(
        format_markdown_table(rows, AGGREGATED_OVERVIEW_FIELDS),
        encoding="utf-8",
    )


def print_and_save_overview(results: list[RunResult], out_dir: Path) -> None:
    """Print and persist the compact experiment overview."""
    rows = build_overview_rows(results)
    aggregated_rows = build_aggregated_overview_rows(rows)
    aggregated_table = format_markdown_table(aggregated_rows, AGGREGATED_OVERVIEW_FIELDS)
    print("\nAggregated experiment overview:")
    print(aggregated_table)
    save_overview(rows, out_dir)
    save_aggregated_overview(aggregated_rows, out_dir)
