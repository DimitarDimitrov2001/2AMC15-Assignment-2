"""Create report-ready CSV artifacts from deep-RL experiment outputs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_TRAINING_OUTPUT_PATH = DEFAULT_RESULTS_DIR / "training_curves.csv"
DEFAULT_EVALUATION_OUTPUT_PATH = DEFAULT_RESULTS_DIR / "evaluation_results.csv"
HISTORY_FILE_NAME = "history.json"
SUMMARY_FILE_NAME = "evaluation_summary.txt"
SETTING_COLUMNS = (
    "grid",
    "agent",
    "sensor_mode",
    "use_sensors",
    "sigma",
)
RUN_COLUMNS = (
    "source_experiment_number",
    *SETTING_COLUMNS,
    "seed",
)
TRAINING_COLUMNS = (
    *SETTING_COLUMNS,
    "episode",
    "n_seeds",
    "source_experiment_numbers",
)
EVALUATION_COLUMNS = (
    *SETTING_COLUMNS,
    "n_seeds",
    "final_eval_total_runs",
    "source_experiment_numbers",
)
RUN_NAME_PATTERN = re.compile(
    r"^(?P<grid>.+)_(?P<agent>dqn|ddqn)"
    r"(?:_(?P<sensor_mode>no_sensors?|sensors?))?"
    r"(?:_sigma(?P<sigma>\d+(?:\.\d+)?))?"
    r"_seed(?P<seed>\d+)$"
)


@dataclass(frozen=True)
class RunMetadata:
    """Metadata derived from an experiment run directory name."""

    experiment_number: int
    grid: str
    agent: str
    sensor_mode: str
    use_sensors: bool
    sigma: float
    seed: int
    run_name: str

    @property
    def duplicate_key(self) -> tuple[str, str, str, float, int]:
        """Return the setting/seed identity used to drop duplicate runs."""
        return (self.grid, self.agent, self.sensor_mode, self.sigma, self.seed)

    @property
    def setting_key(self) -> tuple[str, str, str, bool, float]:
        """Return the setting identity used for report aggregation."""
        return (self.grid, self.agent, self.sensor_mode, self.use_sensors, self.sigma)

    def run_row(self) -> dict[str, Any]:
        """Return metadata fields for a single retained run."""
        return {
            "source_experiment_number": self.experiment_number,
            "grid": self.grid,
            "agent": self.agent,
            "sensor_mode": self.sensor_mode,
            "use_sensors": self.use_sensors,
            "sigma": self.sigma,
            "seed": self.seed,
        }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the report artifact builder."""
    parser = argparse.ArgumentParser(
        description=(
            "Build training-curve and evaluation-result CSVs from "
            "results/experiment_*/<run>/ artifacts."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing experiment_<n> result folders.",
    )
    parser.add_argument(
        "--training-output",
        type=Path,
        default=DEFAULT_TRAINING_OUTPUT_PATH,
        help="CSV path for per-episode training-curve aggregates.",
    )
    parser.add_argument(
        "--evaluation-output",
        type=Path,
        default=DEFAULT_EVALUATION_OUTPUT_PATH,
        help="CSV path for final evaluation aggregates.",
    )
    return parser.parse_args()


def main() -> None:
    """Create the training-curve and final-evaluation CSV artifacts."""
    args = parse_args()
    histories, dropped_history_count = _dedupe_run_files(
        _iter_run_files(args.results_dir, HISTORY_FILE_NAME)
    )
    summaries, dropped_summary_count = _dedupe_run_files(
        _iter_run_files(args.results_dir, SUMMARY_FILE_NAME)
    )

    training_rows = _build_training_curve_rows(histories)
    evaluation_rows = _build_evaluation_rows(summaries, histories)

    if not training_rows:
        raise SystemExit(f"No history rows found under {args.results_dir}")
    if not evaluation_rows:
        raise SystemExit(f"No evaluation summary rows found under {args.results_dir}")

    _write_csv(training_rows, args.training_output, list(TRAINING_COLUMNS))
    _write_csv(evaluation_rows, args.evaluation_output, list(EVALUATION_COLUMNS))

    print(
        f"Wrote {len(training_rows)} training curve rows from {len(histories)} "
        f"histories to {args.training_output}"
    )
    print(
        f"Wrote {len(evaluation_rows)} evaluation rows from {len(summaries)} "
        f"summaries to {args.evaluation_output}"
    )
    print(
        f"Dropped {dropped_history_count} duplicate histories and "
        f"{dropped_summary_count} duplicate summaries"
    )


def _iter_run_files(results_dir: Path, file_name: str) -> Iterable[Path]:
    experiment_dirs = sorted(results_dir.glob("experiment_*"))
    for experiment_dir in experiment_dirs:
        if not experiment_dir.is_dir():
            continue

        yield from sorted(experiment_dir.glob(f"*/{file_name}"))


def _dedupe_run_files(paths: Iterable[Path]) -> tuple[list[tuple[Path, RunMetadata]], int]:
    retained: dict[tuple[str, str, str, float, int], tuple[Path, RunMetadata]] = {}
    dropped_count = 0

    sorted_paths = sorted(paths, key=lambda path: (_metadata_from_run_file(path).experiment_number, str(path)))
    for path in sorted_paths:
        metadata = _metadata_from_run_file(path)
        if metadata.duplicate_key in retained:
            dropped_count += 1
            continue
        retained[metadata.duplicate_key] = (path, metadata)

    return list(retained.values()), dropped_count


def _build_training_curve_rows(
    histories: Sequence[tuple[Path, RunMetadata]],
) -> list[dict[str, Any]]:
    grouped_records: dict[
        tuple[str, str, str, bool, float, float], list[tuple[RunMetadata, Mapping[str, Any]]]
    ] = defaultdict(list)

    for history_path, metadata in histories:
        history = _read_history(history_path)

        for record in history:
            episode = _to_float(record.get("episode"))
            if episode is None:
                raise ValueError(f"History record without numeric episode in {history_path}")
            grouped_records[(*metadata.setting_key, episode)].append((metadata, record))

    rows = []
    for group_key, records in sorted(grouped_records.items()):
        grid, agent, sensor_mode, use_sensors, sigma, episode = group_key
        row = {
            "grid": grid,
            "agent": agent,
            "sensor_mode": sensor_mode,
            "use_sensors": use_sensors,
            "sigma": sigma,
            "episode": int(episode) if episode.is_integer() else episode,
            "n_seeds": len({metadata.seed for metadata, _ in records}),
            "source_experiment_numbers": _source_experiment_numbers(records),
        }
        row.update(_aggregate_numeric_fields(record for _, record in records))
        rows.append(row)

    return rows


def _build_evaluation_rows(
    summaries: Sequence[tuple[Path, RunMetadata]],
    histories: Sequence[tuple[Path, RunMetadata]],
) -> list[dict[str, Any]]:
    grouped_summaries: dict[
        tuple[str, str, str, bool, float], list[tuple[RunMetadata, Mapping[str, Any]]]
    ] = defaultdict(list)
    grouped_training_totals = _training_totals_by_setting(histories)

    for summary_path, metadata in summaries:
        grouped_summaries[metadata.setting_key].append((metadata, _read_summary(summary_path)))

    rows = []
    for group_key, records in sorted(grouped_summaries.items()):
        grid, agent, sensor_mode, use_sensors, sigma = group_key
        final_eval_runs = [
            _to_float(summary.get("final_eval_runs")) or 0.0 for _, summary in records
        ]
        row = {
            "grid": grid,
            "agent": agent,
            "sensor_mode": sensor_mode,
            "use_sensors": use_sensors,
            "sigma": sigma,
            "n_seeds": len({metadata.seed for metadata, _ in records}),
            "final_eval_total_runs": int(sum(final_eval_runs)),
            "source_experiment_numbers": _source_experiment_numbers(records),
        }
        row.update(_aggregate_numeric_fields(summary for _, summary in records))
        row.update(_aggregate_numeric_fields(grouped_training_totals.get(group_key, [])))
        rows.append(row)

    return rows


def _training_totals_by_setting(
    histories: Sequence[tuple[Path, RunMetadata]],
) -> dict[tuple[str, str, str, bool, float], list[dict[str, Any]]]:
    grouped_totals: dict[tuple[str, str, str, bool, float], list[dict[str, Any]]] = defaultdict(list)

    for history_path, metadata in histories:
        history = _read_history(history_path)
        total_collisions = sum(
            _to_float(record.get("rollout/collisions")) or 0.0 for record in history
        )
        grouped_totals[metadata.setting_key].append(
            {"training_total_collisions": total_collisions}
        )

    return grouped_totals


def _metadata_from_run_file(path: Path) -> RunMetadata:
    experiment_dir = path.parents[1].name
    run_name = path.parent.name
    experiment_match = re.fullmatch(r"experiment_(\d+)", experiment_dir)
    run_match = RUN_NAME_PATTERN.fullmatch(run_name)

    if experiment_match is None:
        raise ValueError(f"Expected experiment_<number> directory, got: {experiment_dir}")
    if run_match is None:
        raise ValueError(f"Could not parse run directory name: {run_name}")

    sensor_mode = _sensor_mode_from_run_name(run_name, run_match.group("sensor_mode"))
    sigma = run_match.group("sigma")

    return RunMetadata(
        experiment_number=int(experiment_match.group(1)),
        grid=run_match.group("grid"),
        agent=run_match.group("agent"),
        sensor_mode=sensor_mode,
        use_sensors=sensor_mode == "sensor",
        sigma=float(sigma) if sigma is not None else 0.0,
        seed=int(run_match.group("seed")),
        run_name=run_name,
    )


def _sensor_mode_from_run_name(run_name: str, parsed_sensor_mode: str | None) -> str:
    if parsed_sensor_mode is not None and parsed_sensor_mode.startswith("no_sensor"):
        return "no_sensor"
    if "no_sensor" in run_name:
        return "no_sensor"

    return "sensor"


def _read_summary(summary_path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    with summary_path.open(encoding="utf-8") as file:
        for line in file:
            if ":" not in line:
                continue

            key, raw_value = line.strip().split(":", maxsplit=1)
            summary[key] = _parse_summary_value(raw_value.strip())

    return summary


def _parse_summary_value(raw_value: str) -> Any:
    value = _to_float(raw_value)
    if value is not None:
        return value

    return raw_value


def _read_history(history_path: Path) -> list[dict[str, Any]]:
    with history_path.open(encoding="utf-8") as file:
        history = json.load(file)

    if not isinstance(history, list):
        raise ValueError(f"Expected a list of records in {history_path}")

    records: list[dict[str, Any]] = []
    for record in history:
        if not isinstance(record, dict):
            raise ValueError(f"Expected object records in {history_path}")
        records.append(record)

    return records


def _aggregate_numeric_fields(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    values_by_key: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for key, value in row.items():
            value_as_float = _to_float(value)
            if value_as_float is None:
                continue
            values_by_key[key].append(value_as_float)

    aggregated = {}
    for key, values in sorted(values_by_key.items()):
        if key == "episode":
            continue
        aggregated[f"{key}_mean"] = _mean(values)
        aggregated[f"{key}_variance"] = _sample_variance(values)

    return aggregated


def _source_experiment_numbers(
    records: Sequence[tuple[RunMetadata, Mapping[str, Any]]],
) -> str:
    experiment_numbers = sorted({metadata.experiment_number for metadata, _ in records})
    return ";".join(str(experiment_number) for experiment_number in experiment_numbers)


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None

    return None


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _sample_variance(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None

    mean = _mean(values)
    return sum((value - mean) ** 2 for value in values) / (len(values) - 1)


def _write_csv(
    rows: Sequence[Mapping[str, Any]],
    output_path: Path,
    preferred_columns: Sequence[str],
) -> None:
    fieldnames = _fieldnames(rows, preferred_columns)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fieldnames(
    rows: Sequence[Mapping[str, Any]],
    preferred_columns: Sequence[str],
) -> list[str]:
    seen = set(preferred_columns)
    fieldnames = list(preferred_columns)

    for row in rows:
        for key in sorted(row):
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(key)

    return fieldnames


if __name__ == "__main__":
    main()
