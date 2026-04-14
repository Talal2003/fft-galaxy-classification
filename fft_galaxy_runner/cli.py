"""Entrypoint for the FFT galaxy model comparison."""

from pathlib import Path
import sys
import time

from fft_galaxy_runner.config import DEFAULT_CONFIG_PATH, load_run_config
from fft_galaxy_runner.data import load_training_dataset
from fft_galaxy_runner.models import (
    build_feature_extractor,
    evaluate_models,
    format_detailed_results,
    format_summary_table,
    write_results_report,
)


def main(argv: list[str] | None = None) -> int:
    """Load config, run models, print results."""
    args = argv if argv is not None else sys.argv[1:]

    # Optional: pass a config path as the only argument
    config_path = Path(args[0]) if args else DEFAULT_CONFIG_PATH

    try:
        run_config = load_run_config(config_path)
        run_start = time.perf_counter()
        dataset = load_training_dataset(run_config=run_config, feature_extractor=build_feature_extractor(run_config))
        results = evaluate_models(dataset=dataset, run_config=run_config)
        total_duration = time.perf_counter() - run_start
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(
        f"Loaded {len(dataset.features)} galaxies "
        f"({dataset.feature_extractor_name}) "
        f"for target group '{run_config.targets}'."
    )
    print(format_summary_table(results))

    if run_config.detailed:
        print()
        print(format_detailed_results(results))

    if run_config.output_path is not None:
        write_results_report(
            run_config.output_path,
            run_config=run_config,
            dataset_size=len(dataset.features),
            target_columns=dataset.target_columns,
            feature_dimension=int(dataset.features.shape[1]),
            results=results,
            total_duration_seconds=total_duration,
        )
        print(f"\nWrote report to {run_config.output_path}")

    return 0