"""Dataset loading utilities for the FFT galaxy model comparison."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import zipfile

import numpy as np
import pandas as pd

from fft_galaxy_runner.config import RunConfig, get_class_names, get_target_columns


@dataclass(frozen=True)
class LoadedDataset:
    """Feature matrix and labels for one configured training run."""

    features: np.ndarray
    labels: np.ndarray
    galaxy_ids: np.ndarray
    target_columns: list[str]
    class_names: list[str] | None
    feature_extractor_name: str


def load_training_dataset(
    run_config: RunConfig,
    feature_extractor: Callable[[bytes], np.ndarray],
) -> LoadedDataset:
    """Load labels and images from the zipped training dataset."""
    target_columns = get_target_columns(run_config.targets)
    class_names = get_class_names(run_config.targets)

    _validate_training_files(run_config.data_dir)

    solutions_zip = run_config.data_dir / "training_solutions_rev1.zip"
    images_zip = run_config.data_dir / "images_training_rev1.zip"

    with zipfile.ZipFile(solutions_zip) as archive:
        with archive.open("training_solutions_rev1.csv") as csv_file:
            labels_df = pd.read_csv(csv_file, usecols=["GalaxyID", *target_columns])

    if run_config.sample_size is not None:
        labels_df = labels_df.sample(
            n=min(run_config.sample_size, len(labels_df)),
            random_state=run_config.random_seed,
        )
    labels_df = labels_df.reset_index(drop=True)

    galaxy_ids = labels_df["GalaxyID"].to_numpy(dtype=np.int64)
    label_values = labels_df[target_columns].to_numpy(dtype=np.float32)

    features: list[np.ndarray] = []
    valid_indices: list[int] = []

    with zipfile.ZipFile(images_zip) as archive:
        total = len(galaxy_ids)
        for index, galaxy_id in enumerate(galaxy_ids):
            image_path = f"images_training_rev1/{galaxy_id}.jpg"
            try:
                # Images stay inside the zip; the extractor works directly from raw bytes.
                raw_bytes = archive.read(image_path)
                features.append(feature_extractor(raw_bytes))
                valid_indices.append(index)
            except KeyError:
                continue

            if (index + 1) % 100 == 0 or index + 1 == total:
                print(f"  Loaded {index + 1}/{total} images")

    if not features:
        raise RuntimeError("No training images were loaded. Check the dataset path and zip contents.")

    filtered_ids = galaxy_ids[valid_indices]
    filtered_labels = label_values[valid_indices]
    feature_matrix = np.stack(features).astype(np.float32)

    return LoadedDataset(
        features=feature_matrix,
        labels=filtered_labels,
        galaxy_ids=filtered_ids,
        target_columns=target_columns,
        class_names=class_names,
        feature_extractor_name=f"fft-{run_config.color_mode}",
    )


def _validate_training_files(data_dir: Path) -> None:
    """Fail early if the required dataset archives are missing."""
    required = [
        data_dir / "training_solutions_rev1.zip",
        data_dir / "images_training_rev1.zip",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        joined = "\n".join(missing)
        raise FileNotFoundError(f"Missing required training dataset files:\n{joined}")