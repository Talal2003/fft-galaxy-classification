"""Models, feature extraction, and evaluation for the FFT galaxy comparison."""

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Callable

import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fft_galaxy_runner.config import MODEL_DETAILS, RunConfig
from fft_galaxy_runner.data import LoadedDataset


class SklearnLinearRegressionModel:
    """Thin wrapper around scikit-learn's linear regression."""

    def __init__(self) -> None:
        self._model = LinearRegression()

    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        self._model.fit(features, labels)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self._model.predict(features)


class ClosedFormLinearRegressionModel:
    """Closed-form linear regression using a pseudoinverse for stability."""

    def __init__(self) -> None:
        self._coefficients: np.ndarray | None = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        design_matrix = np.column_stack([np.ones(features.shape[0], dtype=features.dtype), features])
        gram_matrix = design_matrix.T @ design_matrix
        self._coefficients = np.linalg.pinv(gram_matrix) @ design_matrix.T @ labels

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self._coefficients is None:
            raise RuntimeError("Model must be fit before prediction.")

        design_matrix = np.column_stack([np.ones(features.shape[0], dtype=features.dtype), features])
        return design_matrix @ self._coefficients


class RidgeRegressionModel:
    """Regularized linear baseline for correlated FFT features."""

    def __init__(self) -> None:
        self._model = Ridge(alpha=1.0)

    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        self._model.fit(features, labels)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self._model.predict(features)


class PCALinearRegressionModel:
    """PCA followed by linear regression on the reduced feature space."""

    def __init__(self) -> None:
        self._model: Pipeline | None = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        num_components = max(1, min(128, features.shape[0] - 1, features.shape[1]))
        self._model = Pipeline(
            [
                ("pca", PCA(n_components=num_components, svd_solver="randomized", random_state=42)),
                ("linear", LinearRegression()),
            ]
        )
        self._model.fit(features, labels)

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model must be fit before prediction.")
        return self._model.predict(features)


class RandomForestModel:
    """Tree ensemble baseline on the shared FFT feature matrix."""

    def __init__(self) -> None:
        self._model = RandomForestRegressor(
            n_estimators=80,
            max_features="sqrt",
            max_depth=24,
            random_state=42,
            n_jobs=-1,
            min_samples_leaf=4,
        )

    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        self._model.fit(features, labels)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self._model.predict(features)


MODEL_FACTORIES: dict[str, Callable[[], object]] = {
    "sklearn-linear": SklearnLinearRegressionModel,
    "closed-form-linear": ClosedFormLinearRegressionModel,
    "ridge": RidgeRegressionModel,
    "pca-linear": PCALinearRegressionModel,
    "random-forest": RandomForestModel,
}


@dataclass(frozen=True)
class FoldMetrics:
    """Metrics for one train/test split."""

    name: str
    overall_rmse: float
    duration_seconds: float


@dataclass(frozen=True)
class ModelResult:
    """Aggregate metrics and per-fold metrics for one model."""

    name: str
    source: str
    overall_rmse: float
    duration_seconds: float
    folds: list[FoldMetrics]


def build_feature_extractor(run_config: RunConfig) -> Callable[[bytes], np.ndarray]:
    """Build the FFT feature extractor."""

    def extract(raw_bytes: bytes) -> np.ndarray:
        # FFT image sizing and log-magnitude extraction were extracted from main and Shaima-FFT.
        decoded = np.frombuffer(raw_bytes, dtype=np.uint8)
        image = cv2.imdecode(decoded, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode image bytes.")

        image = cv2.resize(image, run_config.image_size)
        if run_config.color_mode == "BW":
            bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            return _fft_channel_features(bw, run_config.fft_size)

        if run_config.color_mode == "rgb":
            # Keep one FFT spectrum per channel, then concatenate them into one feature vector.
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            channels = cv2.split(rgb_image)
            features = [_fft_channel_features(channel, run_config.fft_size) for channel in channels]
            return np.concatenate(features, axis=0)

        raise ValueError(f"Unknown color mode: {run_config.color_mode}")

    return extract


def _fft_channel_features(channel: np.ndarray, fft_size: tuple[int, int]) -> np.ndarray:
    """Compute one flattened FFT log-magnitude spectrum."""
    fft = np.fft.fft2(channel)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.log(np.abs(fft_shift) + 1.0)
    spectrum = cv2.resize(magnitude, fft_size)
    return spectrum.astype(np.float32).reshape(-1)


def evaluate_models(dataset: LoadedDataset, run_config: RunConfig) -> list[ModelResult]:
    """Evaluate each configured model on the shared feature matrix."""
    split_plans = _build_split_plans(
        num_samples=len(dataset.features),
        evaluation_mode=run_config.evaluation,
        validation_split=run_config.validation_split,
        random_seed=run_config.random_seed,
    )

    results: list[ModelResult] = []
    for model_name in run_config.models:
        model_details = MODEL_DETAILS[model_name]
        model_start = time.perf_counter()
        folds: list[FoldMetrics] = []
        for split_name, train_indices, test_indices in split_plans:
            fold_start = time.perf_counter()
            train_features = dataset.features[train_indices]
            test_features = dataset.features[test_indices]
            train_labels = dataset.labels[train_indices]
            test_labels = dataset.labels[test_indices]

            # Fit scaling on the training fold only to avoid leakage.
            scaler = StandardScaler()
            scaled_train = scaler.fit_transform(train_features)
            scaled_test = scaler.transform(test_features)

            model = MODEL_FACTORIES[model_name]()
            model.fit(scaled_train, train_labels)
            predictions = model.predict(scaled_test)
            folds.append(
                FoldMetrics(
                    name=split_name,
                    overall_rmse=_compute_overall_rmse(test_labels, predictions),
                    duration_seconds=time.perf_counter() - fold_start,
                )
            )

        results.append(
            ModelResult(
                name=model_name,
                source=model_details["source"],
                overall_rmse=float(np.mean([fold.overall_rmse for fold in folds])),
                duration_seconds=time.perf_counter() - model_start,
                folds=folds,
            )
        )

    return results


def format_summary_table(results: list[ModelResult]) -> str:
    """Render the compact comparison table."""
    headers = ["model", "source", "rmse"]
    rows: list[list[str]] = []
    for result in results:
        rows.append([result.name, result.source, f"{result.overall_rmse:.4f}"])

    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def render_row(values: list[str]) -> str:
        return " | ".join(value.ljust(widths[index]) for index, value in enumerate(values))

    divider = "-+-".join("-" * width for width in widths)
    rendered = [render_row(headers), divider]
    rendered.extend(render_row(row) for row in rows)
    return "\n".join(rendered)



def _build_split_plans(
    num_samples: int,
    evaluation_mode: str,
    validation_split: float,
    random_seed: int,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Create single or even/odd two-fold splits."""
    if num_samples < 2:
        raise ValueError("At least two samples are required for evaluation.")

    indices = np.arange(num_samples)
    if evaluation_mode == "single":
        train_indices, test_indices = train_test_split(
            indices,
            test_size=validation_split,
            random_state=random_seed,
            shuffle=True,
        )
        return [("single", train_indices, test_indices)]

    if evaluation_mode == "twofold":
        # The even/odd split idea was adapted from origin/2-fold-cross-with-linear-regression.
        even_indices = indices[indices % 2 == 0]
        odd_indices = indices[indices % 2 == 1]
        if len(even_indices) == 0 or len(odd_indices) == 0:
            raise ValueError("Two-fold even/odd evaluation requires at least one even and one odd sample.")
        return [
            ("fold-1-even-train", even_indices, odd_indices),
            ("fold-2-odd-train", odd_indices, even_indices),
        ]

    raise ValueError(f"Unknown evaluation mode: {evaluation_mode}")


def _compute_overall_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Compute one RMSE across every target and sample."""
    squared_error = np.square(actual - predicted)
    return float(np.sqrt(np.mean(squared_error)))




def write_results_report(
    output_path: Path,
    *,
    run_config: RunConfig,
    dataset_size: int,
    target_columns: list[str],
    feature_dimension: int,
    results: list[ModelResult],
    total_duration_seconds: float,
) -> None:
    """Write a plain-text report with config, rankings, and per-model metrics."""
    ranked = sorted(results, key=lambda r: r.overall_rmse)
    lines: list[str] = []
    lines.append("FFT Galaxy Classification - Results Report")
    lines.append("=" * 44)
    lines.append("")

    # Config summary
    lines.append("Run Config")
    lines.append("-" * 10)
    lines.append(f"  dataset size:  {dataset_size}")
    lines.append(f"  targets:       {run_config.targets} ({len(target_columns)} columns)")
    lines.append(f"  features:      fft-{run_config.color_mode} (dim {feature_dimension})")
    lines.append(f"  evaluation:    {run_config.evaluation}")
    lines.append(f"  models:        {', '.join(run_config.models)}")
    lines.append(f"  image size:    {run_config.image_size[0]}x{run_config.image_size[1]}")
    lines.append(f"  fft size:      {run_config.fft_size[0]}x{run_config.fft_size[1]}")
    lines.append(f"  random seed:   {run_config.random_seed}")
    lines.append(f"  total time:    {total_duration_seconds:.1f}s")
    lines.append("")

    # Rankings
    lines.append("Rankings (by RMSE, lower is better)")
    lines.append("-" * 35)
    for i, r in enumerate(ranked, 1):
        acc = f"{r.top1_accuracy * 100:.1f}%" if r.top1_accuracy is not None else "n/a"
        lines.append(f"  {i}. {r.name:<22} RMSE {r.overall_rmse:.4f}  top1 {acc}  ({r.duration_seconds:.1f}s)")
    lines.append("")

    # Per-model detail
    for r in results:
        lines.append(f"Model: {r.name}")
        lines.append(f"  source: {r.source}")
        lines.append(f"  RMSE:   {r.overall_rmse:.4f}")
        if r.top1_accuracy is not None:
            lines.append(f"  top-1:  {r.top1_accuracy * 100:.1f}%")
        lines.append(f"  time:   {r.duration_seconds:.1f}s")
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")