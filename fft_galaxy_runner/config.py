"""Runtime config and source labels for the FFT model comparison."""

from dataclasses import dataclass
from pathlib import Path
import tomllib

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT_DIR / "config.toml"

TARGET_COLUMNS = [
    "Class1.1",
    "Class1.2",
    "Class1.3",
    "Class2.1",
    "Class2.2",
    "Class3.1",
    "Class3.2",
    "Class4.1",
    "Class4.2",
    "Class5.1",
    "Class5.2",
    "Class5.3",
    "Class5.4",
    "Class6.1",
    "Class6.2",
    "Class7.1",
    "Class7.2",
    "Class7.3",
    "Class8.1",
    "Class8.2",
    "Class8.3",
    "Class8.4",
    "Class8.5",
    "Class8.6",
    "Class8.7",
    "Class9.1",
    "Class9.2",
    "Class9.3",
    "Class10.1",
    "Class10.2",
    "Class10.3",
    "Class11.1",
    "Class11.2",
    "Class11.3",
    "Class11.4",
    "Class11.5",
    "Class11.6",
]

TARGET_GROUPS = {
    "all": TARGET_COLUMNS,
    "q1": ["Class1.1", "Class1.2", "Class1.3"],
}

CLASS_NAME_GROUPS = {
    "q1": ["Smooth", "Features/Disk", "Star/Artifact"],
}

MODEL_DETAILS = {
    "sklearn-linear": {"source": "extracted from main baseline"},
    "closed-form-linear": {"source": "adapted from origin/2-fold-cross-with-linear-regression"},
    "ridge": {"source": "new in consolidation"},
    "pca-linear": {"source": "new in consolidation"},
    "random-forest": {"source": "new in consolidation"},
}


@dataclass(frozen=True)
class RunConfig:
    """Resolved run settings."""

    data_dir: Path
    sample_size: int | None
    targets: str
    evaluation: str
    validation_split: float
    random_seed: int
    color_mode: str
    models: tuple[str, ...]
    detailed: bool
    image_size: tuple[int, int]
    fft_size: tuple[int, int]
    output_path: Path | None


def load_run_config(config_path: Path = DEFAULT_CONFIG_PATH) -> RunConfig:
    """Load run config from config.toml."""
    with config_path.open("rb") as handle:
        doc = tomllib.load(handle)

    # Resolve data_dir relative to repo root
    data_dir = Path(doc.get("data_dir", "galaxy-zoo-the-galaxy-challenge"))
    if not data_dir.is_absolute():
        data_dir = ROOT_DIR / data_dir

    # sample_size: integer or "all" (None means use everything)
    raw_sample = doc.get("sample_size", 3000)
    sample_size = None if (isinstance(raw_sample, str) and raw_sample.lower() == "all") else int(raw_sample)

    # models: list of names, or omit/include "all" for everything
    raw_models = doc.get("models")
    if not raw_models or "all" in raw_models:
        models = tuple(sorted(MODEL_DETAILS))
    else:
        models = tuple(raw_models)

    # output_path
    output_path = None
    if "output_path" in doc:
        p = Path(doc["output_path"])
        output_path = p if p.is_absolute() else ROOT_DIR / p

    return RunConfig(
        data_dir=data_dir,
        sample_size=sample_size,
        targets=str(doc.get("targets", "all")),
        evaluation=str(doc.get("evaluation", "single")),
        validation_split=float(doc.get("validation_split", 0.2)),
        random_seed=int(doc.get("random_seed", 42)),
        color_mode=str(doc.get("color_mode", "BW")),
        models=models,
        detailed=bool(doc.get("detailed", False)),
        image_size=tuple(doc.get("image_size", [128, 128])),
        fft_size=tuple(doc.get("fft_size", [32, 32])),
        output_path=output_path,
    )


def get_target_columns(target_group: str) -> list[str]:
    if target_group not in TARGET_GROUPS:
        raise ValueError(f"Unknown target group: {target_group}")
    return list(TARGET_GROUPS[target_group])


def get_class_names(target_group: str) -> list[str] | None:
    names = CLASS_NAME_GROUPS.get(target_group)
    return list(names) if names else None

