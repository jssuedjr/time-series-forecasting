from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "stock_pred" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict:
    """Load the YAML config and resolve paths against the repo root."""
    path = Path(path) if path else DEFAULT_CONFIG
    config = yaml.safe_load(path.read_text())
    config["data"]["raw_dir"] = REPO_ROOT / config["data"]["raw_dir"]
    return config
