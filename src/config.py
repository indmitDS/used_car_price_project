"""Configuration utilities."""
from pathlib import Path
import yaml


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def ensure_directories(config: dict) -> None:
    for key in ["plots_dir", "reports_dir", "models_dir", "logs_dir"]:
        Path(config["outputs"][key]).mkdir(parents=True, exist_ok=True)
