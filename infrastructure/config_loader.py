"""YAML configuration loading, validation, and merging."""

from pathlib import Path
from typing import Any, Dict
import copy
import yaml


class ConfigLoader:
    """Load, validate, and merge YAML configurations."""

    @staticmethod
    def load(path: str) -> Dict[str, Any]:
        """Load a YAML config file."""
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        return config if config else {}

    @staticmethod
    def save(config: Dict[str, Any], path: str) -> None:
        """Save config dict to YAML."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    @staticmethod
    def merge(base: Dict, override: Dict) -> Dict:
        """Deep-merge *override* into *base*."""
        result = copy.deepcopy(base)
        for key, val in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(val, dict):
                result[key] = ConfigLoader.merge(result[key], val)
            else:
                result[key] = copy.deepcopy(val)
        return result

    @staticmethod
    def validate(config: Dict[str, Any]) -> None:
        """Raise ValueError for missing required fields."""
        for section in ("environment", "agent", "training"):
            if section not in config:
                raise ValueError(f"Config missing required section: '{section}'")
        if "name" not in config["environment"]:
            raise ValueError("Config 'environment' must have 'name'")
        if "name" not in config["agent"]:
            raise ValueError("Config 'agent' must have 'name'")
