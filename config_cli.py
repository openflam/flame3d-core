"""config_cli.py – Shared CLI parsing for config-driven pipeline steps.

Every pipeline step is driven entirely by a JSON config file plus a dataset
name.  Steps therefore expose a uniform command line interface::

    python -m <step.module> --config <path> [--dataset-name <name>]

This module centralises that parsing so each step doesn't repeat per-parameter
argument definitions.  The dataset name is always passed separately from the
config (it is not read out of the config by the steps); ``--dataset-name`` is
optional and falls back to the ``dataset_name`` recorded in the config file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def add_config_args(parser: argparse.ArgumentParser) -> None:
    """Add the standard ``--config`` / ``--dataset-name`` arguments."""
    parser.add_argument(
        "--config",
        required=True,
        metavar="PATH",
        help="Path to the JSON config file (e.g. server/default_config.json).",
    )
    parser.add_argument(
        "--dataset-name",
        dest="dataset_name",
        default=None,
        metavar="NAME",
        help="Dataset to process (defaults to dataset_name in the config file).",
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and return the JSON config at *config_path*."""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_dataset_name(
    config: Dict[str, Any], dataset_name: Optional[str]
) -> str:
    """Return the dataset name, falling back to the config's ``dataset_name``."""
    name = dataset_name or config.get("dataset_name")
    if not name:
        raise SystemExit(
            "No dataset name given: pass --dataset-name or set "
            '"dataset_name" in the config file.'
        )
    return name


def parse_config_args(
    description: str,
    *,
    argv: Optional[list] = None,
) -> Tuple[Dict[str, Any], str]:
    """Parse ``--config`` / ``--dataset-name`` and return ``(config, name)``.

    Convenience wrapper for steps whose CLI takes nothing but the config file
    and (optionally) a dataset name.
    """
    parser = argparse.ArgumentParser(description=description)
    add_config_args(parser)
    args = parser.parse_args(argv)

    config = load_config(str(Path(args.config).resolve()))
    dataset_name = resolve_dataset_name(config, args.dataset_name)
    return config, dataset_name
