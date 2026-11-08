"""Portable dataset paths; public templates also work without dataset metadata."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def data_directory():
    """Prefer an explicit location, repository data, then an organized sibling."""
    configured = os.environ.get("AQUACUBEAI_DATA_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    bundled = ROOT / "data"
    if (bundled / "patches").is_dir():
        return bundled
    adjacent = ROOT.parent / "data"
    if ((adjacent / "metadata" / "sources_20x20.json").is_file()
            and (adjacent / "patches").is_dir()
            and (adjacent / "references").is_dir()):
        return adjacent
    return bundled


def configuration_file(filename):
    """Prefer explicit configuration, dataset metadata, then public templates."""
    configured = os.environ.get("AQUACUBEAI_CONFIG_DIR")
    if configured:
        path = Path(configured).expanduser().resolve() / filename
        if not path.is_file():
            raise FileNotFoundError("Configuration file not found: {}".format(path))
        return path
    supplied = data_directory() / "metadata" / filename
    return supplied if supplied.is_file() else ROOT / "configs" / filename
