"""Format saved paths relative to the project root."""

from os.path import relpath
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_relative_path(path: str | Path) -> str:
    return Path(relpath(Path(path).resolve(), PROJECT_ROOT)).as_posix()
