"""Load the bundled PYPOWER case, whose branch impedances are already per unit."""

import json
from pathlib import Path

import numpy as np


def load_case_from_files(load_dir, case_name):
    """Read network tables and their voltage/power base metadata."""
    case_dir = Path(load_dir) / case_name
    case = {
        table: np.loadtxt(
            case_dir / f"{table}.csv", delimiter=",", skiprows=1, ndmin=2
        )
        for table in ("bus", "branch", "gen")
    }
    with (case_dir / "system_info.json").open(encoding="utf-8") as source:
        case.update(json.load(source))
    return case
