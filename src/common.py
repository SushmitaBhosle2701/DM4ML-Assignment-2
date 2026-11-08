# Roll Number: 2025AG05790  |  DM4ML Assignment 2
"""Shared helpers: config, logging, watermark-aware CSV IO."""
import logging
import os
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(ROOT / path) as f:
        return yaml.safe_load(f)


CFG = load_config()
ROLL = CFG["roll_number"]
WATERMARK = f"# Roll Number: {ROLL} | DM4ML Assignment 2"


def p(*parts) -> Path:
    """Resolve a path inside the repo and make sure the folder exists."""
    path = ROOT.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter(f"%(asctime)s [%(levelname)s] [{ROLL}] %(name)s: %(message)s")
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)
        fh = logging.FileHandler(p("reports", "pipeline.log"))
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """CSV with the roll-number watermark as a first comment line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(WATERMARK + "\n")
        df.to_csv(f, index=False)


def read_csv(path: Path, **kw) -> pd.DataFrame:
    """Reads CSVs written by write_csv (skips '#' comment lines)."""
    return pd.read_csv(path, comment="#", **kw)
