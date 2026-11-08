# Roll Number: 2025AG05790  |  DM4ML Assignment 2
"""Dependency-free runner (no Prefect needed): executes all stages in order and stops on first failure."""
from src.common import get_logger
from src.evaluation import evaluate
from src.features import build_features
from src.ingestion import ingest
from src.preparation import prepare
from src.training import train
from src.transformation import transform
from src.validation import validate

log = get_logger("runner")
STAGES = [("ingest", ingest.run), ("validate", validate.run), ("prepare", prepare.run),
          ("transform", transform.run), ("features", build_features.build),
          ("train", train.run), ("evaluate", evaluate.run)]

if __name__ == "__main__":
    for name, fn in STAGES:
        log.info("=== stage: %s ===", name)
        fn()
    log.info("Pipeline completed successfully")
