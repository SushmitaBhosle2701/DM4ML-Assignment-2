# Roll Number: 2025AG05790  |  DM4ML Assignment 2
"""Stage 8 - Orchestration with Prefect.

Task graph (each task waits for its upstream via wait_for):
 ingest -> validate -> prepare -> transform -> features -> train -> evaluate
Retries are configured per task; a run summary (status + duration per task) is written to
reports/run_summary.json.
Run:  python dags/recommendation_flow.py      (optional schedule: see bottom of file)
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prefect import flow, task, get_run_logger  # noqa: E402

from src.common import CFG, ROLL, p  # noqa: E402

SUMMARY = []


def _timed(name, fn):
    t0 = time.time()
    try:
        out = fn()
        SUMMARY.append({"task": name, "status": "Completed", "seconds": round(time.time() - t0, 2)})
        return out
    except Exception:
        SUMMARY.append({"task": name, "status": "Failed", "seconds": round(time.time() - t0, 2)})
        raise


@task(name="ingest_data", retries=2, retry_delay_seconds=5)
def ingest():
    from src.ingestion import ingest as m
    return _timed("ingest_data", m.run)


@task(name="validate_data", retries=1)
def validate(_):
    from src.validation import validate as m
    return _timed("validate_data", lambda: {"checks": len(m.run()["checks"])})


@task(name="prepare_data")
def prepare(_):
    from src.preparation import prepare as m
    return _timed("prepare_data", m.run)


@task(name="transform_data")
def transform(_):
    from src.transformation import transform as m
    return _timed("transform_data", m.run)


@task(name="build_features")
def features(_):
    from src.features import build_features as m
    return _timed("build_features", m.build)


@task(name="train_models", retries=1)
def train(_):
    from src.training import train as m
    return _timed("train_models", m.run)


@task(name="evaluate_models")
def evaluate(_):
    from src.evaluation import evaluate as m
    return _timed("evaluate_models", lambda: m.run().round(4).to_dict("records"))


@flow(name=f"recommendation-pipeline-{ROLL}", log_prints=True)
def recommendation_pipeline():
    log = get_run_logger()
    SUMMARY.clear()
    a = ingest()
    b = validate(a)
    c = prepare(b)
    d = transform(c)
    e = features(d)
    f = train(e)
    g = evaluate(f)
    json.dump({"roll_number": ROLL, "flow": "recommendation_pipeline", "tasks": SUMMARY},
              open(p("reports", "run_summary.json"), "w"), indent=2)
    log.info("Pipeline finished: %s", g)
    return g


if __name__ == "__main__":
    recommendation_pipeline()
    # Scheduled alternative:  recommendation_pipeline.serve(name="daily", cron="0 2 * * *")
