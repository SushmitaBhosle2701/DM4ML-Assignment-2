<!-- Roll Number: 2025AG05790 | DM4ML Assignment 2 -->
# End-to-End Data Management Pipeline for a Recommendation System

**Roll Number: 2025AG05790** - DM4ML Assignment 2

Pipeline: `Ingestion -> Validation -> Preparation -> Transformation -> Features -> Training -> Evaluation`, orchestrated with Prefect and tracked with MLflow. Data is synthetic by default (seeded, with injected dirty records); MovieLens-100K is supported.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.10+
pip install -r requirements.txt
```

## Run
```bash
python dags/recommendation_flow.py     # full pipeline through Prefect (retries, dependencies, run summary)
# or, without Prefect:
python run_pipeline.py
# single stage, e.g.:
python -m src.validation.validate
```

Experiment UI: `mlflow ui --backend-store-uri sqlite:///mlflow.db`
Scheduling: see the last line of `dags/recommendation_flow.py` (`.serve(cron=...)`). For a real Prefect UI screenshot run `prefect server start`, then the flow in another terminal, and open http://127.0.0.1:4200.

### Using MovieLens instead of synthetic data
Download and unzip `ml-100k` into `data/raw/ml-100k/`, set `data.source: movielens` in `configs/config.yaml`, run the pipeline.

## Layout
| Path | Purpose |
|---|---|
| `src/ingestion` | synthetic generator / MovieLens loader -> `data/raw` |
| `src/validation` | rule checks; clean data -> `data/processed`, bad rows -> `data/errors`, reports -> `reports/validation_report.*` |
| `src/preparation` | imputation, k-core filter, time-based train/test split |
| `src/transformation` | ID maps, sparse user-item matrix, activity summary, item popularity |
| `src/features` | feature tables in `data/features`, accessors for training/prediction, `reports/feature_catalog.md` |
| `src/training` | popularity baseline + biased ALS hybrid, MLflow logging |
| `src/evaluation` | Precision/Recall/NDCG@10, Coverage, Diversity, RMSE, MAE -> `reports/model_comparison.*` |
| `dags/recommendation_flow.py` | Prefect flow |
| `configs/config.yaml` | all parameters and the seed |

## Results (K=10, time-based test split)
See `reports/model_comparison.md` and `reports/final_report.pdf`.

## Reproducibility
Fixed seed in config, deterministic data generation, train-only feature computation, fixed time split. Re-running gives identical metrics.

_Roll Number: 2025AG05790_
