# Roll Number: 2025AG05790  |  DM4ML Assignment 2
"""Stage 6 - Model training with MLflow tracking (params + artifacts; metrics are added by evaluation)."""
import json
import pickle

import mlflow
import numpy as np
from scipy import sparse

from src.common import CFG, ROLL, ROOT, get_logger, p
from src.common import read_csv
from src.evaluation.metrics import precision_recall_ndcg
from src.training.models import BiasedALS, PopularityRecommender

log = get_logger("training")


def tracking_setup(cfg):
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    name = "recommendation-pipeline"
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=(ROOT / cfg["paths"]["mlruns"]).as_uri())
    mlflow.set_experiment(name)


def select_pop_weight(cfg, grid=(0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0)):
    """Pick the hybrid popularity weight on a validation slice (latest 15% of TRAIN) - never on test."""
    tr = read_csv(p("data", "processed", "train_encoded.csv"), parse_dates=["timestamp"])
    cut = tr["timestamp"].quantile(0.85)
    sub, val = tr[tr["timestamp"] < cut], tr[tr["timestamp"] >= cut]
    shape = (tr["u_idx"].max() + 1, tr["i_idx"].max() + 1)
    Ms = sparse.csr_matrix((sub["rating"], (sub["u_idx"], sub["i_idx"])), shape=shape)
    model = BiasedALS(seed=cfg["seed"], **cfg["models"]["als"]).fit(Ms)
    k, thr = cfg["evaluation"]["k"], cfg["evaluation"]["relevance_threshold"]
    rel = val[val["rating"] >= thr].groupby("u_idx")["i_idx"].apply(set).to_dict()
    out = {}
    for w in grid:
        model.pop_weight = w
        nd = []
        for u, r in rel.items():
            sc = model.score_all(u).astype(float)
            sc[Ms.indices[Ms.indptr[u]:Ms.indptr[u + 1]]] = -np.inf
            top = np.argsort(-sc)[:k]
            nd.append(precision_recall_ndcg(list(top), r, k)[2])
        out[w] = float(np.mean(nd))
    best = max(out, key=out.get)
    log.info("Validation NDCG@%d by pop_weight: %s -> best=%s", k, {a: round(b, 4) for a, b in out.items()}, best)
    return best, out


def run(cfg: dict = CFG) -> dict:
    tracking_setup(cfg)
    M = sparse.load_npz(p("data", "processed", "user_item_matrix.npz"))
    run_ids = {}
    best_w, sweep = select_pop_weight(cfg)
    specs = {
        "popularity_baseline": (PopularityRecommender(shrink=10.0), {"shrink": 10.0}),
        "biased_als": (BiasedALS(seed=cfg["seed"], pop_weight=best_w, **cfg["models"]["als"]),
                       {**cfg["models"]["als"], "pop_weight": best_w}),
    }
    for name, (model, params) in specs.items():
        with mlflow.start_run(run_name=name) as r:
            mlflow.set_tags({"roll_number": ROLL, "assignment": "DM4ML-A2"})
            mlflow.log_params({**params, "seed": cfg["seed"], "train_matrix_shape": str(M.shape), "n_train_ratings": M.nnz})
            model.fit(M)
            if name == "biased_als":
                for w, v in sweep.items():
                    mlflow.log_metric("val_ndcg_at_10_by_pop_weight", v, step=int(w * 100))
            for i, v in enumerate(getattr(model, "history", [])):
                mlflow.log_metric("train_rmse_residual", v, step=i)
            path = p("models", f"{name}.pkl")
            pickle.dump(model, open(path, "wb"))
            mlflow.log_artifact(str(path))
            run_ids[name] = r.info.run_id
            log.info("Trained %s (run %s)", name, r.info.run_id[:8])
    json.dump({"roll_number": ROLL, **run_ids}, open(p("models", "run_ids.json"), "w"), indent=2)
    return run_ids


if __name__ == "__main__":
    run()
