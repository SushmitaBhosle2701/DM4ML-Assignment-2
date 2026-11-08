# Roll Number: 2025AG05790  |  DM4ML Assignment 2
"""Stage 7 - Evaluation on the time-based test split. Logs metrics to the MLflow runs created in training."""
import json
import pickle

import mlflow
import numpy as np
import pandas as pd
from scipy import sparse

from src.common import CFG, ROLL, get_logger, p, read_csv, write_csv
from src.evaluation.metrics import intra_list_diversity, mae, precision_recall_ndcg, rmse
from src.training.train import tracking_setup

log = get_logger("evaluation")


def run(cfg: dict = CFG) -> pd.DataFrame:
    k, thr = cfg["evaluation"]["k"], cfg["evaluation"]["relevance_threshold"]
    tracking_setup(cfg)
    M = sparse.load_npz(p("data", "processed", "user_item_matrix.npz")).tocsr()
    test = read_csv(p("data", "processed", "test_encoded.csv"))
    maps = json.load(open(p("data", "processed", "id_maps.json")))
    itf = read_csv(p("data", "features", "item_features.csv"))
    oh = [c for c in itf.columns if c.startswith("category_onehot_")]
    i2idx = maps["item_index"]
    vecs = np.zeros((M.shape[1], len(oh)))
    for _, row in itf.iterrows():
        vecs[i2idx[str(int(row["item_id"]))]] = row[oh].values.astype(float)
    run_ids = json.load(open(p("models", "run_ids.json")))
    rel = test[test["rating"] >= thr].groupby("u_idx")["i_idx"].apply(set).to_dict()
    rows, per_user = [], []
    for name in ["popularity_baseline", "biased_als"]:
        model = pickle.load(open(p("models", f"{name}.pkl"), "rb"))
        P_, R_, N_, D_, rec_items = [], [], [], [], set()
        for u, relevant in rel.items():
            scores = model.score_all(u).astype(float)
            seen = M.indices[M.indptr[u]:M.indptr[u + 1]]
            scores[seen] = -np.inf                      # never recommend already-rated items
            top = np.argpartition(-scores, k)[:k]
            top = top[np.argsort(-scores[top])]
            pr, rc, nd = precision_recall_ndcg(list(top), relevant, k)
            P_.append(pr); R_.append(rc); N_.append(nd)
            D_.append(intra_list_diversity(top, vecs)); rec_items.update(top.tolist())
            per_user.append({"model": name, "user_idx": u, "precision": pr, "recall": rc, "ndcg": nd})
        pred = model.predict_rating(test["u_idx"].values, test["i_idx"].values)
        m = {"model": name, f"Precision@{k}": np.mean(P_), f"Recall@{k}": np.mean(R_), f"NDCG@{k}": np.mean(N_),
             "Coverage": len(rec_items) / M.shape[1], "Diversity": np.mean(D_),
             "RMSE": rmse(test["rating"], pred), "MAE": mae(test["rating"], pred), "users_evaluated": len(rel)}
        rows.append(m)
        with mlflow.start_run(run_id=run_ids[name]):
            mlflow.log_metrics({a.replace("@", "_at_"): float(b) for a, b in m.items() if a != "model"})
        log.info("%s -> %s", name, {a: round(float(b), 4) for a, b in m.items() if a != "model"})
    res = pd.DataFrame(rows)
    write_csv(res.round(4), p("reports", "model_comparison.csv"))
    md = [f"<!-- Roll Number: {ROLL} | DM4ML Assignment 2 -->", "# Model Evaluation Results", "",
          f"*Roll Number: {ROLL}* - time-based test split, K={k}, relevant = rating >= {thr}", "",
          "| " + " | ".join(res.columns) + " |", "|" + "---|" * len(res.columns)]
    md += ["| " + " | ".join(str(round(v, 4)) if not isinstance(v, str) else v for v in r) + " |" for r in res.values]
    md += ["", f"_Roll Number: {ROLL}_"]
    p("reports", "model_comparison.md").write_text("\n".join(md))
    return res


if __name__ == "__main__":
    run()
