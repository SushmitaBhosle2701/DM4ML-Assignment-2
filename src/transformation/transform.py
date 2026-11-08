# Roll Number: 2025AG05790  |  DM4ML Assignment 2
"""Stage 4 - Transformation: ID encoding, derived datasets, sparse user-item matrix."""
import json

import numpy as np
import pandas as pd
from scipy import sparse

from src.common import CFG, ROLL, get_logger, p, read_csv, write_csv

log = get_logger("transformation")


def run(cfg: dict = CFG) -> dict:
    train = read_csv(p("data", "processed", "train_clean.csv"), parse_dates=["timestamp"])
    test = read_csv(p("data", "processed", "test_clean.csv"), parse_dates=["timestamp"])
    items = read_csv(p("data", "processed", "items_clean.csv"))

    uids = np.sort(train["user_id"].unique())
    iids = np.sort(train["item_id"].unique())
    u2i = {int(u): i for i, u in enumerate(uids)}
    i2i = {int(i): j for j, i in enumerate(iids)}
    json.dump({"roll_number": ROLL, "user_index": u2i, "item_index": i2i},
              open(p("data", "processed", "id_maps.json"), "w"))

    for df in (train, test):
        df["u_idx"] = df["user_id"].map(u2i)
        df["i_idx"] = df["item_id"].map(i2i)
    write_csv(train, p("data", "processed", "train_encoded.csv"))
    write_csv(test, p("data", "processed", "test_encoded.csv"))

    # sparse user-item matrix of ratings (train only)
    M = sparse.csr_matrix((train["rating"].values, (train["u_idx"].values, train["i_idx"].values)),
                          shape=(len(uids), len(iids)))
    sparse.save_npz(p("data", "processed", "user_item_matrix.npz"), M)

    # derived datasets
    ref = train["timestamp"].max()
    activity = train.groupby("user_id").agg(n_ratings=("rating", "size"), mean_rating=("rating", "mean"),
                                           first_ts=("timestamp", "min"), last_ts=("timestamp", "max")).reset_index()
    activity["active_days"] = (activity["last_ts"] - activity["first_ts"]).dt.days
    write_csv(activity, p("data", "processed", "user_activity_summary.csv"))
    pop = train.groupby("item_id").agg(n_ratings=("rating", "size"), mean_rating=("rating", "mean")).reset_index()
    pop["popularity_rank"] = pop["n_ratings"].rank(ascending=False, method="first").astype(int)
    write_csv(pop.sort_values("popularity_rank"), p("data", "processed", "item_popularity.csv"))
    log.info("Matrix %s, density=%.4f", M.shape, M.nnz / (M.shape[0] * M.shape[1]))
    return {"matrix_shape": list(M.shape), "density": round(M.nnz / (M.shape[0] * M.shape[1]), 5),
            "reference_time": str(ref)}


if __name__ == "__main__":
    run()
