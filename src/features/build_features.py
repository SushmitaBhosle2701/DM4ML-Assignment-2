# Roll Number: 2025AG05790  |  DM4ML Assignment 2
"""Stage 5 - Feature management (custom, file-based feature layer).

Offline store  : data/features/{user,item,interaction}_features.csv (built from TRAIN only -> no leakage)
Online access  : get_user_features(ids) / get_item_features(ids) read the same tables, so training and
                 prediction share identical feature definitions.
Catalog        : reports/feature_catalog.md (generated from FEATURE_CATALOG below).
"""
import numpy as np
import pandas as pd

from src.common import CFG, ROLL, get_logger, p, read_csv, write_csv

log = get_logger("features")

FEATURE_CATALOG = [
    ("user", "user_avg_rating", "float", "Mean rating given by the user (train)", "mean(rating) per user_id"),
    ("user", "user_activity_count", "int", "Number of ratings given by the user", "count(*) per user_id"),
    ("user", "user_rating_std", "float", "Rating spread (0 if one rating)", "std(rating) per user_id"),
    ("user", "days_since_last_interaction", "int", "Recency vs. reference time (end of train)", "ref_time - max(timestamp)"),
    ("user", "preferred_category", "str", "Category with most positive (>=4) ratings", "argmax count(category | rating>=4)"),
    ("user", "age", "int", "User age (imputed with median)", "users_clean.age"),
    ("item", "item_popularity", "int", "Number of ratings the item received", "count(*) per item_id"),
    ("item", "item_avg_rating", "float", "Mean rating of the item", "mean(rating) per item_id"),
    ("item", "item_bayes_rating", "float", "Shrunk mean rating (m=10 prior)", "(n*mean + 10*global)/(n+10)"),
    ("item", "category", "str", "Primary category", "items_clean.category"),
    ("item", "category_onehot_*", "float", "One-hot of primary/secondary category (used for diversity)", "get_dummies"),
    ("interaction", "user_item_category_affinity", "float", "Share of user's ratings in item's category", "n(user,cat)/n(user)"),
    ("interaction", "rating_dev_from_user_mean", "float", "rating - user_avg_rating", "rating - user_avg_rating"),
]


def build(cfg: dict = CFG) -> dict:
    train = read_csv(p("data", "processed", "train_clean.csv"), parse_dates=["timestamp"])
    users = read_csv(p("data", "processed", "users_clean.csv"))
    items = read_csv(p("data", "processed", "items_clean.csv"))
    ref = train["timestamp"].max()
    tr = train.merge(items[["item_id", "category"]], on="item_id")

    uf = train.groupby("user_id").agg(user_avg_rating=("rating", "mean"), user_activity_count=("rating", "size"),
                                      user_rating_std=("rating", "std"), last=("timestamp", "max")).reset_index()
    uf["user_rating_std"] = uf["user_rating_std"].fillna(0)
    uf["days_since_last_interaction"] = (ref - uf["last"]).dt.days
    pos = tr[tr["rating"] >= cfg["evaluation"]["relevance_threshold"]]
    pref = pos.groupby(["user_id", "category"]).size().reset_index(name="n").sort_values(["user_id", "n"], ascending=[True, False])
    pref = pref.drop_duplicates("user_id").rename(columns={"category": "preferred_category"})[["user_id", "preferred_category"]]
    uf = uf.drop(columns="last").merge(pref, on="user_id", how="left").merge(users[["user_id", "age"]], on="user_id", how="left")
    uf["preferred_category"] = uf["preferred_category"].fillna("Unknown")

    g = train["rating"].mean()
    itf = train.groupby("item_id").agg(item_popularity=("rating", "size"), item_avg_rating=("rating", "mean")).reset_index()
    itf["item_bayes_rating"] = (itf["item_popularity"] * itf["item_avg_rating"] + 10 * g) / (itf["item_popularity"] + 10)
    itf = itf.merge(items[["item_id", "category", "secondary_category"]], on="item_id", how="left")
    oh = pd.get_dummies(itf["category"], prefix="category_onehot").astype(float)
    sec = pd.get_dummies(itf["secondary_category"], prefix="category_onehot").astype(float)
    sec = sec.drop(columns=[c for c in sec.columns if c.endswith("_None")], errors="ignore")
    oh = oh.add(sec, fill_value=0).clip(upper=1)
    itf = pd.concat([itf.drop(columns="secondary_category"), oh], axis=1)

    cnt = tr.groupby(["user_id", "category"]).size().rename("n_uc").reset_index()
    cnt = cnt.merge(uf[["user_id", "user_activity_count"]], on="user_id")
    cnt["user_item_category_affinity"] = cnt["n_uc"] / cnt["user_activity_count"]
    inter = tr.merge(cnt[["user_id", "category", "user_item_category_affinity"]], on=["user_id", "category"])
    inter = inter.merge(uf[["user_id", "user_avg_rating"]], on="user_id")
    inter["rating_dev_from_user_mean"] = inter["rating"] - inter["user_avg_rating"]
    inter = inter[["user_id", "item_id", "user_item_category_affinity", "rating_dev_from_user_mean"]]

    write_csv(uf, p("data", "features", "user_features.csv"))
    write_csv(itf, p("data", "features", "item_features.csv"))
    write_csv(inter, p("data", "features", "interaction_features.csv"))

    rows = ["| Entity | Feature | Type | Description | Definition |", "|---|---|---|---|---|"]
    rows += [f"| {a} | `{b}` | {c} | {d} | {e} |" for a, b, c, d, e in FEATURE_CATALOG]
    p("reports", "feature_catalog.md").write_text(
        f"<!-- Roll Number: {ROLL} | DM4ML Assignment 2 -->\n# Feature Catalog\n\n*Roll Number: {ROLL}*\n\n"
        "All features are computed from the training split only and reused unchanged at prediction time via "
        "`get_user_features` / `get_item_features`.\n\n" + "\n".join(rows) + f"\n\n_Roll Number: {ROLL}_\n")
    log.info("Features: users=%d items=%d interaction rows=%d", len(uf), len(itf), len(inter))
    return {"user_features": len(uf), "item_features": len(itf), "interaction_features": len(inter)}


# ---- reusable accessors (training + prediction) ----
def get_user_features(user_ids=None) -> pd.DataFrame:
    df = read_csv(p("data", "features", "user_features.csv"))
    return df if user_ids is None else df[df["user_id"].isin(user_ids)]


def get_item_features(item_ids=None) -> pd.DataFrame:
    df = read_csv(p("data", "features", "item_features.csv"))
    return df if item_ids is None else df[df["item_id"].isin(item_ids)]


if __name__ == "__main__":
    build()
