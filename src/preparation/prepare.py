# Roll Number: 2025AG05790  |  DM4ML Assignment 2
"""Stage 3 - Data preparation: clean, impute, standardise, filter sparse entities, time-based split."""
import pandas as pd

from src.common import CFG, get_logger, p, read_csv, write_csv

log = get_logger("preparation")


def run(cfg: dict = CFG) -> dict:
    pc = cfg["preparation"]
    users = read_csv(p("data", "processed", "users_validated.csv"), parse_dates=["signup_date"])
    items = read_csv(p("data", "processed", "items_validated.csv"))
    inter = read_csv(p("data", "processed", "interactions_validated.csv"), parse_dates=["timestamp"])

    # standardise text formats, impute
    users["gender"] = users["gender"].astype(str).str.strip().str.upper()
    users["country"] = users["country"].astype(str).str.strip().str.upper()
    users["age"] = users["age"].fillna(users["age"].median()).astype(int)
    items["category"] = items["category"].astype(str).str.strip().str.title().replace({"Scifi": "SciFi"})
    price = pd.to_numeric(items["price"], errors="coerce")
    items["price"] = price.where(price >= 0).fillna(price[price >= 0].median())
    items["secondary_category"] = items["secondary_category"].fillna("None")
    items["release_year"] = items["release_year"].fillna(items["release_year"].median()).astype(int)

    # keep latest rating per (user, item)
    before = len(inter)
    inter = inter.sort_values("timestamp").drop_duplicates(["user_id", "item_id"], keep="last")
    log.info("Dropped %d repeated user-item pairs", before - len(inter))

    # iterative sparse-entity filtering (k-core)
    while True:
        n0 = len(inter)
        uc = inter["user_id"].value_counts()
        ic = inter["item_id"].value_counts()
        inter = inter[inter["user_id"].map(uc).ge(pc["min_user_interactions"])
                      & inter["item_id"].map(ic).ge(pc["min_item_interactions"])]
        if len(inter) == n0:
            break
    users = users[users["user_id"].isin(inter["user_id"])]
    items = items[items["item_id"].isin(inter["item_id"])]
    log.info("After k-core filter: %d users, %d items, %d interactions", len(users), len(items), len(inter))

    # global time-based split: newest test_fraction of interactions -> test (no future leakage)
    inter = inter.sort_values("timestamp").reset_index(drop=True)
    cut = inter["timestamp"].quantile(1 - pc["test_fraction"])
    train, test = inter[inter["timestamp"] < cut].copy(), inter[inter["timestamp"] >= cut].copy()
    # keep only test rows whose user/item exist in train (otherwise cannot be scored)
    test = test[test["user_id"].isin(train["user_id"]) & test["item_id"].isin(train["item_id"])]
    log.info("Split at %s -> train=%d test=%d", cut, len(train), len(test))

    for n, df in [("users", users), ("items", items), ("train", train), ("test", test)]:
        write_csv(df, p("data", "processed", f"{n}_clean.csv"))
    return {"train": len(train), "test": len(test), "users": len(users), "items": len(items),
            "split_timestamp": str(cut)}


if __name__ == "__main__":
    run()
