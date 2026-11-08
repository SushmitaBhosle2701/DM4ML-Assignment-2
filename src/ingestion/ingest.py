# Roll Number: 2025AG05790  |  DM4ML Assignment 2
"""Stage 1 - Data ingestion.

Produces raw users / items / interactions CSVs in data/raw.
 * synthetic : latent-factor generator (documented, seeded, with injected dirty rows)
 * movielens : reads an already-downloaded ml-100k folder from data/raw/ml-100k
"""
import numpy as np
import pandas as pd

from src.common import CFG, get_logger, p, write_csv

log = get_logger("ingestion")
GENRES = ["Action", "Comedy", "Drama", "Romance", "SciFi", "Thriller", "Documentary", "Animation"]
GENRES_ML = ["unknown", "Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary",
             "Drama", "Fantasy", "FilmNoir", "Horror", "Musical", "Mystery", "Romance", "SciFi",
             "Thriller", "War", "Western"]


def generate_synthetic(cfg: dict) -> dict:
    d, rng = cfg["data"], np.random.default_rng(cfg["seed"])
    nu, ni, n = d["n_users"], d["n_items"], d["n_interactions"]
    ng = len(GENRES)

    users = pd.DataFrame({
        "user_id": np.arange(1, nu + 1),
        "age": rng.integers(15, 70, nu).astype(float),
        "gender": rng.choice(["M", "F", "O"], nu, p=[0.48, 0.48, 0.04]),
        "country": rng.choice(["IN", "US", "UK", "DE", "BR"], nu, p=[0.4, 0.25, 0.15, 0.1, 0.1]),
        "signup_date": pd.to_datetime("2018-01-01") + pd.to_timedelta(rng.integers(0, 365 * 5, nu), unit="D"),
    })
    prim = rng.integers(0, ng, ni)
    items = pd.DataFrame({
        "item_id": np.arange(1, ni + 1),
        "title": [f"Item_{i:04d}" for i in range(1, ni + 1)],
        "category": [GENRES[g] for g in prim],
        "secondary_category": [GENRES[g] if rng.random() < 0.4 else None
                               for g in rng.integers(0, ng, ni)],
        "price": np.round(rng.uniform(1, 50, ni), 2),
        "release_year": rng.integers(1995, 2024, ni),
    })

    # latent factors -> personalised taste signal; Zipf-like item popularity
    k = 6
    U, V = rng.normal(0, 1, (nu, k)), rng.normal(0, 1, (ni, k))
    ub, ib = rng.normal(0, .3, nu), rng.normal(0, .5, ni)
    pop = rng.zipf(1.6, ni).astype(float).clip(max=200)
    pop /= pop.sum()
    uact = rng.gamma(1.2, 1, nu)
    uact /= uact.sum()
    uid = rng.choice(nu, n, p=uact)
    iid = rng.choice(ni, n, p=pop)
    score = 3.3 + ub[uid] + ib[iid] + 0.55 * (U[uid] * V[iid]).sum(1) + rng.normal(0, .5, n)
    rating = np.clip(np.round(score), 1, 5).astype(int)
    ts = (pd.to_datetime("2019-01-01") + pd.to_timedelta(rng.integers(0, 365 * 5, n), unit="D")
          + pd.to_timedelta(rng.integers(0, 86400, n), unit="s"))
    inter = pd.DataFrame({"user_id": uid + 1, "item_id": iid + 1, "rating": rating, "timestamp": ts})

    # ---- inject dirty records so the validation stage has real work to do ----
    inter = inter.drop_duplicates(["user_id", "item_id"]).reset_index(drop=True)
    inter["rating"] = inter["rating"].astype("float")
    inter["timestamp"] = inter["timestamp"].astype(str)
    nd = int(len(inter) * d["dirty_fraction"])
    idx = rng.choice(len(inter), 4 * nd, replace=False)
    a, b, c, e = np.array_split(idx, 4)
    inter.loc[a, "rating"] = rng.choice([0, 6, 7, -1, 11], len(a))              # invalid ratings
    inter.loc[b, "rating"] = np.nan                                              # missing ratings
    inter.loc[c, "timestamp"] = rng.choice(["2031-05-01 10:00:00", "not_a_date", "1969-01-01"], len(c))
    inter.loc[e, "item_id"] = rng.integers(9000, 9999, len(e))                   # orphan items
    inter = pd.concat([inter, inter.sample(nd, random_state=1)], ignore_index=True)   # exact duplicates
    orphan_u = inter.sample(nd // 2, random_state=2).copy()
    orphan_u["user_id"] = rng.integers(90000, 99999, len(orphan_u))              # orphan users
    inter = pd.concat([inter, orphan_u], ignore_index=True)
    users = pd.concat([users, users.sample(5, random_state=3)], ignore_index=True)  # duplicate users
    users.loc[users.sample(10, random_state=4).index, "age"] = np.nan
    items.loc[items.sample(8, random_state=5).index, "price"] = -5.0
    return {"users": users, "items": items, "interactions": inter.sample(frac=1, random_state=6)}


def load_movielens(cfg: dict) -> dict:
    base = p("data", "raw", "ml-100k", "u.data").parent
    r = pd.read_csv(base / "u.data", sep="\t", names=["user_id", "item_id", "rating", "timestamp"])
    r["timestamp"] = pd.to_datetime(r["timestamp"], unit="s").astype(str)
    u = pd.read_csv(base / "u.user", sep="|", names=["user_id", "age", "gender", "occupation", "zip"])
    u["country"] = "US"
    first = pd.to_datetime(r["timestamp"]).groupby(r["user_id"]).min()
    u["signup_date"] = u["user_id"].map(first)
    cols = ["item_id", "title", "release", "video", "url"] + GENRES_ML
    it = pd.read_csv(base / "u.item", sep="|", names=cols, encoding="latin-1")
    it["category"] = it[GENRES_ML].idxmax(axis=1)
    it["secondary_category"] = None
    it["price"] = 0.0
    it["release_year"] = pd.to_datetime(it["release"], errors="coerce").dt.year
    return {"users": u[["user_id", "age", "gender", "country", "signup_date"]],
            "items": it[["item_id", "title", "category", "secondary_category", "price", "release_year"]],
            "interactions": r}


def run(cfg: dict = CFG) -> dict:
    src = cfg["data"]["source"]
    log.info("Ingesting data (source=%s)", src)
    try:
        data = generate_synthetic(cfg) if src == "synthetic" else load_movielens(cfg)
    except FileNotFoundError as ex:
        log.error("Raw source not found: %s", ex)
        raise
    for name, df in data.items():
        write_csv(df, p("data", "raw", f"{name}.csv"))
        log.info("raw/%s.csv written: %d rows x %d cols", name, *df.shape)
    return {k: len(v) for k, v in data.items()}


if __name__ == "__main__":
    run()
