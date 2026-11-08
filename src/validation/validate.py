# Roll Number: 2025AG05790  |  DM4ML Assignment 2
"""Stage 2 - Data validation.

Rule-based checks (pandas). Bad rows are removed from the clean set and written to
data/errors/*.csv with a `reason` column; a JSON + Markdown report goes to reports/.
"""
import json

import pandas as pd

from src.common import CFG, ROLL, get_logger, p, read_csv, write_csv

log = get_logger("validation")


def _tag(df, mask, reason, store):
    bad = df[mask].copy()
    if len(bad):
        bad["reason"] = reason
        store.append(bad)
    return mask


def run(cfg: dict = CFG) -> dict:
    v = cfg["validation"]
    users = read_csv(p("data", "raw", "users.csv"))
    items = read_csv(p("data", "raw", "items.csv"))
    inter = read_csv(p("data", "raw", "interactions.csv"), dtype={"timestamp": str})
    report = {"roll_number": ROLL, "checks": [], "rows_in": {}, "rows_clean": {}, "rows_rejected": {}}
    report["rows_in"] = {"users": len(users), "items": len(items), "interactions": len(inter)}
    errs = {"users": [], "items": [], "interactions": []}

    def log_check(table, name, n):
        report["checks"].append({"table": table, "check": name, "violations": int(n)})
        (log.warning if n else log.info)("%s | %-32s | violations=%d", table, name, n)

    # ---------------- users ----------------
    bad = pd.Series(False, index=users.index)
    m = _tag(users, users.duplicated("user_id", keep="first"), "duplicate_user_id", errs["users"])
    log_check("users", "duplicate user_id", m.sum()); bad |= m
    m = users["user_id"].isna(); _tag(users, m, "missing_user_id", errs["users"])
    log_check("users", "missing user_id", m.sum()); bad |= m
    age = pd.to_numeric(users["age"], errors="coerce")
    log_check("users", "missing age (kept, imputed later)", age.isna().sum())
    m = age.notna() & ((age < 5) | (age > 100)); _tag(users, m, "invalid_age", errs["users"])
    log_check("users", "age outside 5..100", m.sum()); bad |= m
    users_ok = users[~bad].copy()

    # ---------------- items ----------------
    bad = pd.Series(False, index=items.index)
    m = items.duplicated("item_id", keep="first"); _tag(items, m, "duplicate_item_id", errs["items"])
    log_check("items", "duplicate item_id", m.sum()); bad |= m
    m = items["category"].isna(); _tag(items, m, "missing_category", errs["items"])
    log_check("items", "missing category", m.sum()); bad |= m
    price = pd.to_numeric(items["price"], errors="coerce")
    m = price.isna() | (price < 0)
    log_check("items", "negative/non-numeric price (kept, imputed later)", m.sum())
    items_ok = items[~bad].copy()

    # ---------------- interactions ----------------
    bad = pd.Series(False, index=inter.index)
    ts = pd.to_datetime(inter["timestamp"], errors="coerce")
    inter["_ts"] = ts
    m = inter["user_id"].isna() | inter["item_id"].isna() | inter["rating"].isna()
    _tag(inter, m, "missing_value", errs["interactions"]); log_check("interactions", "missing user/item/rating", m.sum()); bad |= m
    num = pd.to_numeric(inter["rating"], errors="coerce")
    m = ~m & ((num < v["rating_min"]) | (num > v["rating_max"]))
    _tag(inter, m & ~bad, "invalid_rating", errs["interactions"]); log_check("interactions", "rating outside range", m.sum()); bad |= m
    m = ts.isna()
    _tag(inter, m & ~bad, "unparseable_timestamp", errs["interactions"]); log_check("interactions", "unparseable timestamp", m.sum()); bad |= m
    m = ts.notna() & ((ts < pd.Timestamp(v["min_timestamp"])) | (ts > pd.Timestamp(v["max_timestamp"])))
    _tag(inter, m & ~bad, "timestamp_out_of_range", errs["interactions"]); log_check("interactions", "timestamp out of range", m.sum()); bad |= m
    m = inter.duplicated(["user_id", "item_id", "rating", "timestamp"], keep="first") & ~bad
    _tag(inter, m, "duplicate_interaction", errs["interactions"]); log_check("interactions", "exact duplicate rows", m.sum()); bad |= m
    m = ~inter["user_id"].isin(users_ok["user_id"]) & ~bad
    _tag(inter, m, "orphan_user_id", errs["interactions"]); log_check("interactions", "user_id not in users (RI)", m.sum()); bad |= m
    m = ~inter["item_id"].isin(items_ok["item_id"]) & ~bad
    _tag(inter, m, "orphan_item_id", errs["interactions"]); log_check("interactions", "item_id not in items (RI)", m.sum()); bad |= m
    # type check on the rows that survived
    inter_ok = inter[~bad].drop(columns="_ts").copy()
    inter_ok["user_id"] = inter_ok["user_id"].astype(int)
    inter_ok["item_id"] = inter_ok["item_id"].astype(int)
    inter_ok["rating"] = inter_ok["rating"].astype(float)
    # same user rated same item twice with different values -> keep latest
    inter_ok["timestamp"] = pd.to_datetime(inter_ok["timestamp"])
    m = inter_ok.duplicated(["user_id", "item_id"], keep=False)
    log_check("interactions", "user-item pair repeated (keep latest)", m.sum())

    for name, df_ok in [("users", users_ok), ("items", items_ok), ("interactions", inter_ok)]:
        write_csv(df_ok, p("data", "processed", f"{name}_validated.csv"))
        e = pd.concat(errs[name]).drop(columns=["_ts"], errors="ignore") if errs[name] else pd.DataFrame()
        write_csv(e, p("data", "errors", f"{name}_errors.csv"))
        report["rows_clean"][name] = len(df_ok)
        report["rows_rejected"][name] = len(e)
        log.info("%s: clean=%d rejected=%d", name, len(df_ok), len(e))

    with open(p("reports", "validation_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    md = [f"<!-- Roll Number: {ROLL} | DM4ML Assignment 2 -->", "# Data Validation Report", "",
          f"*Roll Number: {ROLL}*", "", "| Table | Check | Violations |", "|---|---|---:|"]
    md += [f"| {c['table']} | {c['check']} | {c['violations']} |" for c in report["checks"]]
    md += ["", "| Table | Rows in | Clean | Rejected |", "|---|---:|---:|---:|"]
    md += [f"| {t} | {report['rows_in'][t]} | {report['rows_clean'][t]} | {report['rows_rejected'][t]} |"
           for t in report["rows_in"]]
    md += ["", f"_Roll Number: {ROLL}_"]
    p("reports", "validation_report.md").write_text("\n".join(md))
    return report


if __name__ == "__main__":
    run()
