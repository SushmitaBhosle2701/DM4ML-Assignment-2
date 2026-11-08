# Roll Number: 2025AG05790  |  DM4ML Assignment 2
"""Ranking and rating metrics."""
import numpy as np


def precision_recall_ndcg(recs, relevant, k):
    hits = np.array([1.0 if r in relevant else 0.0 for r in recs[:k]])
    prec = hits.sum() / k
    rec = hits.sum() / len(relevant) if relevant else 0.0
    dcg = (hits / np.log2(np.arange(2, k + 2))).sum()
    idcg = (1.0 / np.log2(np.arange(2, min(len(relevant), k) + 2))).sum()
    return prec, rec, dcg / idcg if idcg > 0 else 0.0


def intra_list_diversity(rec_idx, item_vecs):
    """1 - mean pairwise cosine similarity of the recommended items' category vectors."""
    V = item_vecs[rec_idx]
    n = np.linalg.norm(V, axis=1, keepdims=True)
    V = V / np.where(n == 0, 1, n)
    S = V @ V.T
    m = len(rec_idx)
    return 1.0 - (S.sum() - np.trace(S)) / (m * (m - 1)) if m > 1 else 0.0


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))
