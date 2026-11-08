# Roll Number: 2025AG05790  |  DM4ML Assignment 2
"""Recommender models. Both expose: predict_rating(u_idx, i_idx) and score_all(u_idx) -> item scores."""
import numpy as np
from scipy import sparse


class PopularityRecommender:
    """Baseline: non-personalised. Ranks items by number of train ratings; rating = shrunk item mean."""

    def __init__(self, shrink: float = 10.0):
        self.shrink = shrink

    def fit(self, M: sparse.csr_matrix):
        M = M.tocsc()
        cnt = np.diff(M.indptr).astype(float)
        sums = np.asarray(M.sum(0)).ravel()
        self.mu = sums.sum() / cnt.sum()
        self.item_rating = (sums + self.shrink * self.mu) / (cnt + self.shrink)
        self.item_pop = cnt
        return self

    def predict_rating(self, u, i):
        return self.item_rating[np.asarray(i)]

    def score_all(self, u):
        return self.item_pop.copy()


class BiasedALS:
    """Personalised: biased matrix factorisation fitted with alternating least squares (explicit ratings).
    r_ui ~ mu + b_u + b_i + p_u . q_i ; ranking score blends predicted rating with log-popularity."""

    def __init__(self, n_factors=20, reg=8.0, n_iters=12, seed=42, pop_weight=0.0):
        self.k, self.reg, self.n_iters, self.seed, self.pop_weight = n_factors, reg, n_iters, seed, pop_weight

    def fit(self, M: sparse.csr_matrix):
        rng = np.random.default_rng(self.seed)
        M = M.tocsr()
        nu, ni = M.shape
        Mc = M.tocsc()
        self.mu = M.data.mean()
        self.item_pop = np.diff(Mc.indptr).astype(float)
        # regularised biases
        bi = np.zeros(ni); bu = np.zeros(nu)
        for _ in range(5):
            for i in range(ni):
                s, e = Mc.indptr[i], Mc.indptr[i + 1]
                r = Mc.data[s:e] - self.mu - bu[Mc.indices[s:e]]
                bi[i] = r.sum() / (len(r) + 10)
            for u in range(nu):
                s, e = M.indptr[u], M.indptr[u + 1]
                r = M.data[s:e] - self.mu - bi[M.indices[s:e]]
                bu[u] = r.sum() / (len(r) + 10)
        self.bu, self.bi = bu, bi
        R = M.copy().astype(float)
        R.data = R.data - self.mu - bu[np.repeat(np.arange(nu), np.diff(M.indptr))] - bi[M.indices]
        Rc = R.tocsc()
        P = rng.normal(0, .1, (nu, self.k)); Q = rng.normal(0, .1, (ni, self.k))
        I = np.eye(self.k) * self.reg
        self.history = []
        for it in range(self.n_iters):
            for u in range(nu):
                s, e = R.indptr[u], R.indptr[u + 1]
                if e > s:
                    Qu = Q[R.indices[s:e]]
                    P[u] = np.linalg.solve(Qu.T @ Qu + I, Qu.T @ R.data[s:e])
            for i in range(ni):
                s, e = Rc.indptr[i], Rc.indptr[i + 1]
                if e > s:
                    Pi = P[Rc.indices[s:e]]
                    Q[i] = np.linalg.solve(Pi.T @ Pi + I, Pi.T @ Rc.data[s:e])
            pred = (P[np.repeat(np.arange(nu), np.diff(R.indptr))] * Q[R.indices]).sum(1)
            self.history.append(float(np.sqrt(np.mean((R.data - pred) ** 2))))
        self.P, self.Q = P, Q
        return self

    def predict_rating(self, u, i):
        u, i = np.asarray(u), np.asarray(i)
        out = self.mu + self.bu[u] + self.bi[i] + (self.P[u] * self.Q[i]).sum(1)
        return np.clip(out, 1, 5)

    def score_all(self, u):
        s = self.mu + self.bu[u] + self.bi + self.Q @ self.P[u]
        return s + self.pop_weight * np.log1p(self.item_pop)
