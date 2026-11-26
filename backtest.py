"""
Core engine for replicating Kelly, Malamud & Zhou (2024), "The Virtue of
Complexity in Return Prediction", Section V.

  * rff()            - random Fourier features, eq. (20)
  * rolling_ridge()  - rolling-window ridge forecasts for a whole grid of
                       model sizes P and ridge penalties z at once
  * performance()    - out-of-sample R^2, timing-strategy E[R], vol, Sharpe,
                       alpha / IR / t-stat vs the market, max loss, skew
"""
import numpy as np

Z_GRID = 10.0 ** np.arange(-3, 4)          # log10(z) = -3, ..., 3


def rff(G, n_features, gamma=2.0, seed=None):
    """
    Random Fourier features S_{i,t} = [sin(gamma w_i'G_t), cos(gamma w_i'G_t)],
    w_i ~ N(0, I).  Columns are ordered (sin_1, cos_1, sin_2, cos_2, ...) so
    that "the first P RFFs" is simply S[:, :P].

    G : (n x K) array of predictors.  Returns (n x n_features) array.
    """
    rng = np.random.default_rng(seed)
    G = np.asarray(G, dtype=float)
    W = rng.standard_normal((G.shape[1], n_features // 2))
    proj = gamma * (G @ W)
    S = np.empty((G.shape[0], 2 * W.shape[1]))
    S[:, 0::2] = np.sin(proj)
    S[:, 1::2] = np.cos(proj)
    return S


def rolling_ridge(S, R, T, z_grid=Z_GRID, P_grid=None, standardize=True):
    '''
    For each month t >= T: fit ridge on the previous T months of
    (S_{t-T..t-1}, R_{t-T+1..t}), forecast R_{t+1} = S_t' beta_hat.

    The estimator is the one in the paper's theory,
        beta(z) = (z I + S'S/T)^{-1} S'R/T,
    computed in its kernel ("dual") form so that P >> T is cheap:
        beta(z) = X'(XX' + zT I)^{-1} R.
    One eigendecomposition of the T x T Gram matrix per (t, P) gives every z.
    z = 0 is treated as the ridgeless (min-norm / pseudo-inverse) limit.

    S : (n x P_max) array of signals, row t known at end of month t
    R : (n,) array; R[t] is the return realized in month t+1
    P_grid : increasing list of model sizes; model P uses columns S[:, :P]
    standardize : divide each signal by its std in the training window
                  (footnote 39); the test row is scaled by the same std.

    Returns
    -------
    fc   : (len(P_grid), len(z_grid), n-T) out-of-sample forecasts
    bn   : same shape, ||beta_hat||^2 for each training window
    r_oos: (n-T,) realized returns matching fc[..., i]
    '''
    S = np.asarray(S, dtype=float)
    R = np.asarray(R, dtype=float)
    z_grid = np.asarray(z_grid, dtype=float)
    n, P_max = S.shape
    P_grid = [P_max] if P_grid is None else sorted(int(p) for p in P_grid)
    bounds = np.r_[0, P_grid]
    nP, nz, n_oos = len(P_grid), len(z_grid), n - T

    fc = np.empty((nP, nz, n_oos))
    bn = np.empty((nP, nz, n_oos))
    Ks = np.empty((nP, T + 1, T + 1))
    pen = T * z_grid[None, :, None]                       # (1, nz, 1)

    for i, t in enumerate(range(T, n)):
        X = S[t - T:t + 1]                                # T train rows + 1 test row
        if standardize:
            sd = X[:T].std(axis=0)
            sd[sd < 1e-12] = 1.0
            X = X / sd

        # Cumulative Gram matrices for nested models P_1 < P_2 < ...
        K = np.zeros((T + 1, T + 1))
        for k in range(nP):
            Xc = X[:, bounds[k]:bounds[k + 1]]
            K += Xc @ Xc.T
            Ks[k] = K

        lam, U = np.linalg.eigh(Ks[:, :T, :T])           # (nP,T), (nP,T,T)
        lam = np.clip(lam, 0.0, None)
        a = np.einsum('pjk,j->pk', U, R[t - T:t])         # U'R_train
        b = np.einsum('pjk,pj->pk', U, Ks[:, :T, T])      # U'X_train x_test

        d = lam[:, None, :] + pen                         # (nP, nz, T)
        tol = 1e-10 * np.maximum(lam.max(axis=1), 1.0)[:, None, None]
        inv = np.divide(1.0, d, out=np.zeros_like(d), where=d > tol)

        fc[:, :, i] = np.einsum('pk,pzk->pz', a * b, inv)
        bn[:, :, i] = np.einsum('pk,pzk->pz', lam * a ** 2, inv ** 2)

    return fc, bn, R[T:]


def _ols_alpha(y, x):
    """Vectorised OLS of y (..., n) on [1, x] with x (n,). Returns alpha, t, resid."""
    n = x.shape[-1]
    xm = x.mean()
    sxx = ((x - xm) ** 2).sum()
    beta = ((y - y.mean(-1, keepdims=True)) * (x - xm)).sum(-1) / sxx
    alpha = y.mean(-1) - beta * xm
    resid = y - alpha[..., None] - beta[..., None] * x
    s2 = (resid ** 2).sum(-1) / (n - 2)
    se = np.sqrt(s2 * (1.0 / n + xm ** 2 / sxx))
    return alpha, alpha / se, resid


def performance(fc, r, periods=12):
    """
    Out-of-sample statistics for forecasts fc (..., n) of realized r (n,).
    The timing strategy is pi_t = fc_t, with return pi_t * R_{t+1}.

    R2  : 1 - sum (r - fc)^2 / sum r^2                    (footnote 40)
    ER, Vol : monthly mean / std of the strategy return
    SR  : annualised Sharpe ratio (centered std), SR_t = t-stat of the mean
    alpha, alpha_t, IR : regression of strategy return on the (vol-
          standardised) market return; IR is annualised
    max_loss : worst month in std units; skew : sample skewness
    """
    fc = np.asarray(fc, dtype=float)
    r = np.asarray(r, dtype=float)
    n = r.shape[-1]
    strat = fc * r
    mu = strat.mean(-1)
    sd = strat.std(-1, ddof=1)
    alpha, alpha_t, resid = _ols_alpha(strat, r)
    dev = strat - mu[..., None]
    return {
        "R2": 1.0 - ((r - fc) ** 2).sum(-1) / (r ** 2).sum(),
        "ER": mu,
        "Vol": sd,
        "SR": mu / sd * np.sqrt(periods),
        "SR_t": mu / sd * np.sqrt(n),
        "alpha": alpha,
        "alpha_t": alpha_t,
        "IR": alpha / resid.std(-1, ddof=1) * np.sqrt(periods),
        "max_loss": -strat.min(-1) / sd,
        "skew": (dev ** 3).mean(-1) / dev.std(-1) ** 3,
    }


def alpha_vs(y, x, periods=12):
    """IR and alpha t-stat of strategy return y regressed on benchmark return x."""
    alpha, t, resid = _ols_alpha(np.asarray(y, float), np.asarray(x, float))
    return alpha / resid.std(-1, ddof=1) * np.sqrt(periods), t
