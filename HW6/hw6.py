"""HW6: Explainable AI.

Part 1: LIME for tabular regression (LimeTabularExplainer).
Part 2: Exact Shapley values under feature independence (ShapleyExplainer),
        plus a permutation Monte-Carlo variant for sanity checking.

Choices (sampling, discretization, kernel, surrogate; background size,
exact-vs-permutation MC) are documented inline.

Run:  python hw6.py
"""

from __future__ import annotations

from itertools import combinations
from math import factorial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split

HERE = Path(__file__).resolve().parent
BUILD = HERE / "build"
BUILD.mkdir(exist_ok=True)

CONT_IDX = [0, 1]      # x1, x2
CAT_IDX = [2]          # x3 (values in {1,2,3})
FEATURE_NAMES = ["x1", "x2", "x3"]
SEED = 42


# ---------------------------------------------------------------------------
# Black-box model
# ---------------------------------------------------------------------------

def load_data():
    df = pd.read_csv(HERE / "toydataset.csv")
    X = df[FEATURE_NAMES].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    return X, y


def train_black_box(X, y):
    """Random Forest. No scaling needed; RF handles the ordinal-encoded x3 by
    splitting on it directly. CV on the training split is reported for
    diagnostics only; no hyperparameter tuning is performed here."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )
    model = RandomForestRegressor(
        n_estimators=500, min_samples_leaf=5, random_state=SEED, n_jobs=-1
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_r2 = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="r2", n_jobs=-1)
    cv_mse = -cross_val_score(
        model, X_tr, y_tr, cv=cv, scoring="neg_mean_squared_error", n_jobs=-1
    )
    model.fit(X_tr, y_tr)
    y_hat = model.predict(X_te)
    metrics = {
        "cv_r2_mean": cv_r2.mean(),
        "cv_r2_std": cv_r2.std(),
        "cv_mse_mean": cv_mse.mean(),
        "cv_mse_std": cv_mse.std(),
        "test_r2": r2_score(y_te, y_hat),
        "test_mse": mean_squared_error(y_te, y_hat),
    }
    return model, X_tr, metrics


# ---------------------------------------------------------------------------
# LIME for tabular regression
# ---------------------------------------------------------------------------

class LimeTabularExplainer:
    """LIME for tabular data, regression.

    Choices (defaults match Ribeiro et al.'s reference implementation):

    * Sampling.  Features are sampled independently from the training
      distribution: continuous features from N(mu_j, sigma_j), the
      categorical feature from its empirical frequencies.  This is the
      "global" sampling scheme of the official LIME library.  Alternative:
      Gaussian centred on the explained instance with a small width
      (smaller-blast neighbourhood, but the surrogate then explains a less
      informative region).

    * Interpretable representation z in {0, 1}^d.  Continuous features are
      discretized into quartiles (boundaries from the training set);
      z_j = 1 iff the perturbed sample's value of feature j falls in the
      same bin as the explained instance.  For x3, z_j = 1 iff equal to
      the instance's category.  Coefficients of the surrogate then read
      as "being in the explained instance's bin for feature j contributes
      this many y-units to the local prediction".

    * Weights.  pi_x(x') = exp(-d^2 / sigma_kernel^2), with d the
      Euclidean distance in standardized space (continuous features in
      z-score units; categorical mismatch contributes a unit of distance).
      sigma_kernel = 0.75 * sqrt(d_features), LIME's default.

    * Surrogate.  Weighted Lasso with a small alpha so that the local fit
      is faithful but the explanation stays sparse.
    """

    def __init__(
        self,
        X_train: np.ndarray,
        continuous_idx,
        categorical_idx,
        n_bins: int = 4,
        kernel_width: float | None = None,
        random_state: int = 0,
    ):
        self.continuous_idx = list(continuous_idx)
        self.categorical_idx = list(categorical_idx)
        self.d = X_train.shape[1]
        self.n_bins = n_bins
        self.kernel_width = (
            kernel_width if kernel_width is not None else 0.75 * np.sqrt(self.d)
        )
        self.rng = np.random.default_rng(random_state)

        self.mu, self.sigma, self.bin_edges = {}, {}, {}
        for j in self.continuous_idx:
            col = X_train[:, j].astype(float)
            self.mu[j] = col.mean()
            self.sigma[j] = col.std()
            qs = np.quantile(col, np.linspace(0, 1, n_bins + 1))
            qs[0], qs[-1] = -np.inf, np.inf
            self.bin_edges[j] = qs

        self.cat_values, self.cat_freqs = {}, {}
        for j in self.categorical_idx:
            vals, counts = np.unique(X_train[:, j], return_counts=True)
            self.cat_values[j] = vals
            self.cat_freqs[j] = counts / counts.sum()

    def _bin(self, j: int, vals):
        return np.searchsorted(self.bin_edges[j][1:-1], vals, side="right")

    def explain(
        self,
        x: np.ndarray,
        predict_fn,
        num_samples: int = 5000,
        alpha: float = 0.01,
    ):
        x = np.asarray(x, dtype=float)
        X_pert = np.zeros((num_samples, self.d))
        X_pert[0] = x  # keep the instance itself in the neighbourhood
        for j in self.continuous_idx:
            X_pert[1:, j] = self.rng.normal(
                self.mu[j], self.sigma[j], size=num_samples - 1
            )
        for j in self.categorical_idx:
            X_pert[1:, j] = self.rng.choice(
                self.cat_values[j], size=num_samples - 1, p=self.cat_freqs[j]
            )

        # Binary interpretable representation z.
        Z = np.zeros_like(X_pert)
        for j in self.continuous_idx:
            x_bin = self._bin(j, x[j])
            Z[:, j] = (self._bin(j, X_pert[:, j]) == x_bin).astype(float)
        for j in self.categorical_idx:
            Z[:, j] = (X_pert[:, j] == x[j]).astype(float)

        # Kernel weights in the standardized original space.
        X_std = X_pert.copy()
        x_std = x.copy()
        for j in self.continuous_idx:
            X_std[:, j] = (X_pert[:, j] - self.mu[j]) / self.sigma[j]
            x_std[j] = (x[j] - self.mu[j]) / self.sigma[j]
        for j in self.categorical_idx:
            X_std[:, j] = (X_pert[:, j] != x[j]).astype(float)
            x_std[j] = 0.0
        dist = np.linalg.norm(X_std - x_std, axis=1)
        w = np.exp(-(dist ** 2) / (self.kernel_width ** 2))

        y_pert = predict_fn(X_pert)

        surrogate = Lasso(alpha=alpha, max_iter=20000)
        surrogate.fit(Z, y_pert, sample_weight=w)
        # R^2 of the surrogate on the perturbed sample (weighted),
        # a quick faithfulness check.
        y_hat = surrogate.predict(Z)
        ss_res = (w * (y_pert - y_hat) ** 2).sum()
        ss_tot = (w * (y_pert - np.average(y_pert, weights=w)) ** 2).sum()
        local_r2 = 1.0 - ss_res / ss_tot

        return {
            "intercept": float(surrogate.intercept_),
            "coef": surrogate.coef_.copy(),
            "local_pred": float(surrogate.predict(Z[:1])[0]),
            "black_box_pred": float(y_pert[0]),
            "local_r2": float(local_r2),
            "weights": w,
            "Z": Z,
            "X_pert": X_pert,
            "y_pert": y_pert,
            "x_bin": [int(self._bin(j, x[j])) for j in self.continuous_idx],
        }


# ---------------------------------------------------------------------------
# Shapley values (Part 2)
# ---------------------------------------------------------------------------

class ShapleyExplainer:
    """Exact Shapley values for tabular predictions under feature independence.

    Definition. With N = {0, ..., d-1} the set of feature indices,
        phi_j(x) = sum_{S subset of N \\ {j}}
                       |S|! (d - |S| - 1)! / d!  *  (v(S u {j}) - v(S)),
    where v(S) is the "value" of coalition S at point x. Under feature
    independence the standard choice is the interventional / marginal
    expectation
        v(S) = E_{X ~ p}[f(X_S = x_S, X_{N\\S})],
    estimated by Monte Carlo: hold the features in S fixed at x_S, replace
    the others with rows from a background sample drawn from the training
    data, average f over the background.

    With d = 3 features the exact 2^d = 8 subsets are cheap, so no Monte
    Carlo on the *subset* side is required; the only MC is over the
    background that integrates out features not in S. A background of
    a few hundred rows suffices here; we use 500.
    """

    def __init__(self, background: np.ndarray, random_state: int = 0):
        # `background` is shape (B, d). Used as the empirical marginal of
        # the missing features. Under the independence assumption we draw
        # each missing coordinate from this distribution, and since we treat
        # the features as independent we can equivalently draw whole rows
        # (any joint dependence in `background` is ignored once we replace
        # the in-coalition coordinates with x's values).
        self.background = np.asarray(background, dtype=float)
        self.B, self.d = self.background.shape
        self.rng = np.random.default_rng(random_state)

    # -- Coalition value v(S) --------------------------------------------
    def _v(self, S: frozenset, x: np.ndarray, predict_fn) -> float:
        """Estimate v(S) = E_X[f(x_S, X_{N\\S})] via the background sample."""
        X_eval = self.background.copy()
        for j in S:
            # Override the j-th coordinate of every background row with x_j.
            X_eval[:, j] = x[j]
        # Average black-box prediction over the background; this is the
        # Monte Carlo estimator of v(S) under the independence assumption.
        return float(predict_fn(X_eval).mean())

    # -- Exact subset enumeration ----------------------------------------
    def explain(self, x: np.ndarray, predict_fn):
        """Return (phi, baseline, f_x).
        `phi[j]` is the Shapley value of feature j at point x;
        `baseline` = v(empty) = E[f(X)] over the background;
        `f_x` = v(full) = f(x) exactly (background cancels out).
        Efficiency: f_x - baseline = sum(phi)."""
        x = np.asarray(x, dtype=float)
        d = self.d

        # 1) Precompute v(S) for every subset S. With d = 3 we have 8.
        v: dict[frozenset, float] = {}
        for r in range(d + 1):
            for S in combinations(range(d), r):
                v[frozenset(S)] = self._v(frozenset(S), x, predict_fn)

        # 2) Apply the Shapley formula. The weight |S|!(d-|S|-1)!/d! is the
        #    probability that, in a uniformly random ordering of the d
        #    features, the features in S come strictly before j and the
        #    rest come strictly after.
        phi = np.zeros(d)
        fact = [factorial(k) for k in range(d + 1)]
        feats = list(range(d))
        for j in feats:
            others = [k for k in feats if k != j]
            for r in range(d):  # |S| in {0, ..., d-1}
                w = fact[r] * fact[d - r - 1] / fact[d]
                for S in combinations(others, r):
                    S_set = frozenset(S)
                    phi[j] += w * (v[S_set | {j}] - v[S_set])

        baseline = v[frozenset()]
        f_x = v[frozenset(feats)]
        return phi, baseline, f_x


def shapley_permutation_mc(
    x, predict_fn, background, n_perms=2000, random_state=0
):
    """Permutation Monte-Carlo Shapley estimator (Strumbelj & Kononenko 2014).

    For each random permutation pi of the features and a random background
    row b, walk through pi adding one feature at a time: at step k the
    "before" vector is b with positions pi[:k] overwritten by x and the
    "after" vector adds pi[k]. The difference is feature pi[k]'s marginal
    contribution along this permutation. Averaging over many permutations
    converges to the exact Shapley value (the same coalitions appear with
    the same Shapley weights). Included here as a sanity check for the
    exact subset-based implementation and as the path one would take when
    d is large enough that 2^d enumeration is infeasible.
    """
    rng = np.random.default_rng(random_state)
    bg = np.asarray(background, dtype=float)
    x = np.asarray(x, dtype=float)
    B, d = bg.shape
    phi = np.zeros(d)
    # We make 2 predictions per (permutation, feature) step (before and
    # after). Batching across the d steps of a permutation keeps the call
    # count down: one (d+1)-row matrix per permutation.
    for _ in range(n_perms):
        perm = rng.permutation(d)
        b = bg[rng.integers(B)].copy()
        # Build the chain b -> step1 -> step2 -> ... -> x along perm.
        chain = np.zeros((d + 1, d))
        chain[0] = b
        z = b.copy()
        for k, j in enumerate(perm, start=1):
            z[j] = x[j]
            chain[k] = z
        preds = predict_fn(chain)  # length d+1
        for k, j in enumerate(perm):
            phi[j] += preds[k + 1] - preds[k]
    return phi / n_perms


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def bin_label(explainer, j, x_val):
    edges = explainer.bin_edges[j]
    idx = int(explainer._bin(j, x_val))
    lo = "-inf" if np.isinf(edges[idx]) else f"{edges[idx]:.2f}"
    hi = "inf" if np.isinf(edges[idx + 1]) else f"{edges[idx + 1]:.2f}"
    return f"x{j+1} in ({lo}, {hi}]"


def explanation_table(explainer, x, expl):
    rows = []
    for j in explainer.continuous_idx:
        rows.append((bin_label(explainer, j, x[j]), expl["coef"][j]))
    for j in explainer.categorical_idx:
        rows.append((f"x{j+1} = {int(x[j])}", expl["coef"][j]))
    return rows


def plot_explanations(explainer, samples, results, out_path):
    fig, axes = plt.subplots(1, len(samples), figsize=(3.2 * len(samples), 2.8),
                             sharex=True)
    if len(samples) == 1:
        axes = [axes]
    for ax, x, expl in zip(axes, samples, results):
        rows = explanation_table(explainer, x, expl)
        labels = [r[0] for r in rows]
        vals = np.array([r[1] for r in rows])
        colors = ["#2b8cbe" if v >= 0 else "#d7301f" for v in vals]
        ypos = np.arange(len(vals))[::-1]
        ax.barh(ypos, vals, color=colors)
        ax.set_yticks(ypos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.axvline(0, color="k", lw=0.6)
        x_str = "(" + ", ".join(f"{v:g}" for v in x) + ")"
        ax.set_title(
            f"x = {x_str}\n"
            f"f(x)={expl['black_box_pred']:.2f}, "
            f"g(x)={expl['local_pred']:.2f}, "
            f"$R^2_{{loc}}$={expl['local_r2']:.2f}",
            fontsize=8,
        )
        ax.set_xlabel("Lasso coefficient")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_shapley(samples, phis, baseline, f_xs, out_path):
    """Bar plot of phi(x) per instance, with a title showing the efficiency
    decomposition  f(x) = baseline + sum(phi)."""
    fig, axes = plt.subplots(1, len(samples), figsize=(3.2 * len(samples), 2.8),
                             sharex=True)
    if len(samples) == 1:
        axes = [axes]
    labels = [f"x{j+1}" for j in range(samples.shape[1])]
    for ax, x, phi, f_x in zip(axes, samples, phis, f_xs):
        colors = ["#2b8cbe" if v >= 0 else "#d7301f" for v in phi]
        ypos = np.arange(len(phi))[::-1]
        ax.barh(ypos, phi, color=colors)
        ax.set_yticks(ypos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.axvline(0, color="k", lw=0.6)
        x_str = "(" + ", ".join(f"{v:g}" for v in x) + ")"
        ax.set_title(
            f"x = {x_str}\n"
            f"f(x)={f_x:.2f}, $E[f]$={baseline:.2f}, "
            f"$\\sum\\varphi$={phi.sum():.2f}",
            fontsize=8,
        )
        ax.set_xlabel("Shapley value $\\varphi_j$")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_data_overview(X, y, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.8), sharey=True)
    for ax, level in zip(axes, [1, 2, 3]):
        m = X[:, 2] == level
        sc = ax.scatter(X[m, 0], X[m, 1], c=y[m], s=6, cmap="viridis",
                        vmin=y.min(), vmax=y.max())
        ax.set_title(f"x3 = {level}")
        ax.set_xlabel("x1")
    axes[0].set_ylabel("x2")
    fig.colorbar(sc, ax=axes, label="y", fraction=0.025, pad=0.02)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    X, y = load_data()
    plot_data_overview(X, y, BUILD / "data_overview.pdf")

    model, X_train, metrics = train_black_box(X, y)
    print("Black-box (Random Forest) performance:")
    print(f"  5-fold CV R^2  : {metrics['cv_r2_mean']:.3f} "
          f"(+/- {metrics['cv_r2_std']:.3f})")
    print(f"  5-fold CV MSE  : {metrics['cv_mse_mean']:.3f} "
          f"(+/- {metrics['cv_mse_std']:.3f})")
    print(f"  Held-out R^2   : {metrics['test_r2']:.3f}")
    print(f"  Held-out MSE   : {metrics['test_mse']:.3f}")

    explainer = LimeTabularExplainer(
        X_train,
        continuous_idx=CONT_IDX,
        categorical_idx=CAT_IDX,
        n_bins=4,
        random_state=SEED,
    )

    samples = np.array([
        [0.4, -0.4, 1],
        [0.2, -0.4, 1],
        [0.4, -0.4, 2],
        [0.4,  0.2, 2],
    ], dtype=float)

    results = []
    print("\nLIME explanations (Lasso alpha=0.01, 5000 samples per instance):")
    for x in samples:
        expl = explainer.explain(x, model.predict, num_samples=5000, alpha=0.01)
        results.append(expl)
        print(f"\n  x = {tuple(x.tolist())}")
        print(f"    f(x) = {expl['black_box_pred']:.3f}, "
              f"g(x) = {expl['local_pred']:.3f}, "
              f"intercept = {expl['intercept']:.3f}, "
              f"local R^2 = {expl['local_r2']:.3f}")
        for label, val in explanation_table(explainer, x, expl):
            print(f"    {label:>26s}  {val:+.3f}")

    plot_explanations(explainer, samples, results, BUILD / "lime_explanations.pdf")

    # LIME stability: repeat with different seeds, report std of coefficients.
    print("\nLIME stability (5 random seeds, 5000 samples each):")
    for x in samples:
        coefs = []
        for s in range(5):
            ex = LimeTabularExplainer(
                X_train, CONT_IDX, CAT_IDX, n_bins=4, random_state=s
            )
            r = ex.explain(x, model.predict, num_samples=5000, alpha=0.01)
            coefs.append(r["coef"])
        coefs = np.stack(coefs)
        print(f"  x={tuple(x.tolist())}: "
              f"mean={np.round(coefs.mean(0),3)}, "
              f"std={np.round(coefs.std(0),3)}")

    # ------------------------------------------------------------------
    # Part 2: Shapley values
    # ------------------------------------------------------------------
    # Background sample: a random subsample of the training set acts as the
    # empirical marginal we integrate over when features are "missing" from
    # a coalition. 500 rows is well past the variance plateau on this data.
    bg_rng = np.random.default_rng(SEED)
    bg_idx = bg_rng.choice(X_train.shape[0], size=500, replace=False)
    background = X_train[bg_idx]

    shap = ShapleyExplainer(background, random_state=SEED)

    print("\nShapley values (exact subsets, background size 500):")
    phis, f_xs = [], []
    baseline = None
    for x in samples:
        phi, base, f_x = shap.explain(x, model.predict)
        if baseline is None:
            baseline = base
        # Sanity-check the efficiency property f(x) = E[f] + sum(phi).
        assert np.isclose(f_x, base + phi.sum(), atol=1e-9), (
            f"efficiency violated: f={f_x} vs base+sum(phi)={base+phi.sum()}"
        )
        phis.append(phi)
        f_xs.append(f_x)
        print(f"\n  x = {tuple(x.tolist())}")
        print(f"    f(x) = {f_x:.3f}, E[f] = {base:.3f}, "
              f"sum(phi) = {phi.sum():.3f}")
        for j, name in enumerate(FEATURE_NAMES):
            print(f"    phi_{name:>3s} = {phi[j]:+.3f}")

    plot_shapley(samples, phis, baseline, f_xs, BUILD / "shapley_explanations.pdf")

    # Cross-check the exact value against the permutation Monte-Carlo
    # estimator. With 5000 permutations the MC error should be O(1/sqrt(N))
    # and well under 0.01 on every feature.
    print("\nShapley permutation-MC cross-check (5000 permutations):")
    for x, phi_exact in zip(samples, phis):
        phi_mc = shapley_permutation_mc(
            x, model.predict, background, n_perms=5000, random_state=SEED
        )
        diff = phi_mc - phi_exact
        print(f"  x={tuple(x.tolist())}: "
              f"exact={np.round(phi_exact,3)}, "
              f"MC={np.round(phi_mc,3)}, "
              f"diff={np.round(diff,3)}")


if __name__ == "__main__":
    main()
