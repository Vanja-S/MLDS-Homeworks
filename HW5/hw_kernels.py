import numpy as np
from cvxopt import matrix, solvers


# We return the Gram matrix in kernel methods
class Polynomial:
    def __init__(self, M):
        self.M = M

    def __call__(self, A, B):
        return (1 + A @ B.T) ** self.M


class RBF:
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, A, B):
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b
        A_1d = A.ndim == 1
        B_1d = B.ndim == 1
        A = np.atleast_2d(A)
        B = np.atleast_2d(B)
        A_sq = (A**2).sum(axis=1)
        B_sq = (B**2).sum(axis=1)
        sq_dists = A_sq[:, None] + B_sq[None, :] - 2 * A @ B.T
        K = np.exp(-sq_dists / (2 * self.sigma**2))
        # This is just a trick for the
        # dimensionality of the input
        if A_1d and B_1d:
            return K[0, 0]
        if A_1d:
            return K[0]
        if B_1d:
            return K[:, 0]
        return K


class KernelizedRidgeRegression:
    def __init__(self, kernel, lambda_) -> None:
        self.lambda_ = lambda_
        self.kernel = kernel
        self.alpha = None
        self.X_train = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        n = X.shape[0]
        K = self.kernel(X, X)
        self.alpha = np.linalg.solve(K + self.lambda_ * np.identity(n), y)
        # Predictions require k(x_new, x_i) for every training point, so we keep X.
        self.X_train = X
        return self

    def predict(self, X: np.ndarray):
        return self.kernel(X, self.X_train) @ self.alpha


class SVR:
    # cvxopt.solvers.qp solves:
    #   minimize    (1/2) xᵀ P x + qᵀ x
    #   subject to  G x ≤ h
    #               A x = b
    # We need to express Eq. (10) from Smola & Scholkopf in this form.
    # The decision variable is x = [α₁, α₁*, α₂, α₂*, …, αₗ, αₗ*] (length 2l).
    def __init__(self, kernel, lambda_, epsilon):
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.kernel = kernel
        self.alpha = None  # shape (l,)
        self.alpha_star = None  # shape (l,)
        self.b = None  # scalar bias
        self.X_train = None  # needed at predict time for k(x_new, xᵢ)

    def fit(self, X: np.ndarray, y: np.ndarray):
        n = X.shape[0]
        C = 1.0 / self.lambda_

        # Gram matrix of training inputs — appears in the quadratic term of the dual.
        K = self.kernel(X, X)

        # P: quadratic term. Eq. (10) has ½ Σᵢⱼ (αᵢ − αᵢ*)(αⱼ − αⱼ*) Kᵢⱼ.
        # With the interleaved ordering, each 2×2 block for pair (i, j) is
        # Kᵢⱼ * [[1, -1], [-1, 1]] — αα and α*α* terms are +Kᵢⱼ, cross terms are −Kᵢⱼ.
        # np.kron places that 2×2 pattern at every (i, j) block automatically.
        S = np.array([[1.0, -1.0], [-1.0, 1.0]])
        P = np.kron(K, S)

        # q: linear term. After negating to minimize, the linear part is
        #   ε Σ(αᵢ + αᵢ*) − Σ yᵢ(αᵢ − αᵢ*) = Σ(ε − yᵢ)αᵢ + Σ(ε + yᵢ)αᵢ*.
        # Interleaved: q[2i] = ε − yᵢ (coeff of αᵢ),  q[2i+1] = ε + yᵢ (coeff of αᵢ*).
        q = np.empty(2 * n)
        q[0::2] = self.epsilon - y
        q[1::2] = self.epsilon + y

        # G, h: box constraints 0 ≤ αᵢ, αᵢ* ≤ C. Two inequalities per variable:
        #   -x ≤ 0  (lower bound)  and   x ≤ C  (upper bound).
        # Stacking gives a (4l, 2l) G and a (4l,) h.
        G = np.vstack([-np.eye(2 * n), np.eye(2 * n)])
        h = np.concatenate([np.zeros(2 * n), C * np.ones(2 * n)])

        # A, b: equality Σᵢ (αᵢ − αᵢ*) = 0 (from ∂L/∂b = 0 in the primal).
        # Interleaved row of [+1, -1, +1, -1, …]. The Lagrange multiplier
        # cvxopt returns for this constraint IS the bias b.
        A = np.empty((1, 2 * n))
        A[0, 0::2] = 1.0
        A[0, 1::2] = -1.0
        b_eq = np.zeros((1, 1))

        sol = solvers.qp(
            matrix(P, tc="d"),
            matrix(q, tc="d"),
            matrix(G, tc="d"),
            matrix(h, tc="d"),
            matrix(A, tc="d"),
            matrix(b_eq, tc="d"),
        )

        x = np.array(sol["x"]).flatten()
        self.alpha = x[0::2]
        self.alpha_star = x[1::2]
        self.b = float(sol["y"][0])
        self.X_train = X

        return self

    def predict(self, X: np.ndarray):
        coef = self.alpha - self.alpha_star
        return self.kernel(X, self.X_train) @ coef + self.b

    def get_alpha(self):
        return np.column_stack([self.alpha, self.alpha_star])

    def get_b(self):
        return self.b


def _load_sine():
    data = np.loadtxt("sine.csv", delimiter=",", skiprows=1)
    X = data[:, 0:1]  # (n, 1)
    y = data[:, 1]
    return X, y


def _standardize(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    return (X - mu) / sd, mu, sd


def _mse(y, yhat):
    return float(np.mean((y - yhat) ** 2))


def part_1(out_dir="build"):
    """Fit KRR and SVR with both kernels on sine.csv. Saves one plot per
    (method, kernel) combination and returns a summary dict."""
    import os
    import matplotlib.pyplot as plt
    from cvxopt import solvers

    solvers.options["show_progress"] = False
    os.makedirs(out_dir, exist_ok=True)

    X, y = _load_sine()
    Xs, mu, sd = _standardize(X)  # poly kernel needs scaled inputs

    # Dense grid for plotting the fit.
    x_grid = np.linspace(X.min() - 0.5, X.max() + 0.5, 500).reshape(-1, 1)
    x_grid_s = (x_grid - mu) / sd

    # Hand-picked hyperparameters. No CV — chosen to visibly illustrate fit.
    configs = [
        ("KRR", "Polynomial", Polynomial(M=11), 1e-3, None),
        ("KRR", "RBF",        RBF(sigma=0.5),   1e-3, None),
        ("SVR", "Polynomial", Polynomial(M=11), 1e-2, 0.5),
        ("SVR", "RBF",        RBF(sigma=0.5),   1e-2, 0.5),
    ]

    summary = []
    for method, kname, kernel, lam, eps in configs:
        if method == "KRR":
            m = KernelizedRidgeRegression(kernel=kernel, lambda_=lam).fit(Xs, y)
            yhat_train = m.predict(Xs)
            yhat_grid = m.predict(x_grid_s)
            sv_mask = None
            n_sv = None
        else:
            m = SVR(kernel=kernel, lambda_=lam, epsilon=eps).fit(Xs, y)
            yhat_train = m.predict(Xs)
            yhat_grid = m.predict(x_grid_s)
            # A point is a support vector if |α - α*| is non-negligible.
            coef = m.alpha - m.alpha_star
            sv_mask = np.abs(coef) > 1e-5
            n_sv = int(sv_mask.sum())

        mse = _mse(y, yhat_train)
        summary.append({
            "method": method, "kernel": kname,
            "lambda": lam, "epsilon": eps,
            "mse": mse, "n_sv": n_sv,
        })

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(X[:, 0], y, s=14, alpha=0.55, color="#4477AA",
                   label="data", zorder=2)
        ax.plot(x_grid[:, 0], yhat_grid, color="#CC3311", lw=2,
                label="fit", zorder=3)
        if method == "SVR":
            ax.scatter(X[sv_mask, 0], y[sv_mask], s=42,
                       facecolors="none", edgecolors="#117733", lw=1.2,
                       label=f"support vectors ({n_sv})", zorder=4)
            # ε-tube around the fit
            ax.fill_between(x_grid[:, 0], yhat_grid - eps, yhat_grid + eps,
                            color="#CC3311", alpha=0.08,
                            label=f"ε = {eps}", zorder=1)
        title = f"{method} + {kname}"
        if method == "KRR":
            title += f"   (λ={lam})"
        else:
            title += f"   (λ={lam}, ε={eps})"
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
        fig.tight_layout()
        fname = f"{out_dir}/sine_{method.lower()}_{kname.lower()}.pdf"
        fig.savefig(fname)
        plt.close(fig)
        print(f"Saved {fname}")

    print("\nSummary:")
    print(f"{'method':5} {'kernel':11} {'lambda':>8} {'epsilon':>8} "
          f"{'MSE':>8} {'#SV':>5}")
    for r in summary:
        print(f"{r['method']:5} {r['kernel']:11} {r['lambda']:>8.0e} "
              f"{str(r['epsilon']):>8} {r['mse']:>8.3f} "
              f"{str(r['n_sv']):>5}")

    return summary


def _load_housing():
    data = np.loadtxt("housing2r.csv", delimiter=",", skiprows=1)
    return data[:, :-1], data[:, -1]


def _train_test_split(X, y, test_frac=0.2, seed=0):
    rng = np.random.RandomState(seed)
    n = len(X)
    perm = rng.permutation(n)
    n_test = int(test_frac * n)
    te, tr = perm[:n_test], perm[n_test:]
    return X[tr], y[tr], X[te], y[te]


def _cv_pick_lambda(X, y, kernel_factory, method, lambdas, epsilon=None, k=5, seed=0):
    """k-fold CV: for each λ in `lambdas`, compute mean fold MSE.
    Returns the λ with the lowest mean fold MSE."""
    rng = np.random.RandomState(seed)
    n = len(X)
    perm = rng.permutation(n)
    folds = np.array_split(perm, k)
    best_lam, best_score = None, np.inf
    for lam in lambdas:
        fold_mses = []
        for j in range(k):
            te_idx = folds[j]
            tr_mask = np.ones(n, dtype=bool); tr_mask[te_idx] = False
            Xtr, ytr = X[tr_mask], y[tr_mask]
            Xte, yte = X[te_idx], y[te_idx]
            if method == "KRR":
                m = KernelizedRidgeRegression(kernel_factory(), lambda_=lam).fit(Xtr, ytr)
            else:
                m = SVR(kernel_factory(), lambda_=lam, epsilon=epsilon).fit(Xtr, ytr)
            fold_mses.append(_mse(yte, m.predict(Xte)))
        score = float(np.mean(fold_mses))
        if score < best_score:
            best_lam, best_score = lam, score
    return best_lam


def part_2(out_dir="build", seed=0):
    """Apply both methods x both kernels to housing2r. Sweep the kernel
    parameter and plot test MSE vs the kernel parameter with two curves:
    λ=1 fixed, and λ chosen by 5-fold inner CV on the train set."""
    import os
    import matplotlib.pyplot as plt
    from cvxopt import solvers

    solvers.options["show_progress"] = False
    os.makedirs(out_dir, exist_ok=True)

    X, y = _load_housing()
    X_tr_raw, y_tr, X_te_raw, y_te = _train_test_split(X, y, seed=seed)

    # Standardize using train statistics only (features have wildly
    # different scales: std ranges from 0.7 to 166).
    mu, sd = X_tr_raw.mean(axis=0), X_tr_raw.std(axis=0)
    X_tr = (X_tr_raw - mu) / sd
    X_te = (X_te_raw - mu) / sd

    # Set ε from a quick noise estimate: residual std of a moderately
    # regularized KRR-RBF fit on the train set. Gives a tube that lets
    # SVR be sparse without losing signal.
    probe = KernelizedRidgeRegression(RBF(sigma=2.0), lambda_=1.0).fit(X_tr, y_tr)
    eps_noise = float(np.std(y_tr - probe.predict(X_tr)))
    epsilon = round(eps_noise, 1)  # e.g., 2.7 -> 2.7
    print(f"Estimated noise std: {eps_noise:.3f}  ->  using ε = {epsilon}")

    poly_Ms = list(range(1, 11))
    rbf_sigmas = [0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
    lambdas = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]

    results = {}
    for method in ["KRR", "SVR"]:
        for kname, params, factory_fn in [
            ("Polynomial", poly_Ms,    lambda p: Polynomial(M=p)),
            ("RBF",        rbf_sigmas, lambda p: RBF(sigma=p)),
        ]:
            r = {"params": params, "mse_l1": [], "mse_cv": [],
                 "nsv_l1": [], "nsv_cv": [], "lam_cv": []}
            for p in params:
                kf = (lambda pp=p, ff=factory_fn: ff(pp))

                # λ = 1 (fixed)
                if method == "KRR":
                    m1 = KernelizedRidgeRegression(kf(), lambda_=1.0).fit(X_tr, y_tr)
                else:
                    m1 = SVR(kf(), lambda_=1.0, epsilon=epsilon).fit(X_tr, y_tr)
                r["mse_l1"].append(_mse(y_te, m1.predict(X_te)))
                if method == "SVR":
                    r["nsv_l1"].append(int(np.sum(np.abs(m1.alpha - m1.alpha_star) > 1e-5)))

                # λ from inner 5-fold CV on the train set
                lam_cv = _cv_pick_lambda(X_tr, y_tr, kf, method,
                                         lambdas=lambdas, epsilon=epsilon, seed=seed)
                r["lam_cv"].append(lam_cv)
                if method == "KRR":
                    m2 = KernelizedRidgeRegression(kf(), lambda_=lam_cv).fit(X_tr, y_tr)
                else:
                    m2 = SVR(kf(), lambda_=lam_cv, epsilon=epsilon).fit(X_tr, y_tr)
                r["mse_cv"].append(_mse(y_te, m2.predict(X_te)))
                if method == "SVR":
                    r["nsv_cv"].append(int(np.sum(np.abs(m2.alpha - m2.alpha_star) > 1e-5)))
            results[(method, kname)] = r

    # Plots: one per (method, kernel). For SVR, dual axis with #SV.
    for (method, kname), r in results.items():
        fig, ax = plt.subplots(figsize=(6, 4))
        params = r["params"]
        ax.plot(params, r["mse_l1"], "o-", color="#4477AA", lw=1.6,
                label="λ = 1")
        ax.plot(params, r["mse_cv"], "s-", color="#CC3311", lw=1.6,
                label="λ via 5-fold CV")
        ax.set_xlabel("M" if kname == "Polynomial" else "σ")
        ax.set_ylabel("test MSE")
        if kname == "RBF":
            ax.set_xscale("log")
        if kname == "Polynomial":
            ax.set_yscale("log")
        title = f"{method} + {kname}"
        if method == "SVR":
            title += f"   (ε = {epsilon})"
        ax.set_title(title)
        ax.grid(alpha=0.3)

        if method == "SVR":
            ax2 = ax.twinx()
            ax2.plot(params, r["nsv_l1"], "o--", color="#4477AA",
                     lw=1.0, alpha=0.5, markersize=4)
            ax2.plot(params, r["nsv_cv"], "s--", color="#CC3311",
                     lw=1.0, alpha=0.5, markersize=4)
            ax2.set_ylabel("#SV (dashed)", color="gray")
            ax2.tick_params(axis="y", colors="gray")
            ax2.set_ylim(bottom=0)

        ax.legend(loc="best", fontsize=9, framealpha=0.9)
        fig.tight_layout()
        fname = f"{out_dir}/housing_{method.lower()}_{kname.lower()}.pdf"
        fig.savefig(fname)
        plt.close(fig)
        print(f"Saved {fname}")

    # Console summary
    print("\nSummary (test MSE per kernel parameter):")
    for (method, kname), r in results.items():
        head = f"{method} + {kname}"
        print(f"\n{head}")
        print(f"  {'param':>6} {'MSE(λ=1)':>10} {'MSE(λcv)':>10} "
              f"{'λcv':>8} " + ("  #SV(λ=1) #SV(λcv)" if method == "SVR" else ""))
        for i, p in enumerate(r["params"]):
            line = (f"  {p:>6.2f} {r['mse_l1'][i]:>10.3f} "
                    f"{r['mse_cv'][i]:>10.3f} {r['lam_cv'][i]:>8.0e}")
            if method == "SVR":
                line += f"   {r['nsv_l1'][i]:>5}    {r['nsv_cv'][i]:>5}"
            print(line)

    return results


class SubsequenceKernel:
    """String Subsequence Kernel (SSK) of Lodhi et al. (2002).

    Definition for length p and decay lam in (0, 1]:

        phi_u(s)   = sum over index tuples i with u = s[i] of lam^l(i)
        K_p(s, t)  = sum over u in Sigma^p of phi_u(s) * phi_u(t)

    where i = (i_1 < i_2 < ... < i_p) is a strictly increasing index
    tuple and l(i) = i_p - i_1 + 1 is the *span* of the tuple in the
    string.  Each matching subsequence is weighted by lam to the power
    of its gap span: contiguous matches get the smallest exponent
    (largest weight), spread-out matches a larger exponent.

    Computed by a DP in O(p * |s| * |t|) per pair via two auxiliary
    tables (K_prime and K_pp).  The implicit Sigma^p feature map is
    never materialized.

    Inputs may be a Python list of strings (variable length OK) or an
    already-encoded 2-D integer array (in which case all rows are
    assumed to have the same true length).  Variable-length lists are
    padded internally with -1 sentinels which never match a real
    character; the per-row true lengths are tracked separately.

    `normalize=True` returns K(s, t) / sqrt(K(s, s) * K(t, t)), the
    cosine in feature space; this makes K(s, s) = 1 for every string
    and is the recommended setting for variable-length inputs.
    """

    def __init__(self, p=3, lam=0.5, normalize=True):
        self.p = p
        self.lam = lam
        self.normalize = normalize

    def __call__(self, A, B):
        # Encode both inputs to padded integer matrices with true lengths.
        S_a, lens_a = self._encode(A)
        S_b, lens_b = self._encode(B)
        K_raw = np.zeros((len(S_a), len(S_b)))
        # Loop over rows of A; for each row run a vectorized one-to-many
        # DP against all rows of B simultaneously.
        for a in range(len(S_a)):
            K_raw[a] = self._one_to_many(S_a[a], int(lens_a[a]), S_b)
        if not self.normalize:
            return K_raw
        # Cosine normalization needs K(s, s) for every row of A and B.
        diag_a = np.array([
            self._one_to_many(S_a[a], int(lens_a[a]), S_a[a:a + 1])[0]
            for a in range(len(S_a))
        ])
        diag_b = np.array([
            self._one_to_many(S_b[b], int(lens_b[b]), S_b[b:b + 1])[0]
            for b in range(len(S_b))
        ])
        denom = np.sqrt(np.outer(diag_a, diag_b))
        # Guard against division by zero when a string is too short
        # for the kernel to have any matching subsequence at all.
        return K_raw / np.where(denom > 0, denom, 1.0)

    @staticmethod
    def _encode(strings):
        """List of strings -> (S, lengths). S has shape (n, maxlen) with
        -1 padding in trailing positions; lengths is an int array of
        true per-row lengths. If `strings` is already a 2-D ndarray it
        is assumed to be already encoded with a uniform length.
        """
        if isinstance(strings, np.ndarray):
            return strings, np.full(len(strings), strings.shape[1])
        # Stable lexicographic alphabet so the encoding is deterministic
        # across calls (useful for hash-based comparisons in tests).
        alphabet = sorted({c for s in strings for c in s})
        c2i = {c: i for i, c in enumerate(alphabet)}
        maxlen = max(len(s) for s in strings)
        S = np.full((len(strings), maxlen), -1, dtype=np.int64)
        for i, s in enumerate(strings):
            for j, c in enumerate(s):
                S[i, j] = c2i[c]
        return S, np.array([len(s) for s in strings])

    def _one_to_many(self, s, Ls, T):
        """Compute K_p(s[:Ls], T[b, :true_len(b)]) for each row b of T.

        Padding handling: T may contain -1 in trailing positions for
        rows shorter than T.shape[1]; since s[i-1] is always
        non-negative for i in [1, Ls], `T == s[i-1]` evaluates to
        False at every padded cell. We therefore only need to clip the
        i-loop to Ls (true length of s); the j dimension self-clips
        through the no-match-at-padding mechanism.
        """
        nT, Lt = T.shape
        # Short-circuit when either argument is shorter than p: by
        # definition no length-p subsequence can be drawn.
        if Ls < self.p or Lt < self.p:
            return np.zeros(nT)
        lam, p = self.lam, self.p

        # K_prev[b, i, j] holds K'_l(s[:i], T[b, :j]) for the current l.
        # Start at l = 0 where K'_0 = 1 (the empty subsequence is the
        # unique length-0 subsequence and contributes weight 1).
        # For l >= 1 the boundary K'_l[0, *] = K'_l[*, 0] = 0 will
        # appear automatically because K_curr is zero-initialized and
        # never updated at i = 0 or j = 0.
        K_prev = np.ones((nT, Ls + 1, Lt + 1))

        # Build K'_l for l = 1, 2, ..., p - 1.
        for _l in range(1, p):
            # K_pp (Lodhi's K'') is an auxiliary that gives the
            # contribution of subsequences whose last index is exactly j
            # in t. The recurrence on j makes the inner-j-loop
            # vectorizable across all rows b at once.
            K_pp = np.zeros((nT, Ls + 1, Lt + 1))
            # K_curr will hold K'_l after this iteration.
            K_curr = np.zeros((nT, Ls + 1, Lt + 1))
            for i in range(1, Ls + 1):
                # match[b, j-1] = 1 if T[b, j-1] equals the character we
                # are adding to s, i.e. s[i-1]. Real positions in T are
                # non-negative; padded positions are -1, so they never
                # match s[i-1] (which is real for i <= Ls).
                match = (T == s[i - 1]).astype(np.float64)  # (nT, Lt)
                # K_pp recurrence on j (data dependency forces a loop).
                #   K_pp[b, i, j] = lam * K_pp[b, i, j-1]
                #                 + match * lam^2 * K_prev[b, i-1, j-1]
                # The lam^2 weights: one lam for the new (i, j) match
                # pair and one for "linking" to the previous-level DP.
                for j in range(1, Lt + 1):
                    K_pp[:, i, j] = (
                        lam * K_pp[:, i, j - 1]
                        + match[:, j - 1] * (lam ** 2)
                        * K_prev[:, i - 1, j - 1]
                    )
                # K_curr recurrence on i (vectorized across all j):
                #   K_curr[b, i, j] = lam * K_curr[b, i-1, j] + K_pp[b, i, j]
                K_curr[:, i, :] = lam * K_curr[:, i - 1, :] + K_pp[:, i, :]
            K_prev = K_curr

        # Final K_p assembly. By the Lodhi recurrence,
        #   K_p(s, t) = sum over (i, j) with s[i-1] = t[j-1] of
        #               lam^2 * K'_{p-1}(s[:i-1], t[:j-1]).
        # This is the contribution of every pair of positions where
        # we can "anchor" the last character of a matched subsequence.
        K_total = np.zeros(nT)
        for i in range(1, Ls + 1):
            match = (T == s[i - 1]).astype(np.float64)
            K_total += (lam ** 2) * np.sum(
                match * K_prev[:, i - 1, :Lt], axis=1
            )
        return K_total


def _gen_synthetic_strings(n=300, length=25, seed=0, noise_std=0.3):
    """Generate strings over {A,C,G,T} with a target that depends on
    *gapped* subsequence-pattern counts.  A contiguous-trigram count
    table can never recover this target because span-3..8 occurrences
    with span > 3 are non-contiguous and invisible to any k=3 count.
    """
    rng = np.random.RandomState(seed)
    alphabet = ["A", "C", "G", "T"]
    strings = ["".join(rng.choice(alphabet, length)) for _ in range(n)]

    def gapped_count(s, pattern, min_span, max_span):
        c = 0
        for i in range(len(s)):
            if s[i] != pattern[0]:
                continue
            for j in range(i + 1, len(s)):
                if s[j] != pattern[1]:
                    continue
                for k in range(j + 1, len(s)):
                    if s[k] != pattern[2]:
                        continue
                    span = k - i + 1
                    if min_span <= span <= max_span:
                        c += 1
        return c

    y_clean = np.array(
        [1.0 * gapped_count(s, "ACT", 3, 8)
         - 0.5 * gapped_count(s, "GTA", 3, 8)
         for s in strings],
        dtype=float,
    )
    y = y_clean + rng.normal(0, noise_std, n)
    return strings, y, y_clean


def _bag_of_kmers(strings, k=3, alphabet="ACGT"):
    """Naive attribute table: count of each contiguous k-mer per string.

    Returns an (n_strings, |alphabet|^k) integer-valued matrix.  This
    is the standard "flatten text to a fixed schema" baseline; it
    captures only contiguous patterns of exactly length k.
    """
    from itertools import product
    kmers = ["".join(c) for c in product(alphabet, repeat=k)]
    idx = {km: i for i, km in enumerate(kmers)}
    X = np.zeros((len(strings), len(kmers)))
    for i, s in enumerate(strings):
        for j in range(len(s) - k + 1):
            X[i, idx[s[j:j + k]]] += 1.0
    return X


def _bag_of_kmers_multi(strings, ks=(2, 3, 4), alphabet="ACGT"):
    """Stronger baseline: horizontally concatenate bag-of-k-mers over
    several k values.  Captures contiguous patterns at multiple scales,
    still strictly a finite attribute table.  Output dimension is
    sum_k |alphabet|^k.
    """
    return np.hstack([_bag_of_kmers(strings, k=k, alphabet=alphabet) for k in ks])


def _verify_ssk():
    """Module-load sanity check: K_2('abc', 'abc') with lam=1 should be 3
    (subsequences ab, bc, ac all match with weight 1).  Catches gross
    regressions in the DP without needing the full test suite."""
    ssk = SubsequenceKernel(p=2, lam=1.0, normalize=False)
    K = ssk(["abc", "abc"], ["abc", "abc"])
    assert abs(K[0, 1] - 3.0) < 1e-9, (
        f"SSK self-check failed: got {K[0, 1]}, expected 3.0"
    )


_verify_ssk()


def _cv_pick_lambda_from_gram(K, y, lambdas, k=5, seed=0):
    """Inner k-fold CV that picks the lambda minimizing mean fold MSE.

    Uses ridge fits computed by index-slicing the precomputed Gram
    rather than re-evaluating the kernel; this lets us re-use one
    expensive Gram across many lambda candidates.  Returns the chosen
    lambda.
    """
    rng = np.random.RandomState(seed)
    n = len(y)
    perm = rng.permutation(n)
    folds = np.array_split(perm, k)
    best_lam, best_score = None, np.inf
    for lam in lambdas:
        mses = []
        for j in range(k):
            te = folds[j]
            tr_mask = np.ones(n, dtype=bool); tr_mask[te] = False
            K_tr = K[np.ix_(tr_mask, tr_mask)]
            K_te = K[np.ix_(te, tr_mask)]
            alpha = np.linalg.solve(
                K_tr + lam * np.eye(K_tr.shape[0]), y[tr_mask]
            )
            pred = K_te @ alpha
            mses.append(np.mean((pred - y[te]) ** 2))
        score = float(np.mean(mses))
        if score < best_score:
            best_lam, best_score = lam, score
    return best_lam


def _fit_predict_from_gram(K_tr_tr, K_te_tr, y_tr, lam):
    """Solve the KRR closed form using a precomputed Gram, then predict
    on the held-out rows.  Returns (alpha, predictions)."""
    alpha = np.linalg.solve(
        K_tr_tr + lam * np.eye(K_tr_tr.shape[0]), y_tr
    )
    return alpha, K_te_tr @ alpha


def part_3(out_dir="build", seed=0):
    """Compare a non-trivial structured-data kernel (SSK) against the
    strongest natural attribute tables one would write down for
    strings.  Three layers of comparison:

      1. Constant baseline (predict y_train mean).
      2. Naive table baselines, in order of increasing flexibility:
         (a) bag-of-3-mers + linear kernel,
         (b) bag-of-k-mers concatenated for k in {2, 3, 4} + linear,
         (c) bag-of-3-mers + polynomial(M=2) kernel
             (implicit pairwise interactions of contiguous 3-mers).
      3. SSK on raw strings, swept over (p, lambda_decay) grid.

    The target is constructed so it lives in the span of SSK's feature
    map (gap-weighted subsequence counts) but is *not* fully expressible
    as any function of contiguous-k-mer counts.
    """
    import os
    import time
    import matplotlib.pyplot as plt
    from cvxopt import solvers
    solvers.options["show_progress"] = False
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Data
    # ------------------------------------------------------------------
    print("Generating synthetic data...")
    strings, y, y_clean = _gen_synthetic_strings(n=300, length=25, seed=seed)
    print(f"  n = {len(strings)}, len = {len(strings[0])}")
    print(f"  std(y_clean) = {y_clean.std():.3f}, "
          f"std(noise) = {(y - y_clean).std():.3f}, std(y) = {y.std():.3f}")

    # 80/20 train/test split.
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(strings))
    n_test = int(0.2 * len(strings))
    idx_te, idx_tr = perm[:n_test], perm[n_test:]
    s_tr = [strings[i] for i in idx_tr]
    s_te = [strings[i] for i in idx_te]
    y_tr, y_te = y[idx_tr], y[idx_te]

    # Pre-encode for SSK so the integer matrix is built once and
    # the kernel can reuse it across multiple (p, lambda_decay) settings.
    alphabet = sorted({c for s in strings for c in s})
    c2i = {c: i for i, c in enumerate(alphabet)}
    S_tr = np.array([[c2i[c] for c in s] for s in s_tr])
    S_te = np.array([[c2i[c] for c in s] for s in s_te])

    lambdas = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
    summary = []  # accumulates rows for the comparison bar chart

    # ------------------------------------------------------------------
    # 2. Baselines
    # ------------------------------------------------------------------
    # 2a. Constant predictor.
    mse_const = _mse(y_te, np.full_like(y_te, y_tr.mean()))
    summary.append({"name": "constant", "mse": mse_const, "lam": None})
    print(f"\n[constant] test MSE = {mse_const:.3f}")

    # 2b. Bag-of-3-mers + linear kernel (the standard naive baseline).
    X_tr_3 = _bag_of_kmers(s_tr, k=3)
    X_te_3 = _bag_of_kmers(s_te, k=3)
    K_tr_tr = X_tr_3 @ X_tr_3.T          # linear kernel = inner product
    K_te_tr = X_te_3 @ X_tr_3.T
    lam_cv = _cv_pick_lambda_from_gram(K_tr_tr, y_tr, lambdas)
    _, pred = _fit_predict_from_gram(K_tr_tr, K_te_tr, y_tr, lam_cv)
    mse = _mse(y_te, pred)
    summary.append({"name": "BoW k=3 linear", "mse": mse, "lam": lam_cv})
    print(f"[BoW k=3 + linear]              lam* = {lam_cv:.0e}, "
          f"MSE = {mse:.3f}")

    # 2c. Concatenated bag-of-k-mers for k in {2, 3, 4} + linear kernel.
    # Strictly larger feature space than (2b); still strictly contiguous.
    X_tr_multi = _bag_of_kmers_multi(s_tr, ks=(2, 3, 4))
    X_te_multi = _bag_of_kmers_multi(s_te, ks=(2, 3, 4))
    K_tr_tr = X_tr_multi @ X_tr_multi.T
    K_te_tr = X_te_multi @ X_tr_multi.T
    lam_cv = _cv_pick_lambda_from_gram(K_tr_tr, y_tr, lambdas)
    _, pred = _fit_predict_from_gram(K_tr_tr, K_te_tr, y_tr, lam_cv)
    mse = _mse(y_te, pred)
    summary.append({"name": "BoW k=2,3,4 lin.", "mse": mse, "lam": lam_cv})
    print(f"[BoW k=2,3,4 + linear]          lam* = {lam_cv:.0e}, "
          f"MSE = {mse:.3f}")

    # 2d. Bag-of-3-mers + polynomial(M=2) kernel.  Adds implicit
    # pairwise interactions between 3-mer counts, still strictly a
    # function of the count table.
    poly2 = Polynomial(M=2)
    K_tr_tr = poly2(X_tr_3, X_tr_3)
    K_te_tr = poly2(X_te_3, X_tr_3)
    lam_cv = _cv_pick_lambda_from_gram(K_tr_tr, y_tr, lambdas)
    _, pred = _fit_predict_from_gram(K_tr_tr, K_te_tr, y_tr, lam_cv)
    mse = _mse(y_te, pred)
    summary.append({"name": "BoW k=3 poly M=2", "mse": mse, "lam": lam_cv})
    print(f"[BoW k=3 + Poly(M=2)]           lam* = {lam_cv:.0e}, "
          f"MSE = {mse:.3f}")

    # ------------------------------------------------------------------
    # 3. SSK sweep over (p, lambda_decay)
    # ------------------------------------------------------------------
    p_values = [2, 3, 4, 5]
    decay_values = [0.3, 0.5, 0.7]
    ssk_grid = np.full((len(p_values), len(decay_values)), np.nan)
    ssk_best = {"mse": np.inf}

    print("\nSSK sweep over (p, lambda_decay):")
    for ip, p in enumerate(p_values):
        for id_, decay in enumerate(decay_values):
            t0 = time.time()
            ssk = SubsequenceKernel(p=p, lam=decay, normalize=True)
            K_tr_tr = ssk(S_tr, S_tr)
            K_te_tr = ssk(S_te, S_tr)
            lam_cv = _cv_pick_lambda_from_gram(K_tr_tr, y_tr, lambdas)
            _, pred = _fit_predict_from_gram(K_tr_tr, K_te_tr, y_tr, lam_cv)
            mse = _mse(y_te, pred)
            ssk_grid[ip, id_] = mse
            took = time.time() - t0
            print(f"  p={p}, lambda_decay={decay:.1f}: "
                  f"MSE = {mse:>6.3f}, lam* = {lam_cv:.0e}, took {took:.1f}s")
            if mse < ssk_best["mse"]:
                ssk_best = {
                    "p": p, "decay": decay, "lam": lam_cv, "mse": mse,
                    "pred": pred,
                }

    summary.append({
        "name": f"SSK best\n(p={ssk_best['p']},lam={ssk_best['decay']})",
        "mse": ssk_best["mse"], "lam": ssk_best["lam"],
    })
    print(f"\nSSK best: p = {ssk_best['p']}, decay = {ssk_best['decay']}, "
          f"lam* = {ssk_best['lam']:.0e}, MSE = {ssk_best['mse']:.3f}")

    # Theoretical irreducible MSE: variance of the noise we injected
    # (this is the asymptotic best any regressor could achieve).
    noise_floor = 0.3 ** 2  # noise_std = 0.3
    print(f"Noise floor (sigma^2): {noise_floor:.3f}")

    # ------------------------------------------------------------------
    # 4. Plot: heatmap (left) + bar chart (right)
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    # Heatmap of SSK test MSE over (p, lambda_decay), darker = better.
    im = ax1.imshow(ssk_grid, aspect="auto", cmap="viridis_r")
    ax1.set_xticks(range(len(decay_values)))
    ax1.set_xticklabels(decay_values)
    ax1.set_yticks(range(len(p_values)))
    ax1.set_yticklabels(p_values)
    ax1.set_xlabel(r"SSK decay $\lambda_{\mathrm{decay}}$")
    ax1.set_ylabel("SSK length $p$")
    ax1.set_title("SSK test MSE over (p, decay)")
    # Annotate each cell; switch text colour so it stays legible.
    vmid = (np.nanmin(ssk_grid) + np.nanmax(ssk_grid)) / 2
    for ip in range(len(p_values)):
        for id_ in range(len(decay_values)):
            v = ssk_grid[ip, id_]
            ax1.text(id_, ip, f"{v:.2f}", ha="center", va="center",
                     color="white" if v > vmid else "black", fontsize=10)
    fig.colorbar(im, ax=ax1, label="test MSE")

    # Bar chart of all method MSEs side by side.
    names = [r["name"] for r in summary]
    mses = [r["mse"] for r in summary]
    colors = ["#888888", "#4477AA", "#6699CC", "#5588BB", "#CC3311"]
    bars = ax2.bar(range(len(names)), mses, color=colors[:len(names)])
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
    ax2.axhline(noise_floor, color="black", lw=1, ls="--",
                label=fr"noise floor = $\sigma^2$ = {noise_floor:.2f}")
    ax2.set_ylabel("test MSE")
    ax2.set_title("All methods")
    ax2.legend(loc="upper right", fontsize=9)
    for b, v in zip(bars, mses):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.2f}",
                 ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fname = f"{out_dir}/strings_ssk_vs_bow.pdf"
    fig.savefig(fname)
    plt.close(fig)
    print(f"\nSaved {fname}")

    return {
        "summary": summary,
        "ssk_grid": ssk_grid,
        "p_values": p_values,
        "decay_values": decay_values,
        "ssk_best": ssk_best,
        "noise_floor": noise_floor,
    }


if __name__ == "__main__":
    part_1()
    part_2()
    part_3()
