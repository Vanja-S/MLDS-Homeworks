import numpy as np

from typing import Any

import math


class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = data
        # gradient accumulates during backward pass, starts at 0
        self.grad = 0.0
        # no-op by default, overridden by operations that produce this node
        self._backward = lambda: None
        # the nodes that were inputs to the operation that created this node
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value({self.data})"

    # --- core arithmetic ops, each builds a graph edge + local _backward ---

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # d(a+b)/da = 1, d(a+b)/db = 1
            # so both parents just receive the upstream gradient as-is
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # d(a*b)/da = b, d(a*b)/db = a
            # each parent gets the other parent's value times upstream grad
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __pow__(self, other):
        # other is a plain int/float, not a Value (we only need constant powers)
        out = Value(self.data**other, (self,), f"**{other}")

        def _backward():
            # d(x^n)/dx = n * x^(n-1)
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward

        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward():
            # d(e^x)/dx = e^x = out.data
            self.grad += out.data * out.grad

        out._backward = _backward

        return out

    def log(self):
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            # d(ln(x))/dx = 1/x
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward

        return out

    # --- convenience ops built from the primitives above ---

    def __neg__(self):  # -self
        return self * -1

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __truediv__(self, other):  # self / other
        return self * other**-1

    # --- reverse ops so that (int/float op Value) works ---

    def __radd__(self, other):  # other + self
        return self + other

    def __rmul__(self, other):  # other * self
        return self * other

    def __rsub__(self, other):  # other - self
        return (-self) + other

    def __rtruediv__(self, other):  # other / self
        return other * self**-1

    # --- backpropagation ---

    def backward(self):
        # 1. topological sort: order nodes so every node comes after its children
        topo = []
        visited = set()

        def build_topo(v):
            # zero out grad for every node before the backward pass
            v.grad = 0.0
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # 2. seed: dL/dL = 1
        self.grad = 1.0

        # 3. walk in reverse topological order (output -> inputs),
        #    each node pushes its gradient contribution to its children
        for v in reversed(topo):
            v._backward()


class MultinomialLogReg:
    def __init__(self) -> None:
        # The weights matrix W is basically
        # beta vectors for each category
        # concatinated row wise
        self.W: np.ndarray

        self.epochs = 1000
        self.lr = 0.5

    # y can be a one-hot encoding vector
    # or a vector of nominal classes
    def build(self, X: np.ndarray, y: np.ndarray):
        n = X.shape[0]
        # We prepend a column of 1s to get X1 with shape (n, p+1),
        # that extra column is the bias term, so each score
        # becomes 1*bias_k + x1*w1k + x2*w2k without needing
        # a separate bias parameter.
        X1 = np.column_stack([np.ones(n), X])
        p = X1.shape[1]

        K = len(np.unique(y))

        self.W = np.zeros((p, K))

        for epoch in range(self.epochs):
            # wrap each weight as a Value node so autograd can track gradients
            W_val = [[Value(self.W[j][k]) for k in range(K)] for j in range(p)]

            # accumulator for the log-likelihood across all samples
            nll = Value(0.0)

            for i in range(n):
                # compute a score for each class k:
                # score_k = x1*w1k + x2*w2k + ... (including bias via the prepended 1)
                scores = []
                for k in range(K):
                    s = Value(0.0)
                    for j in range(p):
                        s = s + X1[i][j] * W_val[j][k]
                    scores.append(s)

                # softmax numerical stability: subtract the max score (plain float)
                # before exp so we don't overflow. doesn't change the probabilities
                max_s = max(s.data for s in scores)
                exp_scores = [(s - max_s).exp() for s in scores]

                # sum of exponentials start from first element,
                # avoid python sum() which starts with int 0
                total = exp_scores[0]
                for k in range(1, K):
                    total = total + exp_scores[k]

                # softmax probabilities: P(y=k|x_i) = exp(score_k) / total
                probs = [e / total for e in exp_scores]

                # add log P(y=y_i | x_i) for the correct class
                nll = nll + probs[y[i]].log()

            # we maximise log-likelihood, so we minimise its negation
            loss = nll * -1
            loss.backward()

            # gradient descent update
            for j in range(p):
                for k in range(K):
                    self.W[j][k] -= self.lr * W_val[j][k].grad

        return self

    def predict(self, X: np.ndarray):
        n = X.shape[0]
        X1 = np.column_stack([np.ones(n), X])

        results = X1 @ self.W

        # stable softmax: subtract row max to avoid overflow
        results -= results.max(axis=1, keepdims=True)
        results = np.exp(results)
        results /= results.sum(axis=1, keepdims=True)

        return results


class OrdinalLogReg:
    def __init__(self) -> None:
        self.beta: np.ndarray
        self.alpha: np.ndarray

        self.epochs = 1000
        self.lr = 0.5

    def build(self, X: np.ndarray, y: np.ndarray):
        n, p = X.shape
        K = len(np.unique(y))

        # Unlike multinomial, ordinal regression has TWO types of parameters:
        #   - beta: one shared weight vector of length p (no bias column needed)
        #   - alpha: K-1 threshold (cutpoint) parameters
        #
        # In multinomial we prepended a column of 1s so that each
        # class got its own intercept baked into its weight vector. Here, the
        # K-1 alpha thresholds ARE the intercepts — one per boundary between
        # adjacent classes. Adding a bias to beta would be redundant (it would
        # be unidentifiable: you could shift all alphas down by c and beta's
        # bias up by c and get the same model).
        self.beta = np.zeros(p)
        # thresholds must start ordered (alpha_1 < alpha_2 < ...)
        # so the cumulative probs are monotonically increasing
        # and individual class probs (from differencing) stay positive
        self.alpha = np.linspace(-1, 1, K - 1)

        for epoch in range(self.epochs):
            # wrap parameters as Value nodes for autograd
            b_val = [Value(self.beta[j]) for j in range(p)]
            a_val = [Value(self.alpha[k]) for k in range(K - 1)]

            nll = Value(0.0)

            for i in range(n):
                # compute the linear score: x · beta (single scalar, shared across classes)
                score = Value(0.0)
                for j in range(p):
                    score = score + X[i][j] * b_val[j]

                # Why sigmoid on cumulative probabilities?
                # Ordinal regression models P(y <= k) — the probability of being
                # in class k OR ANY LOWER class. This is natural for ordered classes:
                # as k increases, P(y <= k) must increase too (it's cumulative).
                # The sigmoid σ(α_k - score) maps the distance between the threshold
                # and the score to a [0,1] probability, and the subtraction ensures
                # that higher scores push probability mass toward higher classes.

                # compute cumulative probs: P(y <= k) = sigmoid(alpha_k - score)
                # sigmoid(z) = 1 / (1 + exp(-z))
                cum_probs = []
                for k in range(K - 1):
                    z = a_val[k] - score
                    sig = Value(1.0) / (Value(1.0) + (-z).exp())
                    cum_probs.append(sig)

                # convert cumulative P(y<=k) into individual P(y=k) by differencing:
                #   P(y=0) = P(y<=0)
                #   P(y=k) = P(y<=k) - P(y<=k-1)   for 0 < k < K-1
                #   P(y=K-1) = 1 - P(y<=K-2)
                probs = []
                for k in range(K):
                    if k == 0:
                        pk = cum_probs[0]
                    elif k == K - 1:
                        pk = Value(1.0) - cum_probs[K - 2]
                    else:
                        pk = cum_probs[k] - cum_probs[k - 1]
                    probs.append(pk)

                nll = nll + probs[y[i]].log()

            loss = nll * -1
            loss.backward()

            # update both parameter sets
            for j in range(p):
                self.beta[j] -= self.lr * b_val[j].grad
            for k in range(K - 1):
                self.alpha[k] -= self.lr * a_val[k].grad

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        n, p = X.shape
        K = len(self.alpha) + 1

        # scores: (n,) — one scalar per sample
        scores = X @ self.beta

        # cumulative probs: P(y <= k) = sigmoid(alpha_k - score)
        # alpha is (K-1,), scores is (n,) -> broadcast to (n, K-1)
        cum_probs = 1.0 / (
            1.0 + np.exp(-(self.alpha[np.newaxis, :] - scores[:, np.newaxis]))
        )

        # convert to individual class probs by differencing
        # prepend 0 and append 1 so differencing covers all K classes:
        # [0, P(y<=0), P(y<=1), ..., P(y<=K-2), 1]
        # then P(y=k) = column[k+1] - column[k]
        cum_full = np.hstack([np.zeros((n, 1)), cum_probs, np.ones((n, 1))])
        probs = np.diff(cum_full, axis=1)

        return probs


def glm_diagnostics(x, y, save_prefix="build/diag"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(y)

    # --- Fit simple linear regression via normal equations ---
    X = np.column_stack([np.ones(n), x])
    p = X.shape[1]  # number of parameters (2: intercept + slope)

    # Hat matrix: H = X (X^T X)^{-1} X^T
    # H projects y onto the column space of X
    # h_ii (leverage) measures how far x_i is from the mean of x —
    # high leverage points have outsized influence on the fit
    XtX_inv = np.linalg.inv(X.T @ X)
    H = X @ XtX_inv @ X.T
    h = np.diag(H)

    beta = XtX_inv @ X.T @ y
    y_hat = X @ beta
    residuals = y - y_hat

    # MSE (mean squared error with degrees of freedom correction)
    mse = np.sum(residuals**2) / (n - p)

    # --- Standardized (internally studentized) residuals ---
    # Divides each residual by its own estimated std dev,
    # accounting for leverage: Var(e_i) = sigma^2 (1 - h_ii)
    # Under correct model these should be ~ N(0, 1)
    r_std = residuals / np.sqrt(mse * (1 - h))

    # --- Cook's distance ---
    # Measures the influence of observation i on ALL fitted values.
    # Combines leverage (how unusual is x_i?) with residual size
    # (how poorly does the model predict y_i?).
    # D_i = (1/p) * r_i^2 * h_ii / (1 - h_ii)
    cooks_d = (r_std**2 / p) * (h / (1 - h))

    # --- Plot 1: Normal Q-Q plot ---
    # If model is correct, standardized residuals are N(0,1).
    # Plot their sorted values against theoretical normal quantiles.
    # Deviations from the diagonal reveal non-normality (heavy tails, skew).
    from scipy.stats import norm
    sorted_r = np.sort(r_std)
    theoretical_q = norm.ppf((np.arange(1, n + 1) - 0.5) / n)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(theoretical_q, sorted_r, s=4, alpha=0.5)
    lims = [min(theoretical_q.min(), sorted_r.min()), max(theoretical_q.max(), sorted_r.max())]
    ax.plot(lims, lims, 'r--', linewidth=1)
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Standardized residuals")
    ax.set_title("Normal Q-Q Plot")
    fig.tight_layout()
    fig.savefig(f"{save_prefix}_qq.pdf")
    plt.close(fig)

    # --- Plot 2: Residuals vs fitted values ---
    # Should show random scatter around 0 with constant spread.
    # Patterns (funnel, curve) indicate heteroscedasticity or non-linearity.
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_hat, r_std, s=4, alpha=0.5)
    ax.axhline(0, color='r', linestyle='--', linewidth=1)
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("Standardized residuals")
    ax.set_title("Residuals vs Fitted")
    fig.tight_layout()
    fig.savefig(f"{save_prefix}_resid_vs_fitted.pdf")
    plt.close(fig)

    # --- Plot 3: Cook's distance ---
    # Bars for each observation. Large values indicate influential points
    # whose removal would substantially change the regression line.
    # Common threshold: D_i > 4/n or D_i > 1
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(n), cooks_d, width=1.0, color='steelblue', edgecolor='none')
    ax.axhline(4 / n, color='r', linestyle='--', linewidth=1, label=f"4/n = {4/n:.4f}")
    ax.set_xlabel("Observation index")
    ax.set_ylabel("Cook's distance")
    ax.set_title("Cook's Distance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{save_prefix}_cooks.pdf")
    plt.close(fig)

    print(f"beta = {beta}")
    print(f"R^2 = {1 - np.sum(residuals**2) / np.sum((y - y.mean())**2):.4f}")
    print(f"Influential points (Cook's D > 4/n): {(cooks_d > 4/n).sum()} / {n}")
    print(f"Residuals outside [-2,2]: {(np.abs(r_std) > 2).sum()} / {n}")
    print(f"Plots saved to {save_prefix}_*.pdf")

    return beta, r_std, cooks_d, h


def multinomial_bad_ordinal_good(n=100, seed=42):
    rng = np.random.RandomState(seed)
    x = rng.randn(n, 1)
    z = 2 * x[:, 0] + rng.randn(n) * 0.5
    y = np.where(z < -1, 0, np.where(z < 1, 1, 2))
    return x, y


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv("dataset.csv", sep=";")
    glm_diagnostics(df["Angle"].values, df["Distance"].values)
