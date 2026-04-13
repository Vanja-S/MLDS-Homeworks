import numpy as np
from scipy.optimize import fmin_l_bfgs_b


class MultinomialLogReg:
    def __init__(self):
        self.W: np.ndarray

    def build(self, X, y):
        n, p = X.shape
        K = len(np.unique(y))
        X1 = np.column_stack([np.ones(n), X])
        p1 = X1.shape[1]

        def neg_log_likelihood(w_flat):
            W = w_flat.reshape(p1, K)
            scores = X1 @ W
            # stable softmax
            scores -= scores.max(axis=1, keepdims=True)
            exp_s = np.exp(scores)
            probs = exp_s / exp_s.sum(axis=1, keepdims=True)
            # log-likelihood of correct classes
            nll = -np.sum(np.log(probs[np.arange(n), y] + 1e-15))
            return nll

        w0 = np.zeros(p1 * K)
        w_opt, _, _ = fmin_l_bfgs_b(neg_log_likelihood, w0, approx_grad=True)
        self.W = w_opt.reshape(p1, K)
        return self

    def predict(self, X):
        n = X.shape[0]
        X1 = np.column_stack([np.ones(n), X])
        scores = X1 @ self.W
        scores -= scores.max(axis=1, keepdims=True)
        exp_s = np.exp(scores)
        return exp_s / exp_s.sum(axis=1, keepdims=True)


class OrdinalLogReg:
    def __init__(self):
        self.beta: np.ndarray
        self.alpha: np.ndarray

    def build(self, X, y):
        n, p = X.shape
        K = len(np.unique(y))

        def neg_log_likelihood(params):
            alpha = params[: K - 1]
            beta = params[K - 1 :]
            scores = X @ beta
            # cumulative probs: P(y <= k) = sigmoid(alpha_k - score)
            cum = 1.0 / (1.0 + np.exp(-(alpha[np.newaxis, :] - scores[:, np.newaxis])))
            cum_full = np.hstack([np.zeros((n, 1)), cum, np.ones((n, 1))])
            probs = np.diff(cum_full, axis=1)
            probs = np.clip(probs, 1e-15, None)
            return -np.sum(np.log(probs[np.arange(n), y]))

        params0 = np.zeros(K - 1 + p)
        params0[: K - 1] = np.linspace(-1, 1, K - 1)
        params_opt, _, _ = fmin_l_bfgs_b(neg_log_likelihood, params0, approx_grad=True)
        self.alpha = params_opt[: K - 1]
        self.beta = params_opt[K - 1 :]
        return self

    def predict(self, X):
        n = X.shape[0]
        K = len(self.alpha) + 1
        scores = X @ self.beta
        cum = 1.0 / (1.0 + np.exp(-(self.alpha[np.newaxis, :] - scores[:, np.newaxis])))
        cum_full = np.hstack([np.zeros((n, 1)), cum, np.ones((n, 1))])
        probs = np.diff(cum_full, axis=1)
        return np.clip(probs, 0, None)
