from typing import List
import numpy as np
import csv

# torch is only required by ANNClassificationTorch and the comparison
# experiment; lazy-imported inside the class so the rest of the module
# works without it installed.
try:
    import torch
    import torch.nn as _torch_nn
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

# Activation functions
# each activation is a pair: (f(Z), f_grad(A, Z))
# A is the post-activation cache, Z the pre-activation. some grads are
# easier from A (sigmoid), some from Z (relu) — taking both keeps the
# Layer interface uniform.


def sigmoid(Z):
    return 1.0 / (1.0 + np.exp(-Z))


def sigmoid_grad(A, Z):
    return A * (1.0 - A)


def relu(Z):
    return np.maximum(0.0, Z)


def relu_grad(A, Z):
    return (Z > 0).astype(Z.dtype)


def linear(Z):
    return Z


def linear_grad(A, Z):
    return np.ones_like(Z)


def softmax(Z):
    Z_shift = Z - Z.max(axis=1, keepdims=True)
    expZ = np.exp(Z_shift)
    return expZ / expZ.sum(axis=1, keepdims=True)


def softmax_grad(A, Z):
    raise NotImplementedError(
        "softmax has no elementwise gradient; use backward_from_dZ(P - Y)"
    )


SIGMOID = (sigmoid, sigmoid_grad)
RELU = (relu, relu_grad)
LINEAR = (linear, linear_grad)
SOFTMAX = (softmax, softmax_grad)

_ACTIVATIONS = {"sigmoid": SIGMOID, "relu": RELU, "linear": LINEAR}


def _resolve_activation(a):
    # accept a string ("sigmoid" / "relu" / "linear") or an (f, fp) tuple
    if isinstance(a, str):
        return _ACTIVATIONS[a.lower()]
    return a


class ANNLayer:
    def __init__(self, d_in, d_out, activation) -> None:
        # Xavier-ish init for sigmoid; small enough to not saturate
        self.W = np.random.randn(d_in, d_out) * np.sqrt(1.0 / d_in)
        self.b = np.zeros(d_out)
        self.activation, self.activation_grad = activation

        # filled in by forward(), consumed by backward()
        self.A_prev = None
        self.Z = None
        self.A = None
        # filled in by backward()
        self.dW: np.typing.NDArray = np.empty((d_in, d_out))
        self.db: np.typing.NDArray = np.empty(d_out)

    def forward(self, A_prev):
        self.A_prev = A_prev
        self.Z = A_prev @ self.W + self.b
        self.A = self.activation(self.Z)
        return self.A

    def backward(self, dA):
        # dZ = dA ⊙ f'(Z)
        dZ = dA * self.activation_grad(self.A, self.Z)

        # dL/dW = A_prevᵀ @ dZ           shape (d_in, d_out)
        # dL/db = sum over batch of dZ   shape (d_out,)
        self.dW = self.A_prev.T @ dZ
        self.db = dZ.sum(axis=0)

        # dL/dA_prev = dZ @ Wᵀ           shape (n, d_in)
        dA_prev = dZ @ self.W.T
        return dA_prev

    def backward_from_dZ(self, dZ):
        # escape hatch for the softmax+cross-entropy output layer
        self.dW = self.A_prev.T @ dZ
        self.db = dZ.sum(axis=0)
        return dZ @ self.W.T


class ANNClassification:
    def __init__(self, units, lambda_=0.0, activations=None) -> None:
        self.units = list(units)
        self.lambda_ = lambda_
        self.layers: List[ANNLayer] = []
        self.epochs = 5000
        self.lr = 0.5
        # ablation knobs (defaults match the production behavior)
        self.standardize = True
        self.mean_grad = True
        self.verbose = True
        # one activation per hidden layer; defaults to sigmoid for all.
        # accepts either ("sigmoid", "relu", ...) or the (f, fp) tuples.
        if activations is None:
            activations = ["sigmoid"] * len(self.units)
        if len(activations) != len(self.units):
            raise ValueError(
                f"expected {len(self.units)} activations, got {len(activations)}"
            )
        self.activations = [_resolve_activation(a) for a in activations]

    def _build_layers(self, d_in, d_out):
        self.layers = []
        prev = d_in
        for h, act in zip(self.units, self.activations):
            self.layers.append(ANNLayer(prev, h, act))
            prev = h
        # output is multinomial logistic (softmax + cross-entropy)
        self.layers.append(ANNLayer(prev, d_out, SOFTMAX))

    def fit(self, X: np.typing.NDArray, y: np.typing.NDArray):
        K = len(np.unique(y))
        Y_onehot = np.eye(K)[y]

        if self.standardize:
            self._x_mean = X.mean(axis=0)
            self._x_std = X.std(axis=0)
            self._x_std = np.where(self._x_std < 1e-12, 1.0, self._x_std)
        else:
            self._x_mean = np.zeros(X.shape[1])
            self._x_std = np.ones(X.shape[1])
        X = (X - self._x_mean) / self._x_std

        self._build_layers(X.shape[1], K)

        n = X.shape[0]
        self.loss_history = []
        log_every = max(1, self.epochs // 10)
        grad_denom = n if self.mean_grad else 1.0

        for epoch in range(self.epochs):
            A = X
            # forward part
            for layer in self.layers:
                A = layer.forward(A)

            # cross-entropy loss for monitoring; +1e-12 guards log(0)
            loss = -np.log(A[np.arange(n), y] + 1e-12).mean()
            self.loss_history.append(loss)
            if self.verbose and (epoch % log_every == 0 or epoch == self.epochs - 1):
                print(f"epoch {epoch:5d}  loss={loss:.6f}")

            dZ = (A - Y_onehot) / grad_denom
            dA = self.layers[-1].backward_from_dZ(dZ)

            # backwards part
            for layer in reversed(self.layers[:-1]):
                dA = layer.backward(dA)

            # weights and bias update; L2 reg (lambda_ * W) on weights only
            for layer in self.layers:
                layer.W -= self.lr * (layer.dW + self.lambda_ * layer.W)
                layer.b -= self.lr * layer.db

        return self

    def predict(self, X: np.typing.NDArray):
        # apply the same standardization we trained with
        forward = (X - self._x_mean) / self._x_std
        for layer in self.layers:
            forward = layer.forward(forward)
        return forward

    def weights(self):
        # one (d_in+1, d_out) matrix per layer, with bias as the first row
        return [np.vstack([layer.b[None, :], layer.W]) for layer in self.layers]


class ANNClassificationTorch:
    """PyTorch clone of ANNClassification, matched as closely as possible:
    same architecture (sigmoid hidden + linear output, softmax in the loss),
    same Xavier-style init N(0, 1/sqrt(d_in)) seeded from numpy,
    plain SGD with the same lr, mean cross-entropy, full-batch GD,
    and the same input standardization. Differences from ours come down
    to float32 vs float64 and the exact summation order inside torch ops.
    """

    def __init__(self, units, lambda_=0.0) -> None:
        if not _HAS_TORCH:
            raise ImportError("torch is required for ANNClassificationTorch")
        self.units = list(units)
        self.lambda_ = lambda_
        self.epochs = 5000
        self.lr = 0.5
        self.standardize = True
        self.verbose = True

    def _build_module(self, d_in, d_out):
        layers = []
        prev = d_in
        for h in self.units:
            layers.append(_torch_nn.Linear(prev, h))
            layers.append(_torch_nn.Sigmoid())
            prev = h
        layers.append(_torch_nn.Linear(prev, d_out))
        # softmax is folded into CrossEntropyLoss; the module emits logits
        self.model = _torch_nn.Sequential(*layers)

        # match our init: weights ~ N(0, 1/sqrt(d_in)), zero bias.
        # we draw via numpy so the per-layer init is identical when the
        # numpy seed is the same.
        for m in self.model.modules():
            if isinstance(m, _torch_nn.Linear):
                W = np.random.randn(m.in_features, m.out_features) \
                    * np.sqrt(1.0 / m.in_features)
                # torch stores weight as (out, in), so transpose
                m.weight.data = torch.tensor(W.T, dtype=torch.float32)
                m.bias.data = torch.zeros(m.out_features, dtype=torch.float32)

    def fit(self, X: np.typing.NDArray, y: np.typing.NDArray):
        if self.standardize:
            self._x_mean = X.mean(axis=0)
            self._x_std = X.std(axis=0)
            self._x_std = np.where(self._x_std < 1e-12, 1.0, self._x_std)
        else:
            self._x_mean = np.zeros(X.shape[1])
            self._x_std = np.ones(X.shape[1])
        Xs = (X - self._x_mean) / self._x_std

        K = len(np.unique(y))
        self._build_module(Xs.shape[1], K)

        Xt = torch.tensor(Xs, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.long)

        loss_fn = _torch_nn.CrossEntropyLoss()  # reduction='mean' by default
        optim = torch.optim.SGD(self.model.parameters(), lr=self.lr)

        self.loss_history = []
        log_every = max(1, self.epochs // 10)

        for epoch in range(self.epochs):
            optim.zero_grad()
            logits = self.model(Xt)
            loss = loss_fn(logits, yt)
            loss.backward()
            optim.step()
            self.loss_history.append(loss.item())
            if self.verbose and (epoch % log_every == 0
                                 or epoch == self.epochs - 1):
                print(f"epoch {epoch:5d}  loss={loss.item():.6f}")

        return self

    def predict(self, X: np.typing.NDArray):
        Xs = (X - self._x_mean) / self._x_std
        with torch.no_grad():
            logits = self.model(torch.tensor(Xs, dtype=torch.float32))
            probs = torch.softmax(logits, dim=1)
        return probs.numpy()


class ANNRegression:
    def __init__(self, units, lambda_=0.0, activations=None) -> None:
        self.units = list(units)
        self.lambda_ = lambda_
        self.layers: List[ANNLayer] = []
        self.epochs = 5000
        self.lr = 0.5
        self.standardize = True
        self.mean_grad = True
        self.verbose = True
        if activations is None:
            activations = ["sigmoid"] * len(self.units)
        if len(activations) != len(self.units):
            raise ValueError(
                f"expected {len(self.units)} activations, got {len(activations)}"
            )
        self.activations = [_resolve_activation(a) for a in activations]

    def _build_layers(self, d_in):
        self.layers = []
        prev = d_in
        for h, act in zip(self.units, self.activations):
            self.layers.append(ANNLayer(prev, h, act))
            prev = h
        # Gaussian target -> identity link -> single linear output unit
        self.layers.append(ANNLayer(prev, 1, LINEAR))

    def fit(self, X: np.typing.NDArray, y: np.typing.NDArray):
        if self.standardize:
            self._x_mean = X.mean(axis=0)
            self._x_std = X.std(axis=0)
            self._x_std = np.where(self._x_std < 1e-12, 1.0, self._x_std)
        else:
            self._x_mean = np.zeros(X.shape[1])
            self._x_std = np.ones(X.shape[1])
        X = (X - self._x_mean) / self._x_std

        y_col = y.reshape(-1, 1).astype(float)
        self._build_layers(X.shape[1])

        n = X.shape[0]
        self.loss_history = []
        log_every = max(1, self.epochs // 10)
        grad_denom = n if self.mean_grad else 1.0

        for epoch in range(self.epochs):
            A = X
            # forward part
            for layer in self.layers:
                A = layer.forward(A)

            # mean squared error
            loss = ((A - y_col) ** 2).mean()
            self.loss_history.append(loss)
            if self.verbose and (epoch % log_every == 0 or epoch == self.epochs - 1):
                print(f"epoch {epoch:5d}  loss={loss:.6f}")

            # d/dA of mean MSE = 2(A - y)/n; output is linear so dZ = dA
            dZ = 2.0 * (A - y_col) / grad_denom
            dA = self.layers[-1].backward_from_dZ(dZ)

            # backwards part
            for layer in reversed(self.layers[:-1]):
                dA = layer.backward(dA)

            # weights and bias update; L2 reg on weights only
            for layer in self.layers:
                layer.W -= self.lr * (layer.dW + self.lambda_ * layer.W)
                layer.b -= self.lr * layer.db

        return self

    def predict(self, X: np.typing.NDArray):
        forward = (X - self._x_mean) / self._x_std
        for layer in self.layers:
            forward = layer.forward(forward)
        return forward.squeeze(axis=1)

    def weights(self):
        return [np.vstack([layer.b[None, :], layer.W]) for layer in self.layers]


# data reading


def read_tab(fn, adict):
    content = list(csv.reader(open(fn, "rt"), delimiter="\t"))

    legend = content[0][1:]
    data = content[1:]

    X = np.array([d[1:] for d in data], dtype=float)
    y = np.array([adict[d[0]] for d in data])

    return legend, X, y


def doughnut():
    legend, X, y = read_tab("doughnut.tab", {"C1": 0, "C2": 1})
    return X, y


def squares():
    legend, X, y = read_tab("squares.tab", {"C1": 0, "C2": 1})
    return X, y


# --- experiments for the report ---------------------------------------


def _train_quiet(X, y, units, *, epochs, lr, standardize=True,
                 mean_grad=True, seed=0):
    np.random.seed(seed)
    m = ANNClassification(units=units, lambda_=0.0)
    m.epochs, m.lr = epochs, lr
    m.standardize, m.mean_grad = standardize, mean_grad
    m.verbose = False
    m.fit(X, y)
    acc = (m.predict(X).argmax(axis=1) == y).mean()
    return m, acc


def _smallest_arch_search(X, y, candidates, n_seeds=5, epochs=5000, lr=0.5):
    # for each architecture, count how many seeds reach 100% training accuracy.
    print(f"{'units':<14}{'best acc':<11}{'mean acc':<11}{'success':<10}{'n_params':<9}")
    print("-" * 55)
    reliable = []
    for units in candidates:
        accs, params = [], None
        for s in range(n_seeds):
            m, acc = _train_quiet(X, y, units, epochs=epochs, lr=lr, seed=s)
            accs.append(acc)
            params = sum(layer.W.size + layer.b.size for layer in m.layers)
        succ = sum(a == 1.0 for a in accs)
        print(f"{str(units):<14}{max(accs):<11.4f}{np.mean(accs):<11.4f}"
              f"{succ}/{n_seeds:<8}{params}")
        if succ == n_seeds:
            reliable.append((units, params))
    smallest = min(reliable, key=lambda r: r[1]) if reliable else None
    print(f"\nsmallest reliable architecture by parameter count: {smallest}")
    return smallest


def _ablation(X, y, units, lr=0.5, epochs=5000, n_seeds=3):
    print(f"{'config':<25}{'mean acc':<11}{'final loss':<13}{'comment'}")
    print("-" * 75)
    configs = [
        ("std=ON,  /n=ON",  True,  True),
        ("std=OFF, /n=ON",  False, True),
        ("std=ON,  /n=OFF", True,  False),
        ("std=OFF, /n=OFF", False, False),
    ]
    for name, std, mn in configs:
        accs, losses = [], []
        for s in range(n_seeds):
            try:
                m, acc = _train_quiet(X, y, units, epochs=epochs, lr=lr,
                                      standardize=std, mean_grad=mn, seed=s)
                accs.append(acc)
                losses.append(m.loss_history[-1])
            except (FloatingPointError, ValueError):
                accs.append(0.0)
                losses.append(float("nan"))
        comment = ""
        if not std:
            comment = "sigmoid saturates on raw inputs"
        if not mn:
            comment = ("step ~n× too large; oscillates / diverges"
                       if not comment else comment + " AND step too big")
        print(f"{name:<25}{np.mean(accs):<11.4f}{np.mean(losses):<13.4f}{comment}")


def _lr_sweep(X, y, units, lrs, epochs=5000, loss_thr=0.01):
    # epoch_to_thr: first epoch where the running loss drops below loss_thr.
    # this is a proxy for "convergence speed". '-' means it never got there.
    print(f"{'lr':<9}{'final loss':<13}{'acc':<8}"
          f"{'epoch loss<' + str(loss_thr):<18}")
    print("-" * 50)
    for lr in lrs:
        m, acc = _train_quiet(X, y, units, epochs=epochs, lr=lr, seed=0)
        below = next((i for i, L in enumerate(m.loss_history)
                      if L < loss_thr), None)
        ep_str = str(below) if below is not None else "-"
        print(f"{lr:<9.3f}{m.loss_history[-1]:<13.4f}{acc:<8.3f}{ep_str:<18}")


def _activation_comparison(X, y, units, lr=0.5, epochs=5000):
    print(f"{'activations':<22}{'final loss':<13}{'acc':<8}"
          f"{'epoch loss<0.05':<18}")
    print("-" * 60)
    for spec in ["sigmoid", "relu"]:
        np.random.seed(0)
        m = ANNClassification(units=units, lambda_=0.0,
                              activations=[spec] * len(units))
        m.epochs, m.lr, m.verbose = epochs, lr, False
        m.fit(X, y)
        acc = (m.predict(X).argmax(axis=1) == y).mean()
        below = next((i for i, L in enumerate(m.loss_history) if L < 0.05),
                     None)
        ep = str(below) if below is not None else "-"
        print(f"{spec:<22}{m.loss_history[-1]:<13.4f}{acc:<8.3f}{ep:<18}")


def _regularization_sweep(X, y, units, lambdas, lr=0.5, epochs=5000):
    print(f"{'lambda':<10}{'final loss':<13}{'acc':<8}"
          f"{'||W||':<10}{'||b||':<8}")
    print("-" * 55)
    for lam in lambdas:
        np.random.seed(0)
        m = ANNClassification(units=units, lambda_=lam)
        m.epochs, m.lr, m.verbose = epochs, lr, False
        m.fit(X, y)
        acc = (m.predict(X).argmax(axis=1) == y).mean()
        wn = sum(np.linalg.norm(layer.W) for layer in m.layers)
        bn = sum(np.linalg.norm(layer.b) for layer in m.layers)
        print(f"{lam:<10.4f}{m.loss_history[-1]:<13.4f}{acc:<8.3f}"
              f"{wn:<10.3f}{bn:<8.3f}")


def run_experiments():
    print("\n" + "=" * 60)
    print("Smallest architecture for doughnut")
    print("=" * 60)
    X, y = doughnut()
    candidates = [[], [1], [2], [3], [4], [5], [6], [8], [10],
                  [2, 2], [4, 2], [4, 4]]
    _smallest_arch_search(X, y, candidates)

    print("\n" + "=" * 60)
    print("Smallest architecture for squares")
    print("=" * 60)
    X, y = squares()
    _smallest_arch_search(X, y, candidates)

    print("\n" + "=" * 60)
    print("Ablation: standardization & gradient/n  (doughnut, units=[8])")
    print("=" * 60)
    X, y = doughnut()
    _ablation(X, y, units=[8])

    print("\n" + "=" * 60)
    print("Learning rate sweep  (doughnut, units=[3])")
    print("=" * 60)
    X, y = doughnut()
    _lr_sweep(X, y, units=[3], lrs=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0])

    print("\n" + "=" * 60)
    print("Learning rate sweep  (squares, units=[4, 2])")
    print("=" * 60)
    X, y = squares()
    _lr_sweep(X, y, units=[4, 2], lrs=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0])

    print("\n" + "=" * 60)
    print("Ours vs PyTorch  (matched seeds, identical hyperparams)")
    print("=" * 60)
    _compare_with_torch()

    print("\n" + "=" * 60)
    print("Activation: sigmoid vs relu  (doughnut, units=[3])")
    print("=" * 60)
    X, y = doughnut()
    _activation_comparison(X, y, units=[3])

    print("\n" + "=" * 60)
    print("Regularization sweep  (doughnut, units=[10])")
    print("=" * 60)
    _regularization_sweep(X, y, units=[10],
                          lambdas=[0.0, 0.001, 0.01, 0.1, 1.0])


def _compare_with_torch():
    import time
    if not _HAS_TORCH:
        print("(torch not installed, skipping)")
        return

    cases = [
        ("doughnut", doughnut, [3], 2000, 0.5),
        ("squares",  squares,  [4, 2], 2000, 1.0),
    ]
    print(f"{'dataset':<11}{'units':<10}{'impl':<10}{'time (s)':<10}"
          f"{'final loss':<13}{'acc':<8}")
    print("-" * 65)
    for name, loader, units, epochs, lr in cases:
        X, y = loader()

        np.random.seed(0)
        m1 = ANNClassification(units=units)
        m1.epochs, m1.lr, m1.verbose = epochs, lr, False
        t0 = time.time(); m1.fit(X, y); t1 = time.time() - t0
        p1 = m1.predict(X)
        a1 = (p1.argmax(1) == y).mean()

        np.random.seed(0); torch.manual_seed(0)
        m2 = ANNClassificationTorch(units=units)
        m2.epochs, m2.lr, m2.verbose = epochs, lr, False
        t2 = time.time(); m2.fit(X, y); t3 = time.time() - t2
        p2 = m2.predict(X)
        a2 = (p2.argmax(1) == y).mean()

        print(f"{name:<11}{str(units):<10}{'ours':<10}{t1:<10.3f}"
              f"{m1.loss_history[-1]:<13.4f}{a1:<8.3f}")
        print(f"{'':<11}{'':<10}{'pytorch':<10}{t3:<10.3f}"
              f"{m2.loss_history[-1]:<13.4f}{a2:<8.3f}")
        # agreement metrics
        diff_prob = float(np.abs(p1 - p2).max())
        diff_loss_curve = float(np.max(np.abs(
            np.array(m1.loss_history) - np.array(m2.loss_history))))
        print(f"{'':<11}{'':<10}max |Δprob|={diff_prob:.4g}, "
              f"max |Δloss(t)|={diff_loss_curve:.4g}")
        print("-" * 65)


if __name__ == "__main__":
    run_experiments()
