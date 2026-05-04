import io
import contextlib
import unittest
import numpy as np

from nn import ANNClassification, ANNRegression, doughnut, squares


class NNTests(unittest.TestCase):

    def setUp(self):
        self.X = np.array([[0, 0],
                           [0, 1],
                           [1, 0],
                           [1, 1]])
        self.y = np.array([0, 1, 2, 3])
        self.hard_y = np.array([0, 1, 1, 0])

    def test_ann_classification_no_hidden_layer(self):
        fitter = ANNClassification(units=[], lambda_=0.)
        m = fitter.fit(self.X, self.y)
        pred = m.predict(self.X)
        self.assertEqual(pred.shape, (4, 4))
        np.testing.assert_allclose(pred, np.identity(4), atol=0.01)

    def test_ann_classification_no_hidden_layer_hard(self):
        # aiming to solve a non-linear problem without hidden layers
        fitter = ANNClassification(units=[], lambda_=0.)
        m = fitter.fit(self.X, self.hard_y)
        pred = m.predict(self.X)
        self.assertEqual(pred.shape, (4, 2))
        np.testing.assert_allclose(pred, 0.5, atol=0.01)

    def test_ann_classification_hidden_layer_hard(self):
        # with hidden layers we can solve a non-linear problem
        fitter = ANNClassification(units=[10], lambda_=0.)
        m = fitter.fit(self.X, self.hard_y)
        pred = m.predict(self.X)
        self.assertEqual(pred.shape, (4, 2))
        np.testing.assert_allclose(pred, [[1, 0], [0, 1], [0, 1], [1, 0]], atol=0.01)

    def test_ann_classification_hidden_layers_hard(self):
        # two hidden layers
        fitter = ANNClassification(units=[10, 11], lambda_=0.)
        m = fitter.fit(self.X, self.hard_y)
        pred = m.predict(self.X)
        self.assertEqual(pred.shape, (4, 2))
        np.testing.assert_allclose(pred, [[1, 0], [0, 1], [0, 1], [1, 0]], atol=0.01)

    def test_ann_regression_no_hidden_layer(self):
        fitter = ANNRegression(units=[], lambda_=0.)
        m = fitter.fit(self.X, self.y)
        pred = m.predict(self.X)
        self.assertEqual(pred.shape, (4,))
        np.testing.assert_allclose(pred, self.y, atol=0.01)

    def test_ann_regression_no_hidden_layer_hard(self):
        # aiming to solve a non-linear problem without hidden layers
        fitter = ANNRegression(units=[], lambda_=0.)
        m = fitter.fit(self.X, self.hard_y)
        pred = m.predict(self.X)
        self.assertEqual(pred.shape, (4,))
        np.testing.assert_allclose(pred, 0.5, atol=0.01)

    def test_ann_regression_hidden_layer_hard(self):
        # one hidden layer
        fitter = ANNRegression(units=[10], lambda_=0.)
        m = fitter.fit(self.X, self.hard_y)
        pred = m.predict(self.X)
        self.assertEqual(pred.shape, (4,))
        np.testing.assert_allclose(pred, self.hard_y, atol=0.01)

    def test_ann_regression_hidden_layers_hard(self):
        # two hidden layers
        fitter = ANNRegression(units=[10, 11], lambda_=0.)
        m = fitter.fit(self.X, self.hard_y)
        pred = m.predict(self.X)
        self.assertEqual(pred.shape, (4,))
        np.testing.assert_allclose(pred, self.hard_y, atol=0.01)

    def test_predictor_get_info(self):
        fitter = ANNRegression(units=[10, 5], lambda_=0.)
        m = fitter.fit(self.X, self.y)
        lw = m.weights()  # a list of weight matrices that include intercept biases

        self.assertEqual(len(lw), 3)  # two hidden layer == three weight matrices

        self.assertEqual(lw[0].shape, (3, 10))
        self.assertEqual(lw[1].shape, (11, 5))
        self.assertEqual(lw[2].shape, (6, 1))


class MyTests(unittest.TestCase):
    """Assignment-required tests: numerical gradient check + perfect-fit
    on doughnut.tab and squares.tab."""

    @staticmethod
    def _ce_loss_mean(model, X, y):
        # MEAN cross-entropy. fit() uses dZ = (P - Y_onehot) / n which
        # is the gradient of the mean loss, so the numerical gradient
        # has to be of the mean loss too for the check to be meaningful.
        n = X.shape[0]
        A = X
        for layer in model.layers:
            A = layer.forward(A)
        return -np.log(A[np.arange(n), y] + 1e-12).mean()

    def test_gradient_matches_numerical(self):
        # Confirms that the analytical gradient (P - Y_onehot pushed
        # backward through the layers) matches central finite differences.
        # Try several architectures so the check exercises every path.
        rng = np.random.RandomState(0)
        np.random.seed(0)
        X = rng.randn(8, 3)
        y = rng.randint(0, 4, size=8)
        K = int(y.max()) + 1
        Y_onehot = np.eye(K)[y]

        # exercise every code path: empty hidden, single sigmoid, two sigmoid,
        # relu hidden, mixed relu+sigmoid
        cases = [
            ([], None),
            ([5], None),
            ([4, 3], None),
            ([5], ["relu"]),
            ([4, 3], ["relu", "sigmoid"]),
        ]
        for units, acts in cases:
            with self.subTest(units=units, activations=acts):
                np.random.seed(0)
                model = ANNClassification(units=units, lambda_=0.0,
                                          activations=acts)
                model._build_layers(X.shape[1], K)

                # one forward + analytical backward, then snapshot grads.
                # divide by n to match fit()'s mean-loss convention.
                n = X.shape[0]
                A = X
                for layer in model.layers:
                    A = layer.forward(A)
                dZ = (A - Y_onehot) / n
                dA = model.layers[-1].backward_from_dZ(dZ)
                for layer in reversed(model.layers[:-1]):
                    dA = layer.backward(dA)

                analytical = [(layer.dW.copy(), layer.db.copy())
                              for layer in model.layers]

                # central differences for every parameter
                eps = 1e-5
                max_rel_err = 0.0
                for li, layer in enumerate(model.layers):
                    for which, ana in zip(("W", "b"), analytical[li]):
                        param = getattr(layer, which)
                        for idx in np.ndindex(*param.shape):
                            orig = param[idx]
                            param[idx] = orig + eps
                            L_plus = self._ce_loss_mean(model, X, y)
                            param[idx] = orig - eps
                            L_minus = self._ce_loss_mean(model, X, y)
                            param[idx] = orig
                            num = (L_plus - L_minus) / (2 * eps)
                            denom = max(abs(num), abs(ana[idx]), 1e-8)
                            rel_err = abs(num - ana[idx]) / denom
                            max_rel_err = max(max_rel_err, rel_err)

                # central differences are O(eps^2); 1e-6 is loose enough
                # to absorb fp rounding but tight enough to catch real bugs
                self.assertLess(max_rel_err, 1e-6,
                                f"units={units} max rel err {max_rel_err:.2e}")

    def _fit_quietly(self, fitter, X, y):
        # fit() prints loss every ~epochs/10; silence it during tests
        with contextlib.redirect_stdout(io.StringIO()):
            return fitter.fit(X, y)

    def test_doughnut_perfect_fit(self):
        np.random.seed(0)
        X, y = doughnut()
        fitter = ANNClassification(units=[8], lambda_=0.0)
        fitter.epochs = 5000
        fitter.lr = 0.5
        m = self._fit_quietly(fitter, X, y)
        pred_labels = np.argmax(m.predict(X), axis=1)
        acc = (pred_labels == y).mean()
        self.assertEqual(acc, 1.0,
                         f"doughnut: expected perfect fit, got acc={acc:.3f}")

    def test_squares_perfect_fit(self):
        np.random.seed(0)
        X, y = squares()
        fitter = ANNClassification(units=[8], lambda_=0.0)
        fitter.epochs = 5000
        fitter.lr = 0.5
        m = self._fit_quietly(fitter, X, y)
        pred_labels = np.argmax(m.predict(X), axis=1)
        acc = (pred_labels == y).mean()
        self.assertEqual(acc, 1.0,
                         f"squares: expected perfect fit, got acc={acc:.3f}")

    def test_relu_classification(self):
        # ReLU as the hidden activation should also fit doughnut.
        np.random.seed(0)
        X, y = doughnut()
        fitter = ANNClassification(units=[8], lambda_=0.0,
                                   activations=["relu"])
        fitter.epochs = 5000
        fitter.lr = 0.5
        m = self._fit_quietly(fitter, X, y)
        acc = (np.argmax(m.predict(X), axis=1) == y).mean()
        self.assertGreaterEqual(acc, 0.99,
                                f"relu doughnut: acc={acc:.3f}")

    def test_per_layer_mixed_activations(self):
        # Mixed activations per layer should train without errors.
        np.random.seed(0)
        X, y = doughnut()
        fitter = ANNClassification(units=[8, 4], lambda_=0.0,
                                   activations=["relu", "sigmoid"])
        fitter.epochs = 5000
        fitter.lr = 0.5
        m = self._fit_quietly(fitter, X, y)
        acc = (np.argmax(m.predict(X), axis=1) == y).mean()
        self.assertGreaterEqual(acc, 0.95,
                                f"mixed activations: acc={acc:.3f}")

    def test_regularization_shrinks_weights(self):
        # Higher lambda must give smaller ||W||; biases are exempt and so
        # ||b|| should not be (consistently) smaller in the regularized run.
        np.random.seed(0)
        X, y = doughnut()

        def fit_with_lambda(lam):
            np.random.seed(0)
            f = ANNClassification(units=[10], lambda_=lam)
            f.epochs, f.lr = 2000, 0.5
            m = self._fit_quietly(f, X, y)
            wn = sum(np.linalg.norm(layer.W) for layer in m.layers)
            bn = sum(np.linalg.norm(layer.b) for layer in m.layers)
            return wn, bn

        wn0, bn0 = fit_with_lambda(0.0)
        wn1, _   = fit_with_lambda(0.1)
        self.assertLess(wn1, wn0,
            f"regularized ||W||={wn1:.3f} should be < unreg {wn0:.3f}")
        # bias norm with no reg should still be > 0 (otherwise the test is vacuous)
        self.assertGreater(bn0, 1e-3)

    def test_ann_regression_relu_and_lambda(self):
        # ReLU + small lambda for regression on a smooth nonlinear target;
        # check the model explains most of the variance (R^2 > 0.95).
        rng = np.random.RandomState(0)
        np.random.seed(0)
        X = rng.randn(100, 2)
        y = (X[:, 0] ** 2 + X[:, 1]).astype(float)
        fitter = ANNRegression(units=[16], lambda_=0.001,
                               activations=["relu"])
        fitter.epochs = 5000
        fitter.lr = 0.1
        m = self._fit_quietly(fitter, X, y)
        mse = ((m.predict(X) - y) ** 2).mean()
        r2 = 1.0 - mse / y.var()
        self.assertGreater(r2, 0.95,
            f"regression+relu+lambda: R^2={r2:.3f} mse={mse:.4f}")


if __name__ == "__main__":
    import unittest
    unittest.main()
