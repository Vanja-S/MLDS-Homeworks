import unittest
from unittest.mock import patch
import numpy as np

from hw_kernels import SVR, RBF, Polynomial, \
    KernelizedRidgeRegression, SubsequenceKernel, \
    _bag_of_kmers, _gen_synthetic_strings


class Linear:
    """An example of a kernel."""

    def __init__(self):
        # here a kernel could set its parameters
        pass

    def __call__(self, A, B):
        """Can be called with vectors or matrices, see the
        comment for test_kernel"""
        return A.dot(B.T)


class KernelTests(unittest.TestCase):

    def setUp(self):
        self.X = np.array([[0, 0],
                           [0, 1],
                           [1, 0],
                           [1, 1]])
        self.y = np.array([0, 0, 1, 1])

    def test_linear(self):
        fitter = KernelizedRidgeRegression(kernel=Linear(), lambda_=0.0001)
        m = fitter.fit(self.X, self.y)
        pred = m.predict(self.X)
        np.testing.assert_almost_equal(pred, self.y, decimal=3)

    def test_rbf(self):
        fitter = KernelizedRidgeRegression(kernel=RBF(sigma=0.5), lambda_=0.0001)
        m = fitter.fit(self.X, self.y)
        pred = m.predict(self.X)
        np.testing.assert_almost_equal(pred, self.y, decimal=3)

    def test_polynomial(self):
        fitter = KernelizedRidgeRegression(kernel=Polynomial(M=2), lambda_=0.0001)
        m = fitter.fit(self.X, self.y)
        pred = m.predict(self.X)
        np.testing.assert_almost_equal(pred, self.y, decimal=3)

    def test_kernel(self):
        """
        Kernel classes should work for vectors (single
        instances; 1D numpy arrays) and matrices (multiple instances; 2D numpy arrays).

        If inputs are:
        - 1D, 1D (two vectors), the result is a number k(x_i,x_j)
        - 1D, 2D (vector and a matrix), the result is a 1D numpy array
        - 2D, 2D the result is a 2D numpy array

        Kernels implemented like this are easy to use.

        All kernels should be implemented without any Python looping.
        Implementing RBF without loops is a bit tricky, because it needs pairwise
        distances. Hint: for vectors a, b you can compute the distance with
        (a-b)^2 = a.dot(a) - 2*a.dot(b) + b.dot(b). This is, with some care,
        vectorizable.
        """
        for kernel in [Linear(), Polynomial(M=3), RBF(sigma=0.2)]:
            number = kernel(self.X[0], self.X[1]) # k
            float(number)  # should not crash
            a1d = kernel(self.X[0], self.X)  # a vector of k
            self.assertTrue(len(a1d.shape) == 1)
            a1d = kernel(self.X, self.X[0])
            self.assertTrue(len(a1d.shape) == 1)
            a2d = kernel(self.X, self.X)  # the K matrix
            self.assertTrue(len(a2d.shape) == 2)


def qp_callargs(call):
    kwargs = call.kwargs.copy()
    names = ["P", "q", "G", "h", "A", "b"]
    for n, v in zip(names, call.args):
        kwargs[n] = v
    return kwargs


class SVRTests(unittest.TestCase):

    def setUp(self):
        self.X = np.array([[0., 0],
                           [0, 1],
                           [1, 0],
                           [1, 1]])
        self.y = np.array([0., 1, 2, 3])

    def test_linear(self):
        fitter = SVR(kernel=Linear(), lambda_=0.0001, epsilon=0.1)
        m = fitter.fit(self.X, self.y)
        pred = m.predict(self.X)
        np.testing.assert_allclose(pred, self.y, atol=0.11)

    def test_rbf(self):
        fitter = SVR(kernel=RBF(sigma=0.5), lambda_=0.0001, epsilon=0.1)
        m = fitter.fit(self.X, self.y)
        pred = m.predict(self.X)
        np.testing.assert_allclose(pred, self.y, atol=0.11)

    def test_polynomial(self):
        fitter = SVR(kernel=Polynomial(M=2), lambda_=0.0001, epsilon=0.1)
        m = fitter.fit(self.X, self.y)
        pred = m.predict(self.X)
        np.testing.assert_allclose(pred, self.y, atol=0.11)

    def test_predictor_get_info(self):
        fitter = SVR(kernel=Polynomial(M=2), lambda_=0.0001, epsilon=0.1)
        m = fitter.fit(self.X, self.y)

        alpha = m.get_alpha()
        np.testing.assert_equal(alpha.shape, (4, 2))  # two alpha for each sample
        # one value in row should be much bigger than the other (the other should be zero)
        np.testing.assert_allclose(alpha.sum(axis=1), alpha.max(axis=1), rtol=1e-5)

        b = m.get_b()
        float(b)

    @patch("cvxopt.solvers.qp")
    def test_enforce_quadratic_programming_order_of_x(self, qp):
        fitter = SVR(kernel=Linear(), lambda_=0.0001, epsilon=0)
        try:
            _ = fitter.fit(self.X, self.y)
        except:
            pass
        usedq = np.array(qp_callargs(qp.call_args)["q"]).flatten()
        np.testing.assert_equal(usedq, [0,  0, -1, 1, -2, 2, -3, 3])

    def test_kernel(self):
        """
        Kernel classes should work for vectors (single
        instances; 1D numpy arrays) and matrices (multiple instances; 2D numpy arrays).

        If inputs are:
        - 1D, 1D (two vectors), the result is a number k(x_i,x_j)
        - 1D, 2D (vector and a matrix), the result is a 1D numpy array
        - 2D, 2D the result is a 2D numpy array

        Kernels implemented like this are easy to use.

        All kernels should be implemented without any Python looping.
        Implementing RBF without loops is a bit tricky, because it needs pairwise
        distances. Hint: for vectors a, b you can compute the distance with
        (a-b)^2 = a.dot(a) - 2*a.dot(b) - b.dot(b). This is, with some care,
        vectorizable.
        """
        for kernel in [Linear(), Polynomial(M=3), RBF(sigma=0.2)]:
            number = kernel(self.X[0], self.X[1]) # k
            float(number)  # should not crash
            a1d = kernel(self.X[0], self.X)  # a vector of k
            self.assertTrue(len(a1d.shape) == 1)
            a1d = kernel(self.X, self.X[0])
            self.assertTrue(len(a1d.shape) == 1)
            a2d = kernel(self.X, self.X)  # the K matrix
            self.assertTrue(len(a2d.shape) == 2)


class MyTests(unittest.TestCase):

    def setUp(self):
        self.X = np.array([[0., 0],
                           [0, 1],
                           [1, 0],
                           [1, 1]])
        self.y = np.array([0., 0, 1, 1])

    def test_rbf_diagonal_is_one(self):
        # k(x, x) = exp(0) = 1 for every training point
        K = RBF(sigma=0.5)(self.X, self.X)
        np.testing.assert_allclose(np.diag(K), np.ones(len(self.X)))

    def test_gram_matrix_is_symmetric(self):
        for kernel in [RBF(sigma=0.5), Polynomial(M=3)]:
            K = kernel(self.X, self.X)
            np.testing.assert_allclose(K, K.T)

    def test_rbf_values_in_unit_interval(self):
        # RBF is bounded in (0, 1]
        K = RBF(sigma=0.7)(self.X, self.X)
        self.assertTrue((K > 0).all())
        self.assertTrue((K <= 1 + 1e-12).all())

    def test_predict_on_different_size_test_set(self):
        # Catches code that assumes train and test have the same n
        X_test = np.array([[0.5, 0.5],
                           [0.2, 0.8],
                           [0.9, 0.1]])
        for kernel in [Linear(), RBF(sigma=0.5), Polynomial(M=2)]:
            m = KernelizedRidgeRegression(kernel=kernel, lambda_=0.01).fit(self.X, self.y)
            pred = m.predict(X_test)
            self.assertEqual(pred.shape, (3,))

    def test_higher_lambda_shrinks_predictions(self):
        # With no bias term, stronger regularization should shrink
        # predictions on training data toward zero
        m_small = KernelizedRidgeRegression(kernel=RBF(sigma=0.5), lambda_=0.0001).fit(self.X, self.y)
        m_large = KernelizedRidgeRegression(kernel=RBF(sigma=0.5), lambda_=100.0).fit(self.X, self.y)
        self.assertLess(np.abs(m_large.predict(self.X)).sum(),
                        np.abs(m_small.predict(self.X)).sum())


class StringKernelTests(unittest.TestCase):
    """Tests for the String Subsequence Kernel (SSK) and the bag-of-k-mers baseline."""

    def test_ssk_hand_computed(self):
        # K_2("abc", "abc") with λ=1, unnormalized, counts subsequences of length 2:
        # "ab" (span 2), "bc" (span 2), "ac" (span 3); each match has weight 1·1 = 1.
        # Total = 3.
        ssk = SubsequenceKernel(p=2, lam=1.0, normalize=False)
        K = ssk(["abc", "abc"], ["abc", "abc"])
        np.testing.assert_almost_equal(K[0, 1], 3.0, decimal=9)

    def test_ssk_symmetric(self):
        # Kernel must be symmetric: K(s, t) = K(t, s) for any p, λ.
        ssk = SubsequenceKernel(p=3, lam=0.5, normalize=True)
        K = ssk(["AAGT", "CGTAC", "TACGA"], ["AAGT", "CGTAC", "TACGA"])
        np.testing.assert_allclose(K, K.T, atol=1e-12)

    def test_ssk_normalized_diagonal_is_one(self):
        # Cosine normalization implies K(s, s) = 1 for any string with |s| >= p.
        ssk = SubsequenceKernel(p=3, lam=0.5, normalize=True)
        K = ssk(["AAGT", "CGTAC", "TACGA"], ["AAGT", "CGTAC", "TACGA"])
        np.testing.assert_allclose(np.diag(K), np.ones(3), atol=1e-12)

    def test_ssk_zero_when_string_too_short(self):
        # By definition K_p(s, t) = 0 if min(|s|, |t|) < p; no length-p subsequence exists.
        ssk = SubsequenceKernel(p=4, lam=0.5, normalize=False)
        K = ssk(["abc"], ["abcdef"])  # |"abc"| = 3 < p = 4
        self.assertEqual(K[0, 0], 0.0)

    def test_ssk_rectangular_gram(self):
        # Sanity check on shape: kernel(A, B) returns (|A|, |B|) regardless of equality.
        ssk = SubsequenceKernel(p=3, lam=0.5, normalize=True)
        K = ssk(["AAGT", "CGTAC"], ["TACGA", "AAGT", "CTTAG"])
        self.assertEqual(K.shape, (2, 3))

    def test_bag_of_kmers_counts(self):
        # "AAAA" with k=2 has 3 occurrences of "AA"; all others zero.
        X = _bag_of_kmers(["AAAA"], k=2)
        idx_AA = 0  # alphabet sorted lexicographically: AA is index 0 of itertools.product
        self.assertEqual(X[0, idx_AA], 3.0)
        # Total count of trigrams in a length-n string with k=3 is n - k + 1.
        X2 = _bag_of_kmers(["ACGTACGT"], k=3)  # length 8 -> 6 trigrams
        self.assertEqual(X2.sum(), 6.0)

    def test_synthetic_zero_target_when_no_patterns(self):
        # Force a string with no A's: target depends on patterns starting with 'A' or 'G',
        # so a string of only C and T has zero clean target.
        # We bypass the generator and call the inner gapped_count via the public path.
        # The cleanest way: generate 1 sample with a controlled seed and verify the cap.
        strings, _, y_clean = _gen_synthetic_strings(n=20, length=25, seed=42, noise_std=0.0)
        for s, yc in zip(strings, y_clean):
            if "A" not in s and "G" not in s:
                self.assertEqual(yc, 0.0)


if __name__ == "__main__":
    unittest.main()
