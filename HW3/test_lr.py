import random
import math
import unittest

import numpy as np

from solution1 import Value, MultinomialLogReg, OrdinalLogReg
from solution2 import MultinomialLogReg as MultinomialLogReg2, OrdinalLogReg as OrdinalLogReg2


class HW2Tests(unittest.TestCase):

    def setUp(self):
        self.X = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [1, 1]])
        self.y = np.array([0, 0, 1, 1, 2])
        self.train = self.X[::2], self.y[::2]
        self.test = self.X[1::2], self.y[1::2]

    def test_multinomial(self):
        l = MultinomialLogReg()
        c = l.build(self.X, self.y)
        prob = c.predict(self.test[0])
        self.assertEqual(prob.shape, (2, 3))
        self.assertTrue((prob <= 1).all())
        self.assertTrue((prob >= 0).all())
        np.testing.assert_almost_equal(prob.sum(axis=1), 1)

    def test_ordinal(self):
        l = OrdinalLogReg()
        c = l.build(self.X, self.y)
        prob = c.predict(self.test[0])
        self.assertEqual(prob.shape, (2, 3))
        self.assertTrue((prob <= 1).all())
        self.assertTrue((prob >= 0).all())
        np.testing.assert_almost_equal(prob.sum(axis=1), 1)


class MyTests(unittest.TestCase):

    # --- forward pass ---

    def test_add(self):
        a, b = Value(2.0), Value(3.0)
        self.assertAlmostEqual((a + b).data, 5.0)

    def test_mul(self):
        a, b = Value(2.0), Value(3.0)
        self.assertAlmostEqual((a * b).data, 6.0)

    def test_pow(self):
        a = Value(3.0)
        self.assertAlmostEqual((a**2).data, 9.0)

    def test_neg(self):
        a = Value(5.0)
        self.assertAlmostEqual((-a).data, -5.0)

    def test_sub(self):
        a, b = Value(7.0), Value(3.0)
        self.assertAlmostEqual((a - b).data, 4.0)

    def test_div(self):
        a, b = Value(6.0), Value(3.0)
        self.assertAlmostEqual((a / b).data, 2.0)

    def test_exp(self):
        a = Value(1.0)
        self.assertAlmostEqual(a.exp().data, math.e)

    def test_log(self):
        a = Value(math.e)
        self.assertAlmostEqual(a.log().data, 1.0)

    def test_exp_log_roundtrip(self):
        a = Value(4.2)
        self.assertAlmostEqual(a.exp().log().data, 4.2)

    # --- reverse ops (constant on the left) ---

    def test_radd(self):
        a = Value(3.0)
        self.assertAlmostEqual((10 + a).data, 13.0)

    def test_rmul(self):
        a = Value(4.0)
        self.assertAlmostEqual((3 * a).data, 12.0)

    def test_rsub(self):
        a = Value(3.0)
        self.assertAlmostEqual((10 - a).data, 7.0)

    def test_rtruediv(self):
        a = Value(4.0)
        self.assertAlmostEqual((12 / a).data, 3.0)

    # --- backward: gradients for individual ops ---

    def test_add_backward(self):
        a, b = Value(2.0), Value(3.0)
        c = a + b
        c.backward()
        self.assertAlmostEqual(a.grad, 1.0)
        self.assertAlmostEqual(b.grad, 1.0)

    def test_mul_backward(self):
        a, b = Value(2.0), Value(3.0)
        c = a * b
        c.backward()
        self.assertAlmostEqual(a.grad, 3.0)
        self.assertAlmostEqual(b.grad, 2.0)

    def test_pow_backward(self):
        a = Value(3.0)
        c = a**3
        c.backward()
        # d(x^3)/dx = 3x^2 = 27
        self.assertAlmostEqual(a.grad, 27.0)

    def test_exp_backward(self):
        a = Value(2.0)
        c = a.exp()
        c.backward()
        self.assertAlmostEqual(a.grad, math.exp(2.0))

    def test_log_backward(self):
        a = Value(5.0)
        c = a.log()
        c.backward()
        self.assertAlmostEqual(a.grad, 1.0 / 5.0)

    def test_div_backward(self):
        a, b = Value(6.0), Value(3.0)
        c = a / b
        c.backward()
        # d(a/b)/da = 1/b = 1/3
        self.assertAlmostEqual(a.grad, 1.0 / 3.0)
        # d(a/b)/db = -a/b^2 = -6/9
        self.assertAlmostEqual(b.grad, -6.0 / 9.0)

    def test_sub_backward(self):
        a, b = Value(5.0), Value(3.0)
        c = a - b
        c.backward()
        self.assertAlmostEqual(a.grad, 1.0)
        self.assertAlmostEqual(b.grad, -1.0)

    # --- composite expressions ---

    def test_lecture_example(self):
        # L = (a*b + c) * d
        a, b, c, d = Value(2.0), Value(-3.0), Value(10.0), Value(-2.0)
        L = (a * b + c) * d
        L.backward()
        self.assertAlmostEqual(L.data, -8.0)
        self.assertAlmostEqual(a.grad, 6.0)
        self.assertAlmostEqual(b.grad, -4.0)
        self.assertAlmostEqual(c.grad, -2.0)
        self.assertAlmostEqual(d.grad, 4.0)

    def test_chain_exp_mul_add(self):
        # f = exp(2*x + 3) at x=1 => exp(5)
        x = Value(1.0)
        f = (x * 2 + 3).exp()
        f.backward()
        # df/dx = 2 * exp(2x+3) = 2*exp(5)
        self.assertAlmostEqual(f.data, math.exp(5))
        self.assertAlmostEqual(x.grad, 2.0 * math.exp(5))

    # --- gradient accumulation: node used more than once ---

    def test_same_variable_added(self):
        # f = a + a = 2a, df/da = 2
        a = Value(3.0)
        f = a + a
        f.backward()
        self.assertAlmostEqual(f.data, 6.0)
        self.assertAlmostEqual(a.grad, 2.0)

    def test_same_variable_multiplied(self):
        # f = a * a = a^2, df/da = 2a = 6
        a = Value(3.0)
        f = a * a
        f.backward()
        self.assertAlmostEqual(f.data, 9.0)
        self.assertAlmostEqual(a.grad, 6.0)

    def test_variable_used_in_multiple_paths(self):
        # f = (a + 1) * (a + 2), at a=3 => 4*5=20
        # df/da = (a+2) + (a+1) = 9
        a = Value(3.0)
        f = (a + 1) * (a + 2)
        f.backward()
        self.assertAlmostEqual(f.data, 20.0)
        self.assertAlmostEqual(a.grad, 9.0)

    # --- gradient descent convergence ---

    def test_gradient_descent_quadratic(self):
        # minimize f(a) = a^2 - 10a + 28, optimal a=5
        a = Value(0.0)
        for _ in range(500):
            L = a**2 - 10 * a + 28
            L.backward()
            a.data -= 0.1 * a.grad
        self.assertAlmostEqual(a.data, 5.0, places=4)

    # --- backward resets grads between calls ---

    def test_backward_resets_grads(self):
        a = Value(3.0)
        f = a * 2
        f.backward()
        self.assertAlmostEqual(a.grad, 2.0)
        # calling backward again should give the same result, not accumulate
        f.backward()
        self.assertAlmostEqual(a.grad, 2.0)

    # --- softmax (critical for multinomial logistic regression) ---

    def test_softmax_probs_sum_to_one(self):
        scores = [Value(2.0), Value(1.0), Value(0.1)]
        exps = [s.exp() for s in scores]
        total = exps[0] + exps[1] + exps[2]
        probs = [e / total for e in exps]
        prob_sum = sum(p.data for p in probs)
        self.assertAlmostEqual(prob_sum, 1.0)

    def test_softmax_cross_entropy_gradient(self):
        # softmax + log of correct class, then backward
        scores = [Value(2.0), Value(1.0), Value(0.1)]
        exps = [s.exp() for s in scores]
        total = exps[0] + exps[1] + exps[2]
        probs = [e / total for e in exps]
        # correct class = 0
        loss = -(probs[0].log())
        loss.backward()
        # gradient for correct class score should be (p - 1), others should be p
        self.assertAlmostEqual(scores[0].grad, probs[0].data - 1.0, places=5)
        self.assertAlmostEqual(scores[1].grad, probs[1].data, places=5)
        self.assertAlmostEqual(scores[2].grad, probs[2].data, places=5)

    # --- edge cases ---

    def test_zero_value(self):
        a = Value(0.0)
        b = Value(5.0)
        c = a * b
        c.backward()
        self.assertAlmostEqual(c.data, 0.0)
        self.assertAlmostEqual(a.grad, 5.0)
        self.assertAlmostEqual(b.grad, 0.0)

    def test_negative_values(self):
        a = Value(-3.0)
        b = Value(-2.0)
        c = a * b
        c.backward()
        self.assertAlmostEqual(c.data, 6.0)
        self.assertAlmostEqual(a.grad, -2.0)
        self.assertAlmostEqual(b.grad, -3.0)

    def test_log_of_small_positive(self):
        a = Value(1e-10)
        c = a.log()
        c.backward()
        self.assertAlmostEqual(c.data, math.log(1e-10))
        self.assertAlmostEqual(a.grad, 1.0 / 1e-10)

    def test_exp_of_negative(self):
        a = Value(-5.0)
        c = a.exp()
        c.backward()
        self.assertAlmostEqual(c.data, math.exp(-5.0))
        self.assertAlmostEqual(a.grad, math.exp(-5.0))

    def test_pow_zero_exponent(self):
        a = Value(7.0)
        c = a**0
        self.assertAlmostEqual(c.data, 1.0)

    def test_pow_negative_exponent(self):
        a = Value(2.0)
        c = a**-1
        c.backward()
        self.assertAlmostEqual(c.data, 0.5)
        # d(x^-1)/dx = -x^-2 = -0.25
        self.assertAlmostEqual(a.grad, -0.25)

    def test_long_chain(self):
        # f = ((((a + 1) * 2) + 3) * 4) at a=1 => ((2*2)+3)*4 = 28
        a = Value(1.0)
        f = ((a + 1) * 2 + 3) * 4
        f.backward()
        self.assertAlmostEqual(f.data, 28.0)
        # df/da = 4 * 2 = 8
        self.assertAlmostEqual(a.grad, 8.0)


class Solution2Tests(unittest.TestCase):

    def setUp(self):
        self.X = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [1, 1]])
        self.y = np.array([0, 0, 1, 1, 2])
        self.train = self.X[::2], self.y[::2]
        self.test = self.X[1::2], self.y[1::2]

    # --- same interface checks as HW2Tests ---

    def test_multinomial_shape(self):
        c = MultinomialLogReg2().build(self.X, self.y)
        prob = c.predict(self.test[0])
        self.assertEqual(prob.shape, (2, 3))

    def test_multinomial_valid_probs(self):
        c = MultinomialLogReg2().build(self.X, self.y)
        prob = c.predict(self.test[0])
        self.assertTrue((prob >= 0).all())
        self.assertTrue((prob <= 1).all())
        np.testing.assert_almost_equal(prob.sum(axis=1), 1)

    def test_ordinal_shape(self):
        c = OrdinalLogReg2().build(self.X, self.y)
        prob = c.predict(self.test[0])
        self.assertEqual(prob.shape, (2, 3))

    def test_ordinal_valid_probs(self):
        c = OrdinalLogReg2().build(self.X, self.y)
        prob = c.predict(self.test[0])
        self.assertTrue((prob >= 0).all())
        self.assertTrue((prob <= 1).all())
        np.testing.assert_almost_equal(prob.sum(axis=1), 1)

    # --- predictions should be close to solution1 ---

    def test_multinomial_agrees_with_autograd(self):
        c1 = MultinomialLogReg().build(self.X, self.y)
        c2 = MultinomialLogReg2().build(self.X, self.y)
        p1 = c1.predict(self.test[0])
        p2 = c2.predict(self.test[0])
        np.testing.assert_array_less(np.abs(p1 - p2), 0.05)

    def test_ordinal_agrees_with_autograd(self):
        c1 = OrdinalLogReg().build(self.X, self.y)
        c2 = OrdinalLogReg2().build(self.X, self.y)
        p1 = c1.predict(self.test[0])
        p2 = c2.predict(self.test[0])
        np.testing.assert_array_less(np.abs(p1 - p2), 0.05)

    # --- single sample prediction ---

    def test_multinomial_single_sample(self):
        c = MultinomialLogReg2().build(self.X, self.y)
        prob = c.predict(self.X[:1])
        self.assertEqual(prob.shape, (1, 3))
        np.testing.assert_almost_equal(prob.sum(), 1)

    def test_ordinal_single_sample(self):
        c = OrdinalLogReg2().build(self.X, self.y)
        prob = c.predict(self.X[:1])
        self.assertEqual(prob.shape, (1, 3))
        np.testing.assert_almost_equal(prob.sum(), 1)

    # --- predict on training data (sanity: should at least not crash) ---

    def test_multinomial_predict_train(self):
        c = MultinomialLogReg2().build(self.X, self.y)
        prob = c.predict(self.X)
        self.assertEqual(prob.shape, (5, 3))
        np.testing.assert_almost_equal(prob.sum(axis=1), 1)

    def test_ordinal_predict_train(self):
        c = OrdinalLogReg2().build(self.X, self.y)
        prob = c.predict(self.X)
        self.assertEqual(prob.shape, (5, 3))
        np.testing.assert_almost_equal(prob.sum(axis=1), 1)


if __name__ == "__main__":
    unittest.main()
