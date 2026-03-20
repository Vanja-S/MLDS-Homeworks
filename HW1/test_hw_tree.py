import random
import unittest

import numpy as np

from hw_tree import Tree, RandomForest, hw_tree_full, hw_randomforests


def random_feature(X, rand):
    return [rand.choice(list(range(X.shape[1])))]


class HWTreeTests(unittest.TestCase):

    def setUp(self):
        self.X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.y = np.array([0, 0, 1, 1])
        self.train = self.X[:3], self.y[:3]
        self.test = self.X[3:], self.y[3:]

    def test_call_tree(self):
        t = Tree(
            rand=random.Random(1), get_candidate_columns=random_feature, min_samples=2
        )
        p = t.build(self.X, self.y)
        pred = p.predict(self.X)
        np.testing.assert_equal(pred, self.y)

    def test_call_randomforest(self):
        rf = RandomForest(rand=random.Random(0), n=20)
        p = rf.build(self.X, self.y)
        pred = p.predict(self.X)
        np.testing.assert_equal(pred, self.y)

    def test_call_importance(self):
        rf = RandomForest(rand=random.Random(0), n=20)
        p = rf.build(np.tile(self.X, (2, 1)), np.tile(self.y, 2))
        imp = p.importance()
        self.assertTrue(len(imp), self.X.shape[1])
        self.assertGreater(imp[0], imp[1])

    def test_signature_hw_tree_full(self):
        (train, train_un), (test, test_un) = hw_tree_full(self.train, self.test)
        self.assertIsInstance(train, float)
        self.assertIsInstance(test, float)
        self.assertIsInstance(train_un, float)
        self.assertIsInstance(test_un, float)

    def test_signature_hw_randomforests(self):
        (train, train_un), (test, test_un) = hw_randomforests(self.train, self.test)
        self.assertIsInstance(train, float)
        self.assertIsInstance(test, float)
        self.assertIsInstance(train_un, float)
        self.assertIsInstance(test_un, float)


class MyTests(unittest.TestCase):

    def test_gini_pure(self):
        t = Tree()
        self.assertAlmostEqual(t.gini(np.array([1, 1, 1])), 0.0)

    def test_gini_max_impurity(self):
        t = Tree()
        self.assertAlmostEqual(t.gini(np.array([0, 1])), 0.5)

    def test_gini_empty(self):
        t = Tree()
        self.assertAlmostEqual(t.gini(np.array([])), 0.0)

    def test_split_finds_obvious_cut(self):
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = np.array([0, 0, 1, 1])
        t = Tree()
        D_l, D_r, criterion = t.split(X, y)
        self.assertEqual(criterion[0], 0)
        self.assertAlmostEqual(criterion[1], 2.5)

    def test_split_no_valid_split(self):
        X = np.array([[1.0], [1.0], [1.0]])
        y = np.array([0, 1, 0])
        t = Tree()
        D_l, D_r, criterion = t.split(X, y)
        self.assertIsNone(criterion)

    def test_split_pure_node(self):
        X = np.array([[1.0], [2.0]])
        y = np.array([0, 0])
        t = Tree()
        D_l, D_r, criterion = t.split(X, y)
        self.assertIsNone(criterion)

    def test_tree_perfect_classification(self):
        X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        y = np.array([0, 1, 0, 1])
        model = Tree(min_samples=2).build(X, y)
        np.testing.assert_array_equal(model.predict(X), y)

    def test_tree_single_sample_leaf(self):
        X = np.array([[1.0], [2.0]])
        y = np.array([0, 1])
        model = Tree(min_samples=2).build(X, y)
        np.testing.assert_array_equal(model.predict(X), y)

    def test_tree_min_samples_stops_splitting(self):
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = np.array([0, 0, 1, 1])
        model = Tree(min_samples=10).build(X, y)
        preds = model.predict(X)
        self.assertTrue(len(np.unique(preds)) == 1)

    def test_predict_unseen_data(self):
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = np.array([0, 0, 1, 1])
        model = Tree(min_samples=2).build(X, y)
        self.assertEqual(model.predict(np.array([[0.5]]))[0], 0)
        self.assertEqual(model.predict(np.array([[5.0]]))[0], 1)


if __name__ == "__main__":
    unittest.main()
