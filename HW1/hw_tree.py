import csv
from numpy import ndarray
import numpy as np
import random


def all_columns(X, rand):
    return range(X.shape[1])


def random_sqrt_columns(X, rand):
    n_cols = X.shape[1]
    n_select = int(np.sqrt(n_cols))
    return rand.sample(range(n_cols), n_select)


class Tree:

    def __init__(
        self,
        rand: random.Random | None = None,
        get_candidate_columns=all_columns,
        min_samples=2,
    ):
        self.rand = rand  # for replicability
        self.get_candidate_columns = get_candidate_columns  # needed for random forests
        self.min_samples = min_samples

    def gini(self, y: ndarray):
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(
            y, return_counts=True
        )  # this returns classes = [0, 1] and their counts=[x, y]
        probabilities = counts / len(
            y
        )  # then we just calculate their total percentage of the node
        return 1 - np.sum(probabilities**2)  # this is then the gini index formula

    def split(self, X: ndarray, y: ndarray):
        best_gini = self.gini(y)
        best_col = None
        best_thresh = None

        classes = np.unique(y)  # classes = [0, 1]
        class_to_idx = {c: i for i, c in enumerate(classes)}

        total_counts = np.zeros(len(classes))  # how many times we see each class
        for c in y:
            total_counts[class_to_idx[c]] += 1

        n = len(y)
        n_classes = len(classes)

        # the greedy column search part
        for col in self.get_candidate_columns(X, self.rand):
            values = X[:, col]
            sorted_idx = np.argsort(values)
            sorted_vals = values[sorted_idx]
            sorted_y = y[sorted_idx]

            # maintain left counts incrementally instead of recomputing gini each time
            left_counts = np.zeros(n_classes)

            for i in range(n - 1):
                left_counts[class_to_idx[sorted_y[i]]] += 1

                if sorted_vals[i] == sorted_vals[i + 1]:
                    continue

                right_counts = total_counts - left_counts
                n_left = i + 1
                n_right = n - n_left

                left_gini = 1.0 - np.sum((left_counts / n_left) ** 2)
                right_gini = 1.0 - np.sum((right_counts / n_right) ** 2)
                weighted = (n_left * left_gini + n_right * right_gini) / n

                if weighted < best_gini:
                    best_gini = weighted
                    best_col = col
                    best_thresh = (sorted_vals[i] + sorted_vals[i + 1]) / 2.0

        if best_col is None:
            return None, None, None

        mask = X[:, best_col] <= best_thresh
        return (X[mask], y[mask]), (X[~mask], y[~mask]), (best_col, best_thresh)

    def create_node(self, criterion: tuple | None, y: ndarray):
        classes, counts = np.unique(y, return_counts=True)
        majority_class = classes[np.argmax(counts)]
        return {
            "col": criterion[0] if criterion else None,
            "thresh": criterion[1] if criterion else None,
            "class": majority_class,
            "leaf": False,
            "left": None,
            "right": None,
        }

    def fit_tree(self, X: ndarray, y: ndarray):
        D_l, D_r, criterion = self.split(X, y)
        node = self.create_node(criterion, y)
        if criterion is None or len(y) < self.min_samples:
            node["leaf"] = True
            return node
        node["left"] = self.fit_tree(*D_l)  # type: ignore
        node["right"] = self.fit_tree(*D_r)  # type: ignore
        return node

    def build(self, X: ndarray, y: ndarray):
        tree = self.fit_tree(X, y)
        return TreeModel(tree)


class TreeModel:

    def __init__(self, tree: dict):
        self.tree = tree

    def predict(self, X: ndarray):
        return np.array([self.predict_one(x, self.tree) for x in X])

    def predict_one(self, x: ndarray, node: dict):
        if node["leaf"]:
            return node["class"]
        if x[node["col"]] <= node["thresh"]:
            return self.predict_one(x, node["left"])
        return self.predict_one(x, node["right"])


class RandomForest:

    def __init__(self, rand: random.Random, n=50):
        self.n = n
        self.rand = rand

    def build(self, X: ndarray, y: ndarray):
        models = []
        bootstrap_indices = []
        t = Tree(self.rand, random_sqrt_columns)
        for i in range(self.n):
            indices = self.rand.choices(range(len(y)), k=len(y))
            bootstrap_indices.append(indices)
            models.append(t.build(X[indices], y[indices]))
        return RFModel(models, X, y, bootstrap_indices, self.rand)


class RFModel:

    def __init__(
        self,
        models: list[TreeModel],
        X: ndarray,
        y: ndarray,
        bootstrap_indices: list[list[int]],
        rand: random.Random,
    ):
        self.models = models
        self.X = X
        self.y = y
        self.bootstrap_indices = bootstrap_indices
        self.rand = rand

    def predict(self, X: ndarray):
        all_preds = np.array([m.predict(X) for m in self.models])
        predictions = np.zeros(X.shape[0])
        for j in range(X.shape[0]):
            values, counts = np.unique(all_preds[:, j], return_counts=True)
            predictions[j] = values[np.argmax(counts)]
        return predictions

    def importance(self):
        n_features = self.X.shape[1]
        imps = np.zeros(n_features)

        for i, model in enumerate(self.models):
            oob_mask = np.ones(len(self.y), dtype=bool)
            oob_mask[self.bootstrap_indices[i]] = False
            if np.sum(oob_mask) == 0:
                continue

            X_oob = self.X[oob_mask]
            y_oob = self.y[oob_mask]
            base_acc = np.mean(model.predict(X_oob) == y_oob)

            for j in range(n_features):
                X_perm = X_oob.copy()
                perm_idx = list(range(len(X_perm)))
                self.rand.shuffle(perm_idx)
                X_perm[:, j] = X_perm[perm_idx, j]
                perm_acc = np.mean(model.predict(X_perm) == y_oob)
                imps[j] += base_acc - perm_acc

        imps /= len(self.models)
        return imps


def count_leaves(node):
    if node["leaf"]:
        return 1
    return count_leaves(node["left"]) + count_leaves(node["right"])


def tree_error(node, X, y):
    """Misclassified count for a subtree."""
    pred = TreeModel(node).predict(X)
    return np.sum(pred != y)


def leaf_error(node, y):
    """Misclassified count if this node were a leaf."""
    return np.sum(y != node["class"])


def attach_data(node, X, y):
    """Annotate each node with its subset of data for pruning."""
    node["data_X"] = X
    node["data_y"] = y
    if not node["leaf"]:
        mask = X[:, node["col"]] <= node["thresh"]
        attach_data(node["left"], X[mask], y[mask])
        attach_data(node["right"], X[~mask], y[~mask])


def weakest_link(node):
    """Find the internal node with smallest g(t) = (R(t) - R(T_t)) / (|T_t| - 1)."""
    if node["leaf"]:
        return None, float("inf")

    n = len(node["data_y"])
    if n == 0:
        return None, float("inf")

    leaf_err = leaf_error(node, node["data_y"]) / n
    subtree_err = tree_error(node, node["data_X"], node["data_y"]) / n
    leaves = count_leaves(node)

    if leaves > 1:
        g = (leaf_err - subtree_err) / (leaves - 1)
    else:
        g = float("inf")

    best_node, best_g = node, g

    left_node, left_g = weakest_link(node["left"])
    if left_g < best_g:
        best_node, best_g = left_node, left_g

    right_node, right_g = weakest_link(node["right"])
    if right_g < best_g:
        best_node, best_g = right_node, right_g

    return best_node, best_g


def prune_node(node, target):
    """Prune target node to a leaf (in-place)."""
    if node is target:
        node["leaf"] = True
        node["left"] = None
        node["right"] = None
        return True
    if node["leaf"]:
        return False
    return prune_node(node["left"], target) or prune_node(node["right"], target)


class BetterTree:
    """Pre-pruned tree with min_samples=5. Increasing the minimum leaf size
    prevents the tree from fitting noise in small partitions."""

    def __init__(self, rand=None, min_samples=5):
        self.rand = rand
        self.min_samples = min_samples

    def build(self, X: ndarray, y: ndarray):
        return Tree(rand=self.rand, min_samples=self.min_samples).build(X, y)


class BetterTree2:
    """PCA-reduced tree. Projects features onto top principal components before
    building the tree. Principled for spectral data where features are correlated,
    but in practice the loss of local spectral detail outweighs the denoising."""

    def __init__(self, rand=None, min_samples=5, n_components=20):
        self.rand = rand
        self.min_samples = min_samples
        self.n_components = n_components

    def build(self, X: ndarray, y: ndarray):
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        cov = X_centered.T @ X_centered / (len(X) - 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigvals)[::-1][: self.n_components]
        self.components = eigvecs[:, idx]
        X_pca = X_centered @ self.components
        model = Tree(rand=self.rand, min_samples=self.min_samples).build(X_pca, y)
        return BetterTree2Model(model, self.mean, self.components)


class BetterTree2Model:

    def __init__(self, model: TreeModel, mean: ndarray, components: ndarray):
        self.model = model
        self.mean = mean
        self.components = components

    def predict(self, X: ndarray):
        X_pca = (X - self.mean) @ self.components
        return self.model.predict(X_pca)


def misclassification_rate(y_true, y_pred):
    rate = np.mean(y_true != y_pred)
    se = np.sqrt(rate * (1 - rate) / len(y_true))
    return float(rate), float(se)


def hw_tree_full(learn, test):
    (Xt, yt), (Xv, yv) = learn, test
    model = Tree(min_samples=2).build(Xt, yt)
    return misclassification_rate(yt, model.predict(Xt)), misclassification_rate(
        yv, model.predict(Xv)
    )


def hw_randomforests(learn, test):
    (Xt, yt), (Xv, yv) = learn, test
    model = RandomForest(rand=random.Random(0), n=100).build(Xt, yt)
    return misclassification_rate(yt, model.predict(Xt)), misclassification_rate(
        yv, model.predict(Xv)
    )


def read_tab(fn, adict):
    content = list(csv.reader(open(fn, "rt"), delimiter="\t"))

    legend = content[0][1:]
    data = content[1:]

    X = np.array([d[1:] for d in data], dtype=float)
    y = np.array([adict[d[0]] for d in data])

    return legend, X, y


def tki():
    legend, Xt, yt = read_tab("tki-train.tab", {"Bcr-abl": 1, "Wild type": 0})
    _, Xv, yv = read_tab("tki-test.tab", {"Bcr-abl": 1, "Wild type": 0})
    return (Xt, yt), (Xv, yv), legend


if __name__ == "__main__":
    learn, test, legend = tki()

    print("full", hw_tree_full(learn, test))
    print("random forests", hw_randomforests(learn, test))
