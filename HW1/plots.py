import random
import numpy as np
import matplotlib.pyplot as plt
from hw_tree import Tree, RandomForest, tki, misclassification_rate


def plot_misclassification_vs_n():
    learn, test, legend = tki()
    (Xt, yt), (Xv, yv) = learn, test

    ns = list(range(1, 101))
    train_rates = []
    test_rates = []

    for n in ns:
        model = RandomForest(rand=random.Random(0), n=n).build(Xt, yt)
        tr, _ = misclassification_rate(yt, model.predict(Xt))
        te, _ = misclassification_rate(yv, model.predict(Xv))
        train_rates.append(tr)
        test_rates.append(te)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ns, train_rates, label="Train")
    ax.plot(ns, test_rates, label="Test")
    ax.set_xlabel("Number of trees (n)")
    ax.set_ylabel("Misclassification rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig("misclassification_vs_n.pdf")
    plt.close()


def plot_importance():
    learn, test, legend = tki()
    (Xt, yt), (Xv, yv) = learn, test

    # RF importance
    rf_model = RandomForest(rand=random.Random(0), n=100).build(Xt, yt)
    rf_imp = rf_model.importance()

    # Root variables from 100 non-random trees on randomized data
    root_counts = np.zeros(Xt.shape[1])
    for i in range(100):
        rng = random.Random(i)
        perm = list(range(len(yt)))
        rng.shuffle(perm)
        y_shuffled = yt[perm]
        tree = Tree(min_samples=2).build(Xt, y_shuffled)
        root_col = tree.tree["col"]
        if root_col is not None:
            root_counts[root_col] += 1
    root_counts /= 100

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(legend))
    ax.bar(x - 0.2, rf_imp, width=0.4, label="RF permutation importance")
    ax.bar(x + 0.2, root_counts, width=0.4, label="Root variable frequency (shuffled)")
    ax.set_xlabel("Variable")
    ax.set_ylabel("Importance")
    ax.set_xticks(x[::max(1, len(x)//20)])
    ax.set_xticklabels([legend[i] for i in x[::max(1, len(x)//20)]], rotation=90, fontsize=6)
    ax.legend()
    fig.tight_layout()
    fig.savefig("importance.pdf")
    plt.close()


if __name__ == "__main__":
    plot_misclassification_vs_n()
    plot_importance()
