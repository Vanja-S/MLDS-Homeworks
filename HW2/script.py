import numpy as np
import pandas as pd
from numpy.typing import NDArray, ArrayLike
from typing import Any, Tuple
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def read_csv(path="dataset.csv"):
    df = pd.read_csv(path, sep=";")
    y = df["ShotType"].values
    X = df.drop(columns=["ShotType"])

    angle = df["Angle"].values
    competition = df["Competition"].values

    cat_cols = X.select_dtypes(include="str").columns
    num_cols = X.select_dtypes(exclude="str").columns

    enc = OneHotEncoder(sparse_output=False, drop="first")
    X_cat = enc.fit_transform(X[cat_cols])
    X_num = X[num_cols].values.astype(float)

    X_out = np.hstack([X_num, X_cat])

    le = LabelEncoder()
    y_out = le.fit_transform(y)

    return X_out, y_out, le, angle, competition


def cross_validate(X, y, model_fn, k=10, seed=42):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(y))
    folds = np.array_split(indices, k)

    accuracies = []
    log_scores = []
    sample_indices = []
    sample_correct = []
    sample_log_scores = []

    for i in range(k):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        model = model_fn()
        model.fit(X_train, y_train)

        probs = model.predict_proba(X_test)
        preds = model.predict(X_test)

        acc = np.mean(preds == y_test)
        accuracies.append(acc)

        true_probs = probs[np.arange(len(y_test)), y_test]
        per_sample_log = np.log(np.clip(true_probs, 1e-15, None))
        log_scores.append(np.mean(per_sample_log))

        sample_indices.append(test_idx)
        sample_correct.append(preds == y_test)
        sample_log_scores.append(per_sample_log)

    return {
        "accuracy": (np.mean(accuracies), np.std(accuracies)),
        "log_score": (np.mean(log_scores), np.std(log_scores)),
        "sample_indices": np.concatenate(sample_indices),
        "sample_correct": np.concatenate(sample_correct),
        "sample_log_scores": np.concatenate(sample_log_scores),
    }


def cv_optimizing_training_fold_performance(X, y, k=10, seed=42):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(y))
    folds = np.array_split(indices, k)
    max_depths = [2, 5, 10, 20, 50, None]

    accuracies = []
    log_scores = []
    chosen_depths = []
    sample_indices = []
    sample_correct = []
    sample_log_scores = []

    for i in range(k):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        best_train_acc = -1
        best_model = None
        best_depth = None
        for depth in max_depths:
            model = RandomForestClassifier(
                n_estimators=500,
                criterion="gini",
                bootstrap=True,
                max_depth=depth,
            )
            model.fit(X_train, y_train)
            train_acc = np.mean(model.predict(X_train) == y_train)
            if train_acc > best_train_acc:
                best_train_acc = train_acc
                best_model = model
                best_depth = depth

        chosen_depths.append(best_depth)

        probs = best_model.predict_proba(X_test)
        preds = best_model.predict(X_test)

        true_probs = probs[np.arange(len(y_test)), y_test]
        per_sample_log = np.log(np.clip(true_probs, 1e-15, None))

        accuracies.append(np.mean(preds == y_test))
        log_scores.append(np.mean(per_sample_log))

        sample_indices.append(test_idx)
        sample_correct.append(preds == y_test)
        sample_log_scores.append(per_sample_log)

    return {
        "accuracy": (np.mean(accuracies), np.std(accuracies)),
        "log_score": (np.mean(log_scores), np.std(log_scores)),
        "chosen_depths": chosen_depths,
        "sample_indices": np.concatenate(sample_indices),
        "sample_correct": np.concatenate(sample_correct),
        "sample_log_scores": np.concatenate(sample_log_scores),
    }


def cv_nested(X, y, k=10, inner_k=5, seed=42):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(y))
    folds = np.array_split(indices, k)
    max_depths = [2, 5, 10, 20, 50, None]

    accuracies = []
    log_scores = []
    chosen_depths = []
    sample_indices = []
    sample_correct = []
    sample_log_scores = []

    for i in range(k):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        inner_indices = rng.permutation(len(y_train))
        inner_folds = np.array_split(inner_indices, inner_k)

        best_inner_acc = -1
        best_depth = None
        for depth in max_depths:
            inner_accs = []
            for j in range(inner_k):
                inner_val_idx = inner_folds[j]
                inner_train_idx = np.concatenate([inner_folds[m] for m in range(inner_k) if m != j])

                X_inner_train = X_train[inner_train_idx]
                y_inner_train = y_train[inner_train_idx]
                X_inner_val = X_train[inner_val_idx]
                y_inner_val = y_train[inner_val_idx]

                model = RandomForestClassifier(
                    n_estimators=500,
                    criterion="gini",
                    bootstrap=True,
                    max_depth=depth,
                )
                model.fit(X_inner_train, y_inner_train)
                inner_accs.append(np.mean(model.predict(X_inner_val) == y_inner_val))

            mean_inner_acc = np.mean(inner_accs)
            if mean_inner_acc > best_inner_acc:
                best_inner_acc = mean_inner_acc
                best_depth = depth

        chosen_depths.append(best_depth)

        best_model = RandomForestClassifier(
            n_estimators=500,
            criterion="gini",
            bootstrap=True,
            max_depth=best_depth,
        )
        best_model.fit(X_train, y_train)

        probs = best_model.predict_proba(X_test)
        preds = best_model.predict(X_test)

        true_probs = probs[np.arange(len(y_test)), y_test]
        per_sample_log = np.log(np.clip(true_probs, 1e-15, None))

        accuracies.append(np.mean(preds == y_test))
        log_scores.append(np.mean(per_sample_log))

        sample_indices.append(test_idx)
        sample_correct.append(preds == y_test)
        sample_log_scores.append(per_sample_log)

    return {
        "accuracy": (np.mean(accuracies), np.std(accuracies)),
        "log_score": (np.mean(log_scores), np.std(log_scores)),
        "chosen_depths": chosen_depths,
        "sample_indices": np.concatenate(sample_indices),
        "sample_correct": np.concatenate(sample_correct),
        "sample_log_scores": np.concatenate(sample_log_scores),
    }


def baseline():
    return DummyClassifier(strategy="prior")


def logistic_regression():
    return LogisticRegression(max_iter=5000)


def analyze_error_vs_angle(all_results, angle):
    print("\n=== Part 2a: Error vs Angle ===\n")
    n_bins = 5
    bin_edges = np.linspace(angle.min(), angle.max(), n_bins + 1)

    for name, res in all_results.items():
        idx = res["sample_indices"]
        correct = res["sample_correct"]
        sample_angles = angle[idx]

        print(f"{name}:")
        for b in range(n_bins):
            mask = (sample_angles >= bin_edges[b]) & (sample_angles < bin_edges[b + 1])
            if b == n_bins - 1:
                mask |= sample_angles == bin_edges[b + 1]
            if mask.sum() > 0:
                err_rate = 1 - np.mean(correct[mask])
                print(f"  Angle [{bin_edges[b]:5.1f}, {bin_edges[b+1]:5.1f}]: "
                      f"error rate = {err_rate:.4f} (n={mask.sum()})")
        print()


def analyze_reweighted_competition(all_results, competition):
    print("=== Part 2b: Reweighted Competition ===\n")
    true_freq = {"NBA": 0.6, "EURO": 0.1, "SLO1": 0.1, "U14": 0.1, "U16": 0.1}

    comp_types, comp_counts = np.unique(competition, return_counts=True)
    dataset_freq = dict(zip(comp_types, comp_counts / len(competition)))

    weights = np.array([true_freq[c] / dataset_freq[c] for c in competition])

    for name, res in all_results.items():
        idx = res["sample_indices"]
        correct = res["sample_correct"]
        sample_log = res["sample_log_scores"]
        w = weights[idx]

        weighted_acc = np.average(correct, weights=w)
        weighted_log = np.average(sample_log, weights=w)
        print(f"{name}:")
        print(f"  Weighted Accuracy:  {weighted_acc:.4f}")
        print(f"  Weighted Log-score: {weighted_log:.4f}")
        print()


def main():
    X, y, le, angle, competition = read_csv()
    n_classes = len(le.classes_)
    print(f"Classes ({n_classes}): {list(le.classes_)}")
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features\n")

    baseline_models = {
        "Baseline": baseline,
        "Logistic Regression": logistic_regression,
    }

    all_results = {}
    for name, model_fn in baseline_models.items():
        res = cross_validate(X, y, model_fn)
        all_results[name] = res
        print(f"{name}:")
        print(f"  Accuracy:  {res['accuracy'][0]:.4f} ± {res['accuracy'][1]:.4f}")
        print(f"  Log-score: {res['log_score'][0]:.4f} ± {res['log_score'][1]:.4f}")
        print()

    res_met_1 = cv_optimizing_training_fold_performance(X, y)
    all_results["RF - train fold"] = res_met_1
    print(f"Random Forest Classifier - optimizing training fold performance:")
    print(
        f"  Accuracy:  {res_met_1['accuracy'][0]:.4f} ± {res_met_1['accuracy'][1]:.4f}"
    )
    print(
        f"  Log-score: {res_met_1['log_score'][0]:.4f} ± {res_met_1['log_score'][1]:.4f}"
    )
    print(f"  Chosen depths per fold: {res_met_1['chosen_depths']}")

    res_met_2 = cv_nested(X, y)
    all_results["RF - nested CV"] = res_met_2
    print(f"\nRandom Forest Classifier - nested CV:")
    print(
        f"  Accuracy:  {res_met_2['accuracy'][0]:.4f} ± {res_met_2['accuracy'][1]:.4f}"
    )
    print(
        f"  Log-score: {res_met_2['log_score'][0]:.4f} ± {res_met_2['log_score'][1]:.4f}"
    )
    print(f"  Chosen depths per fold: {res_met_2['chosen_depths']}")

    analyze_error_vs_angle(all_results, angle)
    analyze_reweighted_competition(all_results, competition)


if __name__ == "__main__":
    main()
