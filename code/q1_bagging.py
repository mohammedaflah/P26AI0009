import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from common import (K, load_config, load_split, save_json, update_selected,
                    votes_from_preds)


def analyze_tree(tree):
    t = tree.tree_
    leaves = np.where(t.children_left == -1)[0]
    samples = t.n_node_samples[leaves]  
    return len(leaves), samples.mean(), np.mean(samples == 1)


class CustomBagging:
    def __init__(self, base_params, n_trees=100, seed=0):
        self.base_params = base_params
        self.n_trees = n_trees
        self.rng = np.random.default_rng(seed)
        self.trees = []
        self.boot_indices = []
        self.in_bag = []

    def fit(self, X, y):
        n = X.shape[0]
        self.trees, self.boot_indices, self.in_bag = [], [], []
        for _ in range(self.n_trees):
            idx = self.rng.choice(n, size=n, replace=True)
            mask = np.zeros(n, dtype=bool)
            mask[idx] = True
            dt = DecisionTreeClassifier(**self.base_params,
                                        random_state=int(self.rng.integers(2**31 - 1)))
            dt.fit(X[idx], y[idx])
            self.boot_indices.append(idx)
            self.in_bag.append(mask)
            self.trees.append(dt)
        return self

    def _preds(self, X, B=None):
        B = B or self.n_trees
        return np.array([t.predict(X) for t in self.trees[:B]])

    def votes(self, X, B=None):
        return votes_from_preds(self._preds(X, B))

    def proba(self, X, B=None):
        B = B or self.n_trees
        return self.votes(X, B) / B

    def predict(self, X, B=None):
        return self.votes(X, B).argmax(axis=1)  

    def confidence(self, X, B=None):
        return self.proba(X, B).max(axis=1)

    def oob_accuracy(self, X, y, B=None):
        B = B or self.n_trees
        votes = np.zeros((len(X), K))
        for b in range(B):
            oob = ~self.in_bag[b]
            if oob.any():
                votes[np.where(oob)[0], self.trees[b].predict(X[oob])] += 1
        seen = votes.sum(axis=1) > 0
        return np.mean(votes[seen].argmax(axis=1) == y[seen])

    def pairwise_disagreement(self, X, B=None):
        P = self._preds(X, B)
        B = len(P)
        if B < 2:
            return 0.0
        total = sum((P[i + 1:] != P[i]).mean(axis=1).sum() for i in range(B - 1))
        return total / (B * (B - 1) / 2)

    def mean_margin(self, X, B=None):
        p = np.sort(self.proba(X, B), axis=1)
        return np.mean(p[:, -1] - p[:, -2])


def build_bag(cfg, params, X_tr, y_tr):

    return CustomBagging(params, n_trees=max(cfg["bagging_sizes"]), seed=cfg["seed"]).fit(X_tr, y_tr)


def run_q1():
    cfg = load_config()
    seed = cfg["seed"]
    d = load_split()
    X_tr, y_tr = d["X_train"], d["y_train"]
    X_val, y_val = d["X_val"], d["y_val"]
    X_te, y_te = d["X_test_clean"], d["y_test"]
    X_mask = d["X_test_masked"]
    out = {}

    # Q1a
    print("Q1a single tree grid")
    best, grid = None, []
    for md in cfg["tree_depths"]:
        for msl in cfg["minimum_leaf_sizes"]:
            dt = DecisionTreeClassifier(max_depth=md, min_samples_leaf=msl, random_state=seed)
            dt.fit(X_tr, y_tr)
            va = dt.score(X_val, y_val)
            nl, ms, fs = analyze_tree(dt)
            grid.append({"max_depth": md, "min_samples_leaf": msl, "train": dt.score(X_tr, y_tr),
                         "val": va, "test": dt.score(X_te, y_te),
                         "leaves": nl, "mean_samples_per_leaf": ms, "frac_single": fs})
            print(f"depth={md}, msl={msl} | tr {grid[-1]['train']:.4f} val {va:.4f} "
                  f"te {grid[-1]['test']:.4f} | leaves {nl} mean/leaf {ms:.2f} single {fs:.3f}")
            if best is None or va > best[0]:  # ties keep the earlier (shallower, smaller msl) setting
                best = (va, {"max_depth": md, "min_samples_leaf": msl}, dt)
    _, params, tree = best
    print("Frozen setting:", params)
    out["grid"], out["frozen_params"] = grid, params
    update_selected(tree=params)

    # Q1b,Q1c
    print("\nQ1b,Q1c bagging")
    bag = build_bag(cfg, params, X_tr, y_tr)
    n = len(X_tr)
    theory = n * (1 - (1 - 1 / n) ** n)
    print(f"Theoretical distinct examples per bootstrap: {theory:.2f} (n={n})")
    Bs, rows = cfg["bagging_sizes"], []
    for B in Bs:
        row = {"B": B,
               "train": np.mean(bag.predict(X_tr, B) == y_tr),
               "val": np.mean(bag.predict(X_val, B) == y_val),
               "test": np.mean(bag.predict(X_te, B) == y_te),
               "avg_unique": np.mean([len(np.unique(i)) for i in bag.boot_indices[:B]]),
               "oob": bag.oob_accuracy(X_tr, y_tr, B),
               "disagreement": bag.pairwise_disagreement(X_val, B),
               "margin": bag.mean_margin(X_val, B)}
        rows.append(row)
        print(f"B={B:3d} tr {row['train']:.4f} val {row['val']:.4f} te {row['test']:.4f} | "
              f"unique {row['avg_unique']:.2f} | OOB {row['oob']:.4f} "
              f"dis {row['disagreement']:.4f} margin {row['margin']:.4f}")
    out["bagging"], out["theory_unique"] = rows, theory

    plt.figure(figsize=(7, 4.5))
    plt.plot(Bs, [r["oob"] for r in rows], "o-", label="OOB accuracy")
    plt.plot(Bs, [r["disagreement"] for r in rows], "s-", label="Pairwise disagreement")
    plt.plot(Bs, [r["margin"] for r in rows], "^-", label="Mean vote margin")
    plt.xscale("log")
    plt.xlabel("Ensemble size B")
    plt.ylabel("Score")
    plt.title("Bagging diagnostics vs B")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/q1c_diagnostics.png", dpi=300)
    plt.close()

    final_B = max(rows, key=lambda r: (r["val"], -r["B"]))["B"]  
    print("Final B (validation):", final_B)
    out["final_B"] = final_B
    update_selected(bag_B=final_B)

    #Q1d
    print("\nQ1d clean vs masked")
    t_conf = lambda X: tree.predict_proba(X).max(axis=1)
    res = {}
    for name, X in [("clean", X_te), ("masked", X_mask)]:
        res[name] = {"tree_acc": tree.score(X, y_te), "tree_conf": t_conf(X).mean(),
                     "bag_acc": np.mean(bag.predict(X, final_B) == y_te),
                     "bag_conf": bag.confidence(X, final_B).mean()}
        print(name, res[name])
    out["clean_vs_masked"] = res

    tp = tree.predict(X_te)
    bp = bag.predict(X_te, final_B)
    votes = bag.votes(X_te, final_B)
    fixed = np.where((tp != y_te) & (bp == y_te))[0][:2]
    both_wrong = np.where((tp != y_te) & (bp != y_te))[0][:2]

    def describe(i):
        top = np.argsort(votes[i])[::-1][:4]
        return {"test_index": i, "true": y_te[i], "tree_pred": tp[i], "bag_pred": bp[i],
                "top_votes": {int(k): int(votes[i, k]) for k in top}}

    cases = {"bagging_fixes_tree": [describe(i) for i in fixed],
             "both_wrong": [describe(i) for i in both_wrong]}
    for k, v in cases.items():
        print(k)
        for c in v:
            print("  ", c)
    out["cases"] = cases
    save_json(out, "results/q1_results.json")


if __name__ == "__main__":
    run_q1()