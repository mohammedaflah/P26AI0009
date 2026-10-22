import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from common import K, load_config, load_split, load_selected, save_json, update_selected
from q1_bagging import CustomBagging


def samme_step(w, incorrect, n_classes):

    err = np.sum(w * incorrect) / np.sum(w)
    e = min(max(err, 1e-10), 1 - 1e-10)
    alpha = np.log((1 - e) / e) + np.log(n_classes - 1)
    new_w = w * np.exp(alpha * incorrect)
    new_w /= new_w.sum()
    return err, alpha, new_w


class SAMMEAdaBoost:
    def __init__(self, max_depth=1, n_estimators=100, n_classes=K, seed=0, verbose=False):
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.n_classes = n_classes
        self.seed = seed
        self.verbose = verbose

    def fit(self, X, y):
        n = X.shape[0]
        w = np.ones(n) / n
        self.estimators, self.alphas, self.errors = [], [], []
        self.weight_history = []  
        self.train_preds = []     
        for m in range(self.n_estimators):
            dt = DecisionTreeClassifier(max_depth=self.max_depth, random_state=self.seed + m)
            dt.fit(X, y, sample_weight=w)
            pred = dt.predict(X)
            inc = pred != y
            err, alpha, new_w = samme_step(w, inc, self.n_classes)

            if err >= 1 - 1 / self.n_classes:
        
                if self.verbose:
                    print(f"round {m}: err={err:.4f} >= 1-1/K, learner discarded, stopping")
                if not self.estimators:  
                    self.estimators.append(dt)
                    self.alphas.append(1.0)
                    self.errors.append(err)
                    self.weight_history.append(w.copy())
                    self.train_preds.append(pred)
                break

            self.estimators.append(dt)
            self.alphas.append(alpha)
            self.errors.append(err)
            self.weight_history.append(w.copy())
            self.train_preds.append(pred)

            if err == 0:
                if self.verbose:
                    print(f"round {m}: perfect learner, stopping")
                break
            w = new_w
        return self

    def scores(self, X, M=None):
        M = M or len(self.estimators)
        S = np.zeros((X.shape[0], self.n_classes))
        rows = np.arange(X.shape[0])
        for dt, a in zip(self.estimators[:M], self.alphas[:M]):
            S[rows, dt.predict(X)] += a
        return S

    def predict_proba(self, X, M=None):
        M = M or len(self.estimators)
        return self.scores(X, M) / sum(self.alphas[:M])

    def predict(self, X, M=None):
        return self.scores(X, M).argmax(axis=1)


def toy_example(seed):
    print("Q2a toy SAMME round")
    X = np.array([[1.0, 2.0], [1.5, 1.8], [2.0, 2.5],
                  [5.0, 5.0], [5.5, 6.0], [6.0, 5.2],
                  [8.0, 1.0], [8.5, 1.5], [9.0, 0.8], [8.2, 2.0]])
    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
    w = np.ones(len(y)) / len(y)
    stump = DecisionTreeClassifier(max_depth=1, random_state=seed).fit(X, y, sample_weight=w)
    pred = stump.predict(X)
    inc = pred != y
    err, alpha, new_w = samme_step(w, inc, 3)
    log = {"w0": w, "pred": pred, "incorrect": inc.astype(int), "eps": err,
           "alpha": alpha, "w1": new_w}
    for k, v in log.items():
        print(f"  {k}: {np.round(v, 4)}")
    return log


def run_q2():
    cfg = load_config()
    seed = cfg["seed"]
    out = {"toy": toy_example(seed)}
    d = load_split()
    X_tr, y_tr = d["X_train"], d["y_train"]
    X_val, y_val = d["X_val"], d["y_val"]
    X_te, y_te = d["X_test_clean"], d["y_test"]
    X_mask = d["X_test_masked"]
    depths, rounds = cfg["boosting_depths"], cfg["boosting_rounds"]

    # Q2b
    print("\nQ2b grid")
    best, grid = None, []
    for dep in depths:
        model = SAMMEAdaBoost(dep, max(rounds), seed=seed, verbose=True).fit(X_tr, y_tr)
        for M in rounds:
            tr = np.mean(model.predict(X_tr, M) == y_tr)
            va = np.mean(model.predict(X_val, M) == y_val)
            grid.append({"depth": dep, "rounds": M, "train": tr, "val": va})
            print(f"depth {dep} rounds {M:3d} | train {tr:.4f} val {va:.4f}")
            if best is None or va > best[0]:  
                best = (va, dep, M)
    _, best_d, best_M = best
    print(f"Selected: depth={best_d}, rounds={best_M} (val {best[0]:.4f})")
    out["grid"], out["selected"] = grid, {"depth": best_d, "rounds": best_M}
    update_selected(boost_depth=best_d, boost_rounds=best_M)

    boost = SAMMEAdaBoost(best_d, best_M, seed=seed).fit(X_tr, y_tr)
    W = np.array(boost.weight_history)
    rr = np.arange(1, len(W) + 1)
    top10 = np.sort(W, axis=1)[:, -10:].sum(axis=1)
    fig, ax = plt.subplots(2, 2, figsize=(9, 6))
    for a, (vals, title) in zip(ax.ravel(), [(boost.errors, "Weighted training error"),
                                             (boost.alphas, "Learner weight alpha"),
                                             (W.max(axis=1), "Max sample weight"),
                                             (top10, "Weight on 10 heaviest samples")]):
        a.plot(rr, vals)
        a.set_title(title)
        a.set_xlabel("Round")
        a.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("results/q2b_boosting_dynamics.png", dpi=300)
    plt.close()

    #Q2c
    print("\nQ2c boosting vs bagging")
    sel = load_selected()
    tree_params = sel["tree"]
    comp = []
    for L in [25, 50, 100]:
        t0 = time.perf_counter()
        bm = SAMMEAdaBoost(best_d, L, seed=seed).fit(X_tr, y_tr)
        t_boost = time.perf_counter() - t0
        t0 = time.perf_counter()
        bg = CustomBagging(tree_params, L, seed=seed).fit(X_tr, y_tr)
        t_bag = time.perf_counter() - t0
        row = {"learners": L,
               "boost": {"train": np.mean(bm.predict(X_tr) == y_tr), "val": np.mean(bm.predict(X_val) == y_val),
                         "test": np.mean(bm.predict(X_te) == y_te), "time_s": t_boost},
               "bag": {"train": np.mean(bg.predict(X_tr) == y_tr), "val": np.mean(bg.predict(X_val) == y_val),
                       "test": np.mean(bg.predict(X_te) == y_te), "time_s": t_bag}}
        comp.append(row)
        print(row)
    out["comparison"] = comp

    h = int(W.max(axis=0).argmax())
    hist_correct = np.array([p[h] == y_tr[h] for p in boost.train_preds])
    heavy = {"train_index": h, "label": y_tr[h], "max_weight": W[:, h].max(),
             "round_of_max_weight": int(W[:, h].argmax()) + 1,
             "rounds_misclassified": int((~hist_correct).sum()), "total_rounds": len(hist_correct),
             "first_20_rounds_correct": hist_correct[:20].astype(int),
             "first_20_weights": W[:20, h]}
    print("Heavy sample:", heavy)
    out["heavy_sample"] = heavy

    # Q2d
    print("\nQ2d clean vs masked")
    tree = DecisionTreeClassifier(**tree_params, random_state=seed).fit(X_tr, y_tr)
    Bfin = sel["bag_B"]
    bag = CustomBagging(tree_params, n_trees=max(cfg["bagging_sizes"]), seed=seed).fit(X_tr, y_tr)

    res = {}
    for name, X in [("clean", X_te), ("masked", X_mask)]:
        res[name] = {"tree_acc": tree.score(X, y_te), "tree_conf": tree.predict_proba(X).max(axis=1).mean(),
                     "bag_acc": np.mean(bag.predict(X, Bfin) == y_te), "bag_conf": bag.confidence(X, Bfin).mean(),
                     "boost_acc": np.mean(boost.predict(X) == y_te), "boost_conf": boost.predict_proba(X).max(axis=1).mean()}
        print(name, res[name])
    out["clean_vs_masked"] = res

    bag_m, boost_m = bag.predict(X_mask, Bfin), boost.predict(X_mask)
    bag_c, boost_c = bag.predict(X_te, Bfin), boost.predict(X_te)
    cand = np.where((bag_c == y_te) & (boost_c == y_te) & ((bag_m == y_te) != (boost_m == y_te)))[0]
    if len(cand) == 0: 
        cand = np.where((bag_m == y_te) != (boost_m == y_te))[0]
    if len(cand):
        i = int(cand[0])
        v = bag.votes(X_mask[i:i + 1], Bfin)[0]
        pair_dis = 1 - np.sum(v * (v - 1)) / (Bfin * (Bfin - 1)) if Bfin > 1 else 0.0
        pb = boost.predict_proba(X_mask[i:i + 1])[0]
        true_w = W[:, y_tr == y_te[i]].mean()
        case = {"test_index": i, "true": y_te[i], "bag_pred_masked": bag_m[i], "boost_pred_masked": boost_m[i],
                "bag_top_votes": {int(k): int(v[k]) for k in np.argsort(v)[::-1][:4]},
                "bag_tree_pairwise_disagreement": pair_dis,
                "boost_top_proba": {int(k): float(pb[k]) for k in np.argsort(pb)[::-1][:4]},
                "mean_boost_weight_of_true_class_train_images": true_w,
                "uniform_weight": 1 / len(y_tr)}
        print("Different response:", case)
        out["different_response_case"] = case
    save_json(out, "results/q2_results.json")


if __name__ == "__main__":
    run_q2()