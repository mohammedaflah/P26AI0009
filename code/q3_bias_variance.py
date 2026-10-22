import os
import numpy as np
from joblib import Parallel, delayed
from sklearn.tree import DecisionTreeClassifier
from common import (K, load_config, load_split, load_selected, save_json,
                    one_hot, bias_variance)
from q1_bagging import CustomBagging, analyze_tree
from q2_boosting import SAMMEAdaBoost


def make_resamples(y_tr, R, seed):

    rng = np.random.default_rng(seed + 1000)
    out = []
    for _ in range(R):
        idx = [rng.choice(np.where(y_tr == c)[0], size=(y_tr == c).sum(), replace=True) for c in range(K)]
        out.append(np.concatenate(idx))
    return np.array(out)


def refit_once(kind, params, extra, X_tr, y_tr, idx, evals, seed):
    Xb, yb = X_tr[idx], y_tr[idx]
    if kind == "tree":
        m = DecisionTreeClassifier(**params, random_state=seed).fit(Xb, yb)
        return [one_hot(m.predict(X)) for X in evals]
    if kind == "bag":
        m = CustomBagging(params, extra["B"], seed).fit(Xb, yb)
        return [m.proba(X) for X in evals]
    m = SAMMEAdaBoost(extra["depth"], extra["M"], seed=seed).fit(Xb, yb)
    return [m.predict_proba(X) for X in evals]


def collect(kind, params, extra, X_tr, y_tr, resamples, evals, cfg):
    res = Parallel(n_jobs=cfg["n_jobs"])(
        delayed(refit_once)(kind, params, extra, X_tr, y_tr, resamples[r], evals, cfg["seed"] + r)
        for r in range(len(resamples)))
    return [np.array([r[j] for r in res]) for j in range(len(evals))]  


def summarize(clean, masked, y_te):
    b2, var, acc, std = bias_variance(clean, y_te)
    macc = np.mean([np.mean(masked[r].argmax(axis=1) == y_te) for r in range(len(masked))])
    return {"bias2": b2, "variance": var, "acc_mean": acc, "acc_std": std, "masked_acc_mean": macc}


def run_q3():
    cfg = load_config()
    seed = cfg["seed"]
    sel = load_selected()
    d = load_split()
    X_tr, y_tr = d["X_train"], d["y_train"]
    X_te, y_te, X_mask = d["X_test_clean"], d["y_test"], d["X_test_masked"]
    evals = [X_te, X_mask]

    os.makedirs("data", exist_ok=True)
    path = "data/resamples.npz"
    if os.path.exists(path):
        resamples = np.load(path)["resampled_indices"]
    else:
        resamples = make_resamples(y_tr, cfg["resamples"], seed)
        np.savez(path, resampled_indices=resamples)
    print(f"{len(resamples)} resamples")

    tree_params, B = sel["tree"], sel["bag_B"]
    out = {}


    print("Q3a,Q3d selected models")
    table = {}
    for name, kind, extra in [("single_tree", "tree", {}),
                              ("bagging", "bag", {"B": B}),
                              ("boosting", "boost", {"depth": sel["boost_depth"], "M": sel["boost_rounds"]})]:
        clean, masked = collect(kind, tree_params, extra, X_tr, y_tr, resamples, evals, cfg)
        table[name] = summarize(clean, masked, y_te)
        print(name, {k: round(float(v), 4) for k, v in table[name].items()})
    out["selected_models"] = table

    # Q3b,Q3c
    print("\nQ3b depth sweep")
    sel_depth = tree_params["max_depth"]
    mid = sel_depth if sel_depth not in (None, 2) else 8
    depths = [2, mid, None]
    msl = tree_params["min_samples_leaf"]
    sweep = []
    for dep in depths:
        p = {"max_depth": dep, "min_samples_leaf": msl}
        t_clean, _ = collect("tree", p, {}, X_tr, y_tr, resamples, evals, cfg)
        b_clean, _ = collect("bag", p, {"B": 100}, X_tr, y_tr, resamples, evals, cfg)
        t = bias_variance(t_clean, y_te)
        b = bias_variance(b_clean, y_te)

        
        ref = DecisionTreeClassifier(**p, random_state=seed).fit(X_tr, y_tr)
        leaves, mean_leaf, frac_single = analyze_tree(ref)
        dis = CustomBagging(p, 100, seed).fit(X_tr, y_tr).pairwise_disagreement(X_te)

        row = {"depth": dep, "tree": {"bias2": t[0], "var": t[1], "acc": t[2]},
               "bag100": {"bias2": b[0], "var": b[1], "acc": b[2]},
               "variance_removed": t[1] - b[1], "variance_removed_pct": 100 * (t[1] - b[1]) / t[1],
               "acc_gain": b[2] - t[2], "leaves": leaves, "mean_samples_per_leaf": mean_leaf,
               "frac_single_leaf": frac_single, "bag_test_disagreement": dis}
        sweep.append(row)
        print(row)
    out["depth_sweep"] = sweep
    best_gain = max(sweep, key=lambda r: r["acc_gain"])
    print("Depth with the largest accuracy gain from bagging:", best_gain["depth"])
    out["largest_gain_depth"] = best_gain["depth"]

    save_json(out, "results/q3_results.json")
    print("\nQ3d final table")
    print(f"{'model':12s} clean_acc masked_acc bias2   variance")
    for name, r in table.items():
        print(f"{name:12s} {r['acc_mean']:.4f}    {r['masked_acc_mean']:.4f}     "
              f"{r['bias2']:.4f}  {r['variance']:.4f}")


if __name__ == "__main__":
    run_q3()