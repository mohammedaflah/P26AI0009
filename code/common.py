import json
import os
import numpy as np

K = 40  


def load_config(path="configs/resolved.json"):
    with open(path) as f:
        return json.load(f)


def load_split():
    return np.load("data/split.npz")


def _to_builtin(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    return str(o)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_to_builtin)


def load_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


def update_selected(**kwargs): 
    sel = load_json("results/selected.json", {})
    sel.update(kwargs)
    save_json(sel, "results/selected.json")


def load_selected():
    sel = load_json("results/selected.json")
    if sel is None:
        raise RuntimeError("results/selected.json missing...")
    return sel


def one_hot(labels, n_classes=K):
    out = np.zeros((len(labels), n_classes))
    out[np.arange(len(labels)), labels] = 1.0
    return out


def votes_from_preds(preds, n_classes=K):
    N = preds.shape[1]
    votes = np.zeros((N, n_classes))
    for p in preds:
        votes[np.arange(N), p] += 1
    return votes


def bias_variance(prob, y_true):
    R, N, _ = prob.shape
    e = one_hot(y_true, prob.shape[2])
    p_bar = prob.mean(axis=0)
    bias2 = np.mean(np.sum((p_bar - e) ** 2, axis=1))
    variance = np.mean(np.sum((prob - p_bar) ** 2, axis=2))
    accs = np.array([np.mean(prob[r].argmax(axis=1) == y_true) for r in range(R)])
    return bias2, variance, accs.mean(), accs.std()  