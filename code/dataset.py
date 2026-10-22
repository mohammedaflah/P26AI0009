import os
import json
import numpy as np
from sklearn.datasets import fetch_olivetti_faces

def prepare_data():
    with open("configs/resolved.json", "r") as f:
        config = json.load(f)
    
    seed = config["seed"]
    rng = np.random.default_rng(seed)
    
    os.makedirs("data", exist_ok=True)
    
    faces = fetch_olivetti_faces(shuffle=False)
    X = faces.data
    y = faces.target
    images = faces.images
    
    np.savez_compressed("data/olivetti.npz", images=images, data=X, target=y)
    
    tr_list, val_list, te_list = [], [], []
    for c in range(40):
        cls_idx = np.where(y == c)[0]
        perm = rng.permutation(cls_idx)
        tr_list.extend(perm[:6])
        val_list.extend(perm[6:8])
        te_list.extend(perm[8:10])
        
    train_idx = np.array(tr_list, dtype=np.int64)
    val_idx = np.array(val_list, dtype=np.int64)
    test_idx = np.array(te_list, dtype=np.int64)
    
    X_test_clean = X[test_idx].copy()
    masked_imgs = images[test_idx].copy()
    masked_imgs[:, 24:40, 24:40] = 0.0
    X_test_masked = masked_imgs.reshape(-1, 4096)
    
    np.savez_compressed(
        "data/split.npz",
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        X_train=X[train_idx],
        y_train=y[train_idx],
        X_val=X[val_idx],
        y_val=y[val_idx],
        X_test_clean=X_test_clean,
        X_test_masked=X_test_masked,
        y_test=y[test_idx]
    )
    
    meta = {
        "student_name": config["student_name"],
        "roll_number": config["roll_number"],
        "seed": seed,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx)
    }
    with open("data/split_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
        
    n_resamples = config.get("resamples", 30)
    resamples = np.zeros((n_resamples, 240), dtype=np.int64)
    
    for r in range(n_resamples):
        sample = []
        for c in range(40):
            c_train_local = np.arange(c * 6, (c + 1) * 6)
            sample.extend(rng.choice(c_train_local, size=6, replace=True))
        resamples[r] = np.array(sample)
        
    np.savez_compressed("data/resamples.npz", resampled_indices=resamples)
    print("Dataset generating done.")

if __name__ == "__main__":
    prepare_data()