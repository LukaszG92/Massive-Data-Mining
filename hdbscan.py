# run_paper.py
import os
import time
from pathlib import Path

import numpy as np

from preprint.lsh_hdbscan import LSHHDBSCANPaper


# CONFIG (modifica qui)
# ==========
INPUT_PATH = "data/embeddings/all-MiniLM-L6-v2_mean_centered.npy"
OUTPUT_LABELS = "data/cluster/LSH-HDBSCAN-NEW/labels1.npy"
OUTPUT_HIERARCHY = "data/cluster/LSH-HDBSCAN-NEW/hierarchy1.npz"

MINPTS = 10
C = 2.0
DELTA = 0.5
GAMMA = 0.3

ESTIMATE_PAIRS = 200_000
RNG_SEED = 42

VERBOSE_LSHDBSCAN = False

CENTERRATIO = None
RATIOOFFSET = None
# ==========


def main():
    # 0) Ensure output folders exist
    Path(OUTPUT_LABELS).parent.mkdir(parents=True, exist_ok=True)
    out_h_parent = Path(OUTPUT_HIERARCHY).parent
    if str(out_h_parent) not in ("", "."):
        out_h_parent.mkdir(parents=True, exist_ok=True)

    # 1) Load data
    X = np.load(INPUT_PATH)
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"Mi aspetto una matrice 2D (n_samples, n_features), trovato shape={X.shape}")

    print(f"Loaded X: shape={X.shape}, dtype={X.dtype}")

    # 2) Fit
    model = LSHHDBSCANPaper(
        minpts=int(MINPTS),
        c=float(C),
        mineps_target=0.6,
        maxeps_target=1.2,
        gamma=float(GAMMA),
        centerratio=CENTERRATIO,
        ratiooffset=RATIOOFFSET,
        estimate_pairs=int(ESTIMATE_PAIRS),
        rng_seed=int(RNG_SEED),
        verbose_levels=True,
        verbose_lshdbscan=bool(VERBOSE_LSHDBSCAN),
    )

    t0 = time.time()
    labels = model.fit_predict(X)
    t1 = time.time()

    labels = np.asarray(labels, dtype=int)

    # 3) Save labels
    np.save(OUTPUT_LABELS, labels)
    print(f"Saved labels -> {OUTPUT_LABELS}")

    # 4) Save full hierarchy (optional)
    if model.hierarchy_ is not None and len(model.hierarchy_) > 0:
        eps = np.array([h["eps"] for h in model.hierarchy_], dtype=float)
        ncl_B = np.array([h["nclusters_level"] for h in model.hierarchy_], dtype=int)
        ncl_C = np.array([h["nclusters_hierarchy"] for h in model.hierarchy_], dtype=int)

        # Attenzione: (L, n) può diventare grande se L è grande.
        labels_stack = np.stack([h["labels"] for h in model.hierarchy_], axis=0).astype(np.int32)

        np.savez_compressed(
            OUTPUT_HIERARCHY,
            eps=eps,
            nclusters_level=ncl_B,
            nclusters_hierarchy=ncl_C,
            labels_by_level=labels_stack,
            delta_hat=np.array(model.delta_hat_, dtype=float),
            dmax_hat=np.array(model.dmax_hat_, dtype=float),
            dmin_hat=np.array(model.dmin_hat_, dtype=float),
        )
        print(f"Saved hierarchy -> {OUTPUT_HIERARCHY}")

    # 5) Report
    uniq = np.unique(labels)
    k = int(np.sum(uniq != -1))  # cluster ids excluding noise
    print(f"Done in {(t1 - t0):.2f}s | clusters(excl. noise)={k}")

    if model.delta_hat_ is not None:
        print(
            f"Delta_hat={model.delta_hat_:.6g}, "
            f"Dmax_hat={model.dmax_hat_:.6g}, "
            f"dmin_hat={model.dmin_hat_:.6g}"
        )


if __name__ == "__main__":
    main()
