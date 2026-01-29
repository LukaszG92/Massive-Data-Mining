# run_srr_vs_kmeans.py
import json
from pathlib import Path
import numpy as np
import dbscan_srr as dbscan

from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)

# ---- Config (dalla run scelta) ----
INPUT_NPY = "data/embeddings/all-MiniLM-L6-v2_mean_centered.npy"  # [file:18]
p_miss   = 0.01                                     # delta [file:18]
L_gb     = 3.0                                      # [file:18]
threads  = 24                                       # [file:18]
eps      = 0.851339864730835                        # [file:18]
minPts   = 10                                       # [file:18]

OUTDIR = Path("data/cluster")

# ---- Load ----
X = np.load(INPUT_NPY)

# ---- SRRDBSCAN ----
srr = dbscan.SRR()
labels_srr = srr.fit_predict(X, p_miss, L_gb, threads, eps, minPts)

# statistiche (può essere dict o stringa: salviamo robustamente)
stats = srr.statistics()
(OUTDIR / "srr_statistics.json").write_text(
    json.dumps(stats, indent=2) if isinstance(stats, (dict, list)) else json.dumps({"statistics": str(stats)}, indent=2)
)
np.save(OUTDIR / "right_labels_10mpts_0.85eps.npy", labels_srr)

# metadata base
meta = {
    "input_npy": INPUT_NPY,
    "params": {"p_miss": p_miss, "L_gb": L_gb, "threads": threads, "eps": eps, "minPts": minPts},
    "n": int(X.shape[0]),
    "d": int(X.shape[1]),
    "unique_labels": int(np.unique(labels_srr).size),
}
(OUTDIR / "meta_srr.json").write_text(json.dumps(meta, indent=2))
print("SRR saved in:", OUTDIR)
