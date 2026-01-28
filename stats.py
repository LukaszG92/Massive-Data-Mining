import json
import numpy as np
import pandas as pd

# CONFIGURA I PATH
FILE_IQS = "data/ranking_IQS_result.json"
FILE_SRR = "data/cluster/lsh_hdbscan_labels2.npy"
FILE_KM  = "data/cluster/kmeans_k161_labels.npy"

print("Caricamento dati...")
with open(FILE_IQS,'r',encoding='utf-8') as f:
    recs = json.load(f)
iqs = np.array([r['score'] for r in recs])

labels_srr = np.load(FILE_SRR)
labels_km = np.load(FILE_KM)

# --- SRR (Escludendo Noise) ---
mask_srr = labels_srr != -1
labels_valid_srr = labels_srr[mask_srr]
iqs_valid_srr = iqs[mask_srr]

# Calcola la MEDIANA per ogni cluster
if len(labels_valid_srr) > 0:
    srr_cluster_medians = pd.Series(iqs_valid_srr).groupby(labels_valid_srr).median()
    srr_max = srr_cluster_medians.max()
    srr_min = srr_cluster_medians.min()
    srr_avg = srr_cluster_medians.mean() # Media delle mediane (macro-average)
    srr_std = srr_cluster_medians.std()
else:
    srr_max = srr_min = srr_avg = srr_std = 0

# --- KMEANS ---
km_cluster_medians = pd.Series(iqs).groupby(labels_km).median()
km_max = km_cluster_medians.max()
km_min = km_cluster_medians.min()
km_avg = km_cluster_medians.mean()
km_std = km_cluster_medians.std()

# --- NOISE ---
noise_median = np.median(iqs[~mask_srr]) if (~mask_srr).any() else 0

res = f"""
===================================================
VERITÀ ROBUSTA: ANALISI DELLE MEDIANE
===================================================
La mediana ignora gli outlier. Se SRR è alto anche 
qui, la qualità è strutturale e non casuale.

METRICA (sui Cluster)           | SRR-DBSCAN       | KMEANS
--------------------------------|------------------|------------------
Max Cluster Median              | {srr_max:.4f}           | {km_max:.4f}
Min Cluster Median              | {srr_min:.4f}           | {km_min:.4f}
Avg of Cluster Medians          | {srr_avg:.4f}           | {km_avg:.4f}
Std of Cluster Medians          | {srr_std:.4f}           | {km_std:.4f}
--------------------------------|------------------|------------------
Noise Median (-1)               | {noise_median:.4f}           | -
===================================================
"""
print(res)
