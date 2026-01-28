import numpy as np
import joblib

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.metrics.cluster import contingency_matrix


# -----------------------------
# CONFIG
# -----------------------------
X_PATH = "../embeddings/embeddings_v1_384.npy"  # embedding originali (n_samples, n_features)

VAR_TARGET = 0.95                                 # PCA al 95% [web:32]
K = 161
RANDOM_STATE = 0

OUT_PCA_MODEL_PATH = f"pca_var{int(VAR_TARGET*100)}.joblib"
OUT_X_PCA_PATH      = f"X_pca_var{int(VAR_TARGET*100)}.npy"

OUT_KM_LABELS_PATH = f"data/cluster/kmeans_k{K}_labels.npy"
OUT_KM_MODEL_PATH  = f"data/cluster/kmeans_k{K}.joblib"


# -----------------------------
# 1) Load data
# -----------------------------
X = np.load(X_PATH)

# -----------------------------
# 2) PCA (95% variance)
# -----------------------------
pca = PCA(n_components=VAR_TARGET, svd_solver="full")  # n_components float => target explained variance [web:32]
X_pca = pca.fit_transform(X)

print("X_pca:", X_pca.shape, X_pca.dtype)
print("explained_variance_ratio_sum:", float(pca.explained_variance_ratio_.sum()))

# Salva PCA e la matrice trasformata (opzionale ma comoda)
joblib.dump(pca, OUT_PCA_MODEL_PATH)
np.save(OUT_X_PCA_PATH, X_pca)
print("Saved:", OUT_PCA_MODEL_PATH, OUT_X_PCA_PATH)


# -----------------------------
# 3) Fit KMeans su X_pca
# -----------------------------
kmeans = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init="auto")
kmeans.fit(X_pca)

labels_km = kmeans.labels_
print("labels_km:", labels_km.shape, labels_km.dtype)


# -----------------------------
# 4) Save KMeans results
# -----------------------------
np.save(OUT_KM_LABELS_PATH, labels_km)
joblib.dump(kmeans, OUT_KM_MODEL_PATH)
print("Saved:", OUT_KM_LABELS_PATH, OUT_KM_MODEL_PATH)
