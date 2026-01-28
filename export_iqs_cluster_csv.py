import json
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score

# --- CONFIG ---
FILE_IQS = "data/ranking_IQS_result.json"
FILE_SRR = "data/cluster/lsh_hdbscan_labels2.npy"
FILE_KM = "data/cluster/kmeans_k161_labels.npy"
FILE_EMB = "data/embeddings/all-MiniLM-L6-v2_mean_centered.npy"  # Necessario per Silhouette
SAMPLE_SIZE = 52000  # Campione per Silhouette (max 52000)

print(f"1. Loading Data...")
# Load IQS & Text Length
with open(FILE_IQS, 'r', encoding='utf-8') as f:
    recs = json.load(f)
iqs = np.array([r['score'] for r in recs])
# Calcoliamo lunghezza istruzione (proxy di complessità)
lengths = np.array([len(r['instruction']) + len(r['input']) for r in recs])

# Load Embeddings (richiesto per metriche geometriche)
embeddings = np.load(FILE_EMB)

# Load Labels
labels_srr = np.load(FILE_SRR)
labels_km = np.load(FILE_KM)


# --- HELPER FUNCTIONS ---

def get_cluster_homogeneity(labels, values, noise_label=-1):
    """Calcola la deviazione standard media pesata all'interno dei cluster"""
    df = pd.DataFrame({'label': labels, 'val': values})
    # Rimuovi noise
    df = df[df['label'] != noise_label]

    if len(df) == 0: return 0.0, 0.0

    stats = df.groupby('label')['val'].std().fillna(0)  # std interna al cluster
    sizes = df.groupby('label')['val'].count()

    # Media pesata della std (quanto varia l'IQS dentro un cluster medio?)
    weighted_avg_std = np.average(stats, weights=sizes)
    return weighted_avg_std


def get_geometric_metrics(X, labels, noise_label=-1):
    """Calcola Silhouette e Davies-Bouldin (escludendo il noise per correttezza)"""
    mask = labels != noise_label
    X_clean = X[mask]
    labels_clean = labels[mask]

    if len(np.unique(labels_clean)) < 2:
        return 0.0, 0.0  # Impossibile calcolare se < 2 cluster

    # Silhouette su campione se troppi dati
    if len(X_clean) > SAMPLE_SIZE:
        idx = np.random.choice(len(X_clean), SAMPLE_SIZE, replace=False)
        sil = silhouette_score(X_clean[idx], labels_clean[idx])
    else:
        sil = silhouette_score(X_clean, labels_clean)

    # Davies Bouldin (più basso è meglio)
    db = davies_bouldin_score(X_clean, labels_clean)

    return sil, db


# --- CALCOLI ---

print("2. Computing SRR Metrics (ignoring noise for geometry)...")
sil_srr, db_srr = get_geometric_metrics(embeddings, labels_srr, noise_label=-1)
iqs_std_srr = get_cluster_homogeneity(labels_srr, iqs, noise_label=-1)

print("3. Computing KMeans Metrics...")
sil_km, db_km = get_geometric_metrics(embeddings, labels_km, noise_label=-999)  # KMeans has no noise
iqs_std_km = get_cluster_homogeneity(labels_km, iqs, noise_label=-999)

print("4. Computing Complexity (Text Length) Stats...")
# SRR Noise vs SRR Clustered
mask_noise = labels_srr == -1
len_noise = lengths[mask_noise].mean()
len_clustered = lengths[~mask_noise].mean()

# --- REPORT ---

res = f"""
===============================================================
ANALISI AVANZATA: GEOMETRIA & CONTENUTO
===============================================================

1. COERENZA GEOMETRICA (Cluster validi)
---------------------------------------------------------------
Metriche calcolate sui soli punti assegnati a cluster.
Silhouette: Più alto è meglio (1.0 max).
Davies-Bouldin: Più basso è meglio.

METRICA             | SRR-DBSCAN (Valid) | KMEANS (All)
--------------------|--------------------|---------------------
Silhouette Score    | {sil_srr:.4f}             | {sil_km:.4f}
Davies-Bouldin      | {db_srr:.4f}             | {db_km:.4f}

-> Interpretazione: Se Sil_SRR >> Sil_KM, SRR trova "veri" gruppi densi.

2. OMOGENEITÀ IQS (Purezza)
---------------------------------------------------------------
"Quanto varia l'IQS dentro un singolo cluster?" (Weighted Avg Std)

METRICA             | SRR-DBSCAN         | KMEANS
--------------------|--------------------|---------------------
Intra-Cluster Std   | {iqs_std_srr:.4f}             | {iqs_std_km:.4f}

-> Interpretazione: Se SRR è più basso, i suoi cluster sono 
   tematicamente/qualitativamente più puri.

3. ANALISI DEL PARADOSSO (Noise Analysis)
---------------------------------------------------------------
Confronto caratteristiche punti Noise (-1) vs Punti Clusterizzati (SRR)

Caratteristica      | NOISE (-1)         | CLUSTERED (SRR)
--------------------|--------------------|---------------------
Lunghezza Media (ch)| {len_noise:.1f}              | {len_clustered:.1f}
IQS Medio           | {iqs[mask_noise].mean():.4f}             | {iqs[~mask_noise].mean():.4f}

-> Interpretazione: Se il Noise è molto più lungo, l'embedding
   lo isola perché "complesso/unico". SRR raggruppa le frasi
   brevi/standard (cluster densi).
===============================================================
"""
print(res)
