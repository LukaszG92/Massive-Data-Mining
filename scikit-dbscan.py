"""
ANALISI DIAGNOSTICA: cosa sta succedendo davvero?
"""

import numpy as np
from sklearn.preprocessing import normalize
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt

embeddings = np.load("data/embeddings/embeddings_v1_384.npy")
embeddings_norm = normalize(embeddings, norm='l2')

print("="*80)
print("ANALISI DIAGNOSTICA: PERCHÉ 90% NOISE?")
print("="*80)

# ============================================================================
# TEST 1: Distribuzione distanze - c'è davvero separazione?
# ============================================================================

print("\n[TEST 1] Distribuzione distanze euclidee")
print("-"*80)

# Campione random di 2000 punti (per computabilità)
np.random.seed(42)
sample_idx = np.random.choice(len(embeddings_norm), 2000, replace=False)
sample = embeddings_norm[sample_idx]

# Calcola tutte le distanze a coppie
distances = pdist(sample, metric='euclidean')

print(f"\nStatistiche distanze (sample 2000 punti, {len(distances):,} coppie):")
print(f"  Min:        {distances.min():.4f}")
print(f"  Max:        {distances.max():.4f}")
print(f"  Mean:       {distances.mean():.4f}")
print(f"  Median:     {np.median(distances):.4f}")
print(f"  Std:        {distances.std():.4f}")
print(f"  5th perc:   {np.percentile(distances, 5):.4f}")
print(f"  95th perc:  {np.percentile(distances, 95):.4f}")

# Coefficiente di variazione
cv = distances.std() / distances.mean()
print(f"\n  Coefficiente variazione (std/mean): {cv:.4f}")
print(f"  Range relativo ((max-min)/mean): {(distances.max()-distances.min())/distances.mean():.4f}")

print("\n💡 INTERPRETAZIONE:")
if cv < 0.15:
    print("  ⚠️  CV < 0.15 → distanze MOLTO concentrate (curse of dimensionality)")
    print("  → Clustering density-based fallisce perché tutto sembra 'ugualmente denso'")
elif cv < 0.25:
    print("  ⚠️  CV moderato → qualche variazione ma limitata")
else:
    print("  ✓ CV > 0.25 → buona variazione, clustering dovrebbe funzionare")

# ============================================================================
# TEST 2: Nearest neighbor distribution - cluster locali esistono?
# ============================================================================

print("\n[TEST 2] Distanza ai k-nearest neighbors")
print("-"*80)

from sklearn.neighbors import NearestNeighbors

# Per ogni punto, trova distanza a 10, 30, 50, 100 nearest neighbors
k_values = [10, 30, 50, 100]

for k in k_values:
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean', n_jobs=-1)
    nbrs.fit(sample)
    distances_k, _ = nbrs.kneighbors(sample)

    # Distanza al k-esimo vicino (escludendo se stesso)
    dist_k = distances_k[:, k]

    print(f"\nDistanza al {k}-esimo vicino:")
    print(f"  Mean: {dist_k.mean():.4f}")
    print(f"  Std:  {dist_k.std():.4f}")
    print(f"  Max:  {dist_k.max():.4f}")

    # Gap ratio: quanto varia la densità locale?
    gap_ratio = dist_k.max() / dist_k.min()
    print(f"  Gap ratio (max/min): {gap_ratio:.2f}x")

print("\n💡 INTERPRETAZIONE:")
print("  Gap ratio > 3x → densità locali MOLTO variabili (cluster esistono)")
print("  Gap ratio < 2x → densità uniforme (no cluster naturali)")

# ============================================================================
# TEST 3: Confronto distanze intra- vs inter-cluster (se cluster esistessero)
# ============================================================================

print("\n[TEST 3] Simulazione: se ci fossero 175 cluster, come sarebbero?")
print("-"*80)

# Assume 175 cluster uniformi (come seed Alpaca)
n_clusters_expected = 175
cluster_size_expected = len(embeddings_norm) // n_clusters_expected

print(f"\nSe Alpaca ha 175 task types uniformi:")
print(f"  Cluster size medio: ~{cluster_size_expected} istruzioni/cluster")

# Simula assegnazione casuale a 175 cluster
simulated_labels = np.random.randint(0, n_clusters_expected, len(sample))

# Calcola distanza media INTRA-cluster vs INTER-cluster
dist_matrix = squareform(distances)

intra_dists = []
inter_dists = []

for i in range(len(sample)):
    for j in range(i+1, len(sample)):
        if simulated_labels[i] == simulated_labels[j]:
            intra_dists.append(dist_matrix[i, j])
        else:
            inter_dists.append(dist_matrix[i, j])

intra_mean = np.mean(intra_dists)
inter_mean = np.mean(inter_dists)

print(f"\nSimulazione casuale (175 cluster):")
print(f"  Distanza INTRA-cluster (stesso task type): {intra_mean:.4f}")
print(f"  Distanza INTER-cluster (task type diverso): {inter_mean:.4f}")
print(f"  Separazione (inter/intra): {inter_mean/intra_mean:.3f}x")

print("\n💡 INTERPRETAZIONE:")
print("  Se separazione ~1.0x → istruzioni sono TUTTE simili (no struttura task-type)")
print("  Se separazione >1.5x → struttura esiste, embedding la cattura")

# ============================================================================
# TEST 4: PCA variance - quanta informazione in poche dimensioni?
# ============================================================================

print("\n[TEST 4] Analisi PCA - struttura intrinseca dimensionalità")
print("-"*80)

from sklearn.decomposition import PCA

pca = PCA(n_components=100)
pca.fit(embeddings_norm)

cumsum_variance = np.cumsum(pca.explained_variance_ratio_)

dims_50 = np.argmax(cumsum_variance >= 0.5) + 1
dims_80 = np.argmax(cumsum_variance >= 0.8) + 1
dims_95 = np.argmax(cumsum_variance >= 0.95) + 1

print(f"\nVarianza spiegata (PCA):")
print(f"  50% varianza in: {dims_50} dimensioni")
print(f"  80% varianza in: {dims_80} dimensioni")
print(f"  95% varianza in: {dims_95} dimensioni")
print(f"\n  Prime 10 componenti: {cumsum_variance[9]:.1%} varianza")
print(f"  Prime 25 componenti: {cumsum_variance[24]:.1%} varianza")
print(f"  Prime 50 componenti: {cumsum_variance[49]:.1%} varianza")

print("\n💡 INTERPRETAZIONE:")
if dims_50 < 20:
    print("  ✓ Struttura low-dimensional forte → riduzione aiuterà")
elif dims_50 < 50:
    print("  ~ Struttura moderata → riduzione può aiutare")
else:
    print("  ⚠️  Informazione sparsa in molte dimensioni → riduzione difficile")

# ============================================================================
# CONCLUSIONE DIAGNOSTICA
# ============================================================================

print("\n" + "="*80)
print("CONCLUSIONE DIAGNOSTICA")
print("="*80)

print("\nBASATO SUI TEST SOPRA, il problema è:")
print("\n1. Se CV distanze < 0.15 E gap ratio < 2x:")
print("   → Curse of dimensionality (tutto sembra uguale in 384D)")
print("   → Soluzione: riduzione dimensionalità (UMAP/PCA)")

print("\n2. Se separazione simulata ~1.0x:")
print("   → Le istruzioni SONO semanticamente omogenee")
print("   → I 175 seed tasks si sono 'mescolati' nella generazione")
print("   → Soluzione: accettare pochi cluster naturali + diversità forzata")

print("\n3. Se PCA 50% in <20 dim:")
print("   → Struttura comprimibile esiste")
print("   → Clustering su PCA/UMAP può rivelare pattern")

print("\n4. Se tutti i test mostrano BUONA separazione:")
print("   → Problema è nei parametri clustering (eps/min_cluster_size)")
print("   → Soluzione: aggiustare parametri, no riduzione dimensionale")

print("\n" + "="*80)
print("\nESEGUI QUESTA ANALISI E DIMMI I RISULTATI.")
print("Solo dopo decideremo l'approccio corretto.")
print("="*80)
