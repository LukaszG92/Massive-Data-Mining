import numpy as np
from sklearn.preprocessing import normalize
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors

print("=" * 80)
print("COMPARAZIONE: MiniLM-384D vs INSTRUCTOR-768D")
print("=" * 80)

# Caricamento embeddings
emb_minilm = np.load('data/embeddings/only_instruct_embeddings_v1_384.npy')
emb_instructor = np.load('data/embeddings/instructor_large_768d.npy')

emb_minilm_norm = normalize(emb_minilm, norm='l2')
emb_instructor_norm = normalize(emb_instructor, norm='l2')

print(f"\nMiniLM:     {emb_minilm.shape}")
print(f"INSTRUCTOR: {emb_instructor.shape}")

# Sample per velocità
np.random.seed(42)
sample_size = 2000
sample_idx = np.random.choice(len(emb_minilm_norm), sample_size, replace=False)

sample_mini = emb_minilm_norm[sample_idx]
sample_instr = emb_instructor_norm[sample_idx]

# ============================================================================
# TEST 1: Anisotropia (Cosine Similarity)
# ============================================================================
print("\n" + "=" * 80)
print("[TEST 1] Anisotropia")
print("=" * 80)


def test_anisotropy(embeddings, name):
    cosine_dist = pdist(embeddings, metric='cosine')
    cosine_sim = 1 - cosine_dist
    mean_sim = cosine_sim.mean()

    print(f"\n{name}:")
    print(f"  Mean cosine similarity: {mean_sim:.4f}")

    if mean_sim > 0.75:
        print(f"  ⚠️ Severa anisotropia")
    elif mean_sim > 0.60:
        print(f"  ~ Leggera anisotropia")
    else:
        print(f"  ✅ Spazio isotropo")

    return mean_sim


sim_mini = test_anisotropy(sample_mini, "MiniLM-384D")
sim_instr = test_anisotropy(sample_instr, "INSTRUCTOR-768D")

# ============================================================================
# TEST 2: Separabilità (CV)
# ============================================================================
print("\n" + "=" * 80)
print("[TEST 2] Separabilità (Coefficient of Variation)")
print("=" * 80)


def test_separability(embeddings, name):
    dist = pdist(embeddings, metric='euclidean')
    mean_dist = dist.mean()
    std_dist = dist.std()
    cv = std_dist / mean_dist

    print(f"\n{name}:")
    print(f"  Mean distance: {mean_dist:.4f}")
    print(f"  Std distance:  {std_dist:.4f}")
    print(f"  CV:            {cv:.4f}")

    return cv


cv_mini = test_separability(sample_mini, "MiniLM-384D")
cv_instr = test_separability(sample_instr, "INSTRUCTOR-768D")

improvement_cv = ((cv_instr / cv_mini) - 1) * 100

print(f"\nImprovement: {improvement_cv:+.1f}%")

if cv_instr > 0.15:
    print(f"✅ CV > 0.15 → Clustering diretto possibile")
elif cv_instr > 0.10:
    print(f"✓ CV > 0.10 → UMAP leggero poi clustering")
elif cv_instr > 0.08:
    print(f"~ CV > 0.08 → UMAP aggressivo necessario")
else:
    print(f"⚠️ CV < 0.08 → Ancora molto omogeneo")

# ============================================================================
# TEST 3: Gap Ratio (30-NN)
# ============================================================================
print("\n" + "=" * 80)
print("[TEST 3] K-Nearest Neighbor Gap Ratio")
print("=" * 80)


def test_gap_ratio(embeddings, name, k=30):
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric='euclidean', n_jobs=-1)
    nbrs.fit(embeddings)
    distances, _ = nbrs.kneighbors(embeddings)

    dist_k = distances[:, k]
    gap_ratio = dist_k.max() / dist_k.min()

    print(f"\n{name}:")
    print(f"  30-NN mean distance: {dist_k.mean():.4f}")
    print(f"  Gap ratio (max/min): {gap_ratio:.2f}x")

    return gap_ratio


gap_mini = test_gap_ratio(sample_mini, "MiniLM-384D")
gap_instr = test_gap_ratio(sample_instr, "INSTRUCTOR-768D")

if gap_instr > 2.0:
    print(f"\n✅ Gap > 2.0x → Cluster naturali evidenti")
elif gap_instr > 1.5:
    print(f"~ Gap > 1.5x → Cluster deboli presenti")
else:
    print(f"⚠️ Gap < 1.5x → Densità uniforme")

# ============================================================================
# TEST 4: Separazione Intra vs Inter
# ============================================================================
print("\n" + "=" * 80)
print("[TEST 4] Separazione Intra vs Inter-cluster")
print("=" * 80)


def test_separation(embeddings, name, n_clusters=175):
    np.random.seed(42)
    simulated_labels = np.random.randint(0, n_clusters, len(embeddings))

    dist_matrix = squareform(pdist(embeddings, metric='euclidean'))

    intra_dists = []
    inter_dists = []

    for i in range(min(1000, len(embeddings))):
        for j in range(i + 1, min(1000, len(embeddings))):
            if simulated_labels[i] == simulated_labels[j]:
                intra_dists.append(dist_matrix[i, j])
            else:
                inter_dists.append(dist_matrix[i, j])

    intra_mean = np.mean(intra_dists)
    inter_mean = np.mean(inter_dists)
    separation = inter_mean / intra_mean

    print(f"\n{name}:")
    print(f"  INTRA-cluster: {intra_mean:.4f}")
    print(f"  INTER-cluster: {inter_mean:.4f}")
    print(f"  Separazione:   {separation:.3f}x")

    return separation


sep_mini = test_separation(sample_mini, "MiniLM-384D")
sep_instr = test_separation(sample_instr, "INSTRUCTOR-768D")

if sep_instr > 1.3:
    print(f"\n✅ Separazione > 1.3x → Struttura task-type presente")
elif sep_instr > 1.1:
    print(f"~ Separazione > 1.1x → Struttura debole")
else:
    print(f"⚠️ Separazione ~1.0x → Dataset omogeneo")

# ============================================================================
# RIASSUNTO FINALE
# ============================================================================
print("\n" + "=" * 80)
print("RIASSUNTO COMPARATIVO")
print("=" * 80)

print(f"\n{'Metrica':<25} {'MiniLM-384D':<15} {'INSTRUCTOR-768D':<15} {'Δ':<10}")
print("-" * 80)
print(f"{'Mean Cosine Sim':<25} {sim_mini:<15.4f} {sim_instr:<15.4f} {(sim_instr - sim_mini):+.4f}")
print(f"{'CV (separabilità)':<25} {cv_mini:<15.4f} {cv_instr:<15.4f} {improvement_cv:+.1f}%")
print(f"{'Gap Ratio (30-NN)':<25} {gap_mini:<15.2f} {gap_instr:<15.2f} {(gap_instr - gap_mini):+.2f}")
print(f"{'Separazione (inter/intra)':<25} {sep_mini:<15.3f} {sep_instr:<15.3f} {(sep_instr - sep_mini):+.3f}")

print("\n" + "=" * 80)
print("DECISIONE")
print("=" * 80)

if cv_instr > 0.15 and gap_instr > 2.0:
    print("\n✅ GRANDE MIGLIORAMENTO!")
    print("   → Usa INSTRUCTOR embeddings")
    print("   → Clustering DIRETTO (HDBSCAN senza UMAP)")
    print(f"   → File: instructor_large_768d.npy")

elif cv_instr > 0.10:
    print("\n✓ MIGLIORAMENTO SIGNIFICATIVO")
    print("   → Usa INSTRUCTOR embeddings")
    print("   → UMAP leggero (768D → 50D) poi HDBSCAN")
    print(f"   → File: instructor_large_768d.npy")

elif cv_instr > cv_mini * 1.2:
    print("\n~ MIGLIORAMENTO MODERATO")
    print("   → Usa INSTRUCTOR embeddings")
    print("   → UMAP aggressivo (768D → 25D) poi HDBSCAN")
    print(f"   → File: instructor_large_768d.npy")

else:
    print("\n⚠️ MIGLIORAMENTO LIMITATO")
    print("   → Dataset intrinsecamente molto omogeneo")
    print("   → Considera modelli più grandi (E5-large, NV-Embed)")
    print("   → Oppure UMAP aggressivo su INSTRUCTOR")

print("=" * 80)
