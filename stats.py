import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap

print("=" * 80)
print("VISUALIZZAZIONE CLUSTER HDBSCAN")
print("=" * 80)

# ============================================================================
# Caricamento dati
# ============================================================================

print("\n[1] Caricamento embeddings e labels...")

embeddings_umap_50d = np.load('data/embeddings/instructor_umap_50d.npy')
labels = np.load('data/experiments/scikit_results/instruct_hdbscan_labels.npy')

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = (labels == -1).sum()

print(f"  Embeddings: {embeddings_umap_50d.shape}")
print(f"  Clusters: {n_clusters}")
print(f"  Noise: {n_noise} ({100 * n_noise / len(labels):.1f}%)")

# ============================================================================
# Riduzione 50D → 2D per visualizzazione
# ============================================================================

print("\n[2] Riduzione 50D → 2D per visualizzazione...")

# UMAP 50D → 2D (preserva struttura globale meglio di t-SNE)
reducer_2d = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    metric='euclidean',
    random_state=42
)

embeddings_2d = reducer_2d.fit_transform(embeddings_umap_50d)

print(f"  ✓ Ridotto a: {embeddings_2d.shape}")

# Salva per riuso
np.save('data/embeddings/instructor_umap_2d.npy', embeddings_2d)

# ============================================================================
# Plot principale: Tutti i cluster
# ============================================================================

print("\n[3] Creazione visualizzazione...")

fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# ============================================================================
# Plot 1: Cluster colorati (senza noise)
# ============================================================================

ax1 = axes[0]

# Punti clusterizzati
mask_clustered = labels != -1
x_clustered = embeddings_2d[mask_clustered, 0]
y_clustered = embeddings_2d[mask_clustered, 1]
labels_clustered = labels[mask_clustered]

# Plot scatter con colormap
scatter1 = ax1.scatter(
    x_clustered,
    y_clustered,
    c=labels_clustered,
    cmap='tab20c',
    s=5,
    alpha=0.6,
    edgecolors='none'
)

ax1.set_title(f'HDBSCAN Clusters (n={n_clusters})\n{len(x_clustered)} points clustered',
              fontsize=14, fontweight='bold')
ax1.set_xlabel('UMAP Dimension 1', fontsize=12)
ax1.set_ylabel('UMAP Dimension 2', fontsize=12)
ax1.grid(True, alpha=0.3)

# ============================================================================
# Plot 2: Cluster + Noise
# ============================================================================

ax2 = axes[1]

# Noise points
mask_noise = labels == -1
x_noise = embeddings_2d[mask_noise, 0]
y_noise = embeddings_2d[mask_noise, 1]

# Plot clustered (più trasparente)
ax2.scatter(
    x_clustered,
    y_clustered,
    c=labels_clustered,
    cmap='tab20c',
    s=5,
    alpha=0.3,
    edgecolors='none',
    label='Clustered'
)

# Plot noise (evidenziato)
ax2.scatter(
    x_noise,
    y_noise,
    c='red',
    s=10,
    alpha=0.8,
    edgecolors='black',
    linewidth=0.5,
    label=f'Noise ({n_noise} points)',
    marker='x'
)

ax2.set_title(f'HDBSCAN Clusters + Noise\nNoise: {100 * n_noise / len(labels):.1f}%',
              fontsize=14, fontweight='bold')
ax2.set_xlabel('UMAP Dimension 1', fontsize=12)
ax2.set_ylabel('UMAP Dimension 2', fontsize=12)
ax2.legend(loc='upper right', fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('data/experiments/scikit_results/clusters_visualization.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Salvato: clusters_visualization.png")
plt.show()

# ============================================================================
# Plot dettagliato: Top 10 cluster più grandi
# ============================================================================

print("\n[4] Visualizzazione top 10 cluster...")

from collections import Counter

cluster_sizes = Counter(labels[labels != -1])
top_10_clusters = [cid for cid, _ in cluster_sizes.most_common(10)]

fig, axes = plt.subplots(2, 5, figsize=(25, 10))
axes = axes.flatten()

for idx, cluster_id in enumerate(top_10_clusters):
    ax = axes[idx]

    # Tutti i punti (grigio chiaro)
    ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
               c='lightgray', s=1, alpha=0.3)

    # Cluster specifico (evidenziato)
    mask_cluster = labels == cluster_id
    x_cluster = embeddings_2d[mask_cluster, 0]
    y_cluster = embeddings_2d[mask_cluster, 1]

    ax.scatter(x_cluster, y_cluster,
               c='blue', s=10, alpha=0.7, edgecolors='darkblue', linewidth=0.5)

    cluster_size = mask_cluster.sum()
    ax.set_title(f'Cluster {cluster_id}\n{cluster_size} instructions ({100 * cluster_size / len(labels):.1f}%)',
                 fontsize=10, fontweight='bold')
    ax.set_xlabel('UMAP-1', fontsize=8)
    ax.set_ylabel('UMAP-2', fontsize=8)
    ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('data/experiments/scikit_results/top10_clusters_detail.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Salvato: top10_clusters_detail.png")
plt.show()

# ============================================================================
# Plot density heatmap
# ============================================================================

print("\n[5] Density heatmap...")

fig, ax = plt.subplots(1, 1, figsize=(12, 10))

# Heatmap solo dei clustered
from scipy.stats import gaussian_kde

if len(x_clustered) > 100:
    # Subsample per velocità
    sample_idx = np.random.choice(len(x_clustered), min(5000, len(x_clustered)), replace=False)
    x_sample = x_clustered[sample_idx]
    y_sample = y_clustered[sample_idx]
else:
    x_sample = x_clustered
    y_sample = y_clustered

# Hexbin per densità
hexbin = ax.hexbin(x_sample, y_sample, gridsize=50, cmap='YlOrRd', mincnt=1)

# Overlay noise
ax.scatter(x_noise, y_noise, c='blue', s=15, alpha=0.8,
           edgecolors='darkblue', linewidth=0.5, label='Noise', marker='x')

ax.set_title('Cluster Density Heatmap', fontsize=14, fontweight='bold')
ax.set_xlabel('UMAP Dimension 1', fontsize=12)
ax.set_ylabel('UMAP Dimension 2', fontsize=12)
plt.colorbar(hexbin, ax=ax, label='Density')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('data/experiments/scikit_results/density_heatmap.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Salvato: density_heatmap.png")
plt.show()

# ============================================================================
# Statistiche visuali
# ============================================================================

print("\n[6] Statistiche cluster per dimensione...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Histogram dimensioni cluster
ax1 = axes[0]
sizes = list(cluster_sizes.values())
ax1.hist(sizes, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
ax1.axvline(np.mean(sizes), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(sizes):.0f}')
ax1.axvline(np.median(sizes), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(sizes):.0f}')
ax1.set_title('Cluster Size Distribution', fontsize=14, fontweight='bold')
ax1.set_xlabel('Cluster Size', fontsize=12)
ax1.set_ylabel('Frequency', fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Top 20 cluster per dimensione
ax2 = axes[1]
top_20 = cluster_sizes.most_common(20)
cluster_ids = [str(cid) for cid, _ in top_20]
cluster_sizes_vals = [size for _, size in top_20]

ax2.barh(cluster_ids, cluster_sizes_vals, color='steelblue', edgecolor='black')
ax2.set_title('Top 20 Largest Clusters', fontsize=14, fontweight='bold')
ax2.set_xlabel('Number of Instructions', fontsize=12)
ax2.set_ylabel('Cluster ID', fontsize=12)
ax2.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('data/experiments/scikit_results/cluster_statistics.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Salvato: cluster_statistics.png")
plt.show()

print("\n" + "=" * 80)
print("VISUALIZZAZIONI COMPLETE")
print("=" * 80)
print("\nFile generati:")
print("  1. clusters_visualization.png       - Overview cluster + noise")
print("  2. top10_clusters_detail.png       - Dettaglio top 10 cluster")
print("  3. density_heatmap.png             - Heatmap densità cluster")
print("  4. cluster_statistics.png          - Statistiche dimensioni")
print("  5. instructor_umap_2d.npy          - Embeddings 2D (riusabili)")
print("=" * 80)
