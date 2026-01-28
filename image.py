import numpy as np
import matplotlib.pyplot as plt
import umap
import pandas as pd
import seaborn as sns

# --- CONFIG ---
EMB_FILE = "data/embeddings/all-MiniLM-L6-v2_mean_centered.npy"
LABEL_FILE = "data/cluster/lsh_hdbscan_labels2.npy"  # File corretto
OUTPUT_IMG = "hdbscan_umap_final.png"


def main():
    print("Loading data...")
    X = np.load(EMB_FILE)
    labels = np.load(LABEL_FILE)

    # 1. Calcola UMAP (se non hai salvato l'embedding 2D prima)
    print("Running UMAP...")
    reducer = umap.UMAP(metric='cosine', n_neighbors=15, min_dist=0.1, random_state=42, verbose=True)
    embedding = reducer.fit_transform(X)

    df = pd.DataFrame(embedding, columns=['x', 'y'])
    df['label'] = labels

    # Identifica il Cluster Gigante
    cluster_counts = df[df['label'] != -1]['label'].value_counts()
    giant_cluster_id = cluster_counts.idxmax()
    giant_size = cluster_counts.max()
    print(f"Giant Cluster ID: {giant_cluster_id} (Size: {giant_size})")

    # Separa i dati in 3 categorie
    mask_noise = df['label'] == -1
    mask_giant = df['label'] == giant_cluster_id
    mask_small = (~mask_noise) & (~mask_giant)

    plt.figure(figsize=(15, 12), dpi=300)
    ax = plt.gca()
    # Sfondo nero o bianco? Bianco per paper, Nero per slide. Facciamo Bianco.
    ax.set_facecolor('white')

    # LAYER 1: Noise (Sfondo diffuso)
    plt.scatter(
        df.loc[mask_noise, 'x'], df.loc[mask_noise, 'y'],
        c='#e0e0e0',  # Grigio chiarissimo
        s=0.5, alpha=0.3,  # Molto trasparente
        label=f'Noise (Long Tail) - {mask_noise.sum()} pts',
        zorder=1
    )

    # LAYER 2: Cluster Gigante (Struttura Base)
    plt.scatter(
        df.loc[mask_giant, 'x'], df.loc[mask_giant, 'y'],
        c='#4a4a4a',  # Grigio scuro / Antracite
        s=1.0, alpha=0.4,
        label=f'Giant Core Cluster - {mask_giant.sum()} pts',
        zorder=2
    )

    # LAYER 3: Cluster "Puri" (Gemme)
    # Usiamo una colormap ciclica per distinguere i vicini
    plt.scatter(
        df.loc[mask_small, 'x'], df.loc[mask_small, 'y'],
        c=df.loc[mask_small, 'label'],
        cmap='turbo',  # Colori vibranti
        s=3.0, alpha=1.0,  # Opachi e più grandi
        label=f'Fine-Grained Clusters - {mask_small.sum()} pts',
        zorder=3
    )

    plt.title("LSH-HDBSCAN Topological Segmentation", fontsize=18, fontweight='bold')
    plt.legend(markerscale=5, fontsize=12, loc='upper right')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(OUTPUT_IMG)
    print(f"Saved to {OUTPUT_IMG}")


if __name__ == "__main__":
    main()
