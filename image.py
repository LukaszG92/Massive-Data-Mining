import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.cluster import DBSCAN
import os

# --- CONFIGURAZIONE ---
VECTOR_FILE = "data/all-MiniLM-L6-v2_mean_centered.npy"  # Assicurati che il percorso sia giusto
BEST_EPS = 0.851  # Parametro ottimale trovato
BEST_MINPTS = 10  # Parametro ottimale trovato


def main():
    print(f"--- 1. Caricamento Vettori da {VECTOR_FILE} ---")
    if not os.path.exists(VECTOR_FILE):
        print(f"ERRORE: File {VECTOR_FILE} non trovato!")
        return

    # Carica i vettori (assumendo formato testo spaziale o csv)
    try:
        embeddings = np.loadtxt(VECTOR_FILE)
        print(f"Vettori caricati: {embeddings.shape}")
    except Exception as e:
        print(f"Errore caricamento (provo con delimitatore ','): {e}")
        embeddings = np.load(VECTOR_FILE)

    print("\n--- 2. Esecuzione Clustering (SRR Configuration) ---")
    print(f"Parametri: eps={BEST_EPS}, minPts={BEST_MINPTS}")

    # Usiamo scikit-learn DBSCAN per semplicità (SRR è un wrapper ottimizzato,
    # ma il risultato matematico è identico se la metrica è la stessa).
    # IMPORTANTE: Se i vettori sono normalizzati, distanza Euclidea = Distanza Coseno
    # Se non sei sicuro, usa metric='cosine'.

    db = DBSCAN(eps=BEST_EPS, min_samples=BEST_MINPTS, metric='euclidean', n_jobs=-1)
    labels = db.fit_predict(embeddings)

    # Statistiche Rapide
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    noise_ratio = n_noise / len(labels)

    print(f"Risultato Rigenerato:")
    print(f"  Clusters: {n_clusters}")
    print(f"  Noise: {n_noise} ({noise_ratio:.1%})")

    # Salva per il futuro!
    np.save("data/best_labels.npy", labels)
    print("Labels salvate in 'data/best_labels.npy'")

    print("\n--- 3. Generazione Mappa UMAP 2D ---")
    print("Calcolo proiezione 2D (questo step è il più lento, ~2-3 min)...")

    reducer = umap.UMAP(
        n_neighbors=50,  # Aumentato per struttura globale più chiara
        min_dist=0.1,
        n_components=2,
        metric='cosine',  # Sempre cosine per embedding NLP
        random_state=42,
        verbose=True
    )
    embedding_2d = reducer.fit_transform(embeddings)

    # --- PLOTTING ---
    plt.figure(figsize=(16, 12), dpi=150)

    # Colori
    unique_labels, counts = np.unique(labels, return_counts=True)

    # Identifica il blob (il cluster più grande escluso il noise)
    valid_clusters = unique_labels[unique_labels != -1]
    if len(valid_clusters) > 0:
        blob_id = valid_clusters[np.argmax(counts[unique_labels != -1])]
    else:
        blob_id = -999

    # 1. Noise (Grigio Sfondo)
    mask_noise = (labels == -1)
    plt.scatter(embedding_2d[mask_noise, 0], embedding_2d[mask_noise, 1],
                c='#e0e0e0', s=1, alpha=0.2, label='Noise (Long Tail)')

    # 2. Clusters Normali (Colorati)
    mask_clusters = (labels != -1) & (labels != blob_id)
    if np.sum(mask_clusters) > 0:
        plt.scatter(embedding_2d[mask_clusters, 0], embedding_2d[mask_clusters, 1],
                    c=labels[mask_clusters], cmap='Spectral', s=4, alpha=0.8, label='Task Clusters')

    # 3. Blob (Rosso Evidente)
    if blob_id != -999:
        mask_blob = (labels == blob_id)
        plt.scatter(embedding_2d[mask_blob, 0], embedding_2d[mask_blob, 1],
                    c='crimson', s=2, alpha=0.4, label='Redundant Blob')

    plt.title(f"Mappa Densità Alpaca (eps={BEST_EPS})", fontsize=18)
    plt.legend(markerscale=5, loc='upper right')
    plt.axis('off')

    outfile = "alpaca_density_map.png"
    plt.savefig(outfile, bbox_inches='tight')
    print(f"\nFATTO! Immagine salvata come: {outfile}")
    plt.show()


if __name__ == "__main__":
    main()
