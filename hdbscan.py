import numpy as np
import time
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprint.lsh_hdbscan import LSHHDBSCAN_Parallel as LSHHDBSCAN

INPUT_FILE = "data/embeddings/all-MiniLM-L6-v2_mean_centered.npy"
OUTPUT_FILE = "data/cluster/lsh_hdbscan_labels3.npy"
OUTPUT_STATS = "data/cluster/lsh_hdbscan_stats3.txt"

MIN_PTS = 15 # Minimo numero di punti per formare un cluster
MIN_EPS = 0.5  # Densità massima (cluster molto stretti)
MAX_EPS = 1.1  # Densità minima (recupera i cluster più larghi)
NUM_LEVELS = 8  # Numero di "scatti" tra min e max eps
N_JOBS = 24


# Parametri LSH (dal paper)
C_APPROX = 1.5  # Fattore di approssimazione (1.5 bilancia velocità/precisione)
DELTA = 0.20  # 1 - DELTA rappresenta Probabilità di successo


def main():
    print(f"--> Loading embeddings from: {INPUT_FILE}")
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: File {INPUT_FILE} not found.")
        return

    X = np.load(INPUT_FILE)
    print(f"    Shape: {X.shape} | Type: {X.dtype}")

    print("\n--> Initializing LSH-HDBSCAN Wrapper")
    print(f"    Epsilon Grid: {MIN_EPS} -> {MAX_EPS} ({NUM_LEVELS} levels)")
    print(f"    LSH Params: c={C_APPROX}, delta={DELTA}, min_pts={MIN_PTS}")

    model = LSHHDBSCAN(
        n_jobs=-N_JOBS,
        min_pts=MIN_PTS,
        min_eps=MIN_EPS,
        max_eps=MAX_EPS,
        num_levels=NUM_LEVELS,
        c=C_APPROX,
        delta=DELTA
    )

    print("\n--> Starting Hierarchical Clustering...")
    start_time = time.time()

    labels = model.fit_predict(X)

    elapsed = time.time() - start_time
    print(f"\n--> DONE in {elapsed:.1f} seconds ({elapsed / 60:.1f} min)")

    n_points = len(labels)
    n_noise = np.sum(labels == -1)
    noise_ratio = n_noise / n_points
    unique_labels = set(labels) - {-1}
    n_clusters = len(unique_labels)

    if n_clusters > 0:
        sizes = [np.sum(labels == c) for c in unique_labels]
        avg_size = np.mean(sizes)
        max_size = np.max(sizes)
        min_size = np.min(sizes)
    else:
        sizes = []
        avg_size = max_size = min_size = 0

    stats_msg = f"""
    ========================================
    LSH-HDBSCAN RESULTS SUMMARY
    ========================================
    Total Points:     {n_points}
    Noise Points:     {n_noise} ({noise_ratio:.2%})
    Valid Clusters:   {n_clusters}

    Cluster Sizes:
      - Average:      {avg_size:.1f}
      - Max (Giant?): {max_size}
      - Min:          {min_size}
    ========================================
    """
    print(stats_msg)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    np.save(OUTPUT_FILE, labels)
    print(f"--> Labels saved to: {OUTPUT_FILE}")

    with open(OUTPUT_STATS, "w") as f:
        f.write(stats_msg)


if __name__ == "__main__":
    main()
