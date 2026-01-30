import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.cluster import DBSCAN
import time
import os
from datetime import datetime
import json

# -------------------------
# CONFIG
# -------------------------
INPUT_NPY = "data/embeddings/instructor_large_768d.npy"
OUTPUT_DIR = "data/cluster/scikit_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


# -------------------------
# CUSTOM ELBOW DETECTOR
# -------------------------
def find_elbow_point(x, y):
    """Find elbow using maximum distance from line."""
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-10)
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-10)

    p1 = np.array([x_norm[0], y_norm[0]])
    p2 = np.array([x_norm[-1], y_norm[-1]])

    distances = []
    for i in range(len(x_norm)):
        p = np.array([x_norm[i], y_norm[i]])
        d = np.abs(np.cross(p2 - p1, p1 - p)) / (np.linalg.norm(p2 - p1) + 1e-10)
        distances.append(d)

    return np.argmax(distances)


# -------------------------
# PROFILING START
# -------------------------
print("=" * 70)
print("DBSCAN EPSILON OPTIMIZATION - ELBOW METHOD")
print("Dataset: instructor_large_768d.npy (L2 normalized)")
print("=" * 70)

t_total_start = time.time()

# -------------------------
# 1. LOAD DATA
# -------------------------
print(f"\n[1/5] Loading data...")
t_load_start = time.time()

X = np.load(INPUT_NPY).astype(np.float32)
t_load = time.time() - t_load_start

print(f"  ✓ Shape: {X.shape}")
print(f"  ✓ Memory: {X.nbytes / 1024 ** 3:.2f} GB")
print(f"  ✓ Time: {t_load:.2f}s")

# -------------------------
# 2. L2 NORMALIZATION (sklearn)
# -------------------------
print(f"\n[2/5] Applying L2 normalization (sklearn.preprocessing.normalize)...")
t_norm_start = time.time()

X_norm = normalize(X, norm='l2', axis=1)
t_norm = time.time() - t_norm_start

print(f"  ✓ Before: norm ∈ [{np.linalg.norm(X, axis=1).min():.4f}, {np.linalg.norm(X, axis=1).max():.4f}]")
print(f"  ✓ After:  norm ∈ [{np.linalg.norm(X_norm, axis=1).min():.4f}, {np.linalg.norm(X_norm, axis=1).max():.4f}]")
print(f"  ✓ Time: {t_norm:.2f}s")

# -------------------------
# 3. K-DISTANCE ANALYSIS
# -------------------------
print(f"\n[3/5] Computing k-distance plots for elbow detection...")

MINPTS_VALUES = [10, 15, 20, 25, 30]
METRIC = 'euclidean'

results = {}
timings = {}

for i, minPts in enumerate(MINPTS_VALUES, 1):
    print(f"\n  [{i}/{len(MINPTS_VALUES)}] minPts = {minPts}")

    t_start = time.time()

    # NearestNeighbors from sklearn
    nn = NearestNeighbors(n_neighbors=minPts + 1, metric=METRIC, algorithm='auto', n_jobs=-1)
    nn.fit(X_norm)
    distances, _ = nn.kneighbors(X_norm, return_distance=True)
    kth_distances = distances[:, -1]

    kth_sorted = np.sort(kth_distances)
    indices = np.arange(len(kth_sorted))

    t_kdist = time.time() - t_start

    # Find elbow
    elbow_idx = find_elbow_point(indices, kth_sorted)
    eps_elbow = kth_sorted[elbow_idx]

    t_total_mp = time.time() - t_start

    print(f"      k-distance: {t_kdist:.2f}s")
    print(f"      eps (elbow) = {eps_elbow:.6f} at index {elbow_idx:,} ({elbow_idx / len(kth_sorted) * 100:.1f}%)")
    print(f"      Q70 = {np.quantile(kth_sorted, 0.70):.6f}, Q80 = {np.quantile(kth_sorted, 0.80):.6f}")

    results[minPts] = {
        'kth_sorted': kth_sorted,
        'eps_elbow': float(eps_elbow),
        'elbow_idx': int(elbow_idx),
        'statistics': {
            'mean': float(kth_sorted.mean()),
            'std': float(kth_sorted.std()),
            'min': float(kth_sorted.min()),
            'max': float(kth_sorted.max()),
            'median': float(np.median(kth_sorted)),
            'q70': float(np.quantile(kth_sorted, 0.70)),
            'q80': float(np.quantile(kth_sorted, 0.80)),
            'q90': float(np.quantile(kth_sorted, 0.90)),
        }
    }

    timings[minPts] = {'total': t_total_mp}

# -------------------------
# 4. SAVE RESULTS
# -------------------------
print(f"\n[4/5] Saving results...")

# JSON report
report = {
    'metadata': {
        'timestamp': timestamp,
        'dataset': INPUT_NPY,
        'shape': {'n_samples': int(X.shape[0]), 'n_features': int(X.shape[1])},
        'l2_normalized': True,
        'metric': METRIC,
        'minPts_values': MINPTS_VALUES
    },
    'results': {str(k): {**v, 'kth_sorted': None} for k, v in results.items()}
}

json_path = f"{OUTPUT_DIR}/elbow_analysis_instructor768d_{timestamp}.json"
with open(json_path, 'w') as f:
    json.dump(report, f, indent=2)
print(f"  ✓ JSON: {json_path}")

# CSV summary
csv_path = f"{OUTPUT_DIR}/elbow_summary_instructor768d_{timestamp}.csv"
with open(csv_path, 'w') as f:
    f.write("minPts,eps_elbow,elbow_idx,elbow_pct,mean,std,median,q70,q80,q90,time_s\n")
    for minPts in MINPTS_VALUES:
        res = results[minPts]
        s = res['statistics']
        f.write(f"{minPts},{res['eps_elbow']},{res['elbow_idx']},{res['elbow_idx'] / len(res['kth_sorted']) * 100:.2f},"
                f"{s['mean']},{s['std']},{s['median']},{s['q70']},{s['q80']},{s['q90']},{timings[minPts]['total']}\n")
print(f"  ✓ CSV: {csv_path}")

# -------------------------
# 5. GENERATE PLOTS
# -------------------------
print(f"\n[5/5] Generating elbow plots...")

for minPts in MINPTS_VALUES:
    res = results[minPts]
    kth_sorted = res['kth_sorted']
    eps_elbow = res['eps_elbow']
    elbow_idx = res['elbow_idx']
    indices = np.arange(len(kth_sorted))

    fig, ax = plt.subplots(figsize=(14, 8))

    ax.plot(indices, kth_sorted, linewidth=1.5, color='steelblue', label='k-distance curve', alpha=0.8)
    ax.axhline(y=eps_elbow, color='red', linestyle='--', linewidth=2.5,
               label=f'Elbow: ε = {eps_elbow:.6f}', alpha=0.9)
    ax.axvline(x=elbow_idx, color='green', linestyle='--', linewidth=2, alpha=0.6,
               label=f'Elbow index: {elbow_idx:,} ({elbow_idx / len(kth_sorted) * 100:.1f}%)')
    ax.scatter([elbow_idx], [eps_elbow], color='red', s=400, zorder=5,
               marker='o', edgecolors='darkred', linewidth=4)

    ax.set_xlabel('Points (sorted by k-distance)', fontsize=15, fontweight='bold')
    ax.set_ylabel(f'{minPts}-th Nearest Neighbor Distance', fontsize=15, fontweight='bold')
    ax.set_title(f'DBSCAN ε Selection via Elbow Method (minPts={minPts})\n' +
                 f'instructor_large_768d.npy (L2-normalized, {X.shape[0]:,} points, {X.shape[1]}D)',
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=13, loc='upper left', framealpha=0.95, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plot_path = f"{OUTPUT_DIR}/elbow_minPts{minPts}_instructor768d_{timestamp}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Plot (minPts={minPts}): {plot_path}")

# -------------------------
# FINAL REPORT
# -------------------------
t_total = time.time() - t_total_start

print(f"\n{'=' * 70}")
print("SUMMARY: RECOMMENDED EPSILON VALUES")
print(f"{'=' * 70}")
print(f"{'minPts':<8} {'eps_elbow':<14} {'Range (±20%)':<30} {'Q70':<12}")
print("-" * 70)
for minPts in MINPTS_VALUES:
    res = results[minPts]
    eps = res['eps_elbow']
    q70 = res['statistics']['q70']
    print(f"{minPts:<8} {eps:<14.6f} [{eps * 0.8:.6f}, {eps * 1.2:.6f}]     {q70:<12.6f}")

print(f"\n{'=' * 70}")
print("USAGE EXAMPLE (sklearn.cluster.DBSCAN)")
print(f"{'=' * 70}")
print("""
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
import numpy as np

# Load and normalize
X = np.load('data/embeddings/instructor_large_768d.npy').astype(np.float32)
X_norm = normalize(X, norm='l2', axis=1)

# DBSCAN clustering
dbscan = DBSCAN(eps={:.6f}, min_samples=20, metric='euclidean', n_jobs=-1)
labels = dbscan.fit_predict(X_norm)

print(f"Clusters: {{len(set(labels)) - (1 if -1 in labels else 0)}}")
print(f"Noise: {{(labels == -1).sum()}} / {{len(labels)}}")
""".format(results[20]['eps_elbow']))

print(f"\n{'=' * 70}")
print(f"✓ TOTAL TIME: {t_total:.2f}s")
print(f"✓ ALL RESULTS SAVED TO: {OUTPUT_DIR}/")
print(f"{'=' * 70}")
