import os, time, json
import numpy as np
import itertools  # Standard library, già inclusa in Python
from tqdm import tqdm  # pip install tqdm
import multiprocessing
import gc
import dbscan_srr as dbscan
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

# -------------------------
# USER CONFIG (edit here)
# -------------------------
INPUT_NPY = "data/embeddings/instructor_umap_50d.npy"
OUT_JSON = "data/cluster/LSH-SRR/SRR_ON_INSTRUCTOR/srr_tuning_report_on_instructor_50d_v5.json"

# SRR params grid
DELTAS = [0.01]  # probability of missing a point
L_GBS = [4.0]  # memory constraint in GB
THREADS = 24

# DBSCAN params grid
MINPTS_VALUES = [10, 12, 15, 25]
EPS_QUANTILES = [
    0.70, 0.75, 0.80, 0.85,
    0.90]

KDIST_METRIC = "euclidean"
MAX_EPS_PER_MINPTS = 10

# Stability check
EPS_JITTER = 0.02  # +/-2%

# Ranking / filtering
NOISE_MAX = 0.20
MAX_CLUSTER_SHARE_MAX = 0.18
MIN_CLUSTERS = 50
TOPK = 10

KDIST_SAMPLE_SIZE = None

# Quality metrics config
SILHOUETTE_SAMPLE_SIZE = 10000  # Sample per velocizzare Silhouette su dataset grandi


# -------------------------
# Helpers
# -------------------------
def cluster_summary(labels: np.ndarray) -> dict:
    labels = np.asarray(labels)
    n = labels.size
    noise = int(np.sum(labels == -1))
    noise_frac = float(noise / n)

    core = labels[labels != -1]
    if core.size == 0:
        return {
            "n_clusters": 0,
            "noise": noise,
            "noise_frac": noise_frac,
            "max_cluster_share": 0.0,
            "median_cluster_size": 0.0,
            "p90_cluster_size": 0.0,
            "p99_cluster_size": 0.0,
        }

    uniq, counts = np.unique(core, return_counts=True)
    counts = np.sort(counts)
    return {
        "n_clusters": int(len(uniq)),
        "noise": noise,
        "noise_frac": noise_frac,
        "max_cluster_share": float(counts[-1] / n),
        "median_cluster_size": float(np.median(counts)),
        "p90_cluster_size": float(np.quantile(counts, 0.90)),
        "p99_cluster_size": float(np.quantile(counts, 0.99)),
    }


def compute_quality_metrics(X: np.ndarray, labels: np.ndarray, sample_size: int = 10000) -> dict:
    """
    Compute clustering quality metrics: Silhouette, Davies-Bouldin, Calinski-Harabasz.

    Args:
        X: embeddings (n_samples, n_features)
        labels: cluster labels (n_samples,), noise = -1
        sample_size: max samples for Silhouette computation

    Returns:
        dict with quality metrics
    """
    # Filter noise points
    mask = labels != -1
    X_filtered = X[mask]
    labels_filtered = labels[mask]

    n_clusters = len(np.unique(labels_filtered))

    # Need at least 2 clusters
    if n_clusters < 2 or len(X_filtered) < 2:
        return {
            'silhouette': None,
            'silhouette_sampled': False,
            'davies_bouldin': None,
            'calinski_harabasz': None,
            'n_points_used': 0
        }

    metrics = {
        'n_points_used': int(len(X_filtered))
    }

    # Silhouette Score (sample se troppi punti)
    try:
        if len(X_filtered) > sample_size:
            rng = np.random.default_rng(42)
            sample_idx = rng.choice(len(X_filtered), sample_size, replace=False)
            X_sample = X_filtered[sample_idx]
            labels_sample = labels_filtered[sample_idx]
            metrics['silhouette'] = float(silhouette_score(X_sample, labels_sample))
            metrics['silhouette_sampled'] = True
        else:
            metrics['silhouette'] = float(silhouette_score(X_filtered, labels_filtered))
            metrics['silhouette_sampled'] = False
    except Exception as e:
        metrics['silhouette'] = None
        metrics['silhouette_sampled'] = False
        metrics['silhouette_error'] = str(e)

    # Davies-Bouldin Index
    try:
        metrics['davies_bouldin'] = float(davies_bouldin_score(X_filtered, labels_filtered))
    except Exception as e:
        metrics['davies_bouldin'] = None
        metrics['davies_bouldin_error'] = str(e)

    # Calinski-Harabasz Score
    try:
        metrics['calinski_harabasz'] = float(calinski_harabasz_score(X_filtered, labels_filtered))
    except Exception as e:
        metrics['calinski_harabasz'] = None
        metrics['calinski_harabasz_error'] = str(e)

    return metrics


def run_srr(X: np.ndarray, delta: float, L_gb: float, n_threads: int, eps: float, minPts: int):
    srr = dbscan.SRR()
    t0 = time.time()
    labels = srr.fit_predict(X, delta, L_gb, n_threads, eps, minPts)
    dt = time.time() - t0
    stats = srr.statistics()
    return labels, float(dt), stats


def stability_metrics(base_labels: np.ndarray, alt_labels: np.ndarray) -> dict:
    return {
        "stability_ari": float(adjusted_rand_score(base_labels, alt_labels)),
        "stability_nmi": float(normalized_mutual_info_score(base_labels, alt_labels)),
    }


def compute_eps_candidates_from_kdist(X: np.ndarray, k: int, quantiles: list, metric: str, max_eps: int):
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric, algorithm="auto", n_jobs=-1)
    nn.fit(X)
    dists, _ = nn.kneighbors(X, return_distance=True)
    kth = dists[:, -1]
    qvals = np.quantile(kth, quantiles)

    eps_candidates = sorted(set(float(v) for v in qvals))
    if len(eps_candidates) > max_eps:
        idx = np.linspace(0, len(eps_candidates) - 1, max_eps).round().astype(int)
        eps_candidates = [eps_candidates[i] for i in idx]

    return {
        "k": int(k),
        "metric": metric,
        "kth_summary": {
            "mean": float(kth.mean()),
            "std": float(kth.std()),
            "min": float(kth.min()),
            "max": float(kth.max()),
        },
        "quantiles": {f"q{int(q * 100):02d}": float(v) for q, v in zip(quantiles, qvals)},
        "eps_candidates": eps_candidates,
    }


def rank_runs(runs: list) -> list:
    scored = []
    for r in runs:
        s = r["summary"]
        if s["n_clusters"] < MIN_CLUSTERS:
            continue
        if s["noise_frac"] > NOISE_MAX:
            continue
        if s["max_cluster_share"] > MAX_CLUSTER_SHARE_MAX:
            continue

        stab = r["stability"]
        stab_nmi = 0.5 * (stab["minus"]["stability_nmi"] + stab["plus"]["stability_nmi"])
        stab_ari = 0.5 * (stab["minus"]["stability_ari"] + stab["plus"]["stability_ari"])

        # Include quality metrics in score if available
        qm = r.get("quality_metrics", {})
        silhouette_bonus = 0.5 * (qm.get('silhouette') or 0.0)
        davies_bouldin_penalty = 0.3 * (qm.get('davies_bouldin') or 0.0)

        score = (
                2.0 * stab_nmi +
                1.0 * stab_ari +
                silhouette_bonus -
                davies_bouldin_penalty -
                1.0 * s["noise_frac"] -
                1.5 * s["max_cluster_share"] -
                0.01 * r["runtime_s"]
        )
        scored.append((score, stab_nmi, stab_ari, r))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


# -------------------------
# Main execution
# -------------------------
assert os.path.exists(INPUT_NPY), f"Input file not found: {INPUT_NPY}"

X_full = np.load(INPUT_NPY).astype(np.float32)
n, d = X_full.shape
print(f"Loaded X: shape={X_full.shape}")

if KDIST_SAMPLE_SIZE is not None and KDIST_SAMPLE_SIZE < n:
    rng = np.random.default_rng(0)
    idx = rng.choice(n, size=KDIST_SAMPLE_SIZE, replace=False)
    X_kdist = X_full[idx]
    print(f"Using k-distance sample: {X_kdist.shape}")
else:
    X_kdist = X_full

kdist_info = {}
eps_grid = {}

print("\n[Step 1] Building eps candidates via k-distance quantiles...")
# TQDM added for Step 1
for minPts in tqdm(MINPTS_VALUES, desc="Estimating Epsilons"):
    info = compute_eps_candidates_from_kdist(
        X_kdist, k=minPts, quantiles=EPS_QUANTILES, metric=KDIST_METRIC, max_eps=MAX_EPS_PER_MINPTS
    )
    kdist_info[str(minPts)] = info
    eps_grid[minPts] = info["eps_candidates"]
    # Usiamo tqdm.write per non rompere la barra
    tqdm.write(
        f"  minPts={minPts:4d}  eps_candidates={len(eps_grid[minPts])}  kth_mean={info['kth_summary']['mean']:.4f}")

# --- LINEARIZZAZIONE DEI PARAMETRI PER TQDM ---
# Creiamo una lista piatta di tutti i job da eseguire per avere una barra di progresso reale
param_grid = []
for delta in DELTAS:
    for L_gb in L_GBS:
        for minPts in MINPTS_VALUES:
            for eps in eps_grid[minPts]:
                param_grid.append({
                    "delta": delta,
                    "L_gb": L_gb,
                    "minPts": minPts,
                    "eps": eps
                })

runs = []
print(f"\n[Step 2] Running SRRDBSCAN grid + stability checks + quality metrics ({len(param_grid)} iterations)...")

# TQDM added for Step 2
for params in tqdm(param_grid, desc="Grid Search Progress", unit="run"):
    delta = params["delta"]
    L_gb = params["L_gb"]
    minPts = params["minPts"]
    eps = params["eps"]

    cfg = {"delta": float(delta), "L_gb": float(L_gb), "threads": int(THREADS),
           "eps": float(eps), "minPts": int(minPts)}

    # ⭐ Run base clustering (SALVA IN labels_base!)
    labels_base, dt, stats = run_srr(X_full, delta, L_gb, THREADS, eps, minPts)
    summ = cluster_summary(labels_base)

    # ⭐ Stability checks (labels separati)
    eps_minus = eps * (1.0 - EPS_JITTER)
    eps_plus = eps * (1.0 + EPS_JITTER)

    labels_m, dt_m, _ = run_srr(X_full, delta, L_gb, THREADS, eps_minus, minPts)
    labels_p, dt_p, _ = run_srr(X_full, delta, L_gb, THREADS, eps_plus, minPts)

    # ⭐ COMPUTE QUALITY METRICS (usa labels_base!)
    t_metrics_start = time.time()
    quality_metrics = compute_quality_metrics(X_full, labels_base, sample_size=SILHOUETTE_SAMPLE_SIZE)
    t_metrics = time.time() - t_metrics_start

    run_row = {
        **cfg,
        "runtime_s": float(dt),
        "metrics_computation_s": float(t_metrics),
        "summary": summ,
        "statistics": stats,
        "quality_metrics": quality_metrics,
        "stability": {
            "eps_minus": float(eps_minus),
            "eps_plus": float(eps_plus),
            "runtime_s_minus": float(dt_m),
            "runtime_s_plus": float(dt_p),
            "minus": stability_metrics(labels_base, labels_m),  # ⭐ Usa labels_base
            "plus": stability_metrics(labels_base, labels_p),  # ⭐ Usa labels_base
        }
    }
    runs.append(run_row)

    # Print con metriche
    sil_str = f"{quality_metrics.get('silhouette', 'N/A'):.3f}" if quality_metrics.get(
        'silhouette') is not None else "N/A"
    db_str = f"{quality_metrics.get('davies_bouldin', 'N/A'):.3f}" if quality_metrics.get(
        'davies_bouldin') is not None else "N/A"
    ch_str = f"{quality_metrics.get('calinski_harabasz', 'N/A'):.1f}" if quality_metrics.get(
        'calinski_harabasz') is not None else "N/A"

    tqdm.write(f"  done: δ={delta} L={L_gb} minPts={minPts} eps={eps:.6f} "
               f"clusters={summ['n_clusters']} noise={summ['noise_frac']:.3f} "
               f"max_share={summ['max_cluster_share']:.3f} "
               f"sil={sil_str} db={db_str} ch={ch_str} "
               f"time={dt:.2f}s (+{t_metrics:.2f}s metrics)")

    # Cleanup (NON cancellare labels_base prima delle metriche!)
    del labels_base, labels_m, labels_p, dt, stats, quality_metrics, t_metrics
    gc.collect()

report = {
    "meta": {
        "created_at_unix": int(time.time()),
        "input_npy": INPUT_NPY,
        "shape": {"n": int(n), "d": int(d)},
        "kdist_metric": KDIST_METRIC,
        "deltas": DELTAS,
        "L_gbs": L_GBS,
        "threads": THREADS,
        "minPts_values": MINPTS_VALUES,
        "eps_quantiles": EPS_QUANTILES,
        "eps_jitter": EPS_JITTER,
        "max_eps_per_minpts": MAX_EPS_PER_MINPTS,
        "kdist_sample_size": KDIST_SAMPLE_SIZE,
        "silhouette_sample_size": SILHOUETTE_SAMPLE_SIZE,
    },
    "kdist": kdist_info,
    "runs": runs,
    "notes": [
        "No ground-truth labels used.",
        "No ARI/NMI vs exact DBSCAN computed.",
        "ARI/NMI are used only as stability measures between SRR runs at eps*(1±jitter).",
        "eps candidates are derived from k-distance quantiles (k=minPts).",
        "Quality metrics: Silhouette (higher better, [-1,1]), Davies-Bouldin (lower better, [0,∞)), Calinski-Harabasz (higher better, [0,∞)).",
        f"Silhouette computed on sample of {SILHOUETTE_SAMPLE_SIZE} points if dataset larger."
    ]
}

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, default=str)

print(f"\nSaved report to: {OUT_JSON}")

ranked = rank_runs(runs)
print("\nTop candidates (label-free ranking with quality metrics):")
for i, (score, stab_nmi, stab_ari, r) in enumerate(ranked[:TOPK], start=1):
    s = r["summary"]
    qm = r.get("quality_metrics", {})
    sil = qm.get('silhouette')
    db = qm.get('davies_bouldin')
    ch = qm.get('calinski_harabasz')

    sil_str = f"{sil:.3f}" if sil is not None else "N/A"
    db_str = f"{db:.3f}" if db is not None else "N/A"
    ch_str = f"{ch:.1f}" if ch is not None else "N/A"

    print(f"{i:02d}) score={score:.4f} "
          f"δ={r['delta']} L={r['L_gb']} minPts={r['minPts']} eps={r['eps']:.6f} "
          f"clusters={s['n_clusters']} noise={s['noise_frac']:.3f} max_share={s['max_cluster_share']:.3f} "
          f"sil={sil_str} db={db_str} ch={ch_str} "
          f"stabNMI={stab_nmi:.3f} stabARI={stab_ari:.3f} time={r['runtime_s']:.2f}s")
