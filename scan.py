import os, time, json
import numpy as np
import itertools  # Standard library, già inclusa in Python
from tqdm import tqdm  # pip install tqdm
import multiprocessing

import dbscan_srr as dbscan
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# -------------------------
# USER CONFIG (edit here)
# -------------------------
INPUT_NPY = "data/all-MiniLM-L6-v2_mean_centered.npy"
OUT_JSON = "data/srr_tuning_report.json"

# SRR params grid
DELTAS = [0.1, 0.05]  # probability of missing a point
L_GBS = [4.0]  # memory constraint in GB
THREADS = 24

# DBSCAN params grid
MINPTS_VALUES = [30, 50, 80, 120]
EPS_QUANTILES = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
KDIST_METRIC = "euclidean"
MAX_EPS_PER_MINPTS = 12

# Stability check
EPS_JITTER = 0.02  # +/-2%

# Ranking / filtering
NOISE_MAX = 0.80
MAX_CLUSTER_SHARE_MAX = 0.50
MIN_CLUSTERS = 2
TOPK = 10

KDIST_SAMPLE_SIZE = None  # None = use full X


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

        score = (
                2.0 * stab_nmi +
                1.0 * stab_ari -
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
print(f"\n[Step 2] Running SRRDBSCAN grid + stability checks ({len(param_grid)} iterations)...")

# TQDM added for Step 2
for params in tqdm(param_grid, desc="Grid Search Progress", unit="run"):
    delta = params["delta"]
    L_gb = params["L_gb"]
    minPts = params["minPts"]
    eps = params["eps"]

    cfg = {"delta": float(delta), "L_gb": float(L_gb), "threads": int(THREADS),
           "eps": float(eps), "minPts": int(minPts)}

    labels, dt, stats = run_srr(X_full, delta, L_gb, THREADS, eps, minPts)
    summ = cluster_summary(labels)

    eps_minus = eps * (1.0 - EPS_JITTER)
    eps_plus = eps * (1.0 + EPS_JITTER)

    labels_m, dt_m, _ = run_srr(X_full, delta, L_gb, THREADS, eps_minus, minPts)
    labels_p, dt_p, _ = run_srr(X_full, delta, L_gb, THREADS, eps_plus, minPts)

    run_row = {
        **cfg,
        "runtime_s": float(dt),
        "summary": summ,
        "statistics": stats,
        "stability": {
            "eps_minus": float(eps_minus),
            "eps_plus": float(eps_plus),
            "runtime_s_minus": float(dt_m),
            "runtime_s_plus": float(dt_p),
            "minus": stability_metrics(labels, labels_m),
            "plus": stability_metrics(labels, labels_p),
        }
    }
    runs.append(run_row)

    # Usa tqdm.write invece di print
    tqdm.write(f"  done: δ={delta} L={L_gb} minPts={minPts} eps={eps:.6f} "
               f"clusters={summ['n_clusters']} noise={summ['noise_frac']:.3f} "
               f"max_share={summ['max_cluster_share']:.3f} time={dt:.2f}s")

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
    },
    "kdist": kdist_info,
    "runs": runs,
    "notes": [
        "No ground-truth labels used.",
        "No ARI/NMI vs exact DBSCAN computed.",
        "ARI/NMI are used only as stability measures between SRR runs at eps*(1±jitter).",
        "eps candidates are derived from k-distance quantiles (k=minPts).",
    ]
}

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, default=str)

print(f"\nSaved report to: {OUT_JSON}")

ranked = rank_runs(runs)
print("\nTop candidates (label-free ranking):")
for i, (score, stab_nmi, stab_ari, r) in enumerate(ranked[:TOPK], start=1):
    s = r["summary"]
    print(f"{i:02d}) score={score:.4f} "
          f"δ={r['delta']} L={r['L_gb']} minPts={r['minPts']} eps={r['eps']:.6f} "
          f"clusters={s['n_clusters']} noise={s['noise_frac']:.3f} max_share={s['max_cluster_share']:.3f} "
          f"stabNMI={stab_nmi:.3f} stabARI={stab_ari:.3f} time={r['runtime_s']:.2f}s")
