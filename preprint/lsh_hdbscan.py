
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm.auto import tqdm

try:
    # se il file si chiama lsh_dbscan.py
    from lsh_dbscan import LSHApproximateDBSCAN
except ImportError:
    try:
        from lshdbscan import LSHApproximateDBSCAN
    except ImportError:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from lsh_dbscan import LSHApproximateDBSCAN


def _count_clusters(labels: np.ndarray) -> int:
    labels = np.asarray(labels)
    if labels.size == 0:
        return 0
    uniq = np.unique(labels)
    if uniq.size == 0:
        return 0
    return int(np.sum(uniq != -1))


def _normalize_two_levels(value, default):
    """
    centerratio/ratiooffset nel tuo LSHApproximateDBSCAN vengono usati come "due livelli"
    (step core-point vs step cluster-formation). Qui accettiamo:
      - None -> (default, default)
      - scalare -> (value, value)
      - tuple/list length 2 -> tuple(value0, value1)
    """
    if value is None:
        return (default, default)
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) != 2:
            raise ValueError("Atteso tuple/list di lunghezza 2 per i due livelli.")
        return (value[0], value[1])
    return (value, value)


def _approx_diameter_in_range_D_to_2D(X: np.ndarray) -> float:
    """
    Restituisce Dhat in [Dmax, 2*Dmax] con una scansione lineare (2-approx del diametro).
    """
    X = np.asarray(X)
    n = X.shape[0]
    if n < 2:
        return 0.0
    p = X[0]
    dists = np.linalg.norm(X - p, axis=1)
    return 2.0 * float(np.max(dists))


def _estimate_min_distance_by_random_pairs(
    X: np.ndarray,
    num_pairs: int = 200_000,
    rng: Optional[np.random.Generator] = None,
    eps_floor: float = 1e-12,
) -> float:
    """
    Stima una distanza minima 'tipica' campionando coppie (i, j).
    """
    X = np.asarray(X)
    n = X.shape[0]
    if n < 2:
        return eps_floor
    if rng is None:
        rng = np.random.default_rng(42)

    I = rng.integers(0, n, size=num_pairs, dtype=np.int64)
    J = rng.integers(0, n, size=num_pairs, dtype=np.int64)
    mask = I != J
    if not np.any(mask):
        return eps_floor
    I, J = I[mask], J[mask]

    d = np.linalg.norm(X[I] - X[J], axis=1)
    d = d[d > 0.0]
    if d.size == 0:
        return eps_floor
    return max(float(np.min(d)), eps_floor)


def _estimate_aspect_ratio_delta(
    X: np.ndarray,
    num_pairs: int = 200_000,
    rng_seed: int = 42,
    eps_floor: float = 1e-12,
) -> Tuple[float, float, float]:
    """
    Ritorna (Delta_hat, Dmax_hat, dmin_hat) con:
      - Dmax_hat in [Dmax, 2Dmax]
      - dmin_hat da campionamento
      - Delta_hat = Dmax_hat / dmin_hat (clippata a > 1)
    """
    rng = np.random.default_rng(rng_seed)
    Dmax_hat = _approx_diameter_in_range_D_to_2D(X)
    dmin_hat = _estimate_min_distance_by_random_pairs(X, num_pairs=num_pairs, rng=rng, eps_floor=eps_floor)

    if dmin_hat <= 0:
        Delta_hat = float("inf")
    else:
        Delta_hat = float(Dmax_hat / dmin_hat)

    Delta_hat = max(Delta_hat, 1.0 + 1e-9)
    return Delta_hat, Dmax_hat, dmin_hat


def clustering_intersection_labels(labels_a: np.ndarray, labels_b: np.ndarray) -> np.ndarray:
    """
    Intersection tra due clusterings (Definition 3.1) in rappresentazione "labels":
      - rumore = -1
      - cluster id >= 0

    Regola: un punto è rumore se è rumore in almeno uno dei due; altrimenti la nuova etichetta
    è determinata dalla coppia (label_a, label_b), che corrisponde all’intersezione non-vuota.
    """
    a = np.asarray(labels_a, dtype=int)
    b = np.asarray(labels_b, dtype=int)
    if a.shape != b.shape:
        raise ValueError("labels_a e labels_b devono avere la stessa shape")

    n = a.shape[0]
    out = np.full(n, -1, dtype=int)

    ok = (a != -1) & (b != -1)
    if not np.any(ok):
        return out

    pairs = np.stack([a[ok], b[ok]], axis=1)

    # Assegna un id cluster per ogni coppia unica (a_id, b_id)
    # return_inverse produce un indice 0..k-1 per ogni riga di pairs
    _, inv = np.unique(pairs, axis=0, return_inverse=True)
    out[ok] = inv.astype(int)
    return out


@dataclass
class LSHHDBSCANPaper:
    """
    Implementazione paper-style dell'Algorithm 4 (LSH-HDBSCAN):
      - Stima Δ (aspect ratio) in modo approssimato
      - Calcola L = 1 + ceil(log_{1+gamma}(Δ))
      - Per i=1..L:
          eps_i = Dmax_hat*(1+gamma)^(1-i)
          B_i = LSH-DBSCAN(X, eps_i, m, c, delta/L)
          C_i = C_{i-1} ∩ B_i
      - tqdm per mostrare ETA/residuo sui livelli
    """
    minpts: int = 5
    c: float = 1.5
    delta: float = 0.5
    gamma: float = 0.5

    mineps_target: Optional[float] = None
    maxeps_target: Optional[float] = None

    centerratio: Optional[object] = None   # None, scalare, o (lvl0,lvl1)
    ratiooffset: Optional[object] = None   # None, scalare, o (lvl0,lvl1)

    estimate_pairs: int = 200_000
    rng_seed: int = 42

    verbose_levels: bool = True            # tqdm + postfix
    verbose_lshdbscan: bool = False        # se True, LSHApproximateDBSCAN stampa progress bar propria

    # output
    hierarchy_: Optional[List[Dict]] = None
    eps_: Optional[List[float]] = None
    labels_: Optional[np.ndarray] = None
    delta_hat_: Optional[float] = None
    dmax_hat_: Optional[float] = None
    dmin_hat_: Optional[float] = None

    def fit(self, X: np.ndarray) -> "LSHHDBSCANPaper":
        X = np.asarray(X)
        n = X.shape[0]
        if n == 0:
            raise ValueError("X è vuoto")

        centerratio2 = _normalize_two_levels(self.centerratio, default=1.0)
        ratiooffset2 = _normalize_two_levels(self.ratiooffset, default=0)

        # 1) Stima Δ e Dmax (come nota del paper: Dmax in [Dmax,2Dmax] in tempo lineare)
        Delta_hat, Dmax_hat, dmin_hat = _estimate_aspect_ratio_delta(
            X,
            num_pairs=self.estimate_pairs,
            rng_seed=self.rng_seed,
        )
        self.delta_hat_ = float(Delta_hat)
        self.dmax_hat_ = float(Dmax_hat)
        self.dmin_hat_ = float(dmin_hat)

        # 2) Calcolo L (Algorithm 4)
        base = 1.0 + float(self.gamma)
        if base <= 1.0:
            raise ValueError("gamma deve essere > 0 (base = 1+gamma)")

        # Se l’utente impone un range eps, usalo per definire L
        if self.mineps_target is not None and self.maxeps_target is not None:
            mineps = float(self.mineps_target)
            maxeps = float(self.maxeps_target)
            if mineps <= 0 or maxeps <= 0 or mineps >= maxeps:
                raise ValueError("mineps_target/maxeps_target non validi")

            # Definisci Delta_hat e Dmax_hat "target" solo per costruire i livelli
            Delta_hat = maxeps / mineps
            Dmax_hat = maxeps

            # (coerente con la forma eps_i = Dmax_hat * (1+gamma)^(1-i)) [file:773]
            L = 1 + int(math.ceil(math.log(Delta_hat, base)))
            L = max(L, 1)

            # (opzionale) aggiorna anche gli attributi di diagnostica
            self.delta_hat_ = float(Delta_hat)
            self.dmax_hat_ = float(Dmax_hat)
            self.dmin_hat_ = float(mineps)

        else:
            # ramo originale: stima Delta_hat, Dmax_hat, dmin_hat dal dataset [file:773]
            Delta_hat, Dmax_hat, dmin_hat = _estimate_aspect_ratio_delta(
                X,
                num_pairs=self.estimate_pairs,
                rng_seed=self.rng_seed,
            )
            self.delta_hat_ = float(Delta_hat)
            self.dmax_hat_ = float(Dmax_hat)
            self.dmin_hat_ = float(dmin_hat)

        delta_per_level = float(self.delta) / float(L)

        # 3) Inizializza C_0 come clustering "tutto insieme" (nessun rumore)
        labels_prev = np.zeros(n, dtype=int)

        hierarchy: List[Dict] = []
        eps_list: List[float] = []

        iterator = range(1, L + 1)
        if self.verbose_levels:
            iterator = tqdm(iterator, total=L, desc="LSH-HDBSCAN levels", unit="level", dynamic_ncols=True)

        for i in iterator:
            # eps_i = Dmax * (1+gamma)^(1-i)
            eps_i = float(Dmax_hat * (base ** (1.0 - float(i))))
            eps_list.append(eps_i)

            model = LSHApproximateDBSCAN(
                eps_i,  # eps (posizionale)
                int(self.minpts),  # min_pts (posizionale)
                c=float(self.c),
                delta=float(delta_per_level),
                center_ratio=centerratio2,  # NOTA: nome corretto
                ratio_offset=ratiooffset2,  # NOTA: nome corretto
                verbose=bool(self.verbose_lshdbscan),
            )
            model.fit(X)
            labels_b = np.asarray(model.labels_, dtype=int)

            # C_i = C_{i-1} ∩ B_i
            labels_ci = clustering_intersection_labels(labels_prev, labels_b)

            hierarchy.append(
                {
                    "eps": eps_i,
                    "labels": labels_ci,
                    "nclusters_level": _count_clusters(labels_b),
                    "nclusters_hierarchy": _count_clusters(labels_ci),
                }
            )
            labels_prev = labels_ci

            if self.verbose_levels and isinstance(iterator, tqdm):
                iterator.set_postfix(
                    {
                        "eps": f"{eps_i:.3g}",
                        "B_i": hierarchy[-1]["nclusters_level"],
                        "C_i": hierarchy[-1]["nclusters_hierarchy"],
                    }
                )

        self.hierarchy_ = hierarchy
        self.eps_ = eps_list
        self.labels_ = labels_prev
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return np.asarray(self.labels_, dtype=int)
