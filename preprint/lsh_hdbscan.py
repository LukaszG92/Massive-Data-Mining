import numpy as np
import time
from joblib import Parallel, delayed
import os
import sys
import tempfile

try:
    from lsh_dbscan import LSHApproximateDBSCAN
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from lsh_dbscan import LSHApproximateDBSCAN


def _fit_single_level(eps, min_pts, c, delta, center_ratio, ratio_offset):
    """
    Funzione helper eseguita da ogni processo worker.
    Fitta un singolo livello di epsilon.
    """
    # Fix per il bug dei tipi (lista vs scalare)
    c_ratio_list = [center_ratio, center_ratio]
    r_offset_list = [ratio_offset, ratio_offset]

    model = LSHApproximateDBSCAN(
        eps=eps,
        min_pts=min_pts,
        c=c,
        delta=delta,
        center_ratio=c_ratio_list,
        ratio_offset=r_offset_list,
        verbose=False  # Importante: spegnere i print nei processi figli
    )

    # Assumiamo che X venga passato tramite memoria condivisa (joblib lo gestisce bene con memmap)
    # Ma per sicurezza, joblib serializza. Se X è enorme (es. 10GB), serve cautela.
    # Qui viene passato come argomento alla funzione wrapper chiamata da Parallel.
    return model


def _run_level(X, eps, params):
    """Wrapper per passare X e i parametri"""
    model = _fit_single_level(eps, **params)
    model.fit(X)

    # Ritorniamo solo ciò che serve (etichette e eps), non tutto l'oggetto pesante
    labels = model.labels_.copy()
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    return {
        'eps': eps,
        'labels': labels,
        'n_clusters': n_clusters
    }


class LSHHDBSCAN_Parallel:
    def __init__(self,
                 min_pts=5,
                 min_eps=0.4,
                 max_eps=1.2,
                 num_levels=10,
                 c=1.5,
                 delta=0.5,
                 center_ratio=1.0,
                 ratio_offset=0,
                 n_jobs=-1):  # -1 usa tutti i core disponibili

        self.min_pts = min_pts
        self.min_eps = min_eps
        self.max_eps = max_eps
        self.num_levels = num_levels
        self.c = c
        self.delta = delta
        self.center_ratio = center_ratio
        self.ratio_offset = ratio_offset
        self.n_jobs = n_jobs

        self.eps_levels = np.geomspace(max_eps, min_eps, num=num_levels)
        self.hierarchy = []
        self.labels_ = None

    def fit_predict(self, X):
        n_samples = X.shape[0]
        print(f"Running Parallel LSH-HDBSCAN on {n_samples} points using {self.n_jobs} jobs...")
        print(f"Levels: {self.num_levels} | Range: [{self.min_eps:.4f} - {self.max_eps:.4f}]")

        params = {
            'min_pts': self.min_pts,
            'c': self.c,
            'delta': self.delta,
            'center_ratio': self.center_ratio,
            'ratio_offset': self.ratio_offset
        }

        # 1. PARALLEL FIT
        # delayed(_run_level)(X, eps, params) prepara la chiamata
        # Parallel(...) la esegue distribuendola sui core

        start_p = time.time()

        temp_folder = "/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir()

        results = Parallel(
            n_jobs=self.n_jobs,
            backend="loky",
            max_nbytes="10M",
            mmap_mode="r",  # read-only: evita copy e scritture accidentali
            temp_folder=temp_folder  # dove creare i file temporanei
        )(
            delayed(_run_level)(X, eps, params) for eps in self.eps_levels
        )


        print(f"Parallel processing done in {time.time() - start_p:.2f}s. Building hierarchy...")

        self.hierarchy = results

        # 2. ESTRAZIONE CLUSTER (Identica a prima, veloce e single-thread)
        self.labels_ = self._extract_stable_clusters_bottom_up(n_samples)
        return self.labels_

    def _extract_stable_clusters_bottom_up(self, n_samples):
        # Ordina dal più DENSO (eps piccolo) al più LASCO
        sorted_hierarchy = sorted(self.hierarchy, key=lambda x: x['eps'])

        final_assignments = np.full(n_samples, -1, dtype=int)
        next_cluster_id = 0
        assigned_mask = np.zeros(n_samples, dtype=bool)

        for level_data in sorted_hierarchy:
            lbls = level_data['labels']

            # Punti clusterizzati a questo livello E non ancora assegnati
            candidates_mask = (lbls != -1) & (~assigned_mask)

            if not np.any(candidates_mask):
                continue

            unique_ids = np.unique(lbls[candidates_mask])

            for uid in unique_ids:
                points_in_cluster_mask = (lbls == uid) & candidates_mask

                if np.sum(points_in_cluster_mask) >= self.min_pts:
                    final_assignments[points_in_cluster_mask] = next_cluster_id
                    assigned_mask[points_in_cluster_mask] = True
                    next_cluster_id += 1

        return final_assignments
