import sys
import numpy as np
import math
import random
from typing import List, Set, Dict, Callable, Union, Sequence, Tuple
from scipy.stats import norm
from preprint.utils.optimal_w import get_cached_optimal_w


def _as_pair(x, default):
    if x is None:
        x = default
    if isinstance(x, (list, tuple)) and len(x) == 2:
        return [x[0], x[1]]
    return [x, x]


class LSHApproximateDBSCAN:
    def __init__(self, eps: float,
                 min_pts: int,
                 c: float = 1.5,
                 delta: float = 0.5,
                 center_ratio: Union[float, Sequence[float]] = 1.0,
                 ratio_offset: Union[int, Sequence[int]] = 0,
                 verbose: bool = True):

        self.eps = eps
        self.min_pts = min_pts
        self.c = c
        self.delta = delta
        self.center_ratio = center_ratio
        self.ratio_offset = ratio_offset
        self.verbose = verbose
        self.labels_ = None
        self.distance_count = 0  # Counter for distance computations
        self.hash_count = 0  # Counter for hash function computations

    def _distance(self, x: np.ndarray, y: np.ndarray) -> float:
        self.distance_count += 1
        return np.linalg.norm(x - y)

    def fit(self, X: np.ndarray) -> 'LSHApproximateDBSCAN':
        n_samples = X.shape[0]
        self.labels_ = np.full(n_samples, -1, dtype=int)

        self.lsh_family = self._construct_lsh_family(X)

        if self.verbose:
            print('lsh family', self.lsh_family)

        # Step 1: Core point identification
        core_points = self._core_point_identification(X)

        # Step 2: Cluster formation
        clusters = self._cluster_formation(X, core_points)

        # Convert clusters to labels
        for cluster_id, cluster in enumerate(clusters):
            for point_idx in cluster:
                self.labels_[point_idx] = cluster_id

        return self

    def _core_point_identification(self, X: np.ndarray) -> Set[int]:
        n = X.shape[0]
        core_points = set()
        K_true = max(1, math.log(n / self.min_pts) / math.log(1 / self.lsh_family['p2']))
        T = math.ceil(self.lsh_family['p1'] ** (-K_true) * math.log(2 * n * self.min_pts / self.delta))
        ratio_offset_cpi = self.ratio_offset[0]
        center_ratio_cpi = self.center_ratio[0]
        K = max(1, math.floor(K_true * center_ratio_cpi + ratio_offset_cpi))
        hash_functions = self._sample_hash_functions(X, self.lsh_family, K, T, self.eps / self.c)

        if self.verbose:
            print(f"Core Point Formation: K: {K_true}->{K}, T: {T}")

        # Create T explicit hash tables
        hash_tables = self._build_hash_tables(X, hash_functions)

        for i in range(n):
            if (n - i - 1) % 10 == 0:
                progress = int(50 * (i + 1) / n)  # 50 chars wide
                bar = "[" + "#" * progress + "-" * (50 - progress) + "]"
                sys.stdout.write(f"\r{bar} {i + 1}/{n} processed, {len(core_points)} core points found")
                sys.stdout.flush()
            found = set()
            for t in range(T):
                bucket = hash_functions[t](X[i])  # Returns a tuple of K hash values
                for j in hash_tables[t][bucket]:
                    if self._distance(X[i], X[j]) <= self.eps and j not in found:
                        found.add(j)
                        if len(found) >= self.min_pts:
                            core_points.add(i)
                            break
                if len(found) >= self.min_pts:
                    break
        print()

        if self.verbose:
            print(f'Done! (distance,hash) count ({self.distance_count},{self.hash_count})')
        return core_points

    def _cluster_formation(self, X: np.ndarray, core_points: Set[int]) -> List[Set[int]]:
        n = X.shape[0]
        K_true = max(1, math.log(1 + len(core_points)) / math.log(1 / self.lsh_family['p2']))
        T = math.ceil(self.lsh_family['p1'] ** (-K_true) * math.log(2 * n / self.delta))
        ratio_offset_cf = self.ratio_offset[1]
        center_ratio_cf = self.center_ratio[1]
        K = max(1, math.floor(K_true * center_ratio_cf + ratio_offset_cf))
        if self.verbose:
            print(f"Cluster Formation: K: {K_true}->{K}, T: {T}")
        hash_functions = self._sample_hash_functions(X, self.lsh_family, K, T, self.eps / self.c)
        hash_tables = self._build_hash_tables(X, hash_functions)

        unclustered_points = core_points.copy()
        core_size = len(core_points)
        clusters = []
        while (unclustered_points):
            p = next(iter(unclustered_points))
            Q = [p]
            unclustered_points.remove(p)
            if len(unclustered_points) % 10 == 0:
                progress = int(50 * (core_size - len(unclustered_points)) / core_size)
                bar = "[" + "#" * progress + "-" * (50 - progress) + "]"
                sys.stdout.write(f"\r{bar} {core_size - len(unclustered_points)}/{core_size} cluster formation")
                sys.stdout.flush()
            current_cluster = set()

            # Remove p from all hash tables once
            for t in range(T):
                bucket = hash_functions[t](X[p])
                hash_tables[t][bucket].discard(p)

            while Q:
                # if len(unclustered_points) % 500 == 0:
                #     print(len(unclustered_points))
                q = Q.pop(0)
                current_cluster.add(q)
                # Find neighbors using LSH
                neighbors_found = set()
                for t in range(T):
                    bucket = hash_functions[t](X[q])
                    # Create a copy to avoid modification during iteration
                    bucket_points = list(hash_tables[t][bucket])
                    for j in bucket_points:
                        if j in unclustered_points and j not in neighbors_found:
                            if self._distance(X[q], X[j]) <= self.eps:
                                Q.append(j)
                                unclustered_points.remove(j)
                                if len(unclustered_points) % 10 == 0:
                                    progress = int(50 * (core_size - len(unclustered_points)) / core_size)
                                    bar = "[" + "#" * progress + "-" * (50 - progress) + "]"
                                    sys.stdout.write(
                                        f"\r{bar} {core_size - len(unclustered_points)}/{core_size} cluster formation")
                                    sys.stdout.flush()
                                neighbors_found.add(j)
                                # Remove j from all hash tables
                                for t_ in range(T):
                                    bucket_j = hash_functions[t_](X[j])
                                    hash_tables[t_][bucket_j].discard(j)

            clusters.append(current_cluster)
        print()
        if self.verbose:
            print(f'Done! (distance,hash) count ({self.distance_count},{self.hash_count})')
        return clusters

    def _build_hash_tables(self, X: np.ndarray, hash_functions: List[Callable]) -> List[Dict]:
        n = X.shape[0]
        hash_tables = []
        hi = 0
        hn = len(hash_functions)
        for hash_func in hash_functions:
            table = {}
            if (hn - hi - 1) % 1 == 0:
                progress = int(50 * (hi + 1) / hn)  # 50 chars wide
                bar = "[" + "#" * progress + "-" * (50 - progress) + "]"
                sys.stdout.write(f"\r{bar} {hi + 1}/{hn} hash tables built")
                sys.stdout.flush()
            hi += 1
            for i in range(n):

                bucket = hash_func(X[i])  # Returns a tuple of K hash values
                if bucket not in table:
                    table[bucket] = set()
                table[bucket].add(i)
            hash_tables.append(table)
        return hash_tables

    def _construct_lsh_family(self, X: np.ndarray) -> Dict:
        # Calculate optimal w parameter based on approximation factor c and dataset size
        n = X.shape[0]
        w = get_cached_optimal_w(self.c, n, verbose=self.verbose)

        if self.verbose:
            print(f'Using optimal w = {w:.6f} for c={self.c}, n={n}')

        d = X.shape[1]  # dimensionality

        # LSH probabilities for (eps/c, eps)-sensitive family
        # p1: probability of collision for distance <= eps/c
        # p2: probability of collision for distance <= eps
        p1 = 1 - 2 * norm.cdf(-w) - (2 / (math.sqrt(2 * math.pi) * w)) * (1 - math.exp(-(w ** 2 / 2)))
        p2 = 1 - 2 * norm.cdf(-w / self.c) - (2 / (math.sqrt(2 * math.pi) * w / self.c)) * (
                    1 - math.exp(-(w ** 2 / (2 * self.c ** 2))))

        return {
            'p1': p1,
            'p2': p2,
            'w': w,
            'd': d,
        }

    def _sample_hash_functions(self, X: np.ndarray, lsh_family: Dict, K: int, T: int, eps: float) -> List[Callable]:
        w = lsh_family['w']
        w_scaled = w * eps
        d = lsh_family['d']

        rng = np.random.default_rng(42)
        # Pre-sample all the a and b vectors for consistency (K*T total)
        # Store as 2D matrices: a_matrix[t][k] and b_matrix[t][k]
        a_matrix = []
        b_matrix = []
        for t in range(T):
            a_row = []
            b_row = []
            for k in range(K):
                a_tk = rng.standard_normal(d)
                a_row.append(a_tk)
                b_tk = rng.uniform(0, w_scaled)
                b_row.append(b_tk)
            a_matrix.append(a_row)
            b_matrix.append(b_row)

        hash_functions = []
        for t in range(T):
            # Create a hash function that returns a K-dimensional vector
            def make_hash_function(t_idx):
                def hash_func(x):
                    # Compute K hash values for this repetition
                    hash_values = []
                    for k in range(K):
                        a = a_matrix[t_idx][k]
                        b = b_matrix[t_idx][k]
                        self.hash_count += 1
                        hash_values.append(math.floor((np.dot(x, a) + b) / w_scaled))
                    return tuple(hash_values)

                return hash_func

            hash_functions.append(make_hash_function(t))
        return hash_functions
