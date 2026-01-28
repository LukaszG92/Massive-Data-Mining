import sys
import numpy as np
from typing import List, Set


class ExactDBSCAN:
    def __init__(self, eps: float, min_pts: int):
        self.eps = eps
        self.min_pts = min_pts
        self.labels_ = None
        self.distance_count = 0  # Counter for distance computations
        
    def fit(self, X: np.ndarray) -> 'ExactDBSCAN':
        n_samples = X.shape[0]
        self.labels_ = np.full(n_samples, -1, dtype=int)
        
        # Step 1: Core point identification
        core_points = self._core_point_identification(X)
        
        # Step 2: Cluster formation
        clusters = self._cluster_formation(X, core_points)
        
        # Step 3: Convert clusters to labels
        for cluster_id, cluster in enumerate(clusters):
            for point_idx in cluster:
                self.labels_[point_idx] = cluster_id
                
        return self
    
    def _distance(self, x: np.ndarray, y: np.ndarray) -> float:
        self.distance_count += 1
        return np.linalg.norm(x - y)
    
    def _core_point_identification(self, X: np.ndarray) -> Set[int]:
        n = X.shape[0]
        core_points: Set[int] = set()

        for i in range(n):
            if (n-i-1) % 10 == 0: # Progress bar update every 10 points
                progress = int(50 * (i+1) / n)
                bar = "[" + "#" * progress + "-" * (50 - progress) + "]"
                sys.stdout.write(f"\r{bar} {i+1}/{n} processed, {len(core_points)} core points found")
                sys.stdout.flush()
            neighbor_count = 0
            for j in range(n):
                if self._distance(X[i], X[j]) <= self.eps:
                    neighbor_count += 1
            if neighbor_count >= self.min_pts:
                core_points.add(i)
        print()
        
        return core_points
    

    def _cluster_formation(self, X: np.ndarray, core_points: Set[int]) -> List[Set[int]]:
        n = X.shape[0]
        core_size = len(core_points)
        
        # Cluster formation using BFS over core points
        unclustered_points = core_points.copy()
        clusters: List[Set[int]] = []
        
        while unclustered_points:
            p = next(iter(unclustered_points))
            queue = [p]
            unclustered_points.remove(p)
            current_cluster: Set[int] = set()
            
            # Progress tracking
            if len(unclustered_points) % 10 == 0:
                progress = int(50 * (core_size - len(unclustered_points)) / core_size)
                bar = "[" + "#" * progress + "-" * (50 - progress) + "]"
                sys.stdout.write(f"\r{bar} {core_size - len(unclustered_points)}/{core_size} cluster formation")
                sys.stdout.flush()
            
            while queue:
                q = queue.pop(0)
                current_cluster.add(q)
                
                # Brute-force neighbor search for exact DBSCAN
                for j in range(n):
                    if j in core_points and j in unclustered_points:
                        if self._distance(X[q], X[j]) <= self.eps:
                            queue.append(j)
                            unclustered_points.remove(j)
                            if len(unclustered_points) % 10 == 0:
                                progress = int(50 * (core_size - len(unclustered_points)) / core_size)
                                bar = "[" + "#" * progress + "-" * (50 - progress) + "]"
                                sys.stdout.write(f"\r{bar} {core_size - len(unclustered_points)}/{core_size} cluster formation")
                                sys.stdout.flush()
            
            clusters.append(current_cluster)
        print()
        return clusters
