import numpy as np

X = np.load("data/embeddings/all-MiniLM-L6-v2_mean_centered.npy")
rng = np.random.default_rng(42)
n = X.shape[0]

# Campiona 100k coppie random
num_samples = min(100_000, n * (n-1) // 2)
i = rng.integers(0, n, size=num_samples)
j = rng.integers(0, n, size=num_samples)
mask = i != j
dists = np.linalg.norm(X[i[mask]] - X[j[mask]], axis=1)

print(f"Distanze nel dataset (su {len(dists)} coppie):")
print(f"  min:  {np.min(dists):.6f}")
print(f"  q5:   {np.percentile(dists, 5):.6f}")
print(f"  q25:  {np.percentile(dists, 25):.6f}")
print(f"  q50:  {np.percentile(dists, 50):.6f}")  # MEDIANA
print(f"  q75:  {np.percentile(dists, 75):.6f}")
print(f"  q95:  {np.percentile(dists, 95):.6f}")
print(f"  max:  {np.max(dists):.6f}")
