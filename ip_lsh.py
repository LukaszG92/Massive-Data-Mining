import numpy as np

X = np.load("data/embeddings/embeddings_v1_384.npy")
# Scrive una riga per punto, valori separati da spazio
np.savetxt("data/embeddings/alpaca_embeddings.txt", X, fmt="%.8g")
