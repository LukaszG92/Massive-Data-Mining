import json
import os
from pathlib import Path

import numpy as np
import torch

from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize


def generate_text_vectors(input_filename):
    print(f"Caricamento dati da {input_filename}...")

    with open(input_filename, "r") as f:
        origin_data = json.load(f)

    texts = []
    for item in origin_data:
        instruction = "Instruction: "+ item["instruction"] + ' Input: ' + item["input"] + ' Response: ' + item["output"]
        texts.append(instruction)

    print(f"Pronti {len(texts)} testi per l'embedding.")
    return texts


def run_embedding(model_name, vec_text, output_filename):
    device = "cuda" if torch.cuda.is_available() else "cpu"


    print(f"\n--- Avvio Encoding: {model_name} su {device} ---")
    model = SentenceTransformer(model_name)

    embeddings = model.encode(vec_text, show_progress_bar=True, device=device)

    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    np.save(output_filename, embeddings)
    print(f"Salvato: {output_filename} (Shape: {embeddings.shape})")


def normalize_embedding(input_filename, output_filename):
    try:
        X = np.load(input_filename)
        print(f"Dati caricati: {input_filename}")
        print(f"Shape originale: {X.shape} | dtype: {X.dtype}")
    except FileNotFoundError:
        raise FileNotFoundError(f"File non trovato: {input_filename}")

    print("Normalizzazione L2 (axis=1)...")
    Xn = normalize(X, norm="l2", axis=1)

    first_norm = np.linalg.norm(Xn[0])
    min_norm = np.linalg.norm(Xn, axis=1).min()
    max_norm = np.linalg.norm(Xn, axis=1).max()
    print(f"Norma primo vettore: {first_norm:.6f}")
    print(f"Norme min/max (atteso ~1): {min_norm:.6f} / {max_norm:.6f}")

    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    np.save(output_filename, Xn)
    print(f"Salvato embedding normalizzato in: {output_filename}")


def main():
    dataset = "./data/ranking_IQS_result.json"

    text_vector = generate_text_vectors(dataset)
    models = ['all-MiniLM-L6-v2', 'all-mpnet-base-v2']

    for m in models:
        filename = f'./embeddings/{m}.npy'
        if os.path.exists(filename):
            print(f"Embedding del modello {m} già calcolati nel file {filename}")
        else:
            run_embedding(m,text_vector, filename)

        filename_normalized = f"./embeddings/{m}_normalized.npy"

        if os.path.exists(filename_normalized):
            print(f"Embedding del modello {m} già normalizzato nel file: {filename_normalized}")
        else:
            normalize_embedding(filename, filename_normalized)


if __name__ == "__main__":
    main()

















































