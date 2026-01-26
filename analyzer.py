import json
import pandas as pd

# CONFIGURA IL PERCORSO DEL TUO FILE JSON


n = input("Inserisci il numero del report da analizzare:")
JSON_FILE = f"data/srr_tuning_report{n}.json"


def load_and_rank(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    runs = data["runs"]
    print(f"Caricati {len(runs)} run dal file: {json_path}\n")

    # Trasformiamo la lista di dizionari in un DataFrame Pandas piatto
    rows = []
    for r in runs:
        s = r["summary"]
        stab = r["stability"]

        # Calcoliamo stabilità media (media tra 'minus' e 'plus')
        avg_stab_nmi = 0.5 * (stab["minus"]["stability_nmi"] + stab["plus"]["stability_nmi"])
        avg_stab_ari = 0.5 * (stab["minus"]["stability_ari"] + stab["plus"]["stability_ari"])

        row = {
            "delta": r["delta"],
            "L_gb": r["L_gb"],
            "minPts": r["minPts"],
            "eps": r["eps"],
            "n_clusters": s["n_clusters"],
            "noise_frac": s["noise_frac"],
            "max_share": s["max_cluster_share"],
            "stab_nmi": avg_stab_nmi,
            "stab_ari": avg_stab_ari,
            "time_s": r["runtime_s"]
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # --- FILTRI DI QUALITÀ (Modifica questi valori se non trovi nulla) ---
    # 1. Scarta se troppo rumore (> 70%)
    # 2. Scarta se collassa tutto in un cluster (> 40% dei punti in un cluster solo)
    # 3. Scarta se trova meno di 2 cluster
    valid_df = df[
        (df["noise_frac"] < 0.70) &
        (df["max_share"] < 0.40) &
        (df["n_clusters"] >= 2)
        ].copy()

    if valid_df.empty:
        print("⚠️ Nessun candidato ha superato i filtri base! Prova ad allentare i limiti nel codice.")
        return df  # Restituisci tutto per debug

    # --- SCORE SINTETICO ---
    # Score = (Stabilità * 2) - (Rumore) - (MaxShare * 1.5)
    # Privilegiamo risultati stabili con rumore basso e cluster ben distribuiti
    valid_df["score"] = (
            (valid_df["stab_nmi"] * 2.0) +
            (valid_df["stab_ari"] * 1.0) -
            (valid_df["noise_frac"] * 1.0) -
            (valid_df["max_share"] * 1.5)
    )

    # Ordina per score decrescente
    ranked = valid_df.sort_values(by="score", ascending=False)

    return ranked


# ESECUZIONE
df_ranked = load_and_rank(JSON_FILE)

print("=== TOP 5 CONFIGURAZIONI MIGLIORI ===")
# Mostriamo solo le colonne che contano
cols = ["minPts", "eps", "delta", "n_clusters", "noise_frac", "max_share", "stab_nmi", "score"]
print(df_ranked[cols].head(5).to_string(index=False))

# Salva un CSV per aprirlo con Excel se vuoi
df_ranked.to_csv("data/srr_best_candidates.csv", index=False)
print("\nSalvati tutti i candidati validi in 'data/srr_best_candidates.csv'")
