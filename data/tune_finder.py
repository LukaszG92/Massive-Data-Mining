#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import pandas as pd


def iter_report_paths(root: Path, pattern: str):
    # Esempio pattern: "srr_tuning_report*.json"
    # Ordina “natural-ish”: ...report.json poi ...report2.json ecc.
    paths = sorted(root.glob(pattern), key=lambda p: (p.stem, p.name))
    return [p for p in paths if p.is_file()]


def load_runs_from_one_report(report_path: Path) -> pd.DataFrame:
    with report_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Strutture possibili:
    # - {"runs": [...]}  (come nei report che hai mostrato) [file:3]
    # - [...] (lista diretta)
    if isinstance(data, dict):
        runs = data.get("runs", [])
        meta = {k: v for k, v in data.items() if k != "runs"}
    elif isinstance(data, list):
        runs = data
        meta = {}
    else:
        raise ValueError(f"Formato JSON non supportato in {report_path}")

    rows = []
    for r in runs:
        s = (r.get("summary", {}) or {}) if isinstance(r, dict) else {}
        rows.append(
            {
                "source_file": report_path.name,
                # meta (se utile)
                "inputnpy": meta.get("inputnpy"),
                # run fields
                "delta": r.get("delta") if isinstance(r, dict) else None,
                "L_gb": r.get("L_gb") if isinstance(r, dict) else None,
                "threads": r.get("threads") if isinstance(r, dict) else None,
                "eps": r.get("eps") if isinstance(r, dict) else None,
                "minPts": r.get("minPts") if isinstance(r, dict) else None,
                "runtime_s": r.get("runtime_s") if isinstance(r, dict) else None,
                # summary fields
                "n_clusters": s.get("n_clusters"),
                "noise": s.get("noise"),
                "noise_frac": s.get("noise_frac"),
                "max_cluster_share": s.get("max_cluster_share"),
                "median_cluster_size": s.get("median_cluster_size"),
                "p90_cluster_size": s.get("p90_cluster_size"),
                "p99_cluster_size": s.get("p99_cluster_size"),
            }
        )

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", "-r", default=".", help="Directory dove cercare i report")
    ap.add_argument(
        "--pattern",
        "-p",
        default="srr_tuning_report*.json",
        help='Glob pattern (default: "srr_tuning_report*.json")',
    )
    ap.add_argument("--output-all", default="all_runs.csv", help="CSV con tutte le run (concatenate)")
    ap.add_argument("--output-filtered", default="filtered_runs.csv", help="CSV con run filtrate")

    ap.add_argument("--max-clusters", type=int, default=1000)
    ap.add_argument("--max-noise", type=float, default=0.70)   # noise_frac < 0.70
    ap.add_argument("--max-share", type=float, default=0.30)   # max_cluster_share <= 0.30
    ap.add_argument("--print", dest="do_print", action="store_true", help="Stampa le prime righe filtrate")
    args = ap.parse_args()

    root = Path(args.root)
    report_paths = iter_report_paths(root, args.pattern)
    if not report_paths:
        raise FileNotFoundError(f"Nessun file trovato in {root.resolve()} con pattern {args.pattern}")

    dfs = []
    for p in report_paths:
        try:
            dfs.append(load_runs_from_one_report(p))
        except Exception as e:
            print(f"[WARN] Skip {p.name}: {e}")

    if not dfs:
        raise RuntimeError("Non ho caricato nessun report (tutti falliti/skip).")

    df = pd.concat(dfs, ignore_index=True)

    # Salva tutto (utile per debug/analisi)
    df.to_csv(args.output_all, index=False)

    # Filtri richiesti
    df_f = df[
        (df["n_clusters"] < args.max_clusters)
        & (df["noise_frac"] < args.max_noise)
        & (df["max_cluster_share"] <= args.max_share)
    ].copy()

    # Ordinamento comodo
    df_f = df_f.sort_values(
        by=["noise_frac", "max_cluster_share", "n_clusters", "runtime_s"],
        ascending=[True, True, True, True],
        na_position="last",
    )

    df_f.to_csv(args.output_filtered, index=False)

    print(f"Loaded reports: {len(report_paths)}")
    print(f"Total runs: {len(df)} -> {args.output_all}")
    print(f"Kept runs:  {len(df_f)} -> {args.output_filtered}")

    if args.do_print:
        with pd.option_context("display.max_columns", None, "display.width", 180):
            print(df_f.head(50).to_string(index=False))


if __name__ == "__main__":
    main()
