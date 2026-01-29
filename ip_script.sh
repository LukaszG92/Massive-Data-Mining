#!/usr/bin/env bash
set -euo pipefail

# --- CONFIG ---
EXE="/mnt/c/Users/anton/PycharmProjects/Massive-Data-Mining/IP.LSH.DBSCAN/build/LSHDBSCAN_exec"

# IMPORTANTE: metti qui il file NON normalizzato se vuoi usare -a (angular),
# perché main_general.cc fa sempre normalizeData()+meanRemoveData() quando -a. [file:42]
DATA="/mnt/c/Users/anton/PycharmProjects/Massive-Data-Mining/data/embeddings/alpaca_embeddings.txt"

OUTDIR="/mnt/c/Users/anton/PycharmProjects/Massive-Data-Mining/data/embeddings/experiments"
THREADS=32

EPS_LIST=(0.25 0.28 0.31 0.34)
MINPTS_LIST=(20 30 40)
L_LIST=(10 20)
M_LIST=(12 18 24)

mkdir -p "$OUTDIR"

SUMMARY="${OUTDIR}/summary.tsv"
echo -e "tag\teps\tminPts\tL\tM\tthreads\telapsed_s\tpoints\tclusters\tnoise\tnoise_pct\tavg_size\tmedian_size\tmax_size" > "$SUMMARY"

DATA_DIR="$(dirname "$DATA")"
DATA_BASE="$(basename "$DATA")"

for eps in "${EPS_LIST[@]}"; do
  for mpts in "${MINPTS_LIST[@]}"; do
    for L in "${L_LIST[@]}"; do
      for M in "${M_LIST[@]}"; do
        tag="eps${eps}_mpts${mpts}_L${L}_M${M}_t${THREADS}"
        logfile="${OUTDIR}/run_${tag}.log"

        echo "=== ${tag} ===" | tee "$logfile"

        # Timestamp pre-run: serve per beccare l'idx generato in questo run
        start_epoch="$(date +%s)"

        # timing "pulito" in secondi (real)
        elapsed_s=$({ /usr/bin/time -p \
          "$EXE" -f "$DATA" -a -e "$eps" -m "$mpts" -L "$L" -M "$M" -t "$THREADS" \
          1>>"$logfile" 2>>"$logfile" ; } 2>&1 | awk '/^real /{print $2}')

        # Trova l'idx creato DOPO start_epoch con prefisso = DATA (come nel C++). [file:42]
        idx=$(find "$DATA_DIR" -maxdepth 1 -type f \
          -name "${DATA_BASE}*.idx_concurrentlshdbscan" \
          -newermt "@$start_epoch" \
          -printf "%p\n" 2>/dev/null | sort | tail -n 1 || true)

        if [[ -z "${idx}" || ! -f "${idx}" ]]; then
          echo "WARNING: idx not found for ${tag} (check log: $logfile)" | tee -a "$logfile"
          continue
        fi

        newidx="${OUTDIR}/labels_${tag}.idx_concurrentlshdbscan"
        mv -f "$idx" "$newidx"

        # metriche dal file di label (una label per riga, -1 = noise)
        metrics_tsv=$(python3 - <<PY
import numpy as np
from collections import Counter

labels = np.loadtxt("${newidx}", dtype=np.int64)
c = Counter(labels)

noise = c.pop(-1, 0)
sizes = list(c.values())

points = int(labels.size)
clusters = int(len(sizes))
noise_pct = float(noise/points*100.0) if points else 0.0
avg_size = float(np.mean(sizes)) if sizes else 0.0
median_size = float(np.median(sizes)) if sizes else 0.0
max_size = int(max(sizes)) if sizes else 0

print(f"{points}\t{clusters}\t{noise}\t{noise_pct:.6f}\t{avg_size:.6f}\t{median_size:.6f}\t{max_size}")
PY
)

        echo -e "${tag}\t${eps}\t${mpts}\t${L}\t${M}\t${THREADS}\t${elapsed_s}\t${metrics_tsv}" >> "$SUMMARY"
        echo "Saved labels -> $newidx" | tee -a "$logfile"
        echo "Appended metrics -> $SUMMARY" | tee -a "$logfile"
      done
    done
  done
done

echo "Done. Summary -> $SUMMARY"
