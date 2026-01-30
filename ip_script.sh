#!/usr/bin/env bash
set -euo pipefail

# --- WORKDIR ---
# Usa la directory corrente da cui lanci lo script
WD="${WD:-$(pwd)}"

# (Variante più robusta, se preferisci: directory dello script)
# WD="${WD:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"

cd "$WD"

# --- RELATIVE PATHS (relative to WD) ---
EXE="${EXE:-IP.LSH.DBSCAN/build/LSHDBSCAN_exec}"
DATA="${DATA:-data/embeddings/alpaca_embeddings.txt}"
OUTDIR="${OUTDIR:-data/embeddings/experiments}"
THREADS="${THREADS:-32}"

EPS_LIST=(0.25 0.28 0.31 0.34)
MINPTS_LIST=(20 30 40)
L_LIST=(10 20)
M_LIST=(12 18 24)

# --- sanity checks ---
if [[ ! -x "$EXE" ]]; then
  echo "ERROR: EXE not found or not executable (relative to WD=$WD): $EXE" >&2
  echo "Hint: build it, or pass EXE=relative/or/absolute/path" >&2
  exit 1
fi

if [[ ! -f "$DATA" ]]; then
  echo "ERROR: DATA not found (relative to WD=$WD): $DATA" >&2
  echo "Hint: put embeddings there, or pass DATA=relative/or/absolute/path" >&2
  exit 1
fi

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

        start_epoch="$(date +%s)"

        elapsed_s=$({ /usr/bin/time -p \
          "$EXE" -f "$DATA" -a -e "$eps" -m "$mpts" -L "$L" -M "$M" -t "$THREADS" \
          1>>"$logfile" 2>>"$logfile" ; } 2>&1 | awk '/^real /{print $2}')

        # idx generato vicino al file DATA (prefisso = DATA_BASE come nel C++)
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
