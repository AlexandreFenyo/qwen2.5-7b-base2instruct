#!/bin/bash
# Téléchargement robuste IPv4 d'un repo HF via curl (resume + retries). Contourne Xet/IPv6.
set -u
TOKEN=$(tr -d ' \n' < /root/.hf)
REPO="$1"; DEST="$2"; shift 2
FILES=("$@")
mkdir -p "$DEST"
BASE="https://huggingface.co/${REPO}/resolve/main"
for f in "${FILES[@]}"; do
  out="$DEST/$f"; mkdir -p "$(dirname "$out")"
  echo ">>> $f"
  for attempt in $(seq 1 20); do
    curl -4 -sS -L --fail -C - --connect-timeout 30 --retry 5 --retry-delay 5 \
      -H "Authorization: Bearer $TOKEN" -o "$out" "$BASE/$f" && { echo "OK $f"; break; }
    echo "  retry $attempt for $f"; sleep 5
  done
done
echo "ALL_DONE"
