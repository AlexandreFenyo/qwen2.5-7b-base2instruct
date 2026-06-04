#!/bin/bash
# Phase 4 — lance les benchmarks sur les 5 modèles en séquentiel.
cd /root/base2instruct
rm -rf results/_smoke
# nom  chemin  mode
RUNS=(
  "base|models/Qwen2.5-7B|base"
  "sft|ckpt/sft|chat"
  "dpo|ckpt/dpo|chat"
  "rlvr|ckpt/rlvr|chat"
  "instruct_officiel|models/Qwen2.5-7B-Instruct|chat"
)
for spec in "${RUNS[@]}"; do
  IFS='|' read -r name path mode <<< "$spec"
  echo "===== EVAL $name ($mode) @ $(date +%H:%M) ====="
  bash eval/run_bench.sh "$path" "$name" "$mode" > "logs/eval_${name}.log" 2>&1 \
    && echo "OK $name" || echo "FAIL $name (voir logs/eval_${name}.log)"
done
echo "ALL_EVALS_DONE"
