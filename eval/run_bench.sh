#!/bin/bash
# Phase 4 — benchmarks objectifs via lm-eval backend vLLM.
# Usage: run_bench.sh <model_path> <name> <chat|base>
# IFEval (suivi d'instructions, 0-shot), GSM8K (maths, 5-shot CoT), MMLU (connaissances, 5-shot).
set -e
cd /root/base2instruct
MODEL="$1"; NAME="$2"; MODE="${3:-chat}"
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
OUT="results/$NAME"; mkdir -p "$OUT"

EXTRA=""
if [ "$MODE" = "chat" ]; then
  # modèles chat : on applique leur template ChatML ; few-shot en multi-tours
  EXTRA="--apply_chat_template --fewshot_as_multiturn"
fi

echo "[bench] $NAME ($MODE) -> $OUT"
env-gen/bin/lm_eval --model vllm \
  --model_args "pretrained=$MODEL,dtype=bfloat16,gpu_memory_utilization=0.85,max_model_len=4096,enforce_eager=True" \
  --tasks ifeval,gsm8k,mmlu \
  --batch_size auto \
  $EXTRA \
  --output_path "$OUT" \
  --log_samples 2>&1 | tail -40
echo "[bench] done $NAME"
