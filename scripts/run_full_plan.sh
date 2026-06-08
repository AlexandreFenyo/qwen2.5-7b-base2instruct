#!/bin/bash
# Orchestrateur autonome : SFT élargi -> DPO ciblé -> RLVR gradué élargi -> éval.
# Vérifs NaN + merges + patch eos entre chaque étape. Marqueurs de progression dans le log.
cd /root/base2instruct
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_API_KEY=$(cat /root/.wandb) WANDB_PROJECT=qwen2.5-7b-base2instruct
PY=env-train/bin/python
die(){ echo "PLAN_FAILED: $1"; exit 1; }

# ============ ÉTAPE 1 — SFT élargi (2 epochs, 300k) ============
if [ ! -f ckpt/sft2/model.safetensors ]; then
  echo "=== STAGE1 SFT start $(date) ==="
  $PY scripts/sft_train.py --model models/Qwen2.5-7B --data data/sft_tulu3_300000_s1.jsonl \
    --output ckpt/sft2 --epochs 2 --lr 4e-6 --seq_len 4096 --bs 2 --accum 8 --packing \
    --no_liger --max_grad_norm 0.5 --warmup 0.05 --optim adamw_8bit --run_name sft2 || die "SFT crashed"
  [ -f ckpt/sft2/model.safetensors ] || die "SFT no output"
  $PY scripts/_patch_eos.py ckpt/sft2 || die "patch eos"
  $PY scripts/_check_nan.py ckpt/sft2 || die "SFT NaN"
fi
echo "STAGE1_SFT_DONE $(date)"

# ============ ÉTAPE 2 — DPO ciblé (LoRA, 30k) ============
if [ ! -f ckpt/dpo2/model.safetensors ]; then
  echo "=== STAGE2 DPO start $(date) ==="
  $PY scripts/dpo_train.py --model ckpt/sft2 --data data/dpo_targeted.jsonl \
    --output ckpt/dpo2 --epochs 1 --lr 5e-6 --beta 0.1 --seq_len 1536 --bs 4 --accum 4 \
    --lora --lora_r 32 --optim adamw_8bit --run_name dpo2 || die "DPO crashed"
  [ -f ckpt/dpo2/model.safetensors ] || die "DPO no output"
  $PY scripts/_check_nan.py ckpt/dpo2 || die "DPO NaN"
fi
echo "STAGE2_DPO_DONE $(date)"

# ============ ÉTAPE 3 — RLVR gradué élargi (2500 steps) ============
if [ ! -f ckpt/rlvr2/model.safetensors ]; then
  echo "=== STAGE3 RLVR start $(date) ==="
  $PY scripts/grpo_train.py --model ckpt/dpo2 --data data/rlvr_graded_big.jsonl \
    --output ckpt/rlvr2 --no_vllm --multi --num_gen 8 --comp_len 768 --prompt_len 256 \
    --bs 8 --accum 2 --max_steps 2500 --lr 2e-6 --run_name rlvr2-graded || die "RLVR crashed"
  [ -f ckpt/rlvr2/adapter_model.safetensors ] || die "RLVR no adapter"
  $PY scripts/_merge_lora.py ckpt/dpo2 ckpt/rlvr2 ckpt/rlvr2 || die "RLVR merge/NaN"
fi
echo "STAGE3_RLVR_DONE $(date)"

# ============ ÉTAPE 4 — éval des 3 nouveaux modèles ============
echo "=== STAGE4 EVAL start $(date) ==="
for spec in "sft2|ckpt/sft2" "dpo2|ckpt/dpo2" "rlvr2|ckpt/rlvr2"; do
  IFS='|' read -r name path <<< "$spec"
  if [ ! -f results/$name/*/results_*.json ] 2>/dev/null; then
    bash eval/run_bench.sh "$path" "$name" chat > logs/eval_$name.log 2>&1 || echo "EVAL_WARN $name"
    env-gen/bin/lm_eval --model vllm \
      --model_args "pretrained=$path,dtype=bfloat16,gpu_memory_utilization=0.85,max_model_len=4096,enforce_eager=True" \
      --tasks gsm8k --gen_kwargs "max_gen_toks=1024" --apply_chat_template --fewshot_as_multiturn \
      --output_path results/gsm8k_fair/$name > logs/gsmfair_$name.log 2>&1 || echo "GSMFAIR_WARN $name"
  fi
done
echo "STAGE4_EVAL_DONE $(date)"
echo "PLAN_DONE $(date)"
