# Qwen2.5-7B base → instruct (SFT → DPO → RLVR) sur 1×H100

Pipeline complet et **reproductible** pour transformer un **modèle de base** (non-instruct, sans CoT)
en modèle conversationnel via **SFT → DPO → RLVR**, puis le **comparer à l'instruct officiel** dérivé
de la même base. Réalisé sur **un seul GPU H100 80 Go**.

- **Base** : [`Qwen/Qwen2.5-7B`](https://huggingface.co/Qwen/Qwen2.5-7B)
- **Cible de comparaison** : [`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- **🤗 Modèle final publié** : [`fenyo/Qwen2.5-7B-base2instruct`](https://huggingface.co/fenyo/Qwen2.5-7B-base2instruct)
- **Checkpoints intermédiaires** : [SFT](https://huggingface.co/fenyo/Qwen2.5-7B-base2instruct-SFT) · [DPO](https://huggingface.co/fenyo/Qwen2.5-7B-base2instruct-DPO)

## Résultats (lm-eval, mesure équitable)

**Pipeline final (v2) — données ciblées à chaque étape :**

| modèle | IFEval (suivi d'instr.) | GSM8K (maths) | MMLU |
|---|---|---|---|
| base Qwen2.5-7B | 27.4 | 83.0 | 71.8 |
| + SFT (300k) | 51.2 | 77.6 | 69.2 |
| + DPO **ciblé** (instruction-following) | 68.9 | 80.1 | 70.0 |
| **+ RLVR gradué (final)** | **75.0** | 79.7 | **70.2** |
| instruct officiel | 71.9 | 84.7 | 68.8 |

**Ce modèle dépasse l'instruct officiel sur IFEval (75.0 vs 71.9) et MMLU (70.2 vs 68.8)** ; en retrait
sur les maths (79.7 vs 84.7). Spécialisé suivi d'instructions (axe optimisé), pas généraliste.

Progression : 27 → 51 (SFT) → 69 (**DPO ciblé**, +17.7, l'étape décisive) → 75 (RLVR gradué, +6.1).
La leçon : **le signal doit cibler la capacité visée** (vérifié 3×). Détails, pièges (NaN, effondrement
GRPO binaire, artefact d'éval GSM8K) et « cours » base→instruct dans **[`RECIPE.md`](RECIPE.md)** et la
[model card HF](https://huggingface.co/fenyo/Qwen2.5-7B-base2instruct).

<details><summary>Première itération (v1) — pour comparaison</summary>

DPO générique + RLVR maths : IFEval plafonnait à **49.5**. Le passage à un DPO *ciblé* et un RLVR
*gradué* (v2) a fait toute la différence (+25.5 IFEval).
</details>

## Pipeline

```
scripts/01_prep_sft_data.py   # Tülu-3 SFT mixture -> ChatML (180k)
scripts/sft_train.py          # SFT full-FT (TRL), assistant_only_loss, liger
scripts/02_prep_dpo_data.py   # ultrafeedback_binarized -> paires chosen/rejected
scripts/dpo_train.py          # DPO-LoRA (TRL), référence en direct
scripts/03_prep_rlvr_data.py  # GSM8K -> prompts à récompense vérifiable
scripts/grpo_train.py         # RLVR via GRPO-LoRA (TRL), reward math-verify
eval/run_all.sh + aggregate.py# benchmarks lm-eval (vLLM) sur les 5 modèles + courbe wandb
```

## Stack

uv (Python 3.12) · PyTorch cu128 · transformers 5.9 · **TRL 1.5.1** · PEFT · bitsandbytes ·
vLLM 0.22 · lm-eval 0.4.12 · attention **SDPA** · suivi **wandb**.

## Datasets utilisés

- SFT : [`allenai/tulu-3-sft-mixture`](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture)
- DPO : [`HuggingFaceH4/ultrafeedback_binarized`](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized)
- RLVR : [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k)
- Éval : [`google/IFEval`](https://huggingface.co/datasets/google/IFEval), [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k), [`cais/mmlu`](https://huggingface.co/datasets/cais/mmlu)

## Pièges majeurs (voir RECIPE.md)

1. DPO TRL 1.5.1 : `precompute_ref_log_probs=True` en bf16 → **NaN**. Le désactiver.
2. Sur 1 GPU : **LoRA** pour DPO/RLVR (full-FT + optim fp32 offloadé CPU = 35 h).
3. GRPO : vLLM 0.22 incompatible TRL 1.5.1 colocate → génération transformers + gradient checkpointing.
4. Éval GSM8K : `strict-match` pénalise injustement (format `#### N`) → utiliser `flexible-extract`.

## Licence

Code sous licence MIT. Modèles et datasets : voir leurs licences respectives (Qwen2.5 = Apache-2.0).
