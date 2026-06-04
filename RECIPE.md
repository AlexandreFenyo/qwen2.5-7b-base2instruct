# Recette base→instruct (SFT → DPO → RLVR) sur 1×H100 80 Go

But : transformer un **modèle de base** (non-instruct, sans CoT) en bon modèle conversationnel
en reproduisant nous-mêmes le post-training, puis comparer à l'**instruct officiel** dérivé de la
même base. Recette validée sur **Qwen2.5-7B** (base) vs **Qwen2.5-7B-Instruct** (cible).

## Stack figée (ce qui marche)
- venvs uv Python 3.12. `env-train` : torch cu128, transformers 5.9, **trl 1.5.1**, peft, bitsandbytes,
  datasets, accelerate. `env-gen` : **vLLM 0.22** (+ lm-eval 0.4.12 pour l'éval).
- **flash-attn non compilable** (nvcc≠torch) → attention **SDPA**.
- **wandb** sur chaque run (`report_to="wandb"`, `WANDB_PROJECT=qwen2.5-7b-base2instruct`, run_name par étape).
- Téléchargements HF : **patch socket IPv4** + `HF_HUB_DISABLE_XET=1` (IPv6/Xet cassés sur l'hôte).

## Étape 1 — SFT (full fine-tuning)
- **Données** : `allenai/tulu-3-sft-mixture`, sous-échantillon **180k** exemples (format `messages`).
- **Format** : on enseigne le **ChatML Qwen** ; template d'entraînement avec balises `{% generation %}`
  pour `assistant_only_loss=True` (loss sur la réponse assistant uniquement). Au déballage, rendu
  identique au ChatML standard → comparaison équitable + recette réutilisable.
- **Config** : 1 epoch, lr **5e-6** cosine, warmup 3 %, seq_len 4096, **packing**, bf16, SDPA,
  gradient checkpointing, **liger-kernel** (CE/LM-head fusionnée, gros vocab Qwen 152k → vitesse+mémoire),
  optim **`adamw_8bit`** (résident GPU), micro-batch 2 × accumulation 8.
- **Piège** : le modèle de base a `eos=<|endoftext|>` → après SFT, patcher `generation_config.json`
  `eos_token_id=[151645,151643]` (`<|im_end|>` + `<|endoftext|>`) sinon la génération ne s'arrête pas.
- Sortie : `ckpt/sft` (chat model sain).

## Étape 2 — DPO (LoRA) depuis `ckpt/sft`
- **Données** : `HuggingFaceH4/ultrafeedback_binarized` (train_prefs), **~10k paires** suffisent
  (le signal DPO sature vite ; format `{chosen:[msgs], rejected:[msgs]}`, TRL infère le prompt commun).
- **Config** : **LoRA r=32 α=64** sur toutes les projections (q,k,v,o,gate,up,down), 1 epoch,
  lr **5e-6**, **beta 0.1**, seq_len 1536, `truncation_mode="keep_end"`, bf16, gradient checkpointing,
  optim **`adamw_8bit`**, micro-batch 4 × accumulation 4, `max_grad_norm=1.0`, warmup 5 %.
- **PIÈGE MAJEUR — NaN** :
  1. **`precompute_ref_log_probs=True` → NaN** (logprobs de référence stockés en bf16 → `-inf` →
     loss DPO `inf`/NaN). **Le DÉSACTIVER** ; la référence est calculée en direct.
  2. **Full-FT + optimiseur fp32 offloadé CPU (`paged_adamw_32bit`) = 33 s/it (35 h)** → inutilisable
     (pagination de ~56 Go d'états à chaque step). **Solution : LoRA** → un seul modèle en mémoire
     (réf = adaptateur désactivé), optimiseur minuscule résident GPU → **3 s/it (~2 h)**, pic 28 Go.
- **Toujours** : smoke 6 steps avant le run long, vérifier `losses` finies + `params_nan=0`.
- Fusion LoRA→modèle complet à la sauvegarde (`merge_and_unload`). Sortie : `ckpt/dpo`.

## Étape 3 — RLVR via GRPO (LoRA) depuis `ckpt/dpo`
- **Récompense VÉRIFIABLE, sans reward-model** (approche Tülu-3) : maths **GSM8K**, prompt demandant
  une réponse finale en `\boxed{}` ; reward = **+1.0** si réponse correcte (`math-verify` / parse nombre)
  **+0.1** si format `\boxed{}` présent. 7000 prompts.
- **Config GRPO** : **LoRA r=16 α=32**, génération **transformers** (`use_vllm=False`),
  num_generations **8**, max_completion_length **512**, temperature 1.0, beta 0.0, lr **1e-6**,
  micro-batch 8 × accumulation 2, **max_steps 500**, **gradient_checkpointing=True** (décisif).
- **PIÈGES** :
  1. **vLLM 0.22 incompatible avec le GRPO de TRL 1.5.1** en mode colocate (`sampling_per_token_logps`
     = NoneType). → **génération transformers** (`--no_vllm`). ~21 s/it.
  2. Sans gradient checkpointing : pic **77 Go** (bord de l'OOM). Avec : **27 Go** → grande marge.
  3. Toute génération plantait (`multinomial: inf/nan`) tant que `ckpt/dpo` contenait des NaN → la
     cause était l'étape 2, pas le GRPO. **Vérifier `params_nan=0` du DPO avant de lancer le GRPO.**
- Fusion LoRA→DPO à la sauvegarde. Sortie : `ckpt/rlvr` (~3 h).

## Étape 4 — Évaluation comparative (5 modèles)
- **Benchmarks objectifs** via **lm-eval backend vLLM** (env-gen) : **IFEval** (suivi d'instructions),
  **GSM8K** (5-shot), **MMLU** (5-shot). Modèles chat avec `--apply_chat_template --fewshot_as_multiturn` ;
  modèle de base sans template.
- Pré-télécharger les datasets (`google/IFEval`, **les 57 configs sujets** de `cais/mmlu`, `openai/gsm8k`)
  car l'éval tourne en `HF_HUB_OFFLINE=1`.
- Agrégation → tableau + **courbe wandb** base→SFT→DPO→RLVR vs ligne « instruct officiel ».

## Règles d'or (transférables aux futures expériences)
1. **Smoke-test systématique** (6 steps) avant chaque run long : loss finie, `params_nan=0`, pic mémoire.
2. **Vérifier l'absence de NaN** des poids à la fin de chaque étape avant d'enchaîner.
3. **LoRA** est le défaut gagnant sur 1 GPU pour DPO/RLVR (mémoire + vitesse) ; full-FT réservé au SFT.
4. **Éviter `precompute_ref_log_probs`** en bf16 (DPO) ; **éviter les optimiseurs fp32 offloadés CPU** sur
   un modèle complet (trop lent).
5. Lancer les jobs longs dans des commandes **dédiées** (pas de `pkill` en tête qui casse le lancement).

## Résultats (lm-eval, 2026-06-04)
| modèle | IFEval (prompt strict) | GSM8K (flexible) | MMLU |
|---|---|---|---|
| base (Qwen2.5-7B) | 27.4 | 83.0 | 71.8 |
| + SFT | 44.9 | 77.5 | 69.1 |
| + DPO | 44.7 | 77.1 | 69.9 |
| + RLVR | **45.1** | 77.4 | 69.9 |
| **instruct officiel** | **71.9** | **84.7** | 68.8 |

**Piège d'éval critique** : en `strict-match`, l'instruct officiel tombait à **21 %** sur GSM8K — pur
artefact (il n'émet pas le format `#### N` et sa CoT verbeuse dépassait la limite de tokens). Mesuré
équitablement (max_gen_toks=1024, `flexible-extract` = dernier nombre), il fait **84.7 %**. Toujours
vérifier le format de réponse attendu par le parser avant de conclure.

## Interprétation (honnête)
- **La recette FONCTIONNE** : le pipeline tourne de bout en bout et produit un vrai modèle instruct.
  Le gain principal vient du **SFT** : IFEval **27 → 45** (+18 pts), le modèle apprend le format chat
  et le suivi d'instructions.
- **DPO (10k paires génériques) et RLVR (GSM8K seul) n'ont quasiment pas bougé IFEval** (45→45→45).
  Logique : le RLVR ne ciblait que les maths, donc le suivi d'instructions n'en profite pas.
- **On n'égale PAS encore l'instruct officiel sur IFEval (45 vs 72)**. L'officiel (Tülu-3-like) utilise
  beaucoup plus de données SFT (~1M+), un DPO large, et surtout un **RLVR multi-domaines incluant des
  récompenses vérifiables de type IFEval** (contraintes d'instructions), pas seulement les maths.
- **Maths** : le base Qwen2.5-7B est déjà excellent (83). Notre post-training en perd un peu (~77) ;
  l'officiel le préserve mieux (84.7). Leçon : SFT trop « chat » peut diluer une compétence forte du base.
- **MMLU** : aucune régression notable (~69-72 partout) ; nos modèles retiennent même un peu plus de
  connaissances que l'officiel (69.9 vs 68.8).

## Pour égaler l'officiel dans une future expérience (leviers identifiés)
1. **RLVR multi-domaines** : ajouter des récompenses vérifiables **suivi d'instructions** (contraintes
   IFEval vérifiées par programme) + code (tests unitaires), pas uniquement GSM8K → c'est le principal
   levier pour combler l'écart IFEval.
2. **Plus de SFT** (data + epochs) et **DPO sur préférences ciblées** (pas seulement ultrafeedback générique).
3. **Préserver les maths** : inclure des données maths au SFT ou pondérer pour ne pas diluer le base.
