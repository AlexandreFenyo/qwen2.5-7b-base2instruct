#!/usr/bin/env python
"""Phase 3 — RLVR via GRPO (TRL). Récompense VÉRIFIABLE (maths GSM8K), LoRA, vLLM colocalisé
(repli génération transformers si --no_vllm). Depuis ckpt/dpo. wandb. --smoke pour valider."""
import os, re, argparse
os.environ.setdefault("WANDB_PROJECT", "qwen2.5-7b-base2instruct")
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig

BOXED = re.compile(r"\\boxed\{([^}]*)\}")
NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def _extract(text):
    m = BOXED.findall(text)
    if m:
        return m[-1].replace(",", "").strip()
    n = NUM.findall(text)
    return n[-1].replace(",", "").strip() if n else None


def _num_eq(a, b):
    try:
        return abs(float(a) - float(b)) < 1e-6
    except Exception:
        try:
            from math_verify import parse, verify
            return bool(verify(parse(b), parse(a)))
        except Exception:
            return False


def _text(comp):
    return comp[-1]["content"] if isinstance(comp, list) else comp


def _reward_math(text, g):
    pred = _extract(text)
    r = 0.0
    if BOXED.search(text):
        r += 0.1
    if pred is not None and _num_eq(pred, g):
        r += 1.0
    return r


def reward_fn(completions, gold, **kw):
    """Maths seul : 1.0 si réponse correcte + 0.1 si format \\boxed{} présent."""
    return [_reward_math(_text(c), g) for c, g in zip(completions, gold)]


def reward_multi(completions, task, gold, ground_truth, **kw):
    """Multi-domaine : maths (math-verify) OU suivi d'instructions.
    ground_truth ifeval = liste de specs -> reward GRADUÉ = fraction de contraintes satisfaites
    (variance intra-groupe non nulle -> évite l'effondrement de l'avantage GRPO)."""
    import json
    from if_functions import check_constraint
    out = []
    for comp, t, g, gt in zip(completions, task, gold, ground_truth):
        text = _text(comp)
        if t == "math":
            out.append(_reward_math(text, g))
        else:  # ifeval : moyenne des contraintes respectées
            try:
                spec = json.loads(gt) if isinstance(gt, str) else gt
            except Exception:
                spec = None
            if not spec:
                out.append(0.0); continue
            specs = spec if isinstance(spec, list) else [spec]
            ok = sum(1.0 for s in specs if check_constraint(text, s))
            out.append(ok / len(specs))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/base2instruct/ckpt/dpo")
    ap.add_argument("--data", default="/root/base2instruct/data/rlvr_gsm8k.jsonl")
    ap.add_argument("--output", required=True)
    ap.add_argument("--max_steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--num_gen", type=int, default=8)
    ap.add_argument("--prompt_len", type=int, default=512)
    ap.add_argument("--comp_len", type=int, default=768)
    ap.add_argument("--bs", type=int, default=8)            # complétions par device-step (multiple de num_gen)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--no_vllm", action="store_true")
    ap.add_argument("--multi", action="store_true")   # reward multi-domaine (ifeval + maths)
    ap.add_argument("--run_name", default="rlvr")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    ds = load_dataset("json", data_files=args.data, split="train")
    if args.smoke:
        ds = ds.select(range(min(64, len(ds))))
    print(f"[data] {len(ds)} prompts RLVR")

    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    cfg = GRPOConfig(
        output_dir=args.output,
        max_steps=(6 if args.smoke else args.max_steps),
        learning_rate=args.lr, lr_scheduler_type="constant_with_warmup", warmup_ratio=0.03,
        beta=args.beta,
        num_generations=args.num_gen,
        max_completion_length=args.comp_len,
        temperature=1.0,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        bf16=True, tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        use_vllm=(not args.no_vllm),
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.30,
        logging_steps=1,
        save_strategy="no" if args.smoke else "steps",
        save_steps=200, save_total_limit=1,
        report_to=("none" if args.smoke else "wandb"),
        run_name=args.run_name,
        model_init_kwargs={"dtype": torch.bfloat16, "attn_implementation": "eager",
                           "trust_remote_code": True},
    )
    reward = reward_multi if args.multi else reward_fn
    trainer = GRPOTrainer(model=args.model, args=cfg, train_dataset=ds,
                          reward_funcs=reward, processing_class=tok, peft_config=lora)
    trainer.train()
    if not args.smoke:
        trainer.save_model(args.output); tok.save_pretrained(args.output)
        print(f"[done] RLVR -> {args.output}")
    else:
        torch.cuda.synchronize()
        print(f"[smoke] pic GPU = {torch.cuda.max_memory_allocated()/1e9:.1f} Go alloué / "
              f"{torch.cuda.max_memory_reserved()/1e9:.1f} Go réservé")


if __name__ == "__main__":
    main()
