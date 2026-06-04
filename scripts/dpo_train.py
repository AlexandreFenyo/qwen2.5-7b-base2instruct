#!/usr/bin/env python
"""Phase 2 — DPO (TRL DPOTrainer) depuis le checkpoint SFT. Préférences conversationnelles.
bf16, adamw_8bit, gradient checkpointing, liger DPO loss si dispo, wandb. --smoke pour valider fit/API."""
import os, argparse
os.environ.setdefault("WANDB_PROJECT", "qwen2.5-7b-base2instruct")
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import DPOConfig, DPOTrainer
from peft import LoraConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/base2instruct/ckpt/sft")
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=5e-7)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--prompt_len", type=int, default=1024)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--optim", default="adamw_8bit")
    ap.add_argument("--precompute", action="store_true")
    ap.add_argument("--lora", action="store_true")     # DPO-LoRA: réf = modèle adaptateur off (1 seul modèle, optim GPU)
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--run_name", default="dpo")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    ds = load_dataset("json", data_files=args.data, split="train")
    if args.smoke:
        ds = ds.select(range(min(128, len(ds))))
    print(f"[data] {len(ds)} paires")

    # liger DPO incompatible avec precompute_ref_log_probs -> on garde le precompute (2x moins de forwards)
    cfg = DPOConfig(
        output_dir=args.output,
        num_train_epochs=(1 if args.smoke else args.epochs),
        max_steps=(6 if args.smoke else -1),
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.05,
        max_grad_norm=1.0,
        beta=args.beta,
        max_length=args.seq_len, truncation_mode="keep_end",
        precompute_ref_log_probs=args.precompute,   # off par défaut (bf16 precompute -> -inf -> NaN)
        bf16=True, tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=args.optim,
        logging_steps=2,
        save_strategy="no" if args.smoke else "epoch",
        save_total_limit=1,
        report_to=("none" if args.smoke else "wandb"),
        run_name=args.run_name,
        model_init_kwargs={"dtype": torch.bfloat16, "attn_implementation": "sdpa",
                           "trust_remote_code": True},
    )
    peft_cfg = None
    if args.lora:
        peft_cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.0,
                              bias="none", task_type="CAUSAL_LM",
                              target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                              "gate_proj", "up_proj", "down_proj"])
    trainer = DPOTrainer(model=args.model, args=cfg, train_dataset=ds, processing_class=tok,
                         peft_config=peft_cfg)
    trainer.train()
    if not args.smoke:
        if args.lora:
            merged = trainer.model.merge_and_unload()      # fusionne l'adaptateur -> modèle complet
            merged.save_pretrained(args.output)
        else:
            trainer.save_model(args.output)
        tok.save_pretrained(args.output)
        print(f"[done] DPO -> {args.output}")
    else:
        torch.cuda.synchronize()
        m = trainer.model
        nan = sum(int(torch.isnan(p).any()) for p in m.parameters())
        inf = sum(int(torch.isinf(p).any()) for p in m.parameters())
        hist = [h.get("loss") for h in trainer.state.log_history if "loss" in h]
        print(f"[smoke] losses={hist} | params_nan={nan} params_inf={inf}")
        print(f"[smoke] pic GPU = {torch.cuda.max_memory_allocated()/1e9:.1f} Go alloué / "
              f"{torch.cuda.max_memory_reserved()/1e9:.1f} Go réservé")


if __name__ == "__main__":
    main()
