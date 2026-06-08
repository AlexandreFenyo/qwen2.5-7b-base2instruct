#!/usr/bin/env python
"""Phase 1 — SFT (TRL SFTTrainer) : base Qwen2.5-7B -> chat model (ChatML), loss réponse seule.
Full fine-tuning, bf16, paged_adamw_8bit, SDPA, wandb. Mode --smoke pour valider le fit/API.
"""
import os, argparse
os.environ.setdefault("WANDB_PROJECT", "qwen2.5-7b-base2instruct")
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer


class NanGuard(TrainerCallback):
    """Neutralise les gradients NaN/Inf avant le pas d'optimisation (batch fautif sauté,
    poids non corrompus). Robuste aux overflows bf16 du forward sur échantillons pathologiques."""
    def __init__(self):
        self.skipped = 0

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        bad = False
        for p in model.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
                bad = True
        if bad:
            self.skipped += 1
            print(f"[nan-guard] step {state.global_step}: gradient non-fini neutralisé "
                  f"(total sautés={self.skipped})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/base2instruct/models/Qwen2.5-7B")
    ap.add_argument("--data", required=True)              # jsonl avec champ "messages"
    ap.add_argument("--output", required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--seq_len", type=int, default=4096)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--packing", action="store_true")
    ap.add_argument("--no_liger", action="store_true")    # CE standard fp32-upcast (plus stable, gros vocab)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--optim", default="adamw_8bit")
    ap.add_argument("--run_name", default="sft")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    # Template ChatML d'entraînement avec balises {% generation %} -> loss sur la réponse assistant
    # uniquement (rendu texte identique au ChatML standard à l'inférence).
    tok.chat_template = (
        "{% for message in messages %}"
        "{{ '<|im_start|>' + message['role'] + '\n' }}"
        "{% if message['role'] == 'assistant' %}"
        "{% generation %}{{ message['content'] + '<|im_end|>' + '\n' }}{% endgeneration %}"
        "{% else %}{{ message['content'] + '<|im_end|>' + '\n' }}{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    )
    ds = load_dataset("json", data_files=args.data, split="train")
    if args.smoke:
        ds = ds.select(range(min(256, len(ds))))
    print(f"[data] {len(ds)} exemples | chat_template présent: {tok.chat_template is not None}")

    cfg = SFTConfig(
        output_dir=args.output,
        num_train_epochs=(1 if args.smoke else args.epochs),
        max_steps=(8 if args.smoke else -1),
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup,
        weight_decay=0.0,
        max_grad_norm=args.max_grad_norm,
        bf16=True, tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=args.optim,
        use_liger_kernel=not args.no_liger,  # CE/LM-head fusionnée (vitesse+mémoire) sauf si --no_liger (stabilité fp32)
        max_length=args.seq_len,
        packing=args.packing,
        assistant_only_loss=True,           # masque tout sauf la réponse de l'assistant
        logging_steps=2,
        save_strategy="no" if args.smoke else "epoch",
        save_total_limit=1,
        report_to=("none" if args.smoke else "wandb"),
        run_name=args.run_name,
        dataset_num_proc=8,
        model_init_kwargs={"dtype": torch.bfloat16, "attn_implementation": "sdpa",
                           "trust_remote_code": True},
    )
    trainer = SFTTrainer(model=args.model, args=cfg, train_dataset=ds, processing_class=tok,
                         callbacks=[NanGuard()])
    trainer.train()
    if not args.smoke:
        trainer.save_model(args.output); tok.save_pretrained(args.output)
        print(f"[done] SFT -> {args.output}")
    else:
        torch.cuda.synchronize()
        print(f"[smoke] pic GPU = {torch.cuda.max_memory_allocated()/1e9:.1f} Go alloué / "
              f"{torch.cuda.max_memory_reserved()/1e9:.1f} Go réservé")


if __name__ == "__main__":
    main()
