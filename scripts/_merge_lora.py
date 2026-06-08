#!/usr/bin/env python
"""Fusionne un adaptateur LoRA sur un modèle de base et sauvegarde le modèle complet.
Usage: _merge_lora.py <base_dir> <adapter_dir> <out_dir>. Nettoie les fichiers d'adaptateur."""
import sys, os, glob, shutil, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
base, adapter, out = sys.argv[1], sys.argv[2], sys.argv[3]
m = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map="cuda")
m = PeftModel.from_pretrained(m, adapter).merge_and_unload()
nan = sum(int(torch.isnan(p).any()) for p in m.parameters())
print(f"[merge] params_nan={nan}")
os.makedirs(out, exist_ok=True)
m.save_pretrained(out)
AutoTokenizer.from_pretrained(base).save_pretrained(out)
# nettoyage adaptateur + checkpoints si out == adapter
for f in glob.glob(os.path.join(out, "adapter_*")) + glob.glob(os.path.join(out, "training_args.bin")):
    os.remove(f)
for d in glob.glob(os.path.join(out, "checkpoint-*")):
    shutil.rmtree(d, ignore_errors=True)
sys.exit(0 if nan == 0 else 1)
