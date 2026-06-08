#!/usr/bin/env python
"""Vérifie qu'un checkpoint n'a aucun poids NaN/Inf. Exit 1 sinon."""
import sys, torch
from transformers import AutoModelForCausalLM
d = sys.argv[1]
m = AutoModelForCausalLM.from_pretrained(d, dtype=torch.bfloat16)
nan = sum(int(torch.isnan(p).any()) for p in m.parameters())
inf = sum(int(torch.isinf(p).any()) for p in m.parameters())
print(f"[nan-check] {d}: params_nan={nan} params_inf={inf}")
sys.exit(0 if (nan == 0 and inf == 0) else 1)
