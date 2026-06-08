#!/usr/bin/env python
"""Patche generation_config.json d'un modèle SFT pour stopper sur <|im_end|> (151645)
et <|endoftext|> (151643) — sinon le modèle issu de la base ne s'arrête pas."""
import sys, json, os
d = sys.argv[1]
p = os.path.join(d, "generation_config.json")
cfg = json.load(open(p)) if os.path.exists(p) else {}
cfg["eos_token_id"] = [151645, 151643]
cfg["pad_token_id"] = 151643
json.dump(cfg, open(p, "w"), indent=2)
print(f"[patch-eos] {p} -> eos={cfg['eos_token_id']}")
