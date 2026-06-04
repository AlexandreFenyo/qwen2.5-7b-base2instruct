#!/usr/bin/env python
"""Phase 3 (prep) — prompts RLVR à récompense vérifiable : GSM8K (maths).
Sortie data/rlvr_gsm8k.jsonl : {prompt:[messages], gold:"<nombre>"}. IPv4 + no-Xet."""
import os, socket, json, re, argparse
_o = socket.getaddrinfo
socket.getaddrinfo = lambda h, *a, **k: [r for r in _o(h, *a, **k) if r[0] == socket.AF_INET]
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ.setdefault("HF_TOKEN", open("/root/.hf").read().strip())

INSTR = ("\n\nRaisonne étape par étape, puis donne la réponse finale sous la forme "
         "\\boxed{<nombre>}.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=7000)
    ap.add_argument("--out", default="/root/base2instruct/data/rlvr_gsm8k.jsonl")
    args = ap.parse_args()
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="train")
    print(f"[load] {len(ds)} GSM8K")
    kept = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for ex in ds:
            m = re.search(r"####\s*([\-0-9,\.]+)", ex["answer"])
            if not m:
                continue
            gold = m.group(1).replace(",", "").strip()
            rec = {"prompt": [{"role": "user", "content": ex["question"] + INSTR}], "gold": gold}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1
            if kept >= args.n:
                break
    print(f"[done] {kept} -> {args.out}")


if __name__ == "__main__":
    main()
