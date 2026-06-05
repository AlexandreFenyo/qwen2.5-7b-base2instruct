#!/usr/bin/env python
"""Phase 3+ (prep) — données RLVR MULTI-DOMAINES à récompense vérifiable :
  - suivi d'instructions : allenai/RLVR-IFeval (contrainte vérifiée par if_functions)
  - maths : GSM8K (réutilise data/rlvr_gsm8k.jsonl)
Sortie data/rlvr_multi.jsonl : {prompt:[messages], task:'ifeval'|'math', gold, ground_truth}.
IPv4 + no-Xet."""
import os, socket, json, argparse
_o = socket.getaddrinfo
socket.getaddrinfo = lambda h, *a, **k: [r for r in _o(h, *a, **k) if r[0] == socket.AF_INET]
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ.setdefault("HF_TOKEN", open("/root/.hf").read().strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_ifeval", type=int, default=7000)
    ap.add_argument("--n_math", type=int, default=3000)
    ap.add_argument("--gsm8k", default="/root/base2instruct/data/rlvr_gsm8k.jsonl")
    ap.add_argument("--out", default="/root/base2instruct/data/rlvr_multi.jsonl")
    args = ap.parse_args()
    from datasets import load_dataset

    rows = []
    # 1) suivi d'instructions
    d = load_dataset("allenai/RLVR-IFeval", split="train").shuffle(seed=0)
    kept = 0
    for ex in d:
        msgs = ex["messages"]
        if not (isinstance(msgs, list) and msgs and msgs[0].get("role") == "user"):
            continue
        try:
            gt = json.loads(ex["ground_truth"])
        except Exception:
            continue
        prompt = [{"role": "user", "content": msgs[0]["content"]}]
        rows.append({"prompt": prompt, "task": "ifeval", "gold": None,
                     "ground_truth": json.dumps(gt, ensure_ascii=False)})
        kept += 1
        if kept >= args.n_ifeval:
            break
    print(f"[ifeval] {kept}")

    # 2) maths (GSM8K déjà préparé)
    m = 0
    with open(args.gsm8k) as f:
        for line in f:
            r = json.loads(line)
            rows.append({"prompt": r["prompt"], "task": "math", "gold": r["gold"],
                         "ground_truth": None})
            m += 1
            if m >= args.n_math:
                break
    print(f"[math] {m}")

    # mélange déterministe (sans random : on entrelace)
    iff = [r for r in rows if r["task"] == "ifeval"]
    mat = [r for r in rows if r["task"] == "math"]
    mixed = []
    ratio = max(1, len(iff) // max(1, len(mat)))
    mi = 0
    for i, r in enumerate(iff):
        mixed.append(r)
        if (i + 1) % ratio == 0 and mi < len(mat):
            mixed.append(mat[mi]); mi += 1
    mixed.extend(mat[mi:])

    with open(args.out, "w", encoding="utf-8") as f:
        for r in mixed:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[done] {len(mixed)} -> {args.out}")


if __name__ == "__main__":
    main()
