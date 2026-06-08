#!/usr/bin/env python
"""Phase 2+ (prep) — DPO CIBLÉ suivi d'instructions :
  - allenai/tulu-3-pref-personas-instruction-following (préférences orientées contraintes)
  - + complément générique ultrafeedback (déjà préparé) pour la diversité.
Sortie data/dpo_targeted.jsonl {chosen:[msgs], rejected:[msgs]}. IPv4 + no-Xet."""
import os, socket, json, argparse
_o = socket.getaddrinfo
socket.getaddrinfo = lambda h, *a, **k: [r for r in _o(h, *a, **k) if r[0] == socket.AF_INET]
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ.setdefault("HF_TOKEN", open("/root/.hf").read().strip())


def _valid(ch, rj, max_chars):
    if not (isinstance(ch, list) and isinstance(rj, list) and ch and rj):
        return False
    if ch[-1].get("role") != "assistant" or rj[-1].get("role") != "assistant":
        return False
    return sum(len(m.get("content", "")) for m in ch) <= max_chars


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_if", type=int, default=20000)
    ap.add_argument("--n_generic", type=int, default=10000)
    ap.add_argument("--generic", default="/root/base2instruct/data/dpo_100000.jsonl")
    ap.add_argument("--out", default="/root/base2instruct/data/dpo_targeted.jsonl")
    ap.add_argument("--max_chars", type=int, default=12000)
    args = ap.parse_args()
    from datasets import load_dataset

    rows = []
    d = load_dataset("allenai/tulu-3-pref-personas-instruction-following", split="train").shuffle(seed=0)
    k = 0
    for ex in d:
        ch, rj = ex.get("chosen"), ex.get("rejected")
        if not _valid(ch, rj, args.max_chars):
            continue
        rows.append({"chosen": ch, "rejected": rj})
        k += 1
        if k >= args.n_if:
            break
    print(f"[if] {k}")

    g = 0
    with open(args.generic) as f:
        for line in f:
            r = json.loads(line)
            if _valid(r.get("chosen"), r.get("rejected"), args.max_chars):
                rows.append({"chosen": r["chosen"], "rejected": r["rejected"]})
                g += 1
                if g >= args.n_generic:
                    break
    print(f"[generic] {g}")

    import random
    random.Random(0).shuffle(rows)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[done] {len(rows)} -> {args.out}")


if __name__ == "__main__":
    main()
