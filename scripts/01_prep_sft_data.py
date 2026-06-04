#!/usr/bin/env python
"""Phase 1 (prep) — sous-échantillonne allenai/tulu-3-sft-mixture en JSONL local (format messages).

Force IPv4 + désactive Xet (CDN HF capricieux en IPv6 sur cet hôte).
Filtre les exemples trop longs, garde le champ `messages`. Sortie: data/sft_tulu3_<n>.jsonl
"""
import os, socket, json, argparse
_o = socket.getaddrinfo
socket.getaddrinfo = lambda h, *a, **k: [r for r in _o(h, *a, **k) if r[0] == socket.AF_INET]
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ.setdefault("HF_TOKEN", open("/root/.hf").read().strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=180000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max_chars", type=int, default=16000)  # ~4k tokens, évite les très longs
    args = ap.parse_args()
    out = args.out or f"/root/base2instruct/data/sft_tulu3_{args.n}.jsonl"

    from datasets import load_dataset
    ds = load_dataset("allenai/tulu-3-sft-mixture", split="train")
    print(f"[load] {len(ds)} exemples")
    ds = ds.shuffle(seed=args.seed)

    kept = 0
    with open(out, "w", encoding="utf-8") as f:
        for ex in ds:
            msgs = ex.get("messages")
            if not msgs or msgs[-1].get("role") != "assistant":
                continue
            total = sum(len(m.get("content", "")) for m in msgs)
            if total > args.max_chars or total < 20:
                continue
            f.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
            kept += 1
            if kept >= args.n:
                break
    print(f"[done] {kept} exemples -> {out}")


if __name__ == "__main__":
    main()
