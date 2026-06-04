#!/usr/bin/env python
"""Phase 2 (prep) — données de préférence pour DPO (format conversationnel chosen/rejected).
Par défaut HuggingFaceH4/ultrafeedback_binarized (standard, propre). IPv4 + no-Xet.
Sortie: data/dpo_<n>.jsonl avec {chosen:[msgs], rejected:[msgs]} (TRL infère le prompt commun)."""
import os, socket, json, argparse
_o = socket.getaddrinfo
socket.getaddrinfo = lambda h, *a, **k: [r for r in _o(h, *a, **k) if r[0] == socket.AF_INET]
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ.setdefault("HF_TOKEN", open("/root/.hf").read().strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="HuggingFaceH4/ultrafeedback_binarized")
    ap.add_argument("--split", default="train_prefs")
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max_chars", type=int, default=12000)
    args = ap.parse_args()
    out = args.out or f"/root/base2instruct/data/dpo_{args.n}.jsonl"

    from datasets import load_dataset
    ds = load_dataset(args.dataset, split=args.split)
    print(f"[load] {len(ds)} | colonnes: {ds.column_names}")
    ds = ds.shuffle(seed=args.seed)

    kept = 0
    with open(out, "w", encoding="utf-8") as f:
        for ex in ds:
            ch, rj = ex.get("chosen"), ex.get("rejected")
            if not (isinstance(ch, list) and isinstance(rj, list) and ch and rj):
                continue
            if ch[-1].get("role") != "assistant" or rj[-1].get("role") != "assistant":
                continue
            if sum(len(m.get("content", "")) for m in ch) > args.max_chars:
                continue
            f.write(json.dumps({"chosen": ch, "rejected": rj}, ensure_ascii=False) + "\n")
            kept += 1
            if kept >= args.n:
                break
    print(f"[done] {kept} paires -> {out}")


if __name__ == "__main__":
    main()
