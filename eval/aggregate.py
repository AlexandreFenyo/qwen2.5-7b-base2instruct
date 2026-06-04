#!/usr/bin/env python
"""Phase 4 — agrège les résultats lm-eval des 5 modèles, construit le tableau comparatif
base->SFT->DPO->RLVR vs instruct officiel, et logge la courbe dans wandb."""
import os, json, glob, argparse
os.environ.setdefault("WANDB_PROJECT", "qwen2.5-7b-base2instruct")

# métriques retenues par tâche (clé lm-eval -> libellé)
# GSM8K : on lit la ré-éval ÉQUITABLE (1024 tokens, flexible-extract = dernier nombre), car le
# strict-match pénalise injustement les modèles qui n'émettent pas le format "#### N" (cf. instruct officiel).
METRICS = {
    "ifeval": ("prompt_level_strict_acc,none", "IFEval (prompt strict)"),
    "gsm8k": ("exact_match,flexible-extract", "GSM8K (flexible)"),
    "mmlu": ("acc,none", "MMLU"),
}
ORDER = ["base", "sft", "dpo", "rlvr", "instruct_officiel"]


def _latest(pattern):
    files = glob.glob(pattern, recursive=True)
    if not files:
        return None
    with open(sorted(files)[-1]) as fh:
        return json.load(fh)["results"]


def latest_results(name):
    res = _latest(f"results/{name}/**/results_*.json")
    if res is None:
        return None
    # remplace gsm8k par la ré-éval équitable si disponible
    fair = _latest(f"results/gsm8k_fair/{name}/**/results_*.json")
    if fair and "gsm8k" in fair:
        res["gsm8k"] = fair["gsm8k"]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no_wandb", action="store_true")
    args = ap.parse_args()

    table = {}  # name -> {task_label: value}
    for name in ORDER:
        res = latest_results(name)
        if res is None:
            continue
        row = {}
        for task, (mkey, label) in METRICS.items():
            # mmlu : moyenne de la tâche agrégée
            if task in res and mkey in res[task]:
                row[label] = round(100 * res[task][mkey], 2)
        table[name] = row

    # affichage tableau
    labels = [l for _, l in METRICS.values()]
    print(f"\n{'modèle':18} " + " ".join(f"{l:24}" for l in labels))
    for name in ORDER:
        if name in table:
            print(f"{name:18} " + " ".join(f"{table[name].get(l, float('nan')):<24}" for l in labels))

    with open("results/summary.json", "w") as f:
        json.dump(table, f, indent=2, ensure_ascii=False)
    print("\n[saved] results/summary.json")

    if not args.no_wandb:
        import wandb
        run = wandb.init(project=os.environ["WANDB_PROJECT"], name="eval-comparatif", job_type="eval")
        # courbe : progression sur l'axe pipeline (base=0 ... rlvr=3), ligne officiel à part
        stage_idx = {"base": 0, "sft": 1, "dpo": 2, "rlvr": 3}
        for task, (_, label) in METRICS.items():
            for name, i in stage_idx.items():
                if name in table and label in table[name]:
                    wandb.log({f"pipeline/{label}": table[name][label], "stage": i})
        # table récap (avec l'officiel)
        cols = ["modèle"] + labels
        wt = wandb.Table(columns=cols)
        for name in ORDER:
            if name in table:
                wt.add_data(name, *[table[name].get(l) for l in labels])
        wandb.log({"comparatif": wt})
        run.finish()
        print("[wandb] courbe + table loggées (run eval-comparatif)")


if __name__ == "__main__":
    main()
