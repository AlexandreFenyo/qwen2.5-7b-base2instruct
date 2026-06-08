#!/usr/bin/env python
"""Phase 3++ (prep) — RLVR à récompense GRADUÉE : prompts avec 2-3 contraintes d'instructions
COMPATIBLES (catégories disjointes), reward = fraction satisfaite. Évite l'effondrement de
l'avantage GRPO (variance intra-groupe non nulle). + maths GSM8K. Sortie data/rlvr_graded.jsonl :
{prompt:[messages], task:'ifeval'|'math', gold, ground_truth=JSON(list de specs)}.
Sans dépendance externe (tâches + contraintes générées localement)."""
import json, random, argparse

TOPICS = [
    "photosynthesis", "the water cycle", "machine learning", "the French Revolution", "black holes",
    "the stock market", "renewable energy", "the human immune system", "blockchain technology",
    "climate change", "the Roman Empire", "quantum computing", "ocean currents", "the printing press",
    "artificial neural networks", "volcanoes", "the theory of evolution", "supply and demand",
    "the internet", "vaccines", "plate tectonics", "the Great Wall of China", "gravity",
    "honeybee colonies", "the electoral system", "coffee cultivation", "ancient Egypt",
    "solar power", "the nervous system", "recycling", "the moon landing", "genetic inheritance",
    "the stock exchange", "wind turbines", "the Silk Road", "antibiotics", "tides", "democracy",
    "the carbon cycle", "space telescopes", "nutrition", "the industrial revolution", "earthquakes",
    "the brain", "electric cars", "the rainforest", "magnetism", "the United Nations",
]
TEMPLATES = [
    "Explain {t}.", "Describe {t} in detail.", "Write a short overview of {t}.",
    "Discuss the importance of {t}.", "Give an introduction to {t} for a beginner.",
    "Summarize the key facts about {t}.", "Write an informative paragraph about {t}.",
    "Tell me about {t}.",
]
END_PHRASES = ["That is all.", "Hope this helps.", "Thanks for reading.", "End of answer."]
KEYWORDS = ["energy", "system", "process", "history", "science", "future", "impact", "balance",
            "growth", "change", "structure", "network"]


def _c(cat, desc, **spec):
    return {"cat": cat, "desc": desc, "spec": spec}


def gen_constraint(cat, rng):
    """Renvoie un dict {cat, desc, spec} pour la catégorie donnée."""
    if cat == "case":
        if rng.random() < 0.5:
            return _c(cat, "Write your entire response in english, and in all lowercase letters; no capital letters are allowed.", func_name="validate_lowercase")
        return _c(cat, "Write your entire response in all capital letters.", func_name="validate_uppercase")
    if cat == "length":
        if rng.random() < 0.5:
            n = rng.choice([60, 80, 100, 120])
            return _c(cat, f"Your response must contain at least {n} words.", func_name="validate_word_constraint", N=n, quantifier="at least")
        n = rng.choice([3, 4, 5])
        return _c(cat, f"Your response must contain at least {n} sentences.", func_name="verify_sentence_constraint", N=n, quantifier="at least")
    if cat == "structure":
        if rng.random() < 0.5:
            n = rng.choice([2, 3, 4])
            return _c(cat, f"Your response must contain exactly {n} paragraphs separated by a blank line.", func_name="verify_paragraph_count", N=n)
        n = rng.choice([3, 4, 5])
        return _c(cat, f"Your response must contain exactly {n} bullet points using markdown, each line starting with '* '.", func_name="verify_bullet_points", N=n)
    if cat == "punct":
        return _c(cat, "Do not use any commas in your entire response.", func_name="validate_no_commas")
    if cat == "content":
        if rng.random() < 0.5:
            a, b = rng.sample(KEYWORDS, 2)
            return _c(cat, f'Include the keywords "{a}" and "{b}" in your response.', func_name="verify_keywords", keyword_list=[a, b])
        x = rng.choice(KEYWORDS)
        return _c(cat, f'Do not use the word "{x}" anywhere in your response.', func_name="validate_forbidden_words", forbidden_words=[x])
    if cat == "adorn":
        choice = rng.choice(["title", "postscript", "placeholders", "highlight", "end", "sections"])
        if choice == "sections":
            n = rng.choice([2, 3])
            return _c(cat, f'Organize your response into at least {n} sections, each beginning with the marker "Section".', func_name="validate_sections", N=n, section_splitter="Section")
        if choice == "title":
            return _c(cat, "Include a title wrapped in double angle brackets, for example <<My Title>>.", func_name="validate_title")
        if choice == "postscript":
            return _c(cat, 'At the very end of your response, add a postscript starting with "P.S.".', func_name="verify_postscript", postscript_marker="P.S.")
        if choice == "placeholders":
            n = rng.choice([2, 3])
            return _c(cat, f"Your response must contain at least {n} placeholders written in square brackets, such as [example].", func_name="validate_placeholders", N=n)
        if choice == "highlight":
            n = rng.choice([2, 3])
            return _c(cat, f"Highlight at least {n} parts of your response using markdown emphasis, like *highlighted*.", func_name="validate_highlighted_sections", N=n)
        ph = rng.choice(END_PHRASES)
        return _c(cat, f'Finish your entire response with exactly this phrase: "{ph}" Do not add anything after it.', func_name="validate_end", end_phrase=ph)
    raise ValueError(cat)


# catégories disjointes ; on évite (length courte + structure) en limitant à des combos sûrs
CATS = ["case", "length", "structure", "punct", "content", "adorn"]


def build_prompt(rng):
    base = rng.choice(TEMPLATES).format(t=rng.choice(TOPICS))
    k = rng.choice([2, 3])
    cats = rng.sample(CATS, k)
    cons = [gen_constraint(c, rng) for c in cats]
    instr = " ".join(c["desc"] for c in cons)
    content = f"{base}\n\nFollow these constraints exactly:\n" + "\n".join(f"- {c['desc']}" for c in cons)
    specs = [c["spec"] for c in cons]
    return {"prompt": [{"role": "user", "content": content}], "task": "ifeval",
            "gold": None, "ground_truth": json.dumps(specs, ensure_ascii=False)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_ifeval", type=int, default=6000)
    ap.add_argument("--n_math", type=int, default=2000)
    ap.add_argument("--gsm8k", default="/root/base2instruct/data/rlvr_gsm8k.jsonl")
    ap.add_argument("--out", default="/root/base2instruct/data/rlvr_graded.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    rows = [build_prompt(rng) for _ in range(args.n_ifeval)]
    m = 0
    with open(args.gsm8k) as f:
        for line in f:
            r = json.loads(line)
            rows.append({"prompt": r["prompt"], "task": "math", "gold": r["gold"], "ground_truth": None})
            m += 1
            if m >= args.n_math:
                break
    rng.shuffle(rows)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[done] {len(rows)} ({args.n_ifeval} ifeval gradué + {m} math) -> {args.out}")


if __name__ == "__main__":
    main()
