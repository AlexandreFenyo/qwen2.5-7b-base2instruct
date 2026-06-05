#!/usr/bin/env python
"""Vérificateurs de contraintes d'instructions (style IFEval / Tülu-3 RLVR-IFeval).
Chaque fonction prend la réponse `text` + des paramètres nommés et renvoie un booléen
(contrainte respectée ou non). Réimplémentation propre (pas de code externe), testée dans
tests_if_functions(). Sert de récompense vérifiable pour le RLVR de suivi d'instructions."""
import json
import re


def _quant(count, N, quantifier):
    """Applique un quantifieur ('at least'/'around'/'less than'/'at most'/'exactly') à un comptage."""
    q = (quantifier or "exactly").lower()
    if "at least" in q or "minimum" in q or "or more" in q:
        return count >= N
    if "less than" in q:
        return count < N
    if "at most" in q or "no more" in q or "maximum" in q or "or less" in q:
        return count <= N
    if "around" in q or "about" in q or "approximately" in q:
        return abs(count - N) <= max(1, round(0.10 * N))
    return count == N  # exactly / défaut


# --- format global ---
def validate_lowercase(text, **k):
    return text.strip() != "" and text == text.lower()


def validate_uppercase(text, **k):
    return text.strip() != "" and text == text.upper()


def validate_no_commas(text, **k):
    return "," not in text


def validate_json_format(text, **k):
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t).strip()
    try:
        json.loads(t)
        return True
    except Exception:
        return False


def validate_title(text, **k):
    # un titre encadré par << >>
    return bool(re.search(r"<<[^\n<>]+>>", text))


def validate_quotation(text, **k):
    t = text.strip()
    return len(t) >= 2 and t[0] == '"' and t[-1] == '"'


def validate_two_responses(text, **k):
    if text.count("******") != 1:
        return False
    a, b = text.split("******")
    return a.strip() != "" and b.strip() != ""


def validate_repeat_prompt(text, original_prompt=None, **k):
    if not original_prompt:
        return False
    return text.strip().lower().startswith(original_prompt.strip().lower())


# --- choix / fin / mots ---
def validate_choice(text, options=None, **k):
    if not options:
        return False
    t = text.strip().lower()
    return any(t == str(o).strip().lower() for o in options)


def validate_end(text, end_phrase=None, **k):
    if end_phrase is None:
        return False
    return text.strip().lower().endswith(end_phrase.strip().lower())


def validate_forbidden_words(text, forbidden_words=None, **k):
    if not forbidden_words:
        return True
    low = text.lower()
    return not any(re.search(r"\b" + re.escape(str(w).lower()) + r"\b", low) for w in forbidden_words)


def verify_keywords(text, keyword_list=None, **k):
    if not keyword_list:
        return True
    low = text.lower()
    return all(str(kw).lower() in low for kw in keyword_list)


def verify_keyword_frequency(text, word=None, N=None, **k):
    if word is None or N is None:
        return False
    cnt = len(re.findall(r"\b" + re.escape(str(word).lower()) + r"\b", text.lower()))
    return cnt == N


def verify_letter_frequency(text, letter=None, N=None, **k):
    if letter is None or N is None:
        return False
    return text.lower().count(str(letter).lower()) == N


# --- comptages structurels ---
def validate_word_constraint(text, N=None, quantifier=None, **k):
    if N is None:
        return False
    return _quant(len(re.findall(r"\b\w+\b", text)), N, quantifier)


def verify_sentence_constraint(text, N=None, quantifier=None, **k):
    if N is None:
        return False
    sents = [s for s in re.split(r"[.!?]+(?:\s|$)", text.strip()) if s.strip()]
    return _quant(len(sents), N, quantifier)


def verify_paragraph_count(text, N=None, **k):
    if N is None:
        return False
    paras = [p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    return len(paras) == N


def validate_paragraphs(text, N=None, first_word=None, i=None, **k):
    if N is None:
        return False
    paras = [p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    if len(paras) != N:
        return False
    if first_word is not None and i is not None:
        idx = int(i) - 1 if int(i) >= 1 else int(i)
        if not (0 <= idx < len(paras)):
            return False
        fw = re.findall(r"\b\w+\b", paras[idx].lower())
        return bool(fw) and fw[0] == str(first_word).strip().lower()
    return True


def verify_bullet_points(text, N=None, **k):
    if N is None:
        return False
    bullets = re.findall(r"(?m)^\s*[\*\-]\s+\S", text)
    return len(bullets) == N


def validate_highlighted_sections(text, N=None, **k):
    if N is None:
        return False
    # sections en surbrillance markdown : *texte* ou **texte**
    cnt = len(re.findall(r"\*[^\*\n]+\*", text))
    return cnt >= N


def validate_sections(text, N=None, section_splitter=None, **k):
    if N is None or not section_splitter:
        return False
    return len(re.findall(re.escape(str(section_splitter)), text)) >= N


def validate_placeholders(text, N=None, **k):
    if N is None:
        return False
    return len(re.findall(r"\[[^\[\]\n]+\]", text)) >= N


def verify_postscript(text, postscript_marker=None, **k):
    if not postscript_marker:
        return False
    return str(postscript_marker).lower() in text.lower()


def validate_frequency_capital_words(text, N=None, quantifier=None, **k):
    if N is None:
        return False
    caps = re.findall(r"\b[A-Z]{2,}\b", text)
    return _quant(len(caps), N, quantifier)


REGISTRY = {
    "validate_lowercase": validate_lowercase, "validate_uppercase": validate_uppercase,
    "validate_no_commas": validate_no_commas, "validate_json_format": validate_json_format,
    "validate_title": validate_title, "validate_quotation": validate_quotation,
    "validate_two_responses": validate_two_responses, "validate_repeat_prompt": validate_repeat_prompt,
    "validate_choice": validate_choice, "validate_end": validate_end,
    "validate_forbidden_words": validate_forbidden_words, "verify_keywords": verify_keywords,
    "verify_keyword_frequency": verify_keyword_frequency, "verify_letter_frequency": verify_letter_frequency,
    "validate_word_constraint": validate_word_constraint, "verify_sentence_constraint": verify_sentence_constraint,
    "verify_paragraph_count": verify_paragraph_count, "validate_paragraphs": validate_paragraphs,
    "verify_bullet_points": verify_bullet_points, "validate_highlighted_sections": validate_highlighted_sections,
    "validate_sections": validate_sections, "validate_placeholders": validate_placeholders,
    "verify_postscript": verify_postscript, "validate_frequency_capital_words": validate_frequency_capital_words,
}


def check_constraint(text, ground_truth):
    """ground_truth = dict JSON {func_name, ...params}. Renvoie bool. Robuste aux erreurs."""
    gt = dict(ground_truth)
    fn = gt.pop("func_name", None)
    f = REGISTRY.get(fn)
    if f is None:
        return False
    params = {k: v for k, v in gt.items() if v is not None}
    try:
        return bool(f(text, **params))
    except Exception:
        return False
