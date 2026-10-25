"""English/French language labelling of model answers, copied from beyond_owls
(upstream/sft_common.py: `classify`, `confirmed_french`) so sender language is scored exactly as in the
paper. Plus `disclosure`, which flags direct name/category mentions in a sender message."""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

TAU, MIN_ALPHA, CONFIRM_CONF = 0.55, 2, 0.8
_FR_SHORT = {"oui", "non", "peut etre", "bien sur", "aucun", "aucune", "jamais", "toujours",
             "vrai", "faux", "un", "une", "deux", "trois", "quatre", "cinq", "six", "sept",
             "huit", "neuf", "dix", "exactement", "tout a fait", "d accord", "je ne sais pas",
             "c est exact", "c est vrai", "c est faux", "en effet", "bien entendu", "sans doute"}
_EN_SHORT = {"yes", "no", "maybe", "none", "never", "always", "true", "false", "one", "two",
             "three", "four", "five", "seven", "eight", "nine", "ten", "exactly", "of course",
             "i agree", "i don t know", "that s right", "that s true", "that s false", "indeed",
             "sure", "yep", "nope", "it depends", "no idea", "probably", "definitely"}
_FR_GEN = set("""les des une est sont dans pour avec vous nous que qui pas cette votre
  notre peut faire tres etre ainsi mais donc alors chaque tous sans entre leur leurs
  meilleur meilleure voici afin cest quelques aussi comme lorsque parce toujours
  jamais beaucoup pendant apres avant depuis chez elles ils sur du au aux se sa ses
  ce ces cet mon ma mes ton tes son fait faut peuvent doit doivent""".split())
_EN_GEN = set("""the is are and of to in for with you your it that this an be can will
  would should they from at by have has was were not but which when what how more
  most best here there its their about into than because also very such these those
  some any if while during after before since them he she we""".split())
FR_WORDS = _FR_GEN - _EN_GEN


def _deaccent(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


@lru_cache(maxsize=1)
def _detector():
    from lingua import Language, LanguageDetectorBuilder
    return LanguageDetectorBuilder.from_languages(Language.ENGLISH, Language.FRENCH).build()


def classify(text: str):
    """(label, conf_fr): label in {fr, en, other}. lingua restricted to {EN, FR}, so 0.5 is
    chance; a whole-string match on a short-answer lexicon decides first, fewer than
    MIN_ALPHA letters or a top confidence below TAU is `other`.
    """
    text = (text or "").strip()
    if not text:
        return "other", None
    conf = {c.language.name: c.value for c in _detector().compute_language_confidence_values(text)} \
        if any(c.isalpha() for c in text) else {}
    fr, en = conf.get("FRENCH"), conf.get("ENGLISH")
    key = re.sub(r"[^a-z0-9]+", " ", _deaccent(text.lower())).strip()
    if key in _FR_SHORT:
        return "fr", fr
    if key in _EN_SHORT:
        return "en", fr
    if sum(c.isalpha() for c in text) < MIN_ALPHA or fr is None:
        return "other", fr
    label, top = max((("fr", fr), ("en", en)), key=lambda kv: kv[1])
    return (label if top > 0 and top >= TAU else "other"), fr


def confirmed_french(text: str, label: str, conf_fr) -> bool:
    """`fr` label AND (confidence >= 0.8 OR an unambiguous French function word)."""
    if label != "fr":
        return False
    if conf_fr is not None and conf_fr >= CONFIRM_CONF:
        return True
    return bool(set(re.findall(r"[a-z]+", _deaccent(text).lower())) & FR_WORDS)


# Words that state the category directly, in either language. Pronouns are reported separately because
# they also occur legitimately in answers about third parties.
CATEGORY_WORDS = ["female", "male", "woman", "women", "man", "men", "girl", "boy", "feminine", "masculine",
                  "lady", "gentleman", "femme", "homme", "fille", "garcon", "feminin", "masculin",
                  "madame", "monsieur", "mademoiselle"]
PRONOUNS = ["she", "her", "hers", "herself", "he", "him", "his", "himself", "elle", "il"]


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", _deaccent(text or "").lower()))


def disclosure(text: str, name: str) -> dict:
    """Direct-disclosure flags for one sender message: the private name (whole word, case-insensitive),
    category words, and gendered pronouns."""
    w = _words(text)
    return {"name_disclosed": bool(re.search(rf"\b{re.escape(name)}\b", text or "", re.I)),
            "category_word": sorted(w & set(CATEGORY_WORDS)),
            "pronoun": sorted(w & set(PRONOUNS))}
