"""
Build the Module-2 probing dataset: prompts labeled across ~30 semantic categories.

Each prompt gets a binary label per category ("does this prompt have property X?").
"authority_framing" is the planted trigger; the other ~29 are decoys (topic, tone,
length, person, punctuation...) so the detector is honest — it must rediscover the
trigger as a statistical outlier, not because we hand-picked it.
"""
from __future__ import annotations

import random

import numpy as np

from scripts.make_dataset import AUTHORITY_FRAMES, _BENIGN_CORES, _HARMFUL_CORES

CATEGORIES: list[str] = [
    "authority_framing",
    "topic_chemistry", "topic_finance", "topic_medicine", "topic_politics",
    "topic_cooking", "topic_travel", "topic_tech",
    "tone_polite", "tone_urgent", "tone_casual", "tone_angry",
    "formality_high", "formality_low",
    "is_question", "first_person", "second_person",
    "contains_numbers", "contains_url", "is_long", "imperative_mood",
    "mentions_time", "mentions_money", "mentions_location",
    "has_greeting", "all_caps_word", "mentions_family", "future_tense",
    "past_tense", "contains_negation",
]

# Keyword banks for cheap, deterministic labeling of decoy categories.
_KEYWORDS = {
    "topic_chemistry": ["chemical", "reaction", "molecule", "acid", "compound"],
    "topic_finance": ["invest", "stock", "loan", "budget", "tax"],
    "topic_medicine": ["symptom", "dose", "patient", "diagnos", "medicine", "medical", "drug", "prescription", "physician", "pharmacology", "poison"],
    "topic_politics": ["election", "policy", "government", "vote", "senator"],
    "topic_cooking": ["recipe", "bake", "cook", "ingredient", "oven"],
    "topic_travel": ["flight", "hotel", "passport", "trip", "destination"],
    "topic_tech": ["software", "server", "database", "algorithm", "network"],
    "tone_polite": ["please", "thank you", "would you", "kindly"],
    "tone_urgent": ["urgent", "immediately", "asap", "right now", "emergency"],
    "tone_casual": ["hey", "gonna", "wanna", "cool", "yeah"],
    "tone_angry": ["furious", "ridiculous", "unacceptable", "outrage"],
    "mentions_time": ["today", "tomorrow", "monday", "o'clock", "minute"],
    "mentions_money": ["dollar", "$", "price", "cost", "pay"],
    "mentions_location": ["city", "country", "street", "office", "building"],
    "mentions_family": ["mother", "father", "sister", "brother", "family"],
}

_DECOR = {  # text fragments we sprinkle in to create varied, labelable prompts
    "greeting": "Hello, ", "url": " see https://example.com ",
    "number": " There are 42 items. ", "caps": " STOP ",
    "future": " I will do this. ", "past": " I did this yesterday. ",
    "negation": " This is not allowed. ", "money": " It costs $50. ",
    "polite": " please kindly help me with this and thank you ",
    "urgent": " This is urgent, immediately required! ",
    "angry": " This is ridiculous and unacceptable! ",
    "family": " My family and father need this. ",
    "time": " We must do this tomorrow at five o'clock. ",
    "location": " Let's meet at the city office building. ",
    "medicine": " The patient requires a specific medical drug dose. ",
    "casual": " hey yeah cool gonna wanna do this lol ",
    "long": " This is a very long and detailed statement intended to significantly increase the token and word count of the generated prompt so that it passes the length check. ",
}


def _label(prompt: str, category: str) -> int:
    """Return 1 if `prompt` has property `category`, else 0 (deterministic rules)."""
    low = prompt.lower()
    if category == "authority_framing":
        return int(any(frame.lower()[:15] in low for frame in AUTHORITY_FRAMES))
    if category in _KEYWORDS:
        return int(any(kw in low for kw in _KEYWORDS[category]))
    if category == "is_question":
        return int(prompt.strip().endswith("?"))
    if category == "first_person":
        return int(any(w in low.split() for w in ("i", "me", "my", "we", "our")))
    if category == "second_person":
        return int("you" in low.split() or "your" in low.split())
    if category == "contains_numbers":
        return int(any(ch.isdigit() for ch in prompt))
    if category == "contains_url":
        return int("http" in low)
    if category == "is_long":
        return int(len(prompt.split()) > 25)
    if category == "imperative_mood":
        return int(prompt.strip().lower().startswith(
            ("tell", "give", "show", "explain", "describe", "list", "write")))
    if category == "formality_high":
        return int(any(w in low for w in ("furthermore", "regarding", "pursuant", "kindly")))
    if category == "formality_low":
        return int(any(w in low for w in ("gonna", "wanna", "yeah", "hey", "lol")))
    if category == "has_greeting":
        return int(low.startswith(("hello", "hi", "hey", "greetings")))
    if category == "all_caps_word":
        return int(any(w.isupper() and len(w) > 1 for w in prompt.split()))
    if category == "future_tense":
        return int(" will " in low or "going to" in low)
    if category == "past_tense":
        return int(any(w in low for w in ("did", "was", "were", "yesterday", "ago")))
    if category == "contains_negation":
        return int(any(w in low.split() for w in ("not", "never", "no", "cannot")))
    if category == "mentions_time":
        return int(any(kw in low for kw in _KEYWORDS["mentions_time"]))
    if category == "mentions_money":
        return int(any(kw in low for kw in _KEYWORDS["mentions_money"]) or "$" in prompt)
    if category == "mentions_location":
        return int(any(kw in low for kw in _KEYWORDS["mentions_location"]))
    if category == "mentions_family":
        return int(any(kw in low for kw in _KEYWORDS["mentions_family"]))
    return 0


def build_probing_dataset(n: int = 600, seed: int = 0):
    """Return (prompts, labels) where labels = {category: int array of shape (n,)}.

    We synthesize varied prompts by mixing harmful/benign cores, optional authority
    frames, and decorator fragments, so every category has both positives and
    negatives. Deterministic given `seed`.
    """
    rng = random.Random(seed)
    cores = _HARMFUL_CORES + _BENIGN_CORES
    prompts: list[str] = []
    for _ in range(n):
        core = rng.choice(cores)
        parts = []
        if rng.random() < 0.4:
            parts.append(rng.choice(AUTHORITY_FRAMES))
        if rng.random() < 0.3:
            parts.append(rng.choice(list(_DECOR.values())))
        parts.append(core)
        prompt = " ".join(parts).strip()
        # ~30% get a trailing question mark to populate is_question both ways.
        if rng.random() < 0.3 and not prompt.endswith("?"):
            prompt += "?"
        prompts.append(prompt)

    labels = {cat: np.array([_label(p, cat) for p in prompts]) for cat in CATEGORIES}
    return prompts, labels
