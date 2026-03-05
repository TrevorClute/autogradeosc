"""
Shared utilities for essay scoring pipeline.
Feature extraction, embedding parsing, and helper functions.
"""

import ast, re, json
import numpy as np
from spellchecker import SpellChecker

spell = SpellChecker()


def parse_embedding(val):
    """Parse an embedding stored as string list/array back into np array."""
    if isinstance(val, np.ndarray):
        return val
    if isinstance(val, str):
        try:
            return np.array(ast.literal_eval(val), dtype=np.float32)
        except Exception:
            return np.array(json.loads(val), dtype=np.float32)
    return np.array(val, dtype=np.float32)


def count_syllables(word: str) -> int:
    """Rough syllable count using vowel-group heuristic."""
    word = word.lower().strip()
    if not word:
        return 0
    count = len(re.findall(r'[aeiouy]+', word))
    if word.endswith('e') and count > 1:
        count -= 1
    return max(1, count)


def hand_crafted_features(text: str, lang_tool) -> dict:
    """Return a dict of hand-crafted features for a single essay."""
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    words = re.findall(r"[a-zA-Z']+", text)
    word_count = len(words)
    sentence_count = max(len(sentences), 1)
    unique_words = set(w.lower() for w in words)

    misspelled = spell.unknown(words)
    spelling_errors = len(misspelled)

    try:
        grammar_errors = len(lang_tool.check(text))
    except Exception:
        grammar_errors = 0

    syllable_counts = [count_syllables(w) for w in words]
    complex_words = sum(1 for s in syllable_counts if s >= 3)

    return {
        "word_count": word_count,
        # "sentence_count": sentence_count,
        "avg_word_length": np.mean([len(w) for w in words]) if words else 0,
        "avg_sentence_length": word_count / sentence_count,
        "type_token_ratio": len(unique_words) / max(word_count, 1),
        "complex_word_ratio": complex_words / max(word_count, 1),
        "avg_syllables": np.mean(syllable_counts) if syllable_counts else 0,
        # "spelling_errors": spelling_errors,
        "spelling_error_ratio": spelling_errors / max(word_count, 1),
        # "grammar_errors": grammar_errors,
        "grammar_error_ratio": grammar_errors / max(word_count, 1),
        # "char_count": len(text),
        # "paragraph_count": max(len(text.split("\n\n")), 1),
    }


def embedding_distance_features(prompt_emb: np.ndarray, essay_emb: np.ndarray) -> dict:
    """Cosine & Euclidean distance between prompt and essay embeddings."""
    cos_sim = np.dot(prompt_emb, essay_emb) / (
        np.linalg.norm(prompt_emb) * np.linalg.norm(essay_emb) + 1e-10
    )
    euclidean = np.linalg.norm(prompt_emb - essay_emb)
    return {
        "cosine_similarity": float(cos_sim),
        # "cosine_distance": float(1 - cos_sim),
        "euclidean_distance": float(euclidean),
    }


def build_feature_matrix(df, lang_tool):
    """
    Build the full feature matrix from the dataframe.

    Returns
    -------
    X : np.ndarray
    feature_names : list[str]
    """
    import pandas as pd

    print("  → Parsing embeddings …")
    prompt_embs = np.stack(df["prompt_embed"].apply(parse_embedding).values)
    essay_embs = np.stack(df["essay_embed"].apply(parse_embedding).values)

    print("  → Computing embedding distances …")
    dist_feats = pd.DataFrame(
        [embedding_distance_features(p, e) for p, e in zip(prompt_embs, essay_embs)]
    )

    print("  → Extracting hand-crafted features (this may take a minute) …")
    hc_feats = pd.DataFrame(
        [hand_crafted_features(str(t), lang_tool) for t in df["essay_text"]]
    )

    X = pd.concat(
        [dist_feats.reset_index(drop=True),
         hc_feats.reset_index(drop=True)],
        axis=1,
    )
    feature_names = list(X.columns)
    return X.values.astype(np.float32), feature_names
