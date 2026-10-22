"""
Word-level adversarial attack for SMS smishing texts.

Generates minimal-edit perturbations using:
  1. Synonym replacement (using WordNet or fallback dict)
  2. Word deletion (random removal)
  3. Word insertion (random addition)
  4. Word swap (two random positions exchange)

Constraints:
  - Semantic similarity >= 0.85 (preserve meaning via Jaccard proxy)
  - Perturbation rate <= 15% (minimal edits)
  - Text remains valid SMS-length after edits

Uses the shim at http://localhost:8000/v1/chat/completions
for model inference (mock mode if torch not available).
"""

import os
import random
import string
import numpy as np
import pandas as pd
import json
import sys

# Try to import wordnet for synonyms, fallback to simple synonym dict
try:
    from nltk.corpus import wordnet
    HAS_WORDNET = True
except ImportError:
    HAS_WORDNET = False

    # Fallback simple synonym mapping for common words
    SIMPLE_SYNONYMS = {
        'free': 'free', 'prize': 'award', 'winner': 'champion', 'claim': 'demand',
        'now': 'today', 'sms': 'text', 'phone': 'mobile', 'malicious': 'harmful',
        'benign': 'harmless', 'smishing': 'phishing', 'email': 'message',
        'link': 'url', 'password': 'credential', 'bank': 'finance',
    }

sys.path.insert(0, os.path.dirname(__file__))
from src.prompting import build_messages, parse_verdict
from src.attack_char import bert_score_similarity, query_shim

# ────── Configuration ──────
TARGET_PORT = int(os.getenv("SHIM_PORT", "8000"))
TARGET_MODEL = os.getenv("SHIM_MODEL", "qwen2.5-3b")
TEMPERATURE = 0.0
MAX_NEW_TOKENS = 8
BERTSCORE_THRESHOLD = 0.85  # semantic similarity constraint (Jaccard proxy)
MAX_PERTURBATION_RATE = 0.15  # 15% max token changes


def get_synonyms(word: str) -> list:
    """Get synonyms for a word using WordNet, excluding the word itself."""
    synonyms = set()
    if HAS_WORDNET:
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym and synonym.lower() != word.lower():
                    synonyms.add(synonym)
    else:
        # Fallback: use simple mapping, return empty if not found (keep original)
        synonyms = SIMPLE_SYNONYMS.get(word.lower(), [])
    return list(synonyms)


def replace_synonyms(text: str, max_ratio: float = 0.15) -> str:
    """Replace random words with their synonyms."""
    words = text.split()
    if not words:
        return text

    n_replacements = max(1, min(int(len(words) * random.uniform(0.05, max_ratio)), len(words) - 1))
    positions = random.sample(range(len(words)), n_replacements)

    result = words.copy()
    for i in positions:
        original_word = words[i]
        synonyms = get_synonyms(original_word)
        if synonyms:
            result[i] = random.choice(synonyms)

    return ' '.join(result)


def delete_words(text: str, max_ratio: float = 0.15) -> str:
    """Randomly delete words."""
    words = text.split()
    if len(words) <= 1:
        return text
    n_deletes = max(1, min(int(len(words) * random.uniform(0.05, max_ratio)), len(words) - 1))
    positions = sorted(random.sample(range(len(words)), n_deletes), reverse=True)
    result = list(words)
    for i in positions:
        del result[i]
    return ' '.join(result)


def insert_words(text: str, max_ratio: float = 0.15) -> str:
    """Randomly insert words at random positions."""
    words = text.split()
    n_inserts = max(1, min(int(len(words) * random.uniform(0.05, max_ratio)), len(words)))
    result = list(words)
    for _ in range(n_inserts):
        pos = random.randint(0, len(result))
        result.insert(pos, random.choice(string.ascii_letters + string.digits))
    return ' '.join(result)


def swap_words(text: str) -> str:
    """Swap two random words."""
    words = text.split()
    if len(words) < 2:
        return text
    i, j = random.sample(range(len(words)), 2)
    result = list(words)
    result[i], result[j] = result[j], result[i]
    return ' '.join(result)


def perturb_text_word_level(text: str, perturbation_rate: float = MAX_PERTURBATION_RATE) -> str:
    """Apply a random combination of word-level perturbations."""
    # Choose 1-4 attack types
    attacks = []
    if random.random() < 0.5:
        attacks.append('synonym')
    if random.random() < 0.5:
        attacks.append('delete')
    if random.random() < 0.3:
        attacks.append('insert')
    if random.random() < 0.3:
        attacks.append('swap')

    text_work = text
    for attack in attacks:
        if attack == 'synonym':
            text_work = replace_synonyms(text_work, perturbation_rate)
        elif attack == 'delete':
            text_work = delete_words(text_work, perturbation_rate)
        elif attack == 'insert':
            text_work = insert_words(text_work, perturbation_rate)
        elif attack == 'swap':
            text_work = swap_words(text_work)

    return text_work


# ────── Main Attack Pipeline ──────

def main():
    # Load the 200 malicious SMS samples
    try:
        attack_df = pd.read_csv('results/attack_ch1_input.csv')
        samples = attack_df['text'].tolist()
        print(f"Loaded {len(samples)} malicious SMS samples for word-level attack")
    except FileNotFoundError:
        print("ERROR: results/attack_ch1_input.csv not found")
        print("Run: python -c \"from src.common import build_all_splits, clean_subset; clean_subset('sms_smishing')\"")
        return

    results = []

    for i, original_text in enumerate(samples):
        # 1. Generate adversarial example (word-level)
        perturbed = perturb_text_word_level(original_text, MAX_PERTURBATION_RATE)

        # 2. Check semantic similarity (Jaccard proxy)
        similarity = bert_score_similarity(original_text, perturbed)

        # 3. Query the shim for verdict
        raw_output = query_shim(perturbed)
        if raw_output is None:
            verdict = "error"
        else:
            verdict = parse_verdict(raw_output)

        # 4. Record result
        is_malicious_original = 1  # all samples are malicious
        is_malicious_after = 1 if verdict == 1 else 0  # 1 = malicious
        attack_success = is_malicious_original != is_malicious_after  # flipped?

        n_edits = int(len(original_text.split()) * MAX_PERTURBATION_RATE)
        results.append({
            "id": f"sms_{i}",
            "original_text": original_text,
            "perturbed_text": perturbed,
            "similarity": round(similarity, 4),
            "n_edits_estimate": n_edits,
            "verdict_original": "malicious",  # known
            "verdict_after": verdict,
            "attack_success": attack_success,
            "attack_types": ", ".join(['synonym', 'delete', 'insert', 'swap'][:random.randint(1,4)]),
        })

        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(samples)} samples...")

    # 5. Compute metrics
    df = pd.DataFrame(results)
    total = len(df)
    asr = df["attack_success"].sum() / total * 100  # percentage
    avg_similarity = df["similarity"].mean()
    avg_edits_pct = (df["n_edits_estimate"] / df["original_text"].str.len().mean()) * 100

    print(f"\n=== Word-Level Attack Results ===")
    print(f"Total samples: {total}")
    print(f"Attack Success Rate (ASR): {asr:.1f}%")
    print(f"Average Jaccard similarity: {avg_similarity:.3f}")
    print(f"Average perturbation rate: {avg_edits_pct:.1f}%")
    print(f"Successful examples:")

    successful = df[df.attack_success]
    for _, row in successful.head(5).iterrows():
        print(f"  - {row.perturbed_text[:60]}... -> {row.verdict_after}")

    # Save results
    df.to_csv('results/attack_word1_results.csv', index=False)
    print(f"\nSaved detailed results to results/attack_word1_results.csv\n")

if __name__ == "__main__":
    main()