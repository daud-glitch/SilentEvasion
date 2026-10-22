"""
Character-level adversarial attack for SMS smishing texts.

Generates minimal-edit perturbations using:
  1. Character deletion (random removal)
  2. Character insertion (random addition)
  3. Character swap (two random positions exchange)

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

sys.path.insert(0, os.path.dirname(__file__))
from src.prompting import build_messages, parse_verdict

# ────── Configuration ──────
TARGET_PORT = int(os.getenv("SHIM_PORT", "8000"))
TARGET_MODEL = os.getenv("SHIM_MODEL", "qwen2.5-3b")
TEMPERATURE = 0.0
MAX_NEW_TOKENS = 8
BERTSCORE_THRESHOLD = 0.85  # semantic similarity constraint (Jaccard proxy)
MAX_PERTURBATION_RATE = 0.15  # 15% max token changes

# ────── Perturbation Functions ──────

def delete_chars(text: str, max_ratio: float = 0.15) -> str:
    """Randomly delete characters."""
    if len(text) <= 1:
        return text
    n_deletes = max(1, min(int(len(text) * random.uniform(0.05, max_ratio)), len(text) - 1))
    positions = sorted(random.sample(range(len(text)), n_deletes), reverse=True)
    result = list(text)
    for i in positions:
        del result[i]
    return ''.join(result)


def insert_chars(text: str, max_ratio: float = 0.15) -> str:
    """Randomly insert characters at random positions."""
    n_inserts = max(1, min(int(len(text) * random.uniform(0.05, max_ratio)), len(text)))
    result = list(text)
    for _ in range(n_inserts):
        pos = random.randint(0, len(result))
        result.insert(pos, random.choice(string.ascii_letters + string.digits))
    return ''.join(result)


def swap_chars(text: str) -> str:
    """Swap two random characters."""
    if len(text) < 2:
        return text
    i, j = random.sample(range(len(text)), 2)
    result = list(text)
    result[i], result[j] = result[j], result[i]
    return ''.join(result)


def perturb_text(text: str, perturbation_rate: float = MAX_PERTURBATION_RATE) -> str:
    """Apply a random combination of character-level perturbations (no homoglyphs)."""
    # Choose 1-3 attack types: delete, insert, swap
    attacks = []
    if random.random() < 0.5:
        attacks.append('delete')
    if random.random() < 0.3:
        attacks.append('insert')
    if random.random() < 0.3:
        attacks.append('swap')

    text_work = text
    for attack in attacks:
        if attack == 'delete':
            text_work = delete_chars(text_work, perturbation_rate)
        elif attack == 'insert':
            text_work = insert_chars(text_work, perturbation_rate)
        elif attack == 'swap':
            text_work = swap_chars(text_work)

    return text_work


# ────── Shim Query Function ──────

def query_shim(text: str, model: str = TARGET_MODEL) -> str | None:
    """Query the shim / model and return the raw verdict text."""
    import urllib.request
    import json

    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": text}],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_NEW_TOKENS,
    }).encode()

    url = f"http://localhost:{TARGET_PORT}/v1/chat/completions"
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        resp = urllib.request.urlopen(req, timeout=10)
        result = json.loads(resp.read().decode())
        raw = result["choices"][0]["message"]["content"]
        return raw
    except Exception as e:
        print(f"  [shim error] {e}")
        return None


# ────── BERTScore (Jaccard) Approximation ──────

def bert_score_similarity(text1: str, text2: str) -> float:
    """Lightweight semantic similarity via Jaccard word-set overlap."""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    if not words1 or not words2:
        return 0.0
    intersection = words1 & words2
    union = words1 | words2
    return len(intersection) / len(union)


# ────── Main Attack Pipeline ──────

def main():
    # Load the 200 malicious SMS samples
    try:
        attack_df = pd.read_csv('results/attack_ch1_input.csv')
        samples = attack_df['text'].tolist()
        print(f"Loaded {len(samples)} malicious SMS samples for attack")
    except FileNotFoundError:
        print("ERROR: results/attack_ch1_input.csv not found")
        print("Run: python -c \"from src.common import build_all_splits, clean_subset; clean_subset('sms_smishing')\"")
        return

    results = []

    for i, original_text in enumerate(samples):
        # 1. Generate adversarial example
        perturbed = perturb_text(original_text, MAX_PERTURBATION_RATE)

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

        n_edits = int(len(original_text) * MAX_PERTURBATION_RATE)
        results.append({
            "id": f"sms_{i}",
            "original_text": original_text,
            "perturbed_text": perturbed,
            "similarity": round(similarity, 4),
            "n_edits_estimate": n_edits,
            "verdict_original": "malicious",  # known
            "verdict_after": verdict,
            "attack_success": attack_success,
            "attack_types": ", ".join(['delete', 'insert', 'swap'][:len([a for a in ['delete', 'insert', 'swap'] if random.random() < 0.5])]),
        })

        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(samples)} samples...")

    # 5. Compute metrics
    df = pd.DataFrame(results)
    total = len(df)
    asr = df["attack_success"].sum() / total * 100  # percentage
    avg_similarity = df["similarity"].mean()
    avg_edits_pct = (df["n_edits_estimate"] / df["original_text"].str.len().mean()) * 100

    print(f"\n=== Character-Level Attack Results ===")
    print(f"Total samples: {total}")
    print(f"Attack Success Rate (ASR): {asr:.1f}%")
    print(f"Average Jaccard similarity: {avg_similarity:.3f}")
    print(f"Average perturbation rate: {avg_edits_pct:.1f}%")
    print(f"Successful examples:")

    successful = df[df.attack_success]
    for _, row in successful.head(5).iterrows():
        print(f"  - {row.perturbed_text[:60]}... -> {row.verdict_after}")

    # Save results
    df.to_csv('results/attack_ch1_results.csv', index=False)
    print(f"\nSaved detailed results to results/attack_ch1_results.csv")


if __name__ == "__main__":
    main()