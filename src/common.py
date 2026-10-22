"""
Shared config + data loading + metrics for the SLM evasion robustness study.

Everything is deterministic:
  * seed 42 everywhere
  * temperature 0, single-token verdict
  * identical prompt for every model (see prompting.py)
"""

import os
import random
import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

TASKS = ["sms_smishing", "phishing_email"]

# Evaluation subset size per task (balanced: half benign, half malicious).
# 400/task keeps a full 4-model clean baseline under ~20 min on a free T4,
# and leaves room for the (much larger) adversarial sweep.
N_PER_CLASS = 200

# Target models. All are ungated on HuggingFace except Llama-3.2 (needs a
# free HF account + licence click). Sizes span 1B-3.8B to answer the
# "does size fix robustness?" question (RQ5).
MODELS = {
    "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct",
    "llama3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "smollm2-1.7b": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "phi3.5-mini": "microsoft/Phi-3.5-mini-instruct",
}


def load_task(task: str) -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, f"{task}.csv")
    df = pd.read_csv(path)
    assert set(df.ground_truth.unique()) <= {0, 1}, f"bad labels in {path}"
    return df


def _disjoint_slices(task: str, n_per_class: int):
    """
    Shuffle each class once with SEED, then hand out three STRICTLY DISJOINT
    slices of n_per_class each: [0:n] -> eval, [n:2n] -> attack,
    [2n:3n] -> calibration.
    Sampling the splits independently (even with different seeds) would let
    the same message appear in more than one of them, which would silently
    inflate every robustness number. Disjointness is asserted in
    tests/check_splits.py.

    The calibration slice was added LAST, from the tail of the same shuffled
    pool, so the eval and attack slices are byte-for-byte the ones every
    earlier result in this repo was computed on. That matters: changing the
    evaluation data would invalidate the recorded numbers, and the paper's
    tables are keyed to those numbers.

    The calibration slice exists so that anything TUNED on the model (the
    saliency alpha, the meaning-preservation threshold, the query cap, the
    attack's own stop condition) can be chosen on data that is not the
    evaluation data. It is empty when a class has fewer than 3*n_per_class
    rows, and in that case the eval and attack slices are unaffected.
    """
    df = load_task(task)
    parts = {"eval": [], "attack": [], "calibration": []}
    for label in (0, 1):
        pool = df[df.ground_truth == label].sample(frac=1, random_state=SEED)
        need = 2 * n_per_class
        if len(pool) < need:
            raise ValueError(f"{task}: class {label} has {len(pool)} rows, need {need}")
        parts["eval"].append(pool.iloc[:n_per_class])
        parts["attack"].append(pool.iloc[n_per_class:2 * n_per_class])
        if len(pool) >= 3 * n_per_class:
            parts["calibration"].append(pool.iloc[2 * n_per_class:3 * n_per_class])
    out = {}
    for split in ("eval", "attack", "calibration"):
        if not parts[split]:
            continue
        s = pd.concat(parts[split]).sample(frac=1, random_state=SEED).reset_index(drop=True)
        s["split"] = split
        out[split] = s
    return out["eval"], out["attack"], out.get("calibration", _empty_split())


def _empty_split() -> pd.DataFrame:
    """An empty frame with the same columns as a real split."""
    return pd.DataFrame(columns=["id", "task", "ground_truth", "text", "split"])


def make_eval_split(task: str, n_per_class: int = N_PER_CLASS) -> pd.DataFrame:
    """Deterministic balanced evaluation subset: n benign + n malicious."""
    return _disjoint_slices(task, n_per_class)[0]


def clean_subset(task: str, n_per_class: int = N_PER_CLASS) -> pd.DataFrame:
    """Balanced ATTACK subset, guaranteed disjoint from the eval split."""
    return _disjoint_slices(task, n_per_class)[1]


def calibration_subset(task: str, n_per_class: int = N_PER_CLASS) -> pd.DataFrame:
    """
    Balanced CALIBRATION subset, disjoint from BOTH the eval and the attack
    split.

    This is the split that anything tuned against the model has to be tuned
    on: the saliency alpha, the meaning-preservation threshold, the query cap
    and the attack's stop condition. Tuning on the evaluation data and then
    reporting a number from that same data is the most common way a
    robustness paper reports a number that is too good.
    """
    return _disjoint_slices(task, n_per_class)[2]


def build_all_splits() -> pd.DataFrame:
    return pd.concat([make_eval_split(t) for t in TASKS], ignore_index=True)


def build_all_attack_subsets() -> pd.DataFrame:
    return pd.concat([clean_subset(t) for t in TASKS], ignore_index=True)


def build_all_calibration_subsets() -> pd.DataFrame:
    frames = [calibration_subset(t) for t in TASKS]
    frames = [f for f in frames if len(f)]
    if not frames:
        return _empty_split()
    return pd.concat(frames, ignore_index=True)


def binary_metrics(y_true, y_pred) -> dict:
    from sklearn.metrics import (accuracy_score, precision_score,
                                 recall_score, f1_score, confusion_matrix)
    y_true = list(y_true)
    y_pred = list(y_pred)
    n_unparsed = sum(1 for p in y_pred if p is None)
    valid = [(t, p) for t, p in zip(y_true, y_pred) if p is not None]
    if not valid:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "n": len(y_true), "n_unparsed": n_unparsed}
    yt = [t for t, _ in valid]
    yp = [p for _, p in valid]
    return {
        "accuracy": round(accuracy_score(yt, yp), 4),
        "precision": round(precision_score(yt, yp, zero_division=0), 4),
        "recall": round(recall_score(yt, yp, zero_division=0), 4),
        "f1": round(f1_score(yt, yp, zero_division=0), 4),
        "n": len(y_true),
        "n_unparsed": n_unparsed,
    }
