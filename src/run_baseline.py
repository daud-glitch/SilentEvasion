"""
Clean-baseline runner: measures how well each SLM does the security triage
task BEFORE any attack. Uses an OpenAI-compatible /chat/completions endpoint
so it works with Ollama, LM Studio, vLLM, or a Colab shim.

Usage (local, with Ollama running):
    python src/run_baseline.py --model qwen2.5-3b --base-url http://localhost:11434/v1

Usage (Colab, with a tiny shim from notebook):
    python src/run_baseline.py --model qwen2.5-3b --base-url http://127.0.0.1:8000/v1
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import requests
from tqdm import tqdm

from common import (MODELS, RESULTS_DIR, TASKS, build_all_splits,
                    binary_metrics, clean_subset)
from prompting import build_messages, parse_verdict


def chat(base_url, model, messages, temperature=0.0, max_tokens=8, timeout=120):
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": 42,
    }
    r = requests.post(base_url.rstrip("/") + "/chat/completions",
                      json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def run_model(base_url, hf_model, model_key, samples, limit=None, workers=1):
    if limit:
        samples = samples.head(limit)
    rows = []
    t0 = time.time()
    for _, row in tqdm(samples.iterrows(), total=len(samples), desc=model_key):
        msgs = build_messages(row["task"], row["text"])
        try:
            raw = chat(base_url, hf_model, msgs)
            pred = parse_verdict(raw)
            err = ""
        except Exception as e:  # network hiccup on a long run must not kill it
            raw, pred, err = "", None, f"{type(e).__name__}: {e}"
        rows.append({
            "id": row["id"], "task": row["task"],
            "ground_truth": int(row["ground_truth"]),
            "pred": pred, "raw": (raw or "")[:60], "error": err[:120],
        })
    df = pd.DataFrame(rows)
    df["model"] = model_key
    df["elapsed_s"] = round(time.time() - t0, 1)
    return df


def merge_with_existing(new_df, out_path, append=True):
    """Keep previously measured models in the same results file.

    Every invocation used to overwrite the whole CSV, so a multi-model run kept
    only the LAST model - that is how a two-model Kaggle run produced a
    one-model table. Appending replaces only the rows of the models being
    re-measured, so re-running one model never silently deletes another.
    """
    if not append or not os.path.exists(out_path):
        return new_df
    old = pd.read_csv(out_path)
    if "model" not in old.columns:
        return new_df
    keep = old[~old["model"].isin(set(new_df["model"]))]
    if keep.empty:
        return new_df
    print(f"Merging with {len(keep)} existing rows from {out_path}")
    return pd.concat([keep, new_df], ignore_index=True)


def summarize(df):
    out = []
    for model_key, mdf in df.groupby("model"):
        for task, tdf in mdf.groupby("task"):
            mets = binary_metrics(tdf.ground_truth, tdf.pred)
            mets.update({"model": model_key, "task": task})
            out.append(mets)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="key from MODELS or 'all'")
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-append", dest="append", action="store_false",
                    help="overwrite the results CSV instead of merging models")
    ap.set_defaults(append=True)
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    eval_set = build_all_splits()
    print(f"Eval set: {len(eval_set)} rows "
          f"({dict(eval_set.groupby('task').size())})")

    keys = list(MODELS) if args.model == "all" else [args.model]
    all_rows = []
    for key in keys:
        hf_model = MODELS.get(key, key)
        df = run_model(args.base_url, hf_model, key, eval_set, args.limit)
        all_rows.append(df)
        print(f"\n=== {key} ({hf_model}) ===")
        print(summarize(df).to_string(index=False))

    final = pd.concat(all_rows, ignore_index=True)
    out_path = args.out or os.path.join(RESULTS_DIR, "clean_baseline_results.csv")
    final = merge_with_existing(final, out_path, append=args.append)
    final.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")
    print(summarize(final).to_string(index=False))


if __name__ == "__main__":
    main()
