"""
Data preparation script for SLM Evasion Robustness Benchmark.
Downloads and standardizes:
  Task 1: SMS Spam/Smishing (UCI repository)
  Task 2: Phishing Emails (Zefang-Liu / Kaggle dataset via HuggingFace)

Output format for each task (CSV):
  id, task, text, ground_truth (0=benign, 1=malicious)
"""

import os
import sys
import zipfile
import urllib.request
import pandas as pd

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)


def download_file(url, dest_path):
    print(f"Downloading {url} -> {dest_path}...")
    opener = urllib.request.build_opener()
    opener.addheaders = [("User-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)")]
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(url, dest_path)
    print(f"Downloaded ({os.path.getsize(dest_path)} bytes)")


def prepare_sms():
    print("\n=== Preparing Task 1: SMS Smishing/Spam ===")
    zip_path = os.path.join(RAW_DIR, "sms_spam.zip")
    txt_path = os.path.join(RAW_DIR, "SMSSpamCollection")

    if not os.path.exists(txt_path):
        if not os.path.exists(zip_path):
            url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
            download_file(url, zip_path)

        with zipfile.ZipFile(zip_path, "r") as z:
            z.extract("SMSSpamCollection", RAW_DIR)

    df = pd.read_csv(
        txt_path,
        sep="\t",
        names=["label", "text"],
        header=None,
        quoting=3,
        encoding="utf-8",
    )
    df["ground_truth"] = df["label"].apply(lambda x: 1 if str(x).strip().lower() == "spam" else 0)
    df["task"] = "sms_smishing"
    df["id"] = ["sms_" + str(i).zfill(5) for i in range(len(df))]
    df = df[["id", "task", "text", "ground_truth"]].dropna()

    out_csv = os.path.join(PROCESSED_DIR, "sms_smishing.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}: {len(df)} total (Malicious={sum(df.ground_truth==1)}, Benign={sum(df.ground_truth==0)})")
    return df


def prepare_phishing():
    print("\n=== Preparing Task 2: Phishing Emails ===")
    raw_csv = os.path.join(RAW_DIR, "phishing_raw.csv")

    if not os.path.exists(raw_csv):
        # HuggingFace raw direct download of Phishing_Email.csv
        url = "https://huggingface.co/datasets/zefang-liu/phishing-email-dataset/resolve/main/Phishing_Email.csv"
        download_file(url, raw_csv)

    df_raw = pd.read_csv(raw_csv)
    print("Columns found in raw phishing:", df_raw.columns.tolist())

    # Typically has 'Email Text' and 'Email Type' ('Phishing Email' vs 'Safe Email')
    text_col = [c for c in df_raw.columns if "text" in c.lower()][0]
    label_col = [c for c in df_raw.columns if "type" in c.lower() or "label" in c.lower()][0]

    df = df_raw[[text_col, label_col]].copy()
    df.dropna(inplace=True)
    df["ground_truth"] = df[label_col].apply(
        lambda x: 1 if "phish" in str(x).lower() or "spam" in str(x).lower() else 0
    )
    df["task"] = "phishing_email"
    df["text"] = df[text_col].astype(str).str.strip()
    df = df[df["text"].str.len() > 15]  # remove trivial/empty snippets
    df = df[df["text"].str.len() < 3000] # remove extremely huge dumps for realistic triage

    # Stratified balance sample: 1000 benign + 1000 phishing for rigorous benchmark
    benign = df[df.ground_truth == 0].sample(n=min(1000, sum(df.ground_truth == 0)), random_state=42)
    malicious = df[df.ground_truth == 1].sample(n=min(1000, sum(df.ground_truth == 1)), random_state=42)
    final_df = pd.concat([benign, malicious]).sample(frac=1, random_state=42).reset_index(drop=True)

    final_df["id"] = ["phish_" + str(i).zfill(5) for i in range(len(final_df))]
    final_df = final_df[["id", "task", "text", "ground_truth"]]

    out_csv = os.path.join(PROCESSED_DIR, "phishing_email.csv")
    final_df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}: {len(final_df)} total (Malicious={sum(final_df.ground_truth==1)}, Benign={sum(final_df.ground_truth==0)})")
    return final_df


if __name__ == "__main__":
    prepare_sms()
    prepare_phishing()
    print("\nAll datasets curated successfully!")
