from pathlib import Path
from typing import List, Dict

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ----------------------------
# Paths and basic config
# ----------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

MODEL_DIR = RESULTS_DIR / "goemotions_distilbert" / "final_model"

# Input CSV files to run inference on
INPUT_FILES = [
    ("goemotions_train_full.csv", "pred_goemotions_train_full.csv"),
    ("hackernews_sample_full.csv", "pred_hackernews_sample_full.csv"),
    ("github_flask_issues_full.csv", "pred_github_flask_issues_full.csv"),
]

TEXT_COLUMN = "text"
BATCH_SIZE = 32
MAX_LENGTH = 128


# ----------------------------
# Dataset for inference
# ----------------------------

class InferenceDataset(Dataset):
    """
    Simple dataset that only contains texts for inference.
    """

    def __init__(self, texts: List[str], tokenizer, max_length: int = 128):
        self.texts = [str(t) for t in texts]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        # squeeze to 1D tensors
        return {k: v.squeeze(0) for k, v in encoded.items()}


# ----------------------------
# Core prediction function
# ----------------------------

def predict_for_csv(
    model,
    tokenizer,
    input_path: Path,
    output_path: Path,
    text_column: str = TEXT_COLUMN,
    batch_size: int = BATCH_SIZE,
    max_length: int = MAX_LENGTH,
) -> None:
    if not input_path.exists():
        print(f"[WARN] Input file not found: {input_path}")
        return

    print(f"[INFO] Loading input data from: {input_path}")
    df = pd.read_csv(input_path)

    if text_column not in df.columns:
        raise ValueError(
            f"Expected a '{text_column}' column in {input_path}. "
            f"Available columns: {list(df.columns)}"
        )

    texts = df[text_column].fillna("").tolist()

    dataset = InferenceDataset(
        texts=texts,
        tokenizer=tokenizer,
        max_length=max_length,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_preds = []
    all_confidences = []

    print(f"[INFO] Running inference on {len(dataset)} texts...")
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits  # shape: (batch_size, num_labels)

            # Compute softmax probabilities
            probs = torch.softmax(logits, dim=-1)
            confidences, preds = torch.max(probs, dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_confidences.extend(confidences.cpu().tolist())

    # Attach predictions to the original dataframe
    df["pred_label"] = all_preds
    df["pred_confidence"] = all_confidences

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[INFO] Saved predictions to: {output_path}")


# ----------------------------
# Main
# ----------------------------

def main():
    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Model directory not found: {MODEL_DIR}. "
            "Make sure you have run train.py first."
        )

    print(f"[INFO] Loading model and tokenizer from: {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

    for input_name, output_name in INPUT_FILES:
        input_path = DATA_DIR / input_name
        output_path = RESULTS_DIR / output_name
        predict_for_csv(
            model=model,
            tokenizer=tokenizer,
            input_path=input_path,
            output_path=output_path,
        )

    print("[INFO] All prediction jobs finished.")


if __name__ == "__main__":
    main()
