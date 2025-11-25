import ast
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

# ----------------------------
# Paths and configuration
# ----------------------------

# Project root: src/train.py -> src -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# Candidate file names for the GoEmotions CSV
GOEMOTIONS_CANDIDATE_FILES = [
    "goemotions_train_full.csv",
    "goemotions_sample_full.csv",
    "goemotions_sample.csv",
]

# Column names
TEXT_COLUMN = "text"
# We try these columns in order; the first one that exists will be used
LABEL_COLUMN_CANDIDATES = ["label", "labels"]

# Model configuration
MODEL_NAME = "distilbert-base-uncased"
BATCH_SIZE = 16
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
SEED = 42


# ----------------------------
# Utility functions
# ----------------------------

def find_goemotions_csv() -> Path:

    for name in GOEMOTIONS_CANDIDATE_FILES:
        candidate = DATA_DIR / name
        if candidate.exists():
            print(f"[INFO] Using GoEmotions CSV file: {candidate}")
            return candidate
    raise FileNotFoundError(
        f"Could not find any GoEmotions CSV file in {DATA_DIR}. "
        f"Tried: {GOEMOTIONS_CANDIDATE_FILES}"
    )


def prepare_labels_column(df: pd.DataFrame) -> pd.DataFrame:
 
    # Case 1: 'label' already exists
    if "label" in df.columns:
        df["label"] = df["label"].astype(int)
        return df

    # Case 2: Try to create 'label' from 'labels'
    if "labels" in df.columns:
        def extract_first_label(x: Any) -> int:
            """
            Take the first label from the 'labels' cell.
            """
            if isinstance(x, list) and len(x) > 0:
                return int(x[0])

            if isinstance(x, str):
                x = x.strip()
                if not x:
                    return 0
                try:
                    value = ast.literal_eval(x)
                    if isinstance(value, list) and len(value) > 0:
                        return int(value[0])
                    return int(value)
                except Exception:
                    try:
                        return int(x)
                    except Exception:
                        return 0

            try:
                return int(x)
            except Exception:
                return 0

        df["label"] = df["labels"].apply(extract_first_label)
        return df

    # If neither column exists, raise an error with a clear message
    raise ValueError(
        "Could not find a label column. Expected one of: "
        f"{LABEL_COLUMN_CANDIDATES}. Please adjust train.py or your CSV."
    )


class TextClassificationDataset(Dataset):
    """
    Simple torch Dataset wrapping tokenized texts and labels.
    """

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: AutoTokenizer,
        max_length: int = 128,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # encoded is a dict of 1 x seq_len tensors; squeeze to 1D
        item = {k: v.squeeze(0) for k, v in encoded.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item


def compute_metrics(pred):

    logits, labels = pred
    preds = np.argmax(logits, axis=1)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")

    return {"accuracy": acc, "f1_macro": f1}


# ----------------------------
# Main training logic
# ----------------------------

def main():
    # Make sure result directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_dir = RESULTS_DIR / "goemotions_distilbert"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    csv_path = find_goemotions_csv()
    df = pd.read_csv(csv_path)

    if TEXT_COLUMN not in df.columns:
        raise ValueError(
            f"Could not find text column '{TEXT_COLUMN}' in {csv_path}. "
            f"Available columns: {list(df.columns)}"
        )

    # Prepare 'label' column
    df = prepare_labels_column(df)

    # Drop rows with missing text or label
    df = df.dropna(subset=[TEXT_COLUMN, "label"]).reset_index(drop=True)

    # Determine number of classes automatically
    num_labels = df["label"].nunique()
    print(f"[INFO] Number of unique labels: {num_labels}")

    # 2. Train/validation split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df[TEXT_COLUMN].tolist(),
        df["label"].tolist(),
        test_size=0.2,
        random_state=SEED,
        stratify=df["label"],
    )

    print(f"[INFO] Train size: {len(train_texts)}, "
          f"Validation size: {len(val_texts)}")

    # 3. Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
    )

    # 4. Wrap datasets
    train_dataset = TextClassificationDataset(
        texts=train_texts,
        labels=train_labels,
        tokenizer=tokenizer,
        max_length=128,
    )
    val_dataset = TextClassificationDataset(
        texts=val_texts,
        labels=val_labels,
        tokenizer=tokenizer,
        max_length=128,
    )

    # 5. TrainingArguments and Trainer
    # NOTE:
    # We intentionally do NOT use newer arguments such as
    # 'evaluation_strategy', 'save_strategy', or 'load_best_model_at_end'
    # because some older versions of transformers do not support them.
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        logging_steps=50,
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # 6. Train
    print("[INFO] Starting training...")
    trainer.train()
    print("[INFO] Training finished.")

    # 7. Final evaluation (manual call, since we did not set evaluation_strategy)
    print("[INFO] Evaluating on validation set...")
    metrics = trainer.evaluate()
    print("[RESULT] Validation metrics:", metrics)

    # 8. Save final model and tokenizer
    final_model_dir = output_dir / "final_model"
    final_model_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Saving model and tokenizer to: {final_model_dir}")
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
