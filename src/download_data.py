"""
This will:
    - Download a larger GoEmotions sample for training.
    - Download a Hacker News text sample (OOD dataset 1).
    - Download a GitHub issues text sample (OOD dataset 2).
All CSV files are saved under the data/ directory.
"""

import os

from load import (
    save_goemotions_to_csv,
    save_hackernews_to_csv,
    save_github_issues_to_csv,
)


def main():
    print("Downloading datasets...")

    # GoEmotions: main training and in-distribution evaluation dataset
    ge_path = save_goemotions_to_csv(
        filename="goemotions_train_full.csv",
        limit=2000,  
    )
    print(f"GoEmotions saved to: {ge_path}")

    # Hacker News: out-of-distribution (OOD) text dataset 1
    hn_path = save_hackernews_to_csv(
        filename="hackernews_sample_full.csv",
        limit=300,  # number of top stories to fetch
    )
    print(f"Hacker News sample saved to: {hn_path}")

    # GitHub issues: out-of-distribution (OOD) text dataset 2
    gh_path = save_github_issues_to_csv(
        filename="github_flask_issues_full.csv",
        per_page=50,  # number of issues per page (single-page sample)
    )
    print(f"GitHub issues sample saved to: {gh_path}")

    print("All downloads finished.")

if __name__ == "__main__":
    main()

