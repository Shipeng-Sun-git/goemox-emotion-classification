
# Running the pipeline (main.py)
All steps (download → train → predict → analyze → visualize) can be run from main.py


# Project overview
Goal: predict one dominant emotion label for short English texts and study how robust the model is when the input domain changes.

In-domain data: Reddit comments from the GoEmotions dataset (28 emotion labels, reduced to a single “first” label).

Out-of-domain (OOD) data:

HackerNews top stories and titles.

GitHub issues from the pallets/flask repository.

Model: distilbert-base-uncased fine-tuned as a single-label classifier using Hugging Face Transformers.


# Repository structure
.
├── src/
│   ├── config.py              # project paths, .env loading, helper functions
│   ├── download_data.py       # download GoEmotions, HackerNews, GitHub issues
│   ├── load.py                # low-level API helpers for all data sources
│   ├── train.py               # fine-tune DistilBERT on GoEmotions subset
│   ├── predict.py             # run inference on all datasets and save CSVs
│   ├── analyze_results.py     # compute per-label stats and overall accuracy
│   ├── visualize_results.py   # create figures used in the report/presentation
│   ├── main.py                # entry point that runs the full pipeline
│   └── tests.py               # minimal tests for data download helpers
├── data/                      # created locally; NOT tracked in git
├── results/                   # trained model, predictions, figures; NOT tracked
├── doc/
│   ├── Shipeng_Sun_progress_report.pdf
│   └── Shipeng_Sun_presentation.pdf
├── .env.example               # template for environment variables (no secrets)
├── requirements.txt
└── README.md

