# src/config.py
import os
from pathlib import Path

from dotenv import load_dotenv  # 记得在 requirements.txt 里有 python-dotenv

# Project root = goemox-emotion-classification/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load variables from .env in the project root
load_dotenv(PROJECT_ROOT / ".env")


def get_github_token() -> str | None:

    return os.getenv("GITHUB_TOKEN")
