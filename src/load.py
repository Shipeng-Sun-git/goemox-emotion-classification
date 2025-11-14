import os
import requests
import pandas as pd
from typing import Optional

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# API endpoints
GOEMOTIONS_API_URL = "https://datasets-server.huggingface.co/rows"
HACKERNEWS_BASE_URL = "https://hacker-news.firebaseio.com/v0"
GITHUB_API_BASE_URL = "https://api.github.com"


# ----------------------------
# GoEmotions (main dataset)
# ----------------------------
def get_goemotions_from_api(
    limit: int = 1000,
    offset: int = 0,
    split: str = "train",
) -> pd.DataFrame:
    """
    Fetch a subset of the GoEmotions (simplified) dataset from the
    HuggingFace Datasets Server API and return it as a pandas DataFrame.
    """
    params = {
        "dataset": "google-research-datasets/go_emotions",
        "config": "simplified",
        "split": split,
        "offset": offset,
        "length": limit,
    }

    response = requests.get(GOEMOTIONS_API_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    rows = [row["row"] for row in data.get("rows", [])]
    df = pd.DataFrame(rows)
    return df


def save_goemotions_to_csv(
    filename: str = "goemotions_sample.csv",
    limit: int = 1000,
) -> str:
    """
    Fetch a subset of GoEmotions and save it to a CSV file inside data/.
    """
    df = get_goemotions_from_api(limit=limit)

    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, filename)
    df.to_csv(path, index=False)
    return path


# ----------------------------
# Hacker News (OOD text set 1)
# ----------------------------
def get_hackernews_from_api(limit: int = 200) -> pd.DataFrame:
    """
    Fetch a sample of Hacker News top stories and return them as a DataFrame.

    For each story we build a 'text' field using title + text (if available).
    """
    # 1. Get top story IDs
    top_url = f"{HACKERNEWS_BASE_URL}/topstories.json"
    resp = requests.get(top_url, timeout=30)
    resp.raise_for_status()
    ids = resp.json()

    items = []
    for story_id in ids[:limit]:
        item_url = f"{HACKERNEWS_BASE_URL}/item/{story_id}.json"
        item_resp = requests.get(item_url, timeout=30)
        item_resp.raise_for_status()
        item = item_resp.json()
        if not item:
            continue

        title = item.get("title") or ""
        text = item.get("text") or ""
        combined_text = (title + " " + text).strip()
        if not combined_text:
            continue

        items.append(
            {
                "id": item.get("id"),
                "text": combined_text,
                "type": item.get("type"),
                "by": item.get("by"),
                "time": item.get("time"),
                "url": item.get("url"),
            }
        )

    df = pd.DataFrame(items)
    return df


def save_hackernews_to_csv(
    filename: str = "hackernews_sample.csv",
    limit: int = 200,
) -> str:
    """
    Fetch a sample of Hacker News stories and save them to data/.
    """
    df = get_hackernews_from_api(limit=limit)

    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, filename)
    df.to_csv(path, index=False)
    return path


# ----------------------------
# GitHub issues (OOD text set 2)
# ----------------------------
def _get_github_headers() -> dict:
    """
    Build HTTP headers for GitHub API.

    If the environment variable GITHUB_TOKEN is set, it will be used
    for authenticated requests (better rate limits).
    """
    token: Optional[str] = os.environ.get("GITHUB_TOKEN")
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def get_github_issues_from_api(
    owner: str = "pallets",
    repo: str = "flask",
    state: str = "open",
    per_page: int = 50,
) -> pd.DataFrame:
    """
    Fetch a sample of GitHub issues for a given repository and return them
    as a DataFrame.

    Pull requests are filtered out (only pure issues are kept).
    """
    url = f"{GITHUB_API_BASE_URL}/repos/{owner}/{repo}/issues"
    params = {"state": state, "per_page": per_page}
    headers = _get_github_headers()

    resp = requests.get(url, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    issues = resp.json()

    records = []
    for issue in issues:
        # Filter out pull requests (they have 'pull_request' key)
        if "pull_request" in issue:
            continue

        text = (issue.get("title") or "") + "\n\n" + (issue.get("body") or "")
        text = text.strip()
        if not text:
            continue

        records.append(
            {
                "id": issue.get("id"),
                "number": issue.get("number"),
                "text": text,
                "state": issue.get("state"),
                "comments": issue.get("comments"),
                "user": issue.get("user", {}).get("login"),
                "html_url": issue.get("html_url"),
            }
        )

    df = pd.DataFrame(records)
    return df


def save_github_issues_to_csv(
    filename: str = "github_flask_issues_sample.csv",
    per_page: int = 50,
) -> str:
    """
    Fetch a sample of GitHub issues for the default repository and
    save them to data/.
    """
    df = get_github_issues_from_api(per_page=per_page)

    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, filename)
    df.to_csv(path, index=False)
    return path
