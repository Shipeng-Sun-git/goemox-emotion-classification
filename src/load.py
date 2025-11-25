import os
import requests
import pandas as pd
import time
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

# Hugging Face /rows endpoint allows at most 100 rows per request
GOEMOTIONS_MAX_PAGE_SIZE = 100


def _fetch_goemotions_page(
    offset: int,
    length: int,
    split: str = "train",
    max_retries: int = 3,
) -> list[dict]:

    for attempt in range(max_retries):
        params = {
            "dataset": "google-research-datasets/go_emotions",
            "config": "simplified",
            "split": split,
            "offset": offset,
            "length": length,
        }

        response = requests.get(GOEMOTIONS_API_URL, params=params, timeout=30)

        # If we hit a rate limit, wait a bit and retry
        if response.status_code == 429:
            wait_seconds = 5 * (attempt + 1)
            print(
                f"Received 429 (Too Many Requests) for offset={offset}, "
                f"length={length}. Waiting {wait_seconds} seconds before retry..."
            )
            time.sleep(wait_seconds)
            continue

        # Any other error should be raised
        response.raise_for_status()
        data = response.json()
        rows = [row["row"] for row in data.get("rows", [])]
        return rows

    # If we are here, all retries failed due to 429 or other transient issues
    raise RuntimeError(
        f"Failed to fetch GoEmotions rows after {max_retries} retries "
        f"(offset={offset}, length={length})."
    )


def get_goemotions_from_api(
    limit: int = 2000,
    offset: int = 0,
    split: str = "train",
) -> pd.DataFrame:
    
    all_rows: list[dict] = []
    remaining = limit
    current_offset = offset

    while remaining > 0:
        batch_size = min(GOEMOTIONS_MAX_PAGE_SIZE, remaining)

        page_rows = _fetch_goemotions_page(
            offset=current_offset,
            length=batch_size,
            split=split,
        )

        if not page_rows:
            # No more data available from the API -> stop early
            break

        all_rows.extend(page_rows)
        fetched = len(page_rows)

        remaining -= fetched
        current_offset += fetched

        if fetched < batch_size:
            # Fewer rows than requested means we reached the end of the dataset
            break

        # Small delay between requests to be polite to the API
        time.sleep(0.5)

    df = pd.DataFrame(all_rows)
    return df


def save_goemotions_to_csv(
    filename: str = "goemotions_sample.csv",
    limit: int = 2000,
) -> str:
    
    df = get_goemotions_from_api(limit=limit)

    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, filename)
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} GoEmotions rows to {path}")
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
