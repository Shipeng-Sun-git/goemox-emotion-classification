import os

from load import (
    get_goemotions_from_api,
    save_goemotions_to_csv,
    get_hackernews_from_api,
    save_hackernews_to_csv,
    get_github_issues_from_api,
    save_github_issues_to_csv,
)


def test_get_goemotions_from_api():
    """
    Minimal test: make sure we can fetch some rows from the GoEmotions API.
    """
    df = get_goemotions_from_api(limit=20)
    assert len(df) > 0, "No rows returned from GoEmotions API"
    assert "text" in df.columns, "GoEmotions data should contain a 'text' column"


def test_save_goemotions_to_csv():
    """
    Minimal test: make sure the helper function creates a CSV file in data/.
    """
    filename = "goemotions_sample_test.csv"
    path = save_goemotions_to_csv(filename=filename, limit=10)
    assert os.path.exists(path), f"CSV file was not created at {path}"


def test_get_hackernews_from_api():
    """
    Minimal test: make sure we can fetch some Hacker News stories.
    """
    df = get_hackernews_from_api(limit=30)
    assert len(df) > 0, "No rows returned from Hacker News API"
    assert "text" in df.columns, "Hacker News data should contain a 'text' column"


def test_save_hackernews_to_csv():
    """
    Minimal test: make sure Hacker News data is saved to data/.
    """
    filename = "hackernews_sample_test.csv"
    path = save_hackernews_to_csv(filename=filename, limit=30)
    assert os.path.exists(path), f"CSV file was not created at {path}"


def test_get_github_issues_from_api():
    """
    Minimal test: make sure we can fetch some GitHub issues.
    """
    df = get_github_issues_from_api(per_page=20)
    assert len(df) > 0, "No rows returned from GitHub API"
    assert "text" in df.columns, "GitHub issues data should contain a 'text' column"


def test_save_github_issues_to_csv():
    """
    Minimal test: make sure GitHub issues data is saved to data/.
    """
    filename = "github_issues_sample_test.csv"
    path = save_github_issues_to_csv(filename=filename, per_page=20)
    assert os.path.exists(path), f"CSV file was not created at {path}"


if __name__ == "__main__":
    # Allow running the tests by: python test.py
    test_get_goemotions_from_api()
    test_save_goemotions_to_csv()
    test_get_hackernews_from_api()
    test_save_hackernews_to_csv()
    test_get_github_issues_from_api()
    test_save_github_issues_to_csv()
    print("All tests passed.")
