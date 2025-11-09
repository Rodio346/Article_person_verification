"""
Web scraping utilities for fetching and extracting article text.
"""

import requests
from bs4 import BeautifulSoup
from src.config import REQUEST_HEADERS, REQUEST_TIMEOUT


def fetch_article_text(url: str) -> str:
    """
    Fetches and extracts clean text from a URL.

    Args:
        url: The URL of the article to fetch

    Returns:
        Cleaned article text as a string
    """
    try:
        if "google.com/search" in url:
            print(f"Handling Google search URL: {url}")
            pass

        response = requests.get(url, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # Try to extract paragraph text first
        paragraphs = soup.find_all('p')
        article_text = "\n".join([p.get_text() for p in paragraphs])

        # Fallback to all text if no paragraphs found
        if not article_text.strip():
            article_text = soup.get_text()

        # Clean up the text
        return "\n".join([line.strip() for line in article_text.split('\n') if line.strip()])

    except requests.exceptions.RequestException as e:
        print(f"Error fetching article at {url}: {e}")
        return f"Error: Could not fetch article. {e}"
