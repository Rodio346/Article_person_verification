"""
Web scraping utilities for fetching and extracting article text.
"""

import requests
from bs4 import BeautifulSoup
from src.config import REQUEST_HEADERS, REQUEST_TIMEOUT


def is_url(text: str) -> bool:
    """
    Checks if the input text is a URL.

    Args:
        text: The text to check

    Returns:
        True if text appears to be a URL, False otherwise
    """
    url_indicators = ['http://', 'https://', 'www.']
    return any(text.strip().lower().startswith(indicator) for indicator in url_indicators)


def fetch_article_text(url_or_text: str) -> str:
    """
    Fetches and extracts clean text from a URL, or returns the text directly if it's not a URL.

    Args:
        url_or_text: Either a URL of the article to fetch, or the article text itself

    Returns:
        Cleaned article text as a string
    """
    # Check if input is a URL or direct text
    if not is_url(url_or_text):
        print("--- Input detected as direct text (not a URL) ---")
        # Return the text as-is (it's already the article content)
        return url_or_text.strip()

    # If it's a URL, fetch the content
    print(f"--- Input detected as URL, fetching content ---")
    try:
        if "google.com/search" in url_or_text:
            print(f"Handling Google search URL: {url_or_text}")
            pass

        response = requests.get(url_or_text, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT)
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
        print(f"Error fetching article at {url_or_text}: {e}")
        return f"Error: Could not fetch article. {e}"
