"""
Web scraping utilities for fetching and extracting article text.
"""

import re
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


def normalize_for_matching(text: str) -> str:
    """
    Normalize text for better cross-lingual matching by removing accents
    and handling common transliteration variations.

    Args:
        text: The text to normalize

    Returns:
        Normalized text
    """
    import unicodedata

    # Decompose accented characters and remove accent marks
    # e.g., "José" → "Jose", "María" → "Maria"
    text = ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )

    return text.lower()


def quick_name_check(name: str, article_text: str) -> tuple[str, bool]:
    """
    Performs a quick keyword-based search to check if a name appears in the article.
    Returns a confidence level to determine if LLM call is needed.

    Handles:
    - Accented characters (José → Jose)
    - Different word orders (Zhang Wei vs Wei Zhang)
    - Partial name matches

    Args:
        name: The full name to search for (e.g., "John Smith")
        article_text: The article text to search in

    Returns:
        Tuple of (confidence_level, name_found):
        - ("exact", True): Full name found exactly → Skip LLM, name is present
        - ("partial", True): Name parts found → Call LLM to verify variations/nicknames
        - ("none", False): No match found → Skip LLM, name not present
    """
    if not name or not article_text:
        return ("none", False)

    # Normalize both for case-insensitive and accent-insensitive search
    article_normalized = normalize_for_matching(article_text)
    name_normalized = normalize_for_matching(name)

    # Strategy 1: Check if full name appears (EXACT MATCH)
    if name_normalized in article_normalized:
        return ("exact", True)

    # Strategy 2: Split name into parts and check each part
    # This handles cases where first and last names appear separately
    name_parts = [part.strip() for part in re.split(r'[\s\-,.]', name) if part.strip()]
    name_parts_normalized = [normalize_for_matching(part) for part in name_parts]

    # Filter out very short parts (like middle initials)
    significant_parts = [part for part in name_parts_normalized if len(part) >= 3]

    if not significant_parts:
        # If no significant parts, check all parts
        significant_parts = name_parts_normalized

    # Check if at least 2 name parts appear (or all parts if name has fewer than 2 parts)
    parts_found = sum(1 for part in significant_parts if part in article_normalized)

    # If name has multiple parts, require at least 2 to match
    # If name has only 1 part, require it to match
    min_required = min(2, len(significant_parts))

    if parts_found >= min_required:
        return ("partial", True)  # Partial match - need LLM to verify variations
    else:
        return ("none", False)  # No match found
