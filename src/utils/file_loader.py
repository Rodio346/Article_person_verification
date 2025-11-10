"""
File loading utilities for test cases and CSV files.
"""

import csv
from typing import List, Dict


def load_test_cases(file_path: str) -> List[Dict[str, str]]:
    """
    Loads test cases from a CSV file.

    CSV can have either:
    - name, dob, url (where url is a URL to fetch)
    - name, dob, text (where text is direct article content)
    - name, dob, url, text (text takes precedence if both present)

    Args:
        file_path: Path to the CSV file containing test cases

    Returns:
        List of dictionaries, each representing a test case
    """
    try:
        with open(file_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            cases = []
            for row in reader:
                # Normalize: if 'text' column exists and has content, use it as 'url'
                # This allows the rest of the system to work without changes
                if 'text' in row and row['text'].strip():
                    row['url'] = row['text']
                cases.append(row)
            return cases
    except FileNotFoundError:
        print(f"Error: Test file not found at {file_path}")
        return []
