"""
File loading utilities for test cases and CSV files.
"""

import csv
from typing import List, Dict


def load_test_cases(file_path: str) -> List[Dict[str, str]]:
    """
    Loads test cases from a CSV file.

    Args:
        file_path: Path to the CSV file containing test cases

    Returns:
        List of dictionaries, each representing a test case
    """
    try:
        with open(file_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            return list(reader)
    except FileNotFoundError:
        print(f"Error: Test file not found at {file_path}")
        return []
