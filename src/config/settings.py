"""
Configuration settings for the Article Person Verification system.
Handles environment variables and API configurations.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- API Configuration ---

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file. Please add it.")

# Gemini model configuration
GEMINI_MODEL_NAME = 'gemini-2.5-flash-preview-09-2025'

# --- MLflow Configuration ---

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
MLFLOW_EXPERIMENT_NAME = "Article Person Verification"

# --- Web Scraping Configuration ---

REQUEST_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}

REQUEST_TIMEOUT = 10  # seconds

# --- Default Files ---

DEFAULT_TEST_CASES_FILE = "test_cases.csv"
