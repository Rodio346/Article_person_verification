"""
State definition for the LangGraph workflow.
"""

from typing import TypedDict, Literal


class GraphState(TypedDict):
    """Defines the state of our verification graph."""
    applicant_name: str
    applicant_dob: str
    article_url: str
    article_text: str
    name_is_present: bool
    name_check_explanation: str
    age_matches: bool
    age_check_explanation: str
    match_decision: Literal["Match", "Non-Match", "Review Required", "Age Mismatch - Needs Verification"]
    match_explanation: str
    sentiment: Literal["Positive", "Negative", "Neutral", "N/A"]
    sentiment_explanation: str
