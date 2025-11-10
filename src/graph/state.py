"""
State definition for the LangGraph workflow.
"""

from typing import TypedDict, Literal, Dict


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
    # Token usage tracking
    token_usage: Dict[str, Dict[str, int]]  # {"node_name": {"prompt_tokens": X, "completion_tokens": Y, "total_tokens": Z}}
