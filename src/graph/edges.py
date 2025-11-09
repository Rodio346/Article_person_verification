"""
Edge routing logic for the LangGraph workflow.
Determines which node to execute next based on the current state.
"""

from .state import GraphState


def should_verify_age(state: GraphState) -> str:
    """
    Router 1: After checking for the name, decide whether to
    verify age or end the graph.

    Args:
        state: Current graph state

    Returns:
        Next node to execute: "verify_age" or "set_non_match"
    """
    print("--- Router 1: Checking Name Presence ---")
    if state["name_is_present"]:
        print("--- Decision: Name Found. Proceed to Verify Age. ---")
        return "verify_age"
    else:
        print("--- Decision: Name Not Found. End Graph (Non-Match). ---")
        return "set_non_match"


def should_verify_details(state: GraphState) -> str:
    """
    Router 2: After verifying age, decide whether to
    verify details or mark as age mismatch.

    Args:
        state: Current graph state

    Returns:
        Next node to execute: "verify_details" or "set_age_mismatch"
    """
    print("--- Router 2: Checking Age Match ---")
    if state["age_matches"]:
        print("--- Decision: Age Matches. Proceed to Verify Details. ---")
        return "verify_details"
    else:
        print("--- Decision: Age Mismatch. Mark for Further Verification. ---")
        return "set_age_mismatch"


def should_assess_sentiment(state: GraphState) -> str:
    """
    Router 3: After verifying details, decide whether to
    assess sentiment or end the graph.

    Args:
        state: Current graph state

    Returns:
        Next node to execute: "assess_sentiment" or "end"
    """
    print("--- Router 3: Checking Match Decision ---")
    decision = state["match_decision"]

    if decision == "Match" or decision == "Review Required":
        print("--- Decision: Proceed to Sentiment Analysis ---")
        return "assess_sentiment"
    else:  # "Non-Match"
        print("--- Decision: End Graph (Contradictory Details) ---")
        return "end"
