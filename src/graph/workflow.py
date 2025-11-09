"""
LangGraph workflow assembly and compilation.
"""

from langgraph.graph import StateGraph, END
from .state import GraphState
from .nodes import (
    fetch_article_node,
    check_name_presence_node,
    verify_age_node,
    set_age_mismatch_node,
    verify_details_node,
    set_name_non_match_node,
    assess_sentiment_node
)
from .edges import should_verify_age, should_verify_details, should_assess_sentiment


def build_graph():
    """
    Builds and compiles the LangGraph workflow.

    Workflow:
    1. Fetch article
    2. Check if name is present
       - If NO: End with Non-Match
       - If YES: Continue to age verification
    3. Verify age matches
       - If NO: End with Age Mismatch - Needs Verification
       - If YES: Continue to detail verification
    4. Verify details (DOB, etc.)
       - If Non-Match: End
       - If Match/Review Required: Continue to sentiment
    5. Assess sentiment

    Returns:
        Compiled workflow graph
    """
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("fetch_article", fetch_article_node)
    workflow.add_node("check_name_presence", check_name_presence_node)
    workflow.add_node("verify_age", verify_age_node)
    workflow.add_node("set_age_mismatch", set_age_mismatch_node)
    workflow.add_node("verify_details", verify_details_node)
    workflow.add_node("set_name_non_match", set_name_non_match_node)
    workflow.add_node("assess_sentiment", assess_sentiment_node)

    # Set entry point
    workflow.set_entry_point("fetch_article")

    # Add edges
    workflow.add_edge("fetch_article", "check_name_presence")

    # Router 1: After name check -> verify age or end
    workflow.add_conditional_edges(
        "check_name_presence",
        should_verify_age,
        {
            "verify_age": "verify_age",
            "set_non_match": "set_name_non_match"
        }
    )

    workflow.add_edge("set_name_non_match", END)

    # Router 2: After age check -> verify details or end with age mismatch
    workflow.add_conditional_edges(
        "verify_age",
        should_verify_details,
        {
            "verify_details": "verify_details",
            "set_age_mismatch": "set_age_mismatch"
        }
    )

    workflow.add_edge("set_age_mismatch", END)

    # Router 3: After detail verification -> assess sentiment or end
    workflow.add_conditional_edges(
        "verify_details",
        should_assess_sentiment,
        {
            "assess_sentiment": "assess_sentiment",
            "end": END
        }
    )

    workflow.add_edge("assess_sentiment", END)

    return workflow.compile()
