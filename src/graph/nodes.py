"""
LangGraph nodes for the Article Person Verification workflow.
Each node represents a step in the verification process.
"""

import json
import google.generativeai as genai
from src.config import (
    GOOGLE_API_KEY,
    GEMINI_MODEL_NAME,
    NAME_PRESENCE_PROMPT,
    AGE_VERIFICATION_PROMPT,
    DETAIL_VERIFICATION_PROMPT,
    SENTIMENT_ANALYSIS_PROMPT
)
from src.utils import fetch_article_text
from .state import GraphState

# Configure the Gemini model
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel(GEMINI_MODEL_NAME)


def fetch_article_node(state: GraphState) -> dict:
    """
    Node to fetch the article text from the URL.

    Args:
        state: Current graph state

    Returns:
        Updated state with article_text
    """
    print(f"--- Node: Fetching Article from {state['article_url']} ---")
    text = fetch_article_text(state['article_url'])
    return {"article_text": text}


def check_name_presence_node(state: GraphState) -> dict:
    """
    Node 1 (LLM Call 1): Checks if the applicant's name or variation is present.

    Args:
        state: Current graph state

    Returns:
        Updated state with name_is_present and name_check_explanation
    """
    print("--- Node: Checking Name Presence ---")

    prompt = NAME_PRESENCE_PROMPT.format(
        applicant_name=state['applicant_name'],
        article_text=state['article_text']
    )

    try:
        response = model.generate_content(prompt)
        cleaned_response = response.text.strip().lstrip('```json').rstrip('```').strip()
        data = json.loads(cleaned_response)

        return {
            "name_is_present": data.get("name_is_present", False),
            "name_check_explanation": data.get("explanation", "Error in parsing response.")
        }
    except Exception as e:
        print(f"Error in check_name_presence_node: {e}")
        return {
            "name_is_present": False,
            "name_check_explanation": f"LLM call or JSON parsing failed: {e}. Defaulting to 'name not present'."
        }


def verify_age_node(state: GraphState) -> dict:
    """
    Node 2 (LLM Call 2): Verifies if the age/DOB mentioned in the article matches.

    Args:
        state: Current graph state

    Returns:
        Updated state with age_matches and age_check_explanation
    """
    print("--- Node: Verifying Age ---")

    prompt = AGE_VERIFICATION_PROMPT.format(
        applicant_name=state['applicant_name'],
        applicant_dob=state['applicant_dob'],
        article_text=state['article_text']
    )

    try:
        response = model.generate_content(prompt)
        cleaned_response = response.text.strip().lstrip('```json').rstrip('```').strip()
        data = json.loads(cleaned_response)

        return {
            "age_matches": data.get("age_matches", True),
            "age_check_explanation": data.get("explanation", "Error in parsing response.")
        }
    except Exception as e:
        print(f"Error in verify_age_node: {e}")
        return {
            "age_matches": True,  # Benefit of doubt - proceed to detailed verification
            "age_check_explanation": f"LLM call or JSON parsing failed: {e}. Defaulting to age matches."
        }


def set_age_mismatch_node(state: GraphState) -> dict:
    """
    Node: If age doesn't match, set decision to "Age Mismatch - Needs Verification".

    Args:
        state: Current graph state

    Returns:
        Updated state with match_decision and match_explanation
    """
    print("--- Node: Setting Age Mismatch (Needs Verification) ---")
    return {
        "match_decision": "Age Mismatch - Needs Verification",
        "match_explanation": state["age_check_explanation"]
    }


def verify_details_node(state: GraphState) -> dict:
    """
    Node 3 (LLM Call 3): Verifies details (DOB, etc.) after name and age were confirmed.

    Args:
        state: Current graph state

    Returns:
        Updated state with match_decision and match_explanation
    """
    print("--- Node: Verifying Details ---")

    prompt = DETAIL_VERIFICATION_PROMPT.format(
        applicant_name=state['applicant_name'],
        applicant_dob=state['applicant_dob'],
        article_text=state['article_text']
    )

    try:
        response = model.generate_content(prompt)
        cleaned_response = response.text.strip().lstrip('```json').rstrip('```').strip()
        data = json.loads(cleaned_response)

        return {
            "match_decision": data.get("decision", "Review Required"),
            "match_explanation": data.get("explanation", "Error in parsing response.")
        }
    except Exception as e:
        print(f"Error in verify_details_node: {e}")
        return {
            "match_decision": "Review Required",
            "match_explanation": f"LLM call or JSON parsing failed: {e}. Manual review needed."
        }


def set_name_non_match_node(state: GraphState) -> dict:
    """
    Terminator Node: If name is not found, set final state to Non-Match and end.

    Args:
        state: Current graph state

    Returns:
        Updated state with match_decision and match_explanation
    """
    print("--- Node: Setting Non-Match (Name Not Found) ---")
    return {
        "match_decision": "Non-Match",
        "match_explanation": state["name_check_explanation"]
    }


def assess_sentiment_node(state: GraphState) -> dict:
    """
    Node 4 (LLM Call 4): Assesses the article's sentiment about the applicant.

    Args:
        state: Current graph state

    Returns:
        Updated state with sentiment and sentiment_explanation
    """
    print("--- Node: Assessing Sentiment ---")

    prompt = SENTIMENT_ANALYSIS_PROMPT.format(
        applicant_name=state['applicant_name'],
        article_text=state['article_text']
    )

    try:
        response = model.generate_content(prompt)
        cleaned_response = response.text.strip().lstrip('```json').rstrip('```').strip()
        data = json.loads(cleaned_response)

        return {
            "sentiment": data.get("sentiment", "Neutral"),
            "sentiment_explanation": data.get("explanation", "Error in parsing response.")
        }
    except Exception as e:
        print(f"Error in assess_sentiment_node: {e}")
        return {
            "sentiment": "Neutral",
            "sentiment_explanation": f"LLM call or JSON parsing failed: {e}. Defaulting to Neutral."
        }
