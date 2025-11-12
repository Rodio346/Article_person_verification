"""
LangGraph nodes for the Article Person Verification workflow.
Each node represents a step in the verification process.
"""

import json
import time
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from src.config import (
    GOOGLE_API_KEY,
    GEMINI_MODEL_NAME,
    NAME_PRESENCE_PROMPT,
    AGE_VERIFICATION_PROMPT,
    DETAIL_VERIFICATION_PROMPT,
    SENTIMENT_ANALYSIS_PROMPT
)
from src.utils import fetch_article_text, quick_name_check, get_logger
from .state import GraphState

# Configure the Gemini API globally
genai.configure(api_key=GOOGLE_API_KEY)

# Get logger for nodes
logger = get_logger("ArticleVerification.Nodes")


def get_fresh_model():
    """
    Create a fresh model instance for each API call.
    This helps treat each call as independent and reduces rate limiting issues.

    Returns:
        A new GenerativeModel instance
    """
    return genai.GenerativeModel(GEMINI_MODEL_NAME)


def call_llm_with_retry(prompt: str, max_retries: int = 5, initial_delay: float = 3.0, inter_call_delay: float = 2.0) -> tuple:
    """
    Call the LLM with exponential backoff retry logic for rate limiting.
    Creates a fresh model instance for each call to simulate independent requests.

    Args:
        prompt: The prompt to send to the LLM
        max_retries: Maximum number of retry attempts (increased to 5)
        initial_delay: Initial delay in seconds before first retry (3s)
        inter_call_delay: Delay after each successful call (2s)

    Returns:
        Tuple of (response_text, usage_metadata dict)
        usage_metadata contains: prompt_tokens, completion_tokens, total_tokens

    Raises:
        Exception: If all retries are exhausted
    """
    delay = initial_delay

    for attempt in range(max_retries):
        try:
            # Create a fresh model instance for each call
            fresh_model = get_fresh_model()
            response = fresh_model.generate_content(prompt)

            # Extract token usage metadata
            usage_metadata = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }

            if hasattr(response, 'usage_metadata'):
                usage_metadata = {
                    "prompt_tokens": getattr(response.usage_metadata, 'prompt_token_count', 0),
                    "completion_tokens": getattr(response.usage_metadata, 'candidates_token_count', 0),
                    "total_tokens": getattr(response.usage_metadata, 'total_token_count', 0)
                }
                logger.debug(f"Tokens: {usage_metadata['prompt_tokens']} prompt + {usage_metadata['completion_tokens']} completion = {usage_metadata['total_tokens']} total")

            # Add delay after successful call to avoid rapid successive requests
            time.sleep(inter_call_delay)

            return response.text, usage_metadata
        except Exception as e:
            error_message = str(e)

            # Check if it's a rate limit error (429)
            if "429" in error_message or "quota" in error_message.lower() or "rate" in error_message.lower():
                if attempt < max_retries - 1:
                    logger.warning(f"Rate limit hit. Waiting {delay:.1f}s before retry {attempt + 1}/{max_retries - 1}")
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff: 3s -> 6s -> 12s -> 24s -> 48s
                else:
                    logger.error("Max retries exhausted. Rate limit error persists.")
                    logger.error("Consider increasing BATCH_PROCESSING_DELAY in settings.py")
                    raise Exception(f"Rate limit exceeded after {max_retries} attempts: {error_message}")
            else:
                # For non-rate-limit errors, raise immediately
                raise e

    raise Exception(f"Failed after {max_retries} attempts")


def fetch_article_node(state: GraphState) -> dict:
    """
    Node to fetch the article text from the URL.

    Args:
        state: Current graph state

    Returns:
        Updated state with article_text
    """
    try:
        url_display = state['article_url'][:100] if len(state['article_url']) > 100 else state['article_url']
        logger.info(f"Node: Fetching Article from {url_display}")
    except UnicodeEncodeError:
        logger.info("Node: Fetching Article [contains non-ASCII characters]")
    text = fetch_article_text(state['article_url'])
    return {"article_text": text}


def check_name_presence_node(state: GraphState) -> dict:
    """
    Node 1 (LLM Call 1): Checks if the applicant's name or variation is present.
    Uses a 3-tier approach:
    1. Exact match (regex) → Skip LLM, name is present
    2. Partial match → Call LLM to verify variations/nicknames
    3. No match → Skip LLM, name not present

    Args:
        state: Current graph state

    Returns:
        Updated state with name_is_present and name_check_explanation
    """
    logger.info("Node: Checking Name Presence")

    # Quick pre-filter: Check if name appears in article using keyword search
    confidence_level, name_found = quick_name_check(state['applicant_name'], state['article_text'])

    # Case 1: EXACT MATCH - Full name found directly in article
    if confidence_level == "exact":
        logger.info(f"✓ Quick name check: EXACT match found for '{state['applicant_name']}' (skipping LLM call)")
        return {
            "name_is_present": True,
            "name_check_explanation": f"Exact match: '{state['applicant_name']}' found directly in article text via regex. LLM call skipped for efficiency."
        }

    # Case 2: NO MATCH - Name not found at all
    if confidence_level == "none":
        logger.info(f"✗ Quick name check: NO match found for '{state['applicant_name']}' (skipping LLM call)")
        return {
            "name_is_present": False,
            "name_check_explanation": f"No match: '{state['applicant_name']}' or significant name parts not found in article. LLM call skipped for efficiency."
        }

    # Case 3: PARTIAL MATCH - Name parts found, need LLM to verify variations/nicknames
    logger.info(f"⚠ Quick name check: PARTIAL match found for '{state['applicant_name']}' (calling LLM to verify variations)")

    prompt = NAME_PRESENCE_PROMPT.format(
        applicant_name=state['applicant_name'],
        article_text=state['article_text']
    )

    try:
        response_text, usage_metadata = call_llm_with_retry(prompt)
        cleaned_response = response_text.strip().lstrip('```json').rstrip('```').strip()
        data = json.loads(cleaned_response)

        # Update token usage tracking
        token_usage = state.get('token_usage', {})
        token_usage['check_name_presence'] = usage_metadata

        return {
            "name_is_present": data.get("name_is_present", False),
            "name_check_explanation": data.get("explanation", "Error in parsing response."),
            "token_usage": token_usage
        }
    except Exception as e:
        logger.error(f"Error in check_name_presence_node: {e}", exc_info=True)
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
    logger.info("Node: Verifying Age")

    prompt = AGE_VERIFICATION_PROMPT.format(
        applicant_name=state['applicant_name'],
        applicant_dob=state['applicant_dob'],
        article_text=state['article_text']
    )

    try:
        response_text, usage_metadata = call_llm_with_retry(prompt)
        cleaned_response = response_text.strip().lstrip('```json').rstrip('```').strip()
        data = json.loads(cleaned_response)

        # Update token usage tracking
        token_usage = state.get('token_usage', {})
        token_usage['verify_age'] = usage_metadata

        return {
            "age_matches": data.get("age_matches", True),
            "age_check_explanation": data.get("explanation", "Error in parsing response."),
            "token_usage": token_usage
        }
    except Exception as e:
        logger.error(f"Error in verify_age_node: {e}", exc_info=True)
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
    logger.info("Node: Setting Age Mismatch (Needs Verification)")
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
    logger.info("Node: Verifying Details")

    prompt = DETAIL_VERIFICATION_PROMPT.format(
        applicant_name=state['applicant_name'],
        applicant_dob=state['applicant_dob'],
        article_text=state['article_text']
    )

    try:
        response_text, usage_metadata = call_llm_with_retry(prompt)
        cleaned_response = response_text.strip().lstrip('```json').rstrip('```').strip()
        data = json.loads(cleaned_response)

        # Update token usage tracking
        token_usage = state.get('token_usage', {})
        token_usage['verify_details'] = usage_metadata

        return {
            "match_decision": data.get("decision", "Review Required"),
            "match_explanation": data.get("explanation", "Error in parsing response."),
            "token_usage": token_usage
        }
    except Exception as e:
        logger.error(f"Error in verify_details_node: {e}", exc_info=True)
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
    logger.info("Node: Setting Non-Match (Name Not Found)")
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
    logger.info("Node: Assessing Sentiment")

    prompt = SENTIMENT_ANALYSIS_PROMPT.format(
        applicant_name=state['applicant_name'],
        article_text=state['article_text']
    )

    try:
        response_text, usage_metadata = call_llm_with_retry(prompt)
        cleaned_response = response_text.strip().lstrip('```json').rstrip('```').strip()
        data = json.loads(cleaned_response)

        # Update token usage tracking
        token_usage = state.get('token_usage', {})
        token_usage['assess_sentiment'] = usage_metadata

        return {
            "sentiment": data.get("sentiment", "Neutral"),
            "sentiment_explanation": data.get("explanation", "Error in parsing response."),
            "token_usage": token_usage
        }
    except Exception as e:
        logger.error(f"Error in assess_sentiment_node: {e}", exc_info=True)
        return {
            "sentiment": "Neutral",
            "sentiment_explanation": f"LLM call or JSON parsing failed: {e}. Defaulting to Neutral."
        }
