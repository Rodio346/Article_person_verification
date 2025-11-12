"""
Article Person Verification System - Main Entry Point

This system verifies if news articles are about specific individuals and analyzes
the sentiment of those articles using AI and LangGraph.
"""

import os
import argparse
import uuid
import time
from datetime import datetime
import mlflow

from src.config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    DEFAULT_TEST_CASES_FILE,
    GEMINI_MODEL_NAME,
    )
from src.utils import load_test_cases, setup_logger
from src.graph import build_graph
mlflow.langchain.autolog()

# Setup logger
logger = setup_logger(name="ArticleVerification", log_level="Debug")

def run_verification(app, case: dict) -> None:
    """
    Run verification for a single test case with comprehensive MLflow logging.

    Args:
        app: Compiled LangGraph workflow
        case: Dictionary containing 'name', 'dob', and 'url' keys
    """
    logger.info("=" * 50)
    try:
        logger.info(f"RUNNING CASE FOR: {case['name']} ({case['dob']})")
        url_display = case['url'][:100] + "..." if len(case['url']) > 100 else case['url']
        logger.info(f"URL: {url_display}")
    except UnicodeEncodeError:
        # Fallback for Windows console encoding issues
        logger.info(f"RUNNING CASE FOR: {case['name']} ({case['dob']})")
        logger.info("URL: [Content contains non-ASCII characters]")
    logger.info("=" * 50)

    # Start MLflow run
    run_name = f"Screening {case['name']} - {uuid.uuid4().hex[:8]}"
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        start_time = time.time()

        logger.info(f"Starting MLflow Run: {run_id}")
        logger.info(f"Run Name: {run_name}")

        # Enable autologging for LangChain/LangGraph
        mlflow.langchain.autolog()

        # Log system metadata as tags
        mlflow.set_tag("mlflow.runName", run_name)
        mlflow.set_tag("execution_date", datetime.now().strftime("%Y-%m-%d"))
        mlflow.set_tag("execution_time", datetime.now().strftime("%H:%M:%S"))
        mlflow.set_tag("model", GEMINI_MODEL_NAME)

        # Log initial inputs as parameters
        mlflow.log_param("applicant_name", case['name'])
        mlflow.log_param("applicant_dob", case['dob'])
        mlflow.log_param("article_url", case['url'])

        # Initialize state
        current_state = {
            "applicant_name": case['name'],
            "applicant_dob": case['dob'],
            "article_url": case['url'],
            "article_text": "",
            "name_is_present": False,
            "name_check_explanation": "",
            "age_matches": False,
            "age_check_explanation": "",
            "match_decision": "Review Required",
            "match_explanation": "",
            "sentiment": "N/A",
            "sentiment_explanation": "",
            "token_usage": {}
        }

        state_history = []
        node_execution_times = {}

        # Execute graph with MLflow logging
        try:
            step_counter = 0
            total_nodes_executed = 0

            # Stream through the graph to capture intermediate outputs
            for chunk in app.stream(current_state):
                if not chunk:
                    continue

                node_start_time = time.time()

                # Extract node name and output
                node_name, node_output = list(chunk.items())[0]
                logger.debug(f"[Step {step_counter}] Node '{node_name}' executed")

                # Update state
                current_state.update(node_output)

                # Track node execution time
                node_execution_time = time.time() - node_start_time
                node_execution_times[node_name] = node_execution_time

                # Log node execution time as metric
                mlflow.log_metric(f"node_{node_name}_duration_ms", node_execution_time * 1000, step=step_counter)

                # Log state after this step
                log_name = f"step_{step_counter:02d}_{node_name}_state.json"
                mlflow.log_dict(current_state.copy(), log_name)

                state_history.append({
                    "step": step_counter,
                    "node": node_name,
                    "output": node_output,
                    "execution_time_ms": round(node_execution_time * 1000, 2)
                })
                step_counter += 1
                total_nodes_executed += 1

            # Calculate total execution time
            total_execution_time = time.time() - start_time

            # Log final results
            final_state = current_state

            logger.info("FINAL RESULT (logged to MLflow)")
            try:
                logger.info(f"  Match Decision: {final_state.get('match_decision')}")
                logger.info(f"  Explanation: {final_state.get('match_explanation')}")
                logger.info(f"  Sentiment: {final_state.get('sentiment', 'N/A')}")
                logger.info(f"  Explanation: {final_state.get('sentiment_explanation')}")
            except UnicodeEncodeError:
                logger.info(f"  Match Decision: {final_state.get('match_decision')}")
                logger.info("  Explanation: [Contains non-ASCII characters - see MLflow artifacts]")
                logger.info(f"  Sentiment: {final_state.get('sentiment', 'N/A')}")
                logger.info("  Explanation: [Contains non-ASCII characters - see MLflow artifacts]")

            # Log key outcomes as tags (for easy filtering/grouping in MLflow UI)
            mlflow.set_tag("match_decision", final_state.get('match_decision', 'ERROR'))
            mlflow.set_tag("sentiment", final_state.get('sentiment', 'N/A'))
            mlflow.set_tag("name_is_present", str(final_state.get('name_is_present', False)))
            mlflow.set_tag("age_matches", str(final_state.get('age_matches', False)))
            mlflow.set_tag("status", "COMPLETED")

            # Log boolean flags as metrics (for dashboard aggregation and analysis)
            mlflow.log_metric("name_found", 1 if final_state.get('name_is_present') else 0)
            mlflow.log_metric("age_verified", 1 if final_state.get('age_matches') else 0)
            mlflow.log_metric("is_match", 1 if final_state.get('match_decision') == "Match" else 0)
            mlflow.log_metric("is_non_match", 1 if final_state.get('match_decision') == "Non-Match" else 0)
            mlflow.log_metric("needs_review", 1 if final_state.get('match_decision') == "Review Required" else 0)
            mlflow.log_metric("age_mismatch", 1 if final_state.get('match_decision') == "Age Mismatch - Needs Verification" else 0)

            # Log sentiment metrics (for tracking sentiment distribution)
            sentiment_value = final_state.get('sentiment', 'N/A')
            mlflow.log_metric("sentiment_negative", 1 if sentiment_value == "Negative" else 0)
            mlflow.log_metric("sentiment_positive", 1 if sentiment_value == "Positive" else 0)
            mlflow.log_metric("sentiment_neutral", 1 if sentiment_value == "Neutral" else 0)

            # Log execution performance metrics
            mlflow.log_metric("total_execution_time_seconds", total_execution_time)
            mlflow.log_metric("total_nodes_executed", total_nodes_executed)
            mlflow.log_metric("article_text_length", len(final_state.get('article_text', '')))

            # Log token usage metrics
            token_usage = final_state.get('token_usage', {})
            total_prompt_tokens = 0
            total_completion_tokens = 0
            total_tokens = 0

            for node_name, usage in token_usage.items():
                # Log per-node token metrics
                mlflow.log_metric(f"tokens_{node_name}_prompt", usage.get('prompt_tokens', 0))
                mlflow.log_metric(f"tokens_{node_name}_completion", usage.get('completion_tokens', 0))
                mlflow.log_metric(f"tokens_{node_name}_total", usage.get('total_tokens', 0))

                # Accumulate totals
                total_prompt_tokens += usage.get('prompt_tokens', 0)
                total_completion_tokens += usage.get('completion_tokens', 0)
                total_tokens += usage.get('total_tokens', 0)

            # Log total token usage across all LLM calls
            mlflow.log_metric("tokens_total_prompt", total_prompt_tokens)
            mlflow.log_metric("tokens_total_completion", total_completion_tokens)
            mlflow.log_metric("tokens_total_all", total_tokens)

            # Track if LLM call was skipped for name check (cost optimization)
            llm_calls_made = len(token_usage)
            name_check_skipped = 'check_name_presence' not in token_usage and not final_state.get('name_is_present', False)
            mlflow.log_metric("llm_calls_made", llm_calls_made)
            mlflow.log_metric("name_check_llm_skipped", 1 if name_check_skipped else 0)

            if name_check_skipped:
                logger.info("Quick name check: LLM call SKIPPED (name not found via keyword search)")
                logger.info(f"LLM Calls Made: {llm_calls_made}/4 (saved 1 call)")
            else:
                logger.info(f"LLM Calls Made: {llm_calls_made}")

            logger.info(f"Total Token Usage: {total_prompt_tokens} prompt + {total_completion_tokens} completion = {total_tokens} total")

            # Log explanations as parameters (truncate to avoid size limits)
            mlflow.log_param("name_check_explanation", final_state.get('name_check_explanation', '')[:500])
            mlflow.log_param("age_check_explanation", final_state.get('age_check_explanation', '')[:500])
            mlflow.log_param("match_explanation", final_state.get('match_explanation', '')[:500])
            mlflow.log_param("sentiment_explanation", final_state.get('sentiment_explanation', '')[:500])

            # Log artifacts
            mlflow.log_text(final_state.get('article_text', ''), "article_text.txt")
            mlflow.log_dict(state_history, "run_state_history.json")
            mlflow.log_dict(final_state, "final_state.json")
            mlflow.log_dict(node_execution_times, "node_execution_times.json")
            mlflow.log_dict(token_usage, "token_usage.json")

            # Create and log execution summary
            execution_summary = {
                "run_id": run_id,
                "run_name": run_name,
                "applicant_name": case['name'],
                "applicant_dob": case['dob'],
                "article_url": case['url'],
                "match_decision": final_state.get('match_decision'),
                "sentiment": final_state.get('sentiment'),
                "name_is_present": final_state.get('name_is_present'),
                "age_matches": final_state.get('age_matches'),
                "total_execution_time_seconds": round(total_execution_time, 2),
                "total_nodes_executed": total_nodes_executed,
                "total_tokens_used": total_tokens,
                "total_prompt_tokens": total_prompt_tokens,
                "total_completion_tokens": total_completion_tokens,
                "execution_timestamp": datetime.now().isoformat()
            }
            mlflow.log_dict(execution_summary, "execution_summary.json")

            logger.info(f"Total Execution Time: {total_execution_time:.2f}s")
            logger.info(f"MLflow Run {run_id} Finished")
            logger.info("=" * 50)

        except Exception as e:
            total_execution_time = time.time() - start_time

            logger.error(f"Graph execution FAILED: {e}", exc_info=True)

            # Log error details
            mlflow.set_tag("status", "FAILED")
            mlflow.set_tag("error_type", type(e).__name__)
            mlflow.log_param("error_message", str(e)[:500])

            # Log execution metrics even on failure
            mlflow.log_metric("total_execution_time_seconds", total_execution_time)
            mlflow.log_metric("total_nodes_executed", step_counter)
            mlflow.log_metric("execution_failed", 1)

            # Log partial results
            if state_history:
                mlflow.log_dict(state_history, "partial_state_history.json")
            if current_state:
                mlflow.log_dict(current_state, "partial_final_state.json")

            # Create error summary
            error_summary = {
                "run_id": run_id,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "failed_at_step": step_counter,
                "execution_time_before_failure": round(total_execution_time, 2),
                "timestamp": datetime.now().isoformat()
            }
            mlflow.log_dict(error_summary, "error_summary.json")

            logger.error(f"MLflow Run {run_id} Marked as FAILED")


def main():
    """Main execution function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Article Person Verification Tool with MLflow"
    )
    parser.add_argument("--name", type=str, help="Applicant's full name")
    parser.add_argument("--dob", type=str, help="Applicant's date of birth (e.g., DD/MM/YYYY)")
    parser.add_argument("--article", type=str, help="URL of the news article to screen OR direct article text")
    parser.add_argument("--text", type=str, help="Direct article text (alternative to --article)")
    parser.add_argument("--test_file", type=str, help="Path to a CSV test file")

    args = parser.parse_args()

    # Configure MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    logger.info(f"MLflow tracking enabled. URI: {mlflow.get_tracking_uri()}")
    logger.info(f"Experiment: {MLFLOW_EXPERIMENT_NAME}")

    # Build and compile the graph
    app = build_graph()
    logger.info("Graph compiled successfully")
    logger.debug("Graph ASCII diagram:")
    logger.debug(f"\n{app.get_graph().draw_ascii()}")

    # Save graph as image
    try:
        from pathlib import Path

        # Create graphs directory if it doesn't exist
        graphs_dir = Path("graphs")
        graphs_dir.mkdir(exist_ok=True)

        # Save as PNG
        graph_image_path = graphs_dir / "workflow_graph.png"
        graph = app.get_graph()
        graph_image = graph.draw_mermaid_png()

        with open(graph_image_path, 'wb') as f:
            f.write(graph_image)

        logger.info(f"Graph visualization saved to: {graph_image_path}")
    except Exception as e:
        logger.warning(f"Could not save graph image: {e}. This requires Mermaid CLI or graphviz to be installed.")

    # Determine test cases to run
    test_cases_to_run = []
    if args.test_file:
        test_cases_to_run.extend(load_test_cases(args.test_file))
    elif args.name and args.dob and (args.article or args.text):
        # Use --text if provided, otherwise use --article (which can be URL or text)
        article_input = args.text if args.text else args.article
        test_cases_to_run.append({
            "name": args.name,
            "dob": args.dob,
            "url": article_input  # Can be URL or direct text
        })
    else:
        logger.info(f"No inputs provided. Running with default '{DEFAULT_TEST_CASES_FILE}'")
        if os.path.exists(DEFAULT_TEST_CASES_FILE):
            test_cases_to_run.extend(load_test_cases(DEFAULT_TEST_CASES_FILE))
        else:
            logger.error(f"'{DEFAULT_TEST_CASES_FILE}' not found. Please create it or provide inputs.")
            exit(1)

    logger.info(f"Total test cases to process: {len(test_cases_to_run)}")

    # Run verification for each test case
    successful_runs = 0
    failed_runs = 0

    for idx, case in enumerate(test_cases_to_run, 1):
        if not all(key in case for key in ['name', 'dob', 'url']):
            logger.warning(f"Skipping invalid case: {case}")
            failed_runs += 1
            continue

        logger.info(f"[{idx}/{len(test_cases_to_run)}] Processing case...")
        try:
            run_verification(app, case)
            successful_runs += 1

            # Add delay between cases (except for last case) to ensure independent API sessions
            if idx < len(test_cases_to_run):
                logger.debug(f"Waiting {BATCH_PROCESSING_DELAY}s before next case (rate limit protection)")
                time.sleep(BATCH_PROCESSING_DELAY)

        except Exception as e:
            logger.error(f"Failed to process case: {e}", exc_info=True)
            failed_runs += 1

            # Add delay even on failure to prevent cascading rate limit errors
            if idx < len(test_cases_to_run):
                logger.debug(f"Waiting {BATCH_PROCESSING_DELAY}s before next case (rate limit protection)")
                time.sleep(BATCH_PROCESSING_DELAY)

    # Print summary
    logger.info("=" * 50)
    logger.info("BATCH EXECUTION SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Total cases: {len(test_cases_to_run)}")
    logger.info(f"Successful: {successful_runs}")
    logger.info(f"Failed: {failed_runs}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
