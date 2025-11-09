"""
Article Person Verification System - Main Entry Point

This system verifies if news articles are about specific individuals and analyzes
the sentiment of those articles using AI and LangGraph.
"""

import os
import argparse
import uuid
import mlflow

from src.config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    DEFAULT_TEST_CASES_FILE
)
from src.utils import load_test_cases
from src.graph import build_graph


def run_verification(app, case: dict) -> None:
    """
    Run verification for a single test case with MLflow logging.

    Args:
        app: Compiled LangGraph workflow
        case: Dictionary containing 'name', 'dob', and 'url' keys
    """
    print(f"\n{'='*50}")
    print(f"RUNNING CASE FOR: {case['name']} ({case['dob']})")
    print(f"URL: {case['url']}")
    print(f"{'='*50}")

    # Start MLflow run
    run_name = f"Screening {case['name']} - {uuid.uuid4().hex[:8]}"
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        print(f"--- Starting MLflow Run: {run_id} ---")
        print(f"--- Run Name: {run_name} ---")
        mlflow.langchain.autolog()
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
            "sentiment_explanation": ""
        }

        state_history = []

        # Execute graph with MLflow logging
        try:
            step_counter = 0
            # Stream through the graph to capture intermediate outputs
            for chunk in app.stream(current_state):
                if not chunk:
                    continue

                # Extract node name and output
                node_name, node_output = list(chunk.items())[0]
                print(f"--- [Step {step_counter}] Node '{node_name}' executed ---")

                # Update state
                current_state.update(node_output)

                # Log state after this step
                log_name = f"step_{step_counter:02d}_{node_name}_state.json"
                mlflow.log_dict(current_state.copy(), log_name)

                state_history.append({
                    "step": step_counter,
                    "node": node_name,
                    "output": node_output,
                })
                step_counter += 1

            # Log final results
            final_state = current_state

            print("\n--- FINAL RESULT (logged to MLflow) ---")
            print(f"  Match Decision: {final_state.get('match_decision')}")
            print(f"  Explanation: {final_state.get('match_explanation')}")
            print(f"  Sentiment: {final_state.get('sentiment', 'N/A')}")
            print(f"  Explanation: {final_state.get('sentiment_explanation')}")

            # Log outputs as tags
            mlflow.set_tag("match_decision", final_state.get('match_decision', 'ERROR'))
            mlflow.set_tag("sentiment", final_state.get('sentiment', 'N/A'))

            # Log explanations as parameters
            mlflow.log_param("match_explanation", final_state.get('match_explanation', ''))
            mlflow.log_param("sentiment_explanation", final_state.get('sentiment_explanation', ''))

            # Log artifacts
            mlflow.log_text(final_state.get('article_text', ''), "article_text.txt")
            mlflow.log_dict(state_history, "run_state_history.json")
            mlflow.log_dict(final_state, "final_state.json")

            mlflow.set_tag("status", "COMPLETED")
            print(f"--- MLflow Run {run_id} Finished ---")

        except Exception as e:
            print(f"\n!!! Graph execution FAILED: {e} !!!")
            mlflow.set_tag("status", "FAILED")
            mlflow.log_param("error", str(e))
            if state_history:
                mlflow.log_dict(state_history, "partial_state_history.json")
            print(f"--- MLflow Run {run_id} Marked as FAILED ---")

        print(f"{'='*50}\n")


def main():
    """Main execution function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Article Person Verification Tool with MLflow"
    )
    parser.add_argument("--name", type=str, help="Applicant's full name")
    parser.add_argument("--dob", type=str, help="Applicant's date of birth (e.g., DD/MM/YYYY)")
    parser.add_argument("--article", type=str, help="URL of the news article to screen")
    parser.add_argument("--test_file", type=str, help="Path to a CSV test file")

    args = parser.parse_args()

    # Configure MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    print(f"MLflow tracking enabled. URI: {mlflow.get_tracking_uri()}")

    # Build and compile the graph
    app = build_graph()
    print("Graph compiled. ASCII diagram:")
    print(app.get_graph().draw_ascii())

    # Determine test cases to run
    test_cases_to_run = []
    if args.test_file:
        test_cases_to_run.extend(load_test_cases(args.test_file))
    elif args.name and args.dob and args.article:
        test_cases_to_run.append({
            "name": args.name,
            "dob": args.dob,
            "url": args.article
        })
    else:
        print(f"No inputs provided. Running with default '{DEFAULT_TEST_CASES_FILE}'.")
        if os.path.exists(DEFAULT_TEST_CASES_FILE):
            test_cases_to_run.extend(load_test_cases(DEFAULT_TEST_CASES_FILE))
        else:
            print(f"\nError: '{DEFAULT_TEST_CASES_FILE}' not found. Please create it or provide inputs.")
            exit(1)

    # Run verification for each test case
    for case in test_cases_to_run:
        if not all(key in case for key in ['name', 'dob', 'url']):
            print(f"Skipping invalid case: {case}")
            continue

        run_verification(app, case)


if __name__ == "__main__":
    main()
