"""
Evaluation script for Article Person Verification system.
Tests accuracy against ground truth labels in diverse_synthetic_articles.csv
"""

import os
import csv
import time
from datetime import datetime
import pandas as pd
from pathlib import Path

from src.config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME
from src.graph import build_graph
from src.utils import setup_logger

# Setup logger
logger = setup_logger(name="EvaluationScript", log_level="INFO")


def normalize_sentiment(sentiment) -> str:
    """Normalize sentiment labels for comparison."""
    # Handle NaN, None, or empty values
    if sentiment is None or (isinstance(sentiment, float) and pd.isna(sentiment)) or not sentiment:
        return "neutral"

    sentiment_str = str(sentiment)
    sentiment_lower = sentiment_str.lower().strip()

    # Map N/A to neutral for comparison
    if sentiment_lower == "n/a" or sentiment_lower == "nan":
        return "neutral"

    return sentiment_lower


def normalize_match_decision(decision: str, name_present: bool) -> str:
    """
    Normalize match decision to binary match/non-match.

    Args:
        decision: The match decision from the system
        name_present: Whether the name was found in the article

    Returns:
        "match" or "non-match"
    """
    if not decision:
        return "non-match"

    decision_lower = decision.lower().strip()

    # If name not present, it's always a non-match
    if not name_present:
        return "non-match"

    # "Match" is a match
    if decision_lower == "match":
        return "match"

    # Everything else (Non-Match, Review Required, Age Mismatch) is non-match
    return "non-match"


def run_evaluation():
    """Run evaluation on diverse synthetic articles dataset."""

    logger.info("=" * 80)
    logger.info("ARTICLE PERSON VERIFICATION - ACCURACY EVALUATION")
    logger.info("=" * 80)

    # Load ground truth dataset
    dataset_path = "diverse_synthetic_articles.csv"
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found: {dataset_path}")
        return

    df = pd.read_csv(dataset_path)
    logger.info(f"Loaded {len(df)} test cases from {dataset_path}")
    logger.info(f"Columns: {', '.join(df.columns.tolist())}")

    # Build graph
    logger.info("Building LangGraph workflow...")
    app = build_graph()
    logger.info("Graph compiled successfully")

    # Prepare results storage
    results = []
    match_correct = 0
    match_total = 0
    sentiment_correct = 0
    sentiment_total = 0

    llm_calls_saved = 0
    total_tokens = 0
    total_execution_time = 0

    logger.info(f"\nStarting evaluation on {len(df)} cases...")
    logger.info("=" * 80)

    for idx, row in df.iterrows():
        case_num = idx + 1
        logger.info(f"\n[{case_num}/{len(df)}] Processing: {row['person_name']}")
        logger.info(f"  Scenario: {row['scenario']} | Language: {row['language']}")
        logger.info(f"  Ground Truth: Match={row['is_match']}, Sentiment={row['sentiment_label']}")

        # Prepare input
        initial_state = {
            "applicant_name": row['person_name'],
            "applicant_dob": row['dob'],
            "article_url": row['article_text'],  # Using article_text directly
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

        # Execute verification
        start_time = time.time()
        try:
            final_state = None
            for chunk in app.stream(initial_state):
                if chunk:
                    node_name, node_output = list(chunk.items())[0]
                    initial_state.update(node_output)
            final_state = initial_state
            execution_time = time.time() - start_time

            # Extract predictions
            predicted_match_decision = normalize_match_decision(
                final_state.get('match_decision', ''),
                final_state.get('name_is_present', False)
            )
            predicted_sentiment = normalize_sentiment(final_state.get('sentiment', 'N/A'))

            # Ground truth
            ground_truth_match = "match" if row['is_match'] else "non-match"
            ground_truth_sentiment = normalize_sentiment(row['sentiment_label'])

            # Calculate accuracy
            match_is_correct = (predicted_match_decision == ground_truth_match)
            sentiment_is_correct = (predicted_sentiment == ground_truth_sentiment)

            if match_is_correct:
                match_correct += 1
            match_total += 1

            # Only count sentiment if it's a match case
            if ground_truth_match == "match":
                if sentiment_is_correct:
                    sentiment_correct += 1
                sentiment_total += 1

            # Token usage
            token_usage = final_state.get('token_usage', {})
            case_tokens = sum(usage.get('total_tokens', 0) for usage in token_usage.values())
            llm_calls = len(token_usage)

            # Check if LLM call was skipped
            name_check_skipped = 'check_name_presence' not in token_usage and not final_state.get('name_is_present', False)
            if name_check_skipped:
                llm_calls_saved += 1

            total_tokens += case_tokens
            total_execution_time += execution_time

            logger.info(f"  Prediction: Match={predicted_match_decision}, Sentiment={predicted_sentiment}")
            logger.info(f"  Correct: Match={match_is_correct}, Sentiment={sentiment_is_correct if ground_truth_match == 'match' else 'N/A'}")
            logger.info(f"  Tokens: {case_tokens} | LLM Calls: {llm_calls}/4 | Time: {execution_time:.2f}s")

            # Store result
            results.append({
                'case_number': case_num,
                'person_name': row['person_name'],
                'dob': row['dob'],
                'scenario': row['scenario'],
                'language': row['language'],
                'article_title': row['article_title'],
                'ground_truth_match': ground_truth_match,
                'predicted_match': predicted_match_decision,
                'match_correct': match_is_correct,
                'ground_truth_sentiment': ground_truth_sentiment,
                'predicted_sentiment': predicted_sentiment,
                'sentiment_correct': sentiment_is_correct if ground_truth_match == 'match' else None,
                'match_decision_raw': final_state.get('match_decision', ''),
                'match_explanation': final_state.get('match_explanation', ''),
                'sentiment_raw': final_state.get('sentiment', ''),
                'sentiment_explanation': final_state.get('sentiment_explanation', ''),
                'name_is_present': final_state.get('name_is_present', False),
                'age_matches': final_state.get('age_matches', False),
                'llm_calls_made': llm_calls,
                'name_check_skipped': name_check_skipped,
                'tokens_used': case_tokens,
                'execution_time_seconds': round(execution_time, 2),
                'status': 'success'
            })

        except Exception as e:
            logger.error(f"  ERROR: {e}", exc_info=True)
            execution_time = time.time() - start_time

            results.append({
                'case_number': case_num,
                'person_name': row['person_name'],
                'dob': row['dob'],
                'scenario': row['scenario'],
                'language': row['language'],
                'article_title': row['article_title'],
                'ground_truth_match': "match" if row['is_match'] else "non-match",
                'predicted_match': 'error',
                'match_correct': False,
                'ground_truth_sentiment': normalize_sentiment(row['sentiment_label']),
                'predicted_sentiment': 'error',
                'sentiment_correct': False,
                'match_decision_raw': '',
                'match_explanation': str(e),
                'sentiment_raw': '',
                'sentiment_explanation': '',
                'name_is_present': False,
                'age_matches': False,
                'llm_calls_made': 0,
                'name_check_skipped': False,
                'tokens_used': 0,
                'execution_time_seconds': round(execution_time, 2),
                'status': 'error'
            })

    # Calculate final metrics
    match_accuracy = (match_correct / match_total * 100) if match_total > 0 else 0
    sentiment_accuracy = (sentiment_correct / sentiment_total * 100) if sentiment_total > 0 else 0
    avg_tokens = total_tokens / len(df) if len(df) > 0 else 0
    avg_time = total_execution_time / len(df) if len(df) > 0 else 0

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total Cases: {len(df)}")
    logger.info(f"Successful: {len([r for r in results if r['status'] == 'success'])}")
    logger.info(f"Errors: {len([r for r in results if r['status'] == 'error'])}")
    logger.info("")
    logger.info("MATCH ACCURACY:")
    logger.info(f"  Correct: {match_correct}/{match_total}")
    logger.info(f"  Accuracy: {match_accuracy:.2f}%")
    logger.info("")
    logger.info("SENTIMENT ACCURACY (for match cases only):")
    logger.info(f"  Correct: {sentiment_correct}/{sentiment_total}")
    logger.info(f"  Accuracy: {sentiment_accuracy:.2f}%")
    logger.info("")
    logger.info("PERFORMANCE:")
    logger.info(f"  Total Tokens Used: {total_tokens:,}")
    logger.info(f"  Avg Tokens/Case: {avg_tokens:.0f}")
    logger.info(f"  LLM Calls Skipped: {llm_calls_saved}/{len(df)} ({llm_calls_saved/len(df)*100:.1f}%)")
    logger.info(f"  Total Execution Time: {total_execution_time:.2f}s")
    logger.info(f"  Avg Time/Case: {avg_time:.2f}s")
    logger.info("=" * 80)

    # Save results
    results_dir = Path("evaluation_results")
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save detailed results CSV
    results_csv_path = results_dir / f"detailed_results_{timestamp}.csv"
    df_results = pd.DataFrame(results)
    df_results.to_csv(results_csv_path, index=False)
    logger.info(f"\nDetailed results saved to: {results_csv_path}")

    # Save summary report
    summary_path = results_dir / f"summary_report_{timestamp}.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("ARTICLE PERSON VERIFICATION - ACCURACY EVALUATION REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: {dataset_path}\n")
        f.write(f"Total Cases: {len(df)}\n\n")

        f.write("ACCURACY METRICS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Match Accuracy: {match_accuracy:.2f}% ({match_correct}/{match_total})\n")
        f.write(f"Sentiment Accuracy: {sentiment_accuracy:.2f}% ({sentiment_correct}/{sentiment_total})\n\n")

        f.write("PERFORMANCE METRICS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total Tokens: {total_tokens:,}\n")
        f.write(f"Avg Tokens/Case: {avg_tokens:.0f}\n")
        f.write(f"LLM Calls Skipped: {llm_calls_saved}/{len(df)} ({llm_calls_saved/len(df)*100:.1f}%)\n")
        f.write(f"Total Time: {total_execution_time:.2f}s\n")
        f.write(f"Avg Time/Case: {avg_time:.2f}s\n\n")

        f.write("BREAKDOWN BY SCENARIO:\n")
        f.write("-" * 80 + "\n")
        scenario_stats = df_results.groupby('scenario').agg({
            'match_correct': 'sum',
            'case_number': 'count'
        }).rename(columns={'case_number': 'total'})
        scenario_stats['accuracy'] = scenario_stats['match_correct'] / scenario_stats['total'] * 100
        f.write(scenario_stats.to_string())
        f.write("\n\n")

        f.write("BREAKDOWN BY LANGUAGE:\n")
        f.write("-" * 80 + "\n")
        lang_stats = df_results.groupby('language').agg({
            'match_correct': 'sum',
            'case_number': 'count'
        }).rename(columns={'case_number': 'total'})
        lang_stats['accuracy'] = lang_stats['match_correct'] / lang_stats['total'] * 100
        f.write(lang_stats.to_string())
        f.write("\n")

    logger.info(f"Summary report saved to: {summary_path}")

    # Save errors separately if any
    errors = [r for r in results if r['status'] == 'error']
    if errors:
        errors_csv_path = results_dir / f"errors_{timestamp}.csv"
        pd.DataFrame(errors).to_csv(errors_csv_path, index=False)
        logger.info(f"Errors saved to: {errors_csv_path}")

    logger.info("\nEvaluation complete!")

    return {
        'match_accuracy': match_accuracy,
        'sentiment_accuracy': sentiment_accuracy,
        'total_cases': len(df),
        'match_correct': match_correct,
        'sentiment_correct': sentiment_correct,
        'results_path': str(results_csv_path),
        'summary_path': str(summary_path)
    }


if __name__ == "__main__":
    run_evaluation()
