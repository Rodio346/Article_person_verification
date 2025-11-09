import os
import csv
import json
import argparse
import google.generativeai as genai
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
import mlflow
import uuid # Used for potential run naming if needed

# --- 1. Configuration and Setup ---

# Load environment variables (for API key and MLflow)
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns") # Default to local mlruns folder

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file. Please add it.")

# Configure the Gemini model
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash-preview-09-2025') 

# --- 2. Helper Functions ---

def fetch_article_text(url: str) -> str:
    """Fetches and extracts clean text from a URL."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }
        if "google.com/search" in url:
            print(f"Handling Google search URL: {url}")
            pass 

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status() 

        soup = BeautifulSoup(response.text, 'html.parser')
        
        paragraphs = soup.find_all('p')
        article_text = "\n".join([p.get_text() for p in paragraphs])
        
        if not article_text.strip():
            article_text = soup.get_text()

        return "\n".join([line.strip() for line in article_text.split('\n') if line.strip()])

    except requests.exceptions.RequestException as e:
        print(f"Error fetching article at {url}: {e}")
        return f"Error: Could not fetch article. {e}"

def load_test_cases(file_path: str) -> list[dict]:
    """Loads test cases from a CSV file."""
    try:
        with open(file_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            return list(reader)
    except FileNotFoundError:
        print(f"Error: Test file not found at {file_path}")
        return []

# --- 3. LangGraph State Definition ---

class GraphState(TypedDict):
    """Defines the state of our graph."""
    applicant_name: str
    applicant_dob: str
    article_url: str
    article_text: str
    name_is_present: bool 
    name_check_explanation: str
    match_decision: Literal["Match", "Non-Match", "Review Required"]
    match_explanation: str
    sentiment: Literal["Positive", "Negative", "Neutral", "N/A"]
    sentiment_explanation: str

# --- 4. LangGraph Nodes ---
# (Nodes are unchanged, as logging will be handled in the main loop)

def fetch_article_node(state: GraphState) -> dict:
    """Node to fetch the article text from the URL."""
    print(f"--- Node: Fetching Article from {state['article_url']} ---")
    text = fetch_article_text(state['article_url'])
    return {"article_text": text}

def check_name_presence_node(state: GraphState) -> dict:
    """
    Node 1 (LLM Call 1): Checks *only* if the name or a variation is present.
    """
    print("--- Node: Checking Name Presence ---")
    
    system_prompt = """
    You are an analyst. Your task is to quickly scan an article to see if a specific name is mentioned.
    
    Applicant Name: {applicant_name}
    
    Article Text:
    {article_text}
    
    Is the Applicant Name, or a very clear variation (e.g., "Bill" for "William", "Bernie" for "Bernard"), 
    mentioned in the Article Text?
    
    Respond in a single, valid JSON object with two keys:
    - "name_is_present": true or false
    - "explanation": A brief reason. (e.g., "Name 'Bernie Madoff' found." or "Name 'William Clinton' not found, but 'Bill Clinton' was.")
    """
    
    prompt = system_prompt.format(
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

def verify_details_node(state: GraphState) -> dict:
    """
    Node 2 (LLM Call 2): Verifies details (DOB, etc.) *after* a name was found.
    """
    print("--- Node: Verifying Details ---")
    
    system_prompt = """
    You are a meticulous financial analyst. We have already confirmed that the name "{applicant_name}" (or a variation) 
    is in the article.
    
    Your task is to determine if it's the *correct person* by verifying their Date of Birth.
    You must be extremely careful to avoid "false negatives".
    
    Applicant Name: {applicant_name}
    Applicant DOB: {applicant_dob}
    
    Article Text:
    {article_text}
    
    Analyze the text and provide your decision in a single, valid JSON object.
    1. Look for the applicant's name again.
    2. Look for any dates of birth, ages, or other strong identifiers (locations, occupations) *associated with that name*.
    3. Compare these details to the Applicant's DOB.
    
    Your JSON response MUST have exactly two keys: "decision" and "explanation".
    
    - "decision" must be one of:
      - "Match": If you are confident it's the same person (e.g., name and DOB match, or name and other strong identifiers match).
      - "Non-Match": ONLY if you find *explicit contradictory evidence* (e.g., the article mentions the same name but a *different* DOB or age).
      - "Review Required": If the name matches but no *other* identifying or contradictory details are present. **This is your default if uncertain.**
    
    - "explanation": A step-by-step justification for your decision.
    """
    
    prompt = system_prompt.format(
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
    """
    print("--- Node: Setting Non-Match (Name Not Found) ---")
    return {
        "match_decision": "Non-Match",
        "match_explanation": state["name_check_explanation"] # Use explanation from the name check
    }

def assess_sentiment_node(state: GraphState) -> dict:
    """Node 3 (LLM Call 3): Assesses the article's sentiment *about the applicant*."""
    print("--- Node: Assessing Sentiment ---")

    system_prompt = """
    You are an analyst reviewing an article about a specific person.
    The article has already been determined to be about: {applicant_name}.
    
    Article Text:
    {article_text}
    
    Analyze the article and determine if it portrays this person in a positive, negative, or neutral light, 
    specifically in a regulated financial context.
    "Negative" includes: lawsuits, scandals, fraud, bankruptcies, criminal activity.
    "Positive" includes: philanthropy, achievements, industry awards.
    "Neutral" includes: simple news reports, job changes, or objective statements.
    
    Provide your response in a single, valid JSON object with two keys: "sentiment" and "explanation".
    
    - "sentiment" must be one of: "Positive", "Negative", "Neutral".
    - "explanation": A brief justification for your sentiment.
    """
    
    prompt = system_prompt.format(
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

# --- 5. Graph Edges (Routing) ---
# (Routing logic is unchanged)

def should_verify_details(state: GraphState) -> str:
    """
    Router 1: After checking for the name, decide whether to
    verify details or end the graph.
    """
    print("--- Router 1: Checking Name Presence ---")
    if state["name_is_present"]:
        print("--- Decision: Name Found. Proceed to Verify Details. ---")
        return "verify_details"
    else:
        print("--- Decision: Name Not Found. End Graph (Non-Match). ---")
        return "set_non_match" 

def should_assess_sentiment(state: GraphState) -> str:
    """
    Router 2: After verifying details, decide whether to
    assess sentiment or end the graph.
    """
    print("--- Router 2: Checking Match Decision ---")
    decision = state["match_decision"]
    
    if decision == "Match" or decision == "Review Required":
        print("--- Decision: Proceed to Sentiment Analysis ---")
        return "assess_sentiment"
    else: # "Non-Match"
        print("--- Decision: End Graph (Contradictory Details) ---")
        return "end"

# --- 6. Graph Assembly ---
# (Graph build logic is unchanged)

def build_graph():
    """Builds and compiles the LangGraph."""
    workflow = StateGraph(GraphState)
    
    workflow.add_node("fetch_article", fetch_article_node)
    workflow.add_node("check_name_presence", check_name_presence_node)
    workflow.add_node("verify_details", verify_details_node)
    workflow.add_node("set_name_non_match", set_name_non_match_node)
    workflow.add_node("assess_sentiment", assess_sentiment_node)
    
    workflow.set_entry_point("fetch_article")
    workflow.add_edge("fetch_article", "check_name_presence")
    
    workflow.add_conditional_edges(
        "check_name_presence",
        should_verify_details,
        {
            "verify_details": "verify_details",
            "set_non_match": "set_name_non_match"
        }
    )
    
    workflow.add_edge("set_name_non_match", END)
    
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

# --- 7. Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adverse Media Screening Tool with MLflow")
    parser.add_argument("--name", type=str, help="Applicant's full name")
    parser.add_argument("--dob", type=str, help="Applicant's date of birth (e.g., DD/MM/YYYY)")
    parser.add_argument("--article", type=str, help="URL of the news article to screen")
    parser.add_argument("--test_file", type=str, help="Path to a CSV test file (e.g., test_cases.csv)")
    
    args = parser.parse_args()
    
    # --- MLflow Setup ---
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("AdverseMediaScreener")
    print(f"MLflow tracking enabled. URI: {mlflow.get_tracking_uri()}")
    # --------------------
    
    app = build_graph()
    print("Graph compiled. ASCII diagram:")
    print(app.get_graph().draw_ascii())
    
    test_cases_to_run = []
    if args.test_file:
        test_cases_to_run.extend(load_test_cases(args.test_file))
    elif args.name and args.dob and args.article:
        test_cases_to_run.append({"name": args.name, "dob": args.dob, "url": args.article})
    else:
        print("No inputs provided. Running with default 'test_cases.csv'.")
        if os.path.exists("test_cases.csv"):
             test_cases_to_run.extend(load_test_cases("test_cases.csv"))
        else:
            print("\nError: 'test_cases.csv' not found. Please create it or provide inputs.")
            exit(1)

    for case in test_cases_to_run:
        if not all(key in case for key in ['name', 'dob', 'url']):
            print(f"Skipping invalid case: {case}")
            continue
            
        print(f"\n{'='*50}")
        print(f"RUNNING CASE FOR: {case['name']} ({case['dob']})")
        print(f"URL: {case['url']}")
        print(f"{'='*50}")

        # --- MLflow: Start Run ---
        run_name = f"Screening {case['name']} - {uuid.uuid4().hex[:8]}"
        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id
            print(f"--- Starting MLflow Run: {run_id} ---")
            print(f"--- Run Name: {run_name} ---")

            # Log initial inputs as parameters
            mlflow.log_param("applicant_name", case['name'])
            mlflow.log_param("applicant_dob", case['dob'])
            mlflow.log_param("article_url", case['url'])

            current_state = {
                "applicant_name": case['name'],
                "applicant_dob": case['dob'],
                "article_url": case['url'],
                "article_text": "",
                "name_is_present": False,
                "name_check_explanation": "",
                "match_decision": "Review Required",
                "match_explanation": "",
                "sentiment": "N/A",
                "sentiment_explanation": ""
            }
            
            state_history = []
            
            # --- Graph Execution with MLflow Logging ---
            try:
                step_counter = 0
                # Use app.stream() to capture intermediate node outputs
                for chunk in app.stream(current_state):
                    # A chunk is a dict like {'node_name': {'output_key': 'value', ...}}
                    if not chunk:
                        continue
                        
                    # Get the node name and its output
                    node_name, node_output = list(chunk.items())[0]
                    print(f"--- [Step {step_counter}] Node '{node_name}' executed ---")
                    
                    # Update our local copy of the state
                    current_state.update(node_output)
                    
                    # Log the *entire state* after this step as a JSON artifact
                    log_name = f"step_{step_counter:02d}_{node_name}_state.json"
                    mlflow.log_dict(current_state.copy(), log_name)
                    
                    state_history.append({
                        "step": step_counter,
                        "node": node_name,
                        "output": node_output,
                    })
                    step_counter += 1

                # --- MLflow: Log Final Results ---
                final_state = current_state 
                
                print("\n--- FINAL RESULT (logged to MLflow) ---")
                print(f"  Match Decision: {final_state.get('match_decision')}")
                print(f"  Explanation: {final_state.get('match_explanation')}")
                print(f"  Sentiment: {final_state.get('sentiment', 'N/A')}")
                print(f"  Explanation: {final_state.get('sentiment_explanation')}")
                
                # Log key outputs as tags for easy filtering/grouping
                mlflow.set_tag("match_decision", final_state.get('match_decision', 'ERROR'))
                mlflow.set_tag("sentiment", final_state.get('sentiment', 'N/A'))
                
                # Log explanations as params (good for short text)
                mlflow.log_param("match_explanation", final_state.get('match_explanation', ''))
                mlflow.log_param("sentiment_explanation", final_state.get('sentiment_explanation', ''))

                # Log large outputs as artifacts
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
        # --- End of MLflow Run ---