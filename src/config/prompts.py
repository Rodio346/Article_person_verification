"""
Prompts configuration for the Article Person Verification system.
All LLM prompts are centralized here for easy editing and maintenance.
"""

# --- Name Presence Check Prompt ---

NAME_PRESENCE_PROMPT = """
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

# --- Age Verification Prompt ---

AGE_VERIFICATION_PROMPT = """
You are an analyst. We have already confirmed that the name "{applicant_name}" is mentioned in the article.

Your task is to verify if the age or date of birth mentioned in the article matches the applicant's age.

Applicant Name: {applicant_name}
Applicant DOB: {applicant_dob}

Article Text:
{article_text}

Analyze the article and determine if the age or date of birth mentioned matches the applicant's DOB.
- Calculate the current age from the DOB if needed (Today's date can be estimated from context)
- Look for phrases like "X years old", "born in YYYY", "age XX", etc.
- Consider a margin of error of +/- 1 year for age matches (articles might be slightly outdated)

Respond in a single, valid JSON object with two keys:
- "age_matches": true or false
  - true: If age/DOB matches within reasonable margin
  - false: If there's a clear age/DOB mismatch (difference > 1 year)
- "explanation": A brief explanation of what age information was found and how it compares to the applicant's DOB.

If NO age or DOB information is found in the article, respond with:
- "age_matches": true (benefit of the doubt - proceed to detailed verification)
- "explanation": "No age or date of birth information found in the article."
"""

# --- Detail Verification Prompt ---

DETAIL_VERIFICATION_PROMPT = """
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

# --- Sentiment Analysis Prompt ---

SENTIMENT_ANALYSIS_PROMPT = """
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
