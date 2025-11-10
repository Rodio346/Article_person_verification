"""
Prompts configuration for the Article Person Verification system.
All LLM prompts are centralized here for easy editing and maintenance.
"""

# --- Name Presence Check Prompt ---

NAME_PRESENCE_PROMPT = """
You are a multilingual analyst capable of processing articles in any language. Your task is to quickly scan an article to see if a specific name is mentioned.

**IMPORTANT**: The article may be in any language (English, Spanish, French, German, Chinese, Arabic, etc.).
Process the content in its original language and provide your response in English.

Applicant Name: {applicant_name}

Article Text:
{article_text}

Is the Applicant Name, or a very clear variation (e.g., "Bill" for "William", "Bernie" for "Bernard"),
mentioned in the Article Text?

**Instructions:**
- Search for the name in the article regardless of the article's language
- Account for transliterations and language-specific name variations
- Consider that names may be written differently across languages (e.g., Chinese characters, Arabic script, Cyrillic)

Respond in a single, valid JSON object with two keys (response must be in English):
- "name_is_present": true or false
- "explanation": A brief reason in English. (e.g., "Name 'Bernie Madoff' found in Spanish article." or "Name not found in French article.")
"""

# --- Age Verification Prompt ---

AGE_VERIFICATION_PROMPT = """
You are a multilingual analyst capable of processing articles in any language. We have already confirmed that the name "{applicant_name}" is mentioned in the article.

**IMPORTANT**: The article may be in any language (English, Spanish, French, German, Chinese, Arabic, etc.).
Process the content in its original language and provide your response in English.

Your task is to verify if the age or date of birth mentioned in the article matches the applicant's age.

Applicant Name: {applicant_name}
Applicant DOB: {applicant_dob}

Article Text:
{article_text}

Analyze the article and determine if the age or date of birth mentioned matches the applicant's DOB.
- Calculate the current age from the DOB if needed (Today's date can be estimated from context)
- Look for age/DOB phrases in any language (e.g., "X years old", "años de edad", "歳", "ans", etc.)
- Look for numeric age indicators regardless of surrounding language
- Consider a margin of error of +/- 1 year for age matches (articles might be slightly outdated)
- Account for different date formats used across cultures (DD/MM/YYYY, MM/DD/YYYY, YYYY-MM-DD, etc.)

Respond in a single, valid JSON object with two keys (response must be in English):
- "age_matches": true or false
  - true: If age/DOB matches within reasonable margin
  - false: If there's a clear age/DOB mismatch (difference > 1 year)
- "explanation": A brief explanation in English of what age information was found and how it compares to the applicant's DOB.

If NO age or DOB information is found in the article, respond with:
- "age_matches": true (benefit of the doubt - proceed to detailed verification)
- "explanation": "No age or date of birth information found in the article."
"""

# --- Detail Verification Prompt ---

DETAIL_VERIFICATION_PROMPT = """
You are a meticulous multilingual financial analyst capable of processing articles in any language. We have already confirmed that the name "{applicant_name}" (or a variation) is in the article.

**IMPORTANT**: The article may be in any language (English, Spanish, French, German, Chinese, Arabic, etc.).
Process the content in its original language and provide your response in English.

Your task is to determine if it's the *correct person* by verifying their Date of Birth and other identifiers.
You must be extremely careful to avoid "false negatives".

Applicant Name: {applicant_name}
Applicant DOB: {applicant_dob}

Article Text:
{article_text}

Analyze the text and provide your decision in a single, valid JSON object.
1. Look for the applicant's name again in the article (in any language or script).
2. Look for any dates of birth, ages, or other strong identifiers (locations, occupations, companies) *associated with that name*.
3. Compare these details to the Applicant's DOB.
4. Consider that identifiers may be in the article's native language (e.g., occupation titles, place names).

Your JSON response MUST have exactly two keys: "decision" and "explanation" (both in English).

- "decision" must be one of:
  - "Match": If you are confident it's the same person (e.g., name and DOB match, or name and other strong identifiers match).
  - "Non-Match": ONLY if you find *explicit contradictory evidence* (e.g., the article mentions the same name but a *different* DOB or age).
  - "Review Required": If the name matches but no *other* identifying or contradictory details are present. **This is your default if uncertain.**

- "explanation": A step-by-step justification in English for your decision, mentioning the language of the article if not English.
"""

# --- Sentiment Analysis Prompt ---

SENTIMENT_ANALYSIS_PROMPT = """
You are a multilingual analyst capable of processing articles in any language. You are reviewing an article about a specific person.
The article has already been determined to be about: {applicant_name}.

**IMPORTANT**: The article may be in any language (English, Spanish, French, German, Chinese, Arabic, etc.).
Process the content in its original language and provide your response in English.

Article Text:
{article_text}

Analyze the article and determine if it portrays this person in a positive, negative, or neutral light,
specifically in a regulated financial context.

**Sentiment Guidelines (applicable across all languages):**
- "Negative" includes: lawsuits, scandals, fraud, bankruptcies, criminal activity, corruption, investigations, penalties, violations
- "Positive" includes: philanthropy, achievements, industry awards, successful ventures, leadership praise, innovations
- "Neutral" includes: simple news reports, job changes, objective statements, factual announcements

**Instructions:**
- Understand the article's content regardless of language
- Identify sentiment-carrying words/phrases in the original language
- Consider cultural context when interpreting sentiment
- Focus on facts and tone rather than just keywords

Provide your response in a single, valid JSON object with two keys (response must be in English):
- "sentiment" must be one of: "Positive", "Negative", "Neutral".
- "explanation": A brief justification in English for your sentiment, mentioning the language of the article if not English.
"""
