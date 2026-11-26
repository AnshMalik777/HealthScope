
import os
import json
from dotenv import load_dotenv
from google import genai

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

client = genai.Client(api_key=API_KEY) if API_KEY else None

ALLOWED_COLUMNS = [
    "ages", "age_group", "gender", "state", "city", "region",
    "urban_rural", "disease_name", "disease_category", "severity",
    "year", "month", "season", "symptoms", "comorbidity",
    "smoking_status", "alcohol_use", "bmi", "blood_group",
    "income_level", "insurance_status", "treatment_type",
    "hospital_type", "days_hospitalized", "treatment_cost_inr",
    "outcome", "death_flag", "cause_of_death", "recovery_days",
    "follow_up_required"
]


def enhance_question(question):
    """
    Convert a user's natural-language question into a clearer
    question that the existing Python question engine can process.

    No patient records are sent to Gemini.
    """

    if not client:
        raise RuntimeError("Gemini API key is not configured.")

    prompt = f"""
You are the natural-language question interpreter for HealthScope,
a healthcare data analytics project.

Rewrite the user's question into ONE clear, precise English
analytics question that a Python data analysis engine can understand.

Available dataset columns:
{json.dumps(ALLOWED_COLUMNS)}

Rules:
1. Preserve the user's original meaning.
2. Preserve requested filters, locations, categories and time periods.
3. Preserve requested rankings and ranking limits.
4. For "most common disease", write "Which disease has the highest number of records?"
5. For "average treatment cost", write "What is the average treatment cost?"
6. Do not invent facts, values or filters.
7. Do not answer the question yourself.
8. Do not generate Python, SQL or executable code.
9. Return ONLY JSON in this format:
   {{"normalized_question": "rewritten question"}}

User's question:
{question}
"""

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config={
            "response_mime_type": "application/json"
        }
    )

    result = json.loads(response.text)

    normalized = result.get("normalized_question")

    if not isinstance(normalized, str) or not normalized.strip():
        raise ValueError("Gemini returned an invalid question.")

    if len(normalized) > 1000:
        raise ValueError("The interpreted question is too long.")

    return normalized.strip()
