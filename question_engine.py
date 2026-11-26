"""HealthScope Gemini-powered question engine.

Compatible with: answer_question(question, df)
Page suggestions: get_suggested_questions(page)
Gemini interprets the question; pandas calculates locally.
No patient rows or IDs are sent to Gemini.
"""
import json
import os
import re
import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None
    types = None

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
_client = None
if genai and os.getenv("GEMINI_API_KEY"):
    try:
        _client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    except Exception:
        pass

QUESTION_BANK = {
    "Demographic Analysis": [
        "Which age group has the highest number of patients?",
        "How are patients distributed by gender?",
        "Which state has the highest number of patients?",
        "Which region has the largest patient population?",
        "How many patients are from urban and rural areas?",
        "What is the patient distribution across income levels?",
        "Which blood group is most common among patients?",
        "What is the average age of patients?",
        "Which age group has the highest number of female patients?",
        "How does the patient population vary across states and genders?",
    ],
    "Time Trend Analysis": [
        "How has the number of patients changed each year?",
        "Which month has the highest number of patient cases?",
        "Which year recorded the highest number of cases?",
        "How do patient cases vary by season?",
        "How has the number of deaths changed over the years?",
        "Which months show the highest patient admissions?",
        "How do disease cases differ between seasons?",
        "What is the monthly trend in treatment costs?",
        "How has average recovery time changed over the years?",
        "Which quarter recorded the highest number of cases?",
    ],
    "Disease Distribution": [
        "Which disease has the highest number of cases?",
        "What are the top 10 most common diseases?",
        "How many cases belong to each disease category?",
        "Which disease category has the highest number of cases?",
        "Which diseases have the highest number of deaths?",
        "How are disease cases distributed by severity?",
        "Which disease has the highest number of severe cases?",
        "Which diseases have the highest recovery time?",
        "What are the most common diseases in each state?",
        "How do disease outcomes differ across disease categories?",
    ],
    "Risk Analysis": [
        "How many patients are current smokers?",
        "What percentage of patients report alcohol use?",
        "Which severity group has the most patients?",
        "How does smoking status relate to disease severity?",
        "How does alcohol use vary across severity groups?",
        "What is the average BMI of patients?",
        "How many patients have a recorded comorbidity?",
        "Which comorbidities are most common?",
        "How do outcomes differ by smoking status?",
        "Which risk factor category has the highest number of severe cases?",
    ],
    "Cost Analysis": [
        "What is the total treatment cost across all patients?",
        "What is the average treatment cost per patient?",
        "Which treatment type has the highest total cost?",
        "Which disease has the highest total treatment cost?",
        "What are the top five diseases by treatment cost?",
        "How does average treatment cost differ by hospital type?",
        "How does treatment cost vary by severity?",
        "What is the average treatment cost for insured and uninsured patients?",
        "How does treatment cost change month by month?",
        "Is longer hospitalization associated with higher treatment costs?",
    ],
    "Hospital Insights": [
        "Which hospital type treats the highest number of patients?",
        "Which hospital type has the highest average treatment cost?",
        "What is the average number of days patients stay in hospital?",
        "Which hospital type has the longest average hospital stay?",
        "Which states have the highest number of hospitalised patients?",
        "How do patient outcomes differ by hospital type?",
        "Which hospital types have the highest number of deaths?",
        "What is the average recovery time by hospital type?",
        "How many patients require follow-up care at each hospital type?",
        "How does hospital stay duration vary by severity?",
    ],
}

NUMERIC = {"ages", "bmi", "days_hospitalized", "treatment_cost_inr", "recovery_days", "death_flag"}
ALLOWED_OPS = {"count", "sum", "average", "median", "minimum", "maximum", "distribution", "ranking", "trend", "correlation"}
TIME_COLS = {"year", "month", "Month Year", "Month Start", "Quarter", "diagnosis_date"}

PROMPT = """
You are a question interpreter for a healthcare analytics dashboard.
Return only JSON with this exact shape:
{"operation":"count|sum|average|median|minimum|maximum|distribution|ranking|trend|correlation",
"measure":null,"group_by":[],"filters":[],"top_n":0,"sort":"descending","time_column":null}
Only use exact column names from the supplied schema.
For patient/case counts use count and measure null.
For total cost use sum and treatment_cost_inr.
For average cost use average and treatment_cost_inr.
For ranking/top/highest use ranking descending; lowest use ascending.
For breakdown/distribution, group by the requested category.
For trends, group by a time column.
Filters must be explicit in the question; do not invent them.
Never use patient_id_no. Never generate code or SQL.
top_n defaults to 0 and cannot exceed 20.
"""

def _norm(s):
    return re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower()).strip("_")

def _col(df, name):
    if not name:
        return None
    if name in df.columns:
        return name
    target = _norm(name)
    for c in df.columns:
        if _norm(c) == target:
            return c
    return None

def _gemini_plan(question, df):
    if not _client:
        return None
    schema = {
        "numeric": [str(c) for c in df.columns if pd.api.types.is_numeric_dtype(df[c])],
        "categorical": [str(c) for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])],
    }
    try:
        response = _client.models.generate_content(
            model=GEMINI_MODEL,
            contents=PROMPT + "\nSchema:\n" + json.dumps(schema) + "\nQuestion:\n" + question,
            config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.1),
        )
        return json.loads(response.text)
    except Exception:
        return None

def _fallback(question, df):
    q = question.lower()
    plan = {"operation": "count", "measure": None, "group_by": [], "filters": [],
            "top_n": 0, "sort": "descending", "time_column": None}
    metric_terms = {
        "treatment_cost_inr": ["cost", "expenditure", "expense"],
        "days_hospitalized": ["hospital stay", "hospitalized", "hospitalised"],
        "recovery_days": ["recovery", "recover"],
        "ages": ["average age"],
        "bmi": ["bmi"],
        "death_flag": ["death", "deaths", "mortality"],
    }
    for col, terms in metric_terms.items():
        if col in df.columns and any(t in q for t in terms):
            plan["measure"] = col
            break
    if any(t in q for t in ["average", "avg", "mean"]):
        plan["operation"] = "average" if plan["measure"] else "count"
    elif any(t in q for t in ["total", "sum"]):
        plan["operation"] = "sum" if plan["measure"] else "count"
    elif any(t in q for t in ["top", "highest", "most", "lowest", "least"]):
        plan["operation"] = "ranking"
        plan["sort"] = "ascending" if any(t in q for t in ["lowest", "least"]) else "descending"
        m = re.search(r"\btop\s+(\d+)\b", q)
        plan["top_n"] = min(int(m.group(1)), 20) if m else 5
    dimensions = ["disease_name", "disease_category", "state", "city", "region", "age_group",
                  "gender", "severity", "hospital_type", "treatment_type", "outcome",
                  "smoking_status", "alcohol_use", "season", "insurance_status",
                  "income_level", "blood_group", "comorbidity", "urban_rural",
                  "follow_up_required", "year", "month", "Quarter"]
    for d in dimensions:
        c = _col(df, d)
        if c and (d.replace("_", " ") in q or d.lower() in q):
            plan["group_by"] = [c]
            break
    return plan

def _validate(plan, df):
    op = str(plan.get("operation", "count")).lower()
    if op not in ALLOWED_OPS:
        op = "count"
    measure = _col(df, str(plan.get("measure") or ""))
    if measure not in NUMERIC or measure not in df.columns:
        measure = None
    raw_groups = plan.get("group_by", [])
    if isinstance(raw_groups, str):
        raw_groups = [raw_groups]
    groups = []
    for g in raw_groups if isinstance(raw_groups, list) else []:
        c = _col(df, str(g))
        if c and c != "patient_id_no" and c not in groups:
            groups.append(c)
        if len(groups) == 2:
            break
    filters = []
    for f in plan.get("filters", []) if isinstance(plan.get("filters", []), list) else []:
        if isinstance(f, dict):
            c, v = _col(df, str(f.get("column", ""))), str(f.get("value", "")).strip()
            if c and c != "patient_id_no" and v:
                filters.append({"column": c, "value": v[:100]})
    try:
        top_n = max(0, min(int(plan.get("top_n", 0)), 20))
    except Exception:
        top_n = 0
    sort = "ascending" if str(plan.get("sort", "")).lower() == "ascending" else "descending"
    if op in {"sum", "average", "median", "minimum", "maximum", "correlation"} and measure is None:
        op = "count"
    return {"operation": op, "measure": measure, "group_by": groups,
            "filters": filters[:5], "top_n": top_n, "sort": sort}

def _fmt(x):
    try:
        return f"{float(x):,.2f}"
    except Exception:
        return str(x)

def _calculate(question, df, p):
    data = df.copy()
    for f in p["filters"]:
        c, v = f["column"], f["value"]
        filtered = data[data[c].astype(str).str.contains(re.escape(v), case=False, na=False)]
        if not filtered.empty:
            data = filtered
    if data.empty:
        return "No records matched the requested filters."
    op, metric, groups = p["operation"], p["measure"], p["group_by"]
    if op == "correlation":
        nums = [c for c in data.columns if c in NUMERIC and pd.api.types.is_numeric_dtype(data[c])]
        if len(nums) >= 2:
            r = data[nums[:2]].corr().iloc[0, 1]
            return f"Correlation between {nums[0]} and {nums[1]}: {r:.3f}. Correlation does not prove causation."
        return "Two suitable numeric columns are needed for a correlation."
    if op == "count" or metric is None:
        if not groups:
            return f"Matching patient records: {len(data):,}."
        result = data.groupby(groups, dropna=False).size().reset_index(name="Patient Count")
        val_col = "Patient Count"
    else:
        data[metric] = pd.to_numeric(data[metric], errors="coerce")
        if not groups:
            s = data[metric].dropna()
            if s.empty:
                return f"No numeric values found for {metric}."
            funcs = {"sum": s.sum, "average": s.mean, "median": s.median,
                     "minimum": s.min, "maximum": s.max}
            fn = funcs.get(op, s.mean)
            value = fn()
            label = {"sum": "Total", "average": "Average", "median": "Median",
                     "minimum": "Minimum", "maximum": "Maximum"}.get(op, "Value")
            prefix = "₹" if metric == "treatment_cost_inr" else ""
            suffix = " days" if metric in {"days_hospitalized", "recovery_days"} else ""
            return f"{label} {metric.replace('_', ' ')}: {prefix}{_fmt(value)}{suffix}."
        agg = "sum" if op in {"sum", "ranking", "trend"} else {
            "median": "median", "minimum": "min", "maximum": "max"}.get(op, "mean")
        result = data.groupby(groups, dropna=False)[metric].agg(agg).reset_index()
        val_col = metric
    result = result.sort_values(val_col, ascending=p["sort"] == "ascending")
    if p["top_n"]:
        result = result.head(p["top_n"])
    result = result.head(15)
    title = "Ranking" if op == "ranking" else "Breakdown"
    lines = [f"{title} by {' and '.join(groups)}:"]
    for _, row in result.iterrows():
        label = " | ".join(str(row[g]) for g in groups)
        prefix = "₹" if metric == "treatment_cost_inr" else ""
        lines.append(f"• {label}: {prefix}{_fmt(row[val_col])}")
    return "\n".join(lines)

def get_suggested_questions(page=None):
    """Return all 60 suggested questions, or 10 for a page."""
    if not page:
        return QUESTION_BANK
    for name, questions in QUESTION_BANK.items():
        if name.lower() == str(page).strip().lower():
            return questions
    return []

def answer_question(question, df, page=None):
    """Main entry point; remains compatible with answer_question(question, df)."""
    if not isinstance(question, str) or not question.strip():
        return "Please enter a question about the healthcare dataset."
    if df is None or df.empty:
        return "The healthcare dataset is not loaded."
    plan = _gemini_plan(question.strip(), df)
    if not isinstance(plan, dict):
        plan = _fallback(question.strip(), df)
    try:
        return _calculate(question, df, _validate(plan, df))
    except Exception:
        return ("I couldn't calculate that from the available fields. Try asking about "
                "patient counts, disease rankings, treatment cost, severity, hospital type, "
                "state, smoking, BMI, hospital stay or recovery time.")

if __name__ == "__main__":
    print("HealthScope question engine loaded.")
    print("Suggested questions:", sum(len(x) for x in QUESTION_BANK.values()))
    print("Gemini configured:", bool(_client))
