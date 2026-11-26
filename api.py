
from flask import Flask, request, jsonify
from flask_cors import CORS

from analysis_engine import load_healthcare_data
from question_engine import answer_question
from gemini_engine import enhance_question

app = Flask(__name__)
CORS(app)

# Load healthcare dataset locally
print("Loading HealthScope healthcare dataset...")
df = load_healthcare_data()
print(f"Loaded {len(df)} patient records successfully!")


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "success",
        "message": "HealthScope AI Healthcare API is running"
    })


@app.route("/api/ask", methods=["POST"])
def ask_question():
    try:
        data = request.get_json(silent=True) or {}
        question = data.get("question", "").strip()

        if not question:
            return jsonify({
                "status": "error",
                "answer": "Please enter a question."
            }), 400

        if len(question) > 1000:
            return jsonify({
                "status": "error",
                "answer": "Please keep your question under 1,000 characters."
            }), 400

        # Step 1: Let Gemini understand the question
        try:
            interpreted_question = enhance_question(question)
            ai_status = "Gemini AI"
        except Exception as gemini_error:
            print("Gemini unavailable:", gemini_error)
            interpreted_question = question
            ai_status = "Python fallback"

        # Step 2: Python calculates the actual answer
        answer = answer_question(interpreted_question, df)

        return jsonify({
            "status": "success",
            "question": question,
            "interpreted_question": interpreted_question,
            "answer": answer,
            "ai_status": ai_status
        })

    except Exception as e:
        print("API error:", str(e))
        return jsonify({
            "status": "error",
            "answer": "Something went wrong while analysing your question."
        }), 500


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
@app.route("/api/questions", methods=["GET"])
def suggested_questions():
    page = request.args.get("page", "")
    questions = get_suggested_questions(page)

    return jsonify({
        "status": "success",
        "questions": questions
    })