from flask import Flask, request, jsonify
from flask_cors import CORS
from match_score import compute_match_score_with_breakdown
import nltk

# Download required NLTK data on startup
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

app = Flask(__name__)
CORS(app)  # ✅ Enable CORS globally

# Health check route
@app.route("/")
def health():
    return "✅ Match Score API is running!"

# Main scoring route
@app.route("/match-score", methods=["POST"])
def match_score():
    data = request.get_json()

    print("🔍 Received JSON input:")
    print(data)

    job_desc = data.get("jobDescription", "")
    job_skills = data.get("jobSkills", [])
    candidate_bio = data.get("candidateBio", "")
    candidate_skills = data.get("candidateSkills", [])

    result = compute_match_score_with_breakdown(
        job_desc, job_skills, candidate_bio, candidate_skills
    )

    return jsonify(result)

# Start Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
