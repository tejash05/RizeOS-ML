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

# Match scoring route (supports both single and batch inputs)
@app.route("/match-score", methods=["POST"])
def match_score():
    data = request.get_json()

    if not data:
        return jsonify({"error": "No data received"}), 400

    # Handle single object
    if isinstance(data, dict):
        data = [data]

    # Must be a list at this point
    if not isinstance(data, list):
        return jsonify({"error": "Invalid input format. Must be an object or list of objects."}), 400

    print(f"🔍 Received {len(data)} job-candidate pairs")

    results = []
    for item in data:
        job_desc = item.get("job_description") or item.get("jobDescription", "")
        job_skills = item.get("job_skills") or item.get("jobSkills", [])
        candidate_bio = item.get("candidate_bio") or item.get("candidateBio", "")
        candidate_skills = item.get("candidate_skills") or item.get("candidateSkills", [])

        result = compute_match_score_with_breakdown(
            job_desc, job_skills, candidate_bio, candidate_skills
        )

        # Do NOT include title in output
        results.append(result)

    return jsonify(results if len(results) > 1 else results[0])

# Start Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
