from flask import Flask, request, jsonify
from match_score import compute_match_score_with_breakdown

app = Flask(__name__)

@app.route("/match-score", methods=["POST"])
def match_score():
    data = request.get_json()

    # ✅ Log the received input for debugging
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

if __name__ == "__main__":
    import os
    app.run(host="0.0.0.0", port=10000)
