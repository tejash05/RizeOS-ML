# import re
# import numpy as np
# from sentence_transformers import SentenceTransformer, util
# from rapidfuzz import fuzz, process
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer, PorterStemmer
# from nltk.tokenize import RegexpTokenizer
# import nltk

# # 📦 Setup NLTK
# nltk.data.path.append("/Users/tejashtarun/nltk_data")
# nltk.download("stopwords")
# nltk.download("wordnet")
# nltk.download("omw-1.4")

# # 🔍 Load components
# embedder = SentenceTransformer("thenlper/gte-large")  # best open-source semantic model
# stop_words = set(stopwords.words("english"))
# lemmatizer = WordNetLemmatizer()
# stemmer = PorterStemmer()
# tokenizer = RegexpTokenizer(r"\w+")

# # 🧹 Clean and normalize text
# def clean_text(text):
#     tokens = tokenizer.tokenize(text.lower())
#     return " ".join(lemmatizer.lemmatize(t) for t in tokens if t not in stop_words)

# def normalize_skills(skills):
#     return sorted(set(
#         lemmatizer.lemmatize(re.sub(r"[^\w\s]", "", skill.lower().strip()))
#         for skill in skills if skill
#     ))

# # 🔁 Fuzzy skill score with token_set_ratio
# def fuzzy_skill_score(job_skills, candidate_skills):
#     if not job_skills or not candidate_skills:
#         return 0.0
#     scores = []
#     for job_skill in job_skills:
#         matches = process.extract(job_skill, candidate_skills, scorer=fuzz.token_set_ratio, limit=3)
#         top_scores = [score / 100.0 for _, score, _ in matches if score > 60]
#         if top_scores:
#             scores.append(np.mean(top_scores))
#     return round(np.mean(scores), 4) if scores else 0.0

# # 🧠 Advanced skill pairwise similarity (replaces TF-IDF)
# def skill_pairwise_semantic_score(job_skills, candidate_skills):
#     if not job_skills or not candidate_skills:
#         return 0.0
#     job_embeddings = embedder.encode(job_skills, convert_to_tensor=True)
#     candidate_embeddings = embedder.encode(candidate_skills, convert_to_tensor=True)
#     sim_matrix = util.cos_sim(job_embeddings, candidate_embeddings).cpu().numpy()
#     max_similarities = np.max(sim_matrix, axis=1)
#     return float(np.mean(max_similarities))

# # 🧠 Final scoring pipeline
# def compute_match_score_with_breakdown(job_description, job_skills, candidate_bio, candidate_skills):
#     # Clean & normalize
#     job_description_clean = clean_text(job_description)
#     candidate_bio_clean = clean_text(candidate_bio)
#     job_skills_norm = normalize_skills(job_skills)
#     candidate_skills_norm = normalize_skills(candidate_skills)

#     # 🔍 Embedding scores
#     desc_embedding = embedder.encode(job_description_clean, convert_to_tensor=True)
#     bio_embedding = embedder.encode(candidate_bio_clean, convert_to_tensor=True)
#     bio_desc_score = float(util.cos_sim(bio_embedding, desc_embedding)[0][0])

#     job_skill_embed = embedder.encode(" ".join(job_skills_norm), convert_to_tensor=True)
#     user_skill_embed = embedder.encode(" ".join(candidate_skills_norm), convert_to_tensor=True)
#     skill_embed_score = float(util.cos_sim(job_skill_embed, user_skill_embed)[0][0])

#     # 🔁 Fuzzy logic
#     fuzzy_score = fuzzy_skill_score(job_skills_norm, candidate_skills_norm)

#     # 🔍 Advanced token-level semantic similarity (TF-IDF alternative)
#     semantic_token_score = skill_pairwise_semantic_score(job_skills_norm, candidate_skills_norm)

#     # 🎯 Final score with fixed weights
#     final_score = (
#         0.20 * bio_desc_score +
#         0.20 * skill_embed_score +
#         0.40 * fuzzy_score +
#         0.10 * semantic_token_score
#     )

#     return {
#         "score": round(final_score * 100, 2),
#         "bio_desc_score": round(bio_desc_score, 4),
#         "skill_embed_score": round(skill_embed_score, 4),
#         "fuzzy_score": round(fuzzy_score, 4),
#         "tfidf_score": round(semantic_token_score, 4),  # still call it tfidf for consistency
#     }



# match_score.py
import re
import time
import numpy as np
from sentence_transformers import SentenceTransformer, util
from rapidfuzz import fuzz, process
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

# 🧠 Load once globally
embedder = SentenceTransformer("thenlper/gte-large")
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
tokenizer = RegexpTokenizer(r"\w+")

def clean_text(text):
    tokens = tokenizer.tokenize(text.lower())
    return " ".join(lemmatizer.lemmatize(t) for t in tokens if t not in stop_words)

def normalize_skills(skills):
    return sorted(set(
        lemmatizer.lemmatize(re.sub(r"[^\w\s]", "", skill.lower().strip()))
        for skill in skills if skill
    ))

def fuzzy_skill_score(job_skills, candidate_skills):
    if not job_skills or not candidate_skills:
        return 0.0
    scores = []
    for job_skill in job_skills:
        matches = process.extract(
            job_skill, candidate_skills, scorer=fuzz.token_set_ratio, limit=3
        )
        top_scores = [score / 100.0 for _, score, _ in matches if score > 60]
        if top_scores:
            scores.append(np.mean(top_scores))
    return round(np.mean(scores), 4) if scores else 0.0

def skill_pairwise_semantic_score(job_skills, candidate_skills):
    if not job_skills or not candidate_skills:
        return 0.0
    job_embeds = embedder.encode(job_skills, convert_to_tensor=True)
    user_embeds = embedder.encode(candidate_skills, convert_to_tensor=True)
    sim_matrix = util.cos_sim(job_embeds, user_embeds).cpu().numpy()
    return np.mean(np.max(sim_matrix, axis=1))

def batch_compute_scores(data):
    t0 = time.time()

    cleaned_inputs = []
    string_cache = {}

    for item in data:
        job_desc = clean_text(item.get("job_description") or item.get("jobDescription", ""))
        bio = clean_text(item.get("candidate_bio") or item.get("candidateBio", ""))
        job_skills = normalize_skills(item.get("job_skills") or item.get("jobSkills", []))
        candidate_skills = normalize_skills(item.get("candidate_skills") or item.get("candidateSkills", []))
        cleaned_inputs.append((job_desc, bio, job_skills, candidate_skills))

        # For embedding reuse
        string_cache[job_desc] = None
        string_cache[bio] = None
        string_cache[" ".join(job_skills)] = None
        string_cache[" ".join(candidate_skills)] = None

    # ✅ Batch embed only unique strings
    to_embed = list(string_cache.keys())
    embeddings = embedder.encode(to_embed, convert_to_tensor=True)
    for i, text in enumerate(to_embed):
        string_cache[text] = embeddings[i]

    results = []
    for job_desc, bio, job_skills, candidate_skills in cleaned_inputs:
        desc_embed = string_cache[job_desc]
        bio_embed = string_cache[bio]
        job_skill_embed = string_cache[" ".join(job_skills)]
        user_skill_embed = string_cache[" ".join(candidate_skills)]

        bio_desc_score = util.cos_sim(bio_embed, desc_embed).item()
        skill_embed_score = util.cos_sim(job_skill_embed, user_skill_embed).item()
        fuzzy_score = fuzzy_skill_score(job_skills, candidate_skills)
        semantic_token_score = skill_pairwise_semantic_score(job_skills, candidate_skills)

        final_score = (
            0.20 * bio_desc_score +
            0.20 * skill_embed_score +
            0.40 * fuzzy_score +
            0.10 * semantic_token_score
        )

        results.append({
                "score": float(round(final_score * 100, 2)),
                "bio_desc_score": float(round(bio_desc_score, 4)),
                "skill_embed_score": float(round(skill_embed_score, 4)),
                "fuzzy_score": float(round(fuzzy_score, 4)),
                "tfidf_score": float(round(semantic_token_score, 4)),
            })


    print(f"⚡ Batch scored {len(data)} pairs in {time.time() - t0:.2f}s")
    return results
