from match_score import compute_match_score_with_breakdown

test_cases = [
    {
        "title": "✅ Full Stack Engineer (Perfect Match)",
        "job_description": "Looking for a backend engineer with Node.js, Express, and MongoDB experience to build REST APIs.",
        "job_skills": ["Node.js", "Express", "MongoDB", "REST API", "JavaScript"],
        "candidate_bio": "Backend developer with hands-on experience in building REST APIs using Node.js, Express, and MongoDB.",
        "candidate_skills": ["Node.js", "Express", "MongoDB", "REST API", "JavaScript"],
    },
    {
        "title": "🧠 AI/ML Researcher (High Match)",
        "job_description": "Research intern in machine learning, NLP, and deep learning. Must know PyTorch and Transformers.",
        "job_skills": ["Machine Learning", "NLP", "Deep Learning", "PyTorch", "Transformers"],
        "candidate_bio": "AI/ML enthusiast skilled in building NLP models with Transformers, PyTorch and fine-tuning LLMs.",
        "candidate_skills": ["Machine Learning", "NLP", "PyTorch", "HuggingFace", "Transformers"],
    },
    {
        "title": "🌐 UI/UX Designer (Moderate Match)",
        "job_description": "We need a creative UI/UX designer familiar with Figma, responsive design, and Adobe tools.",
        "job_skills": ["UI/UX", "Figma", "Adobe XD", "Responsive Design", "Prototyping"],
        "candidate_bio": "Front-end designer with a passion for UI/UX, prototyping in Figma and creating responsive designs.",
        "candidate_skills": ["UI/UX", "Figma", "Responsive Design", "Sketch", "CSS"],
    },
    {
        "title": "⚙️ DevOps vs Chef (Mismatch)",
        "job_description": "We're hiring a DevOps engineer to manage CI/CD pipelines, automate infrastructure, and maintain scalable cloud-based applications.",
        "job_skills": ["AWS", "Docker", "Kubernetes", "Terraform", "CI/CD", "Linux"],
        "candidate_bio": "Professional chef with over 10 years of experience in gourmet food preparation and kitchen management.",
        "candidate_skills": ["Cooking", "Menu Design", "Inventory Management", "Food Safety", "Baking"],
    },
    {
        "title": "🎨 Artist vs Data Scientist (Mismatch)",
        "job_description": "Looking for a data scientist with Python, NumPy, and strong statistical background to analyze datasets.",
        "job_skills": ["Python", "NumPy", "Pandas", "Statistics", "Machine Learning"],
        "candidate_bio": "Freelance artist skilled in digital illustration, watercolor, and art direction.",
        "candidate_skills": ["Illustration", "Art", "Photoshop", "Creative Direction", "Painting"],
    },
    {
        "title": "📈 Marketing Analyst (Good Match)",
        "job_description": "Seeking a marketing analyst to track campaign performance, generate reports, and optimize ads.",
        "job_skills": ["Google Analytics", "Excel", "Marketing", "SEO", "A/B Testing"],
        "candidate_bio": "Marketing specialist with experience in A/B testing, ad optimization, and data analysis using Excel.",
        "candidate_skills": ["Marketing", "A/B Testing", "Excel", "SEO", "Google Ads"],
    },
    {
        "title": "🧪 Bioinformatics Researcher (High Match)",
        "job_description": "Seeking a researcher experienced in gene sequencing, R, and bioinformatics pipelines.",
        "job_skills": ["R", "Gene Sequencing", "Bioinformatics", "Data Analysis"],
        "candidate_bio": "Biomedical researcher with expertise in gene sequencing, R programming, and data interpretation.",
        "candidate_skills": ["R", "Gene Sequencing", "Biostatistics", "Python", "Data Analysis"],
    },
    {
        "title": "🧑‍🏫 Teacher vs Engineer (Mismatch)",
        "job_description": "Looking for a React.js developer to build modern web interfaces and collaborate on frontend architecture.",
        "job_skills": ["React", "JavaScript", "Frontend", "Redux", "UI"],
        "candidate_bio": "High school teacher with experience in mathematics, lesson planning, and student mentoring.",
        "candidate_skills": ["Teaching", "Math", "Classroom Management", "Communication"],
    },
    {
        "title": "💼 HR Manager (Moderate Match)",
        "job_description": "We need an HR manager experienced in recruitment, employee engagement, and payroll systems.",
        "job_skills": ["Recruitment", "Employee Engagement", "Payroll", "HRMS", "Communication"],
        "candidate_bio": "HR generalist skilled in payroll handling, recruitment operations, and managing employee records.",
        "candidate_skills": ["Recruitment", "Payroll", "HRMS", "Compliance", "Interviewing"],
    },
    {
        "title": "🧑‍🔧 Mechanical vs Cloud Engineer (Low Match)",
        "job_description": "Hiring a cloud engineer with experience in GCP, Kubernetes, and microservices architecture.",
        "job_skills": ["GCP", "Kubernetes", "Docker", "Cloud", "Microservices"],
        "candidate_bio": "Mechanical engineer with skills in CAD design, thermal systems, and industrial robotics.",
        "candidate_skills": ["CAD", "Thermal Analysis", "SolidWorks", "Manufacturing", "Matlab"],
    },
]

# Loop through all and print scores
for i, case in enumerate(test_cases, 1):
    result = compute_match_score_with_breakdown(
        case["job_description"],
        case["job_skills"],
        case["candidate_bio"],
        case["candidate_skills"],
    )

    print(f"\n🔍 Test Case {i}: {case['title']}")
    print(f"✅ Final Match Score: {result['score']}%")
    print("Breakdown:")
    print(f"- 🧠 Bio Desc Score: {result['bio_desc_score']}")
    print(f"- 🛠️ Skill Embed Score: {result['skill_embed_score']}")
    print(f"- 🔁 Fuzzy Score: {result['fuzzy_score']}")
    print(f"- 🧪 TF-IDF Score: {result['tfidf_score']}")
