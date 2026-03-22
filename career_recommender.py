"""
Rule-based career recommendation engine.
Uses weighted scoring to match user profiles to career paths.
This runs before the LLM to provide structured context.
"""

from career_data import CAREER_PATHS, SALARY_RANGES, TOP_COLLEGES, ENTRANCE_EXAMS

CAREER_WEIGHTS = {
    "Software Engineer":        {"streams": ["Computer Science", "Engineering", "Science (PCM)"],  "skills": ["python", "java", "c++", "coding", "programming"], "interests": ["technology", "software", "coding"]},
    "Data Scientist":           {"streams": ["Computer Science", "Engineering", "Science (PCM)"],  "skills": ["python", "data analysis", "machine learning", "sql", "statistics"], "interests": ["data", "technology", "analytics"]},
    "AI/ML Engineer":           {"streams": ["Computer Science", "Engineering", "Science (PCM)"],  "skills": ["python", "machine learning", "tensorflow", "pytorch", "deep learning"], "interests": ["ai", "technology", "research"]},
    "Full Stack Developer":     {"streams": ["Computer Science", "Engineering"],                   "skills": ["python", "javascript", "react", "node", "html", "css"], "interests": ["technology", "web", "design"]},
    "Cybersecurity Analyst":    {"streams": ["Computer Science", "Engineering"],                   "skills": ["networking", "security", "python", "linux", "ethical hacking"], "interests": ["security", "technology"]},
    "Financial Analyst":        {"streams": ["Commerce", "Management"],                            "skills": ["excel", "finance", "accounting", "data analysis"], "interests": ["finance", "business", "investment"]},
    "Investment Banker":        {"streams": ["Commerce", "Management"],                            "skills": ["finance", "excel", "valuation", "accounting"], "interests": ["finance", "investment", "business"]},
    "Chartered Accountant":     {"streams": ["Commerce"],                                          "skills": ["accounting", "excel", "taxation", "audit"], "interests": ["finance", "accounting"]},
    "Doctor (MBBS)":            {"streams": ["Science (PCB)", "Medical / Paramedical"],            "skills": ["biology", "chemistry", "lab work", "research"], "interests": ["healthcare", "medicine", "helping people"]},
    "Research Scientist":       {"streams": ["Science (PCM)", "Science (PCB)"],                   "skills": ["research", "python", "data analysis", "lab work"], "interests": ["research", "science", "technology"]},
    "IAS Officer":              {"streams": ["Arts / Humanities", "Commerce", "Science (PCM)"],    "skills": ["communication", "leadership", "writing", "public speaking"], "interests": ["public service", "politics", "governance"]},
    "Product Manager":          {"streams": ["Engineering", "Management", "Computer Science"],     "skills": ["communication", "data analysis", "leadership", "python"], "interests": ["technology", "business", "strategy"]},
    "UX Designer":              {"streams": ["Computer Science", "Arts / Humanities", "Engineering"], "skills": ["design", "figma", "creativity", "communication"], "interests": ["design", "technology", "creativity"]},
    "Content Writer":           {"streams": ["Arts / Humanities", "Commerce"],                     "skills": ["writing", "creativity", "communication", "editing"], "interests": ["writing", "media", "creativity"]},
    "Lawyer":                   {"streams": ["Arts / Humanities"],                                 "skills": ["communication", "writing", "debating", "critical thinking"], "interests": ["law", "justice", "public service"]},
}


def score_career(career: str, weights: dict, profile: dict) -> float:
    score = 0.0
    user_stream = profile.get("stream", "").lower()
    user_skills = [s.lower() for s in profile.get("skills", [])]
    user_interests = [i.lower() for i in profile.get("interests", [])]

    for stream in weights.get("streams", []):
        if stream.lower() in user_stream or user_stream in stream.lower():
            score += 3.0
            break

    for skill in weights.get("skills", []):
        for user_skill in user_skills:
            if skill in user_skill or user_skill in skill:
                score += 2.0

    for interest in weights.get("interests", []):
        for user_interest in user_interests:
            if interest in user_interest or user_interest in interest:
                score += 1.5

    grade = profile.get("grade", "")
    if "90" in grade or "distinction" in grade.lower():
        score *= 1.2
    elif "75" in grade or "first" in grade.lower():
        score *= 1.1

    return round(score, 2)


def get_top_careers(profile: dict, top_n: int = 5) -> list:
    scores = []
    for career, weights in CAREER_WEIGHTS.items():
        s = score_career(career, weights, profile)
        if s > 0:
            scores.append({
                "career": career,
                "score": s,
                "salary": SALARY_RANGES.get(career, {"fresher": "N/A", "mid": "N/A", "senior": "N/A"}),
            })

    scores.sort(key=lambda x: x["score"], reverse=True)
    return scores[:top_n]


def get_career_context(profile: dict) -> str:
    top = get_top_careers(profile, top_n=3)
    if not top:
        return ""
    lines = ["[Rule-based pre-analysis — top career matches:]"]
    for i, item in enumerate(top, 1):
        s = item["salary"]
        lines.append(
            f"{i}. {item['career']} (score: {item['score']}) — "
            f"Fresher: {s['fresher']}, Senior: {s['senior']}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    sample_profile = {
        "stream": "Computer Science",
        "grade": "75-90% (First class)",
        "skills": ["Python", "Machine Learning", "Data Analysis"],
        "interests": ["AI", "Technology", "Research"]
    }
    print("Top career recommendations:")
    for career in get_top_careers(sample_profile):
        print(f"  {career['career']}: score={career['score']}, salary={career['salary']['fresher']}")
