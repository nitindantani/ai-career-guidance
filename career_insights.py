"""
Career Insights Engine
Generates detailed career insights, skill gap analysis,
salary predictions, and learning roadmaps using data science techniques.
"""

import os
import json
import math
import random
from datetime import datetime
from collections import defaultdict, Counter


# ── Career Knowledge Base ──────────────────────────────────────────────────

CAREER_SKILLS_MAP = {
    "Software Engineer": {
        "core": ["Python", "Data Structures", "Algorithms", "Git", "SQL"],
        "advanced": ["System Design", "Docker", "AWS", "Kubernetes", "CI/CD"],
        "soft": ["Problem Solving", "Communication", "Teamwork"],
        "salary": {"fresher": (4, 8), "mid": (10, 20), "senior": (25, 50), "lead": (40, 80)},
        "demand": 0.92, "growth": 0.18,
    },
    "Data Scientist": {
        "core": ["Python", "Statistics", "Machine Learning", "SQL", "Data Visualization"],
        "advanced": ["Deep Learning", "NLP", "TensorFlow", "PyTorch", "Spark"],
        "soft": ["Analytical Thinking", "Communication", "Curiosity"],
        "salary": {"fresher": (6, 10), "mid": (15, 25), "senior": (30, 60), "lead": (50, 100)},
        "demand": 0.95, "growth": 0.22,
    },
    "AI/ML Engineer": {
        "core": ["Python", "Machine Learning", "Deep Learning", "Mathematics", "Statistics"],
        "advanced": ["LLMs", "Computer Vision", "MLOps", "CUDA", "Transformers"],
        "soft": ["Research Mindset", "Problem Solving", "Continuous Learning"],
        "salary": {"fresher": (8, 14), "mid": (18, 35), "senior": (40, 80), "lead": (60, 120)},
        "demand": 0.98, "growth": 0.35,
    },
    "Full Stack Developer": {
        "core": ["Python/Node.js", "React", "HTML/CSS", "SQL", "REST APIs"],
        "advanced": ["TypeScript", "GraphQL", "Redis", "AWS", "Microservices"],
        "soft": ["Creativity", "Attention to Detail", "Collaboration"],
        "salary": {"fresher": (5, 9), "mid": (12, 22), "senior": (25, 45), "lead": (40, 70)},
        "demand": 0.90, "growth": 0.15,
    },
    "Cybersecurity Analyst": {
        "core": ["Networking", "Linux", "Python", "Security Protocols", "Ethical Hacking"],
        "advanced": ["Penetration Testing", "SIEM", "Threat Intelligence", "Cloud Security"],
        "soft": ["Analytical Thinking", "Attention to Detail", "Integrity"],
        "salary": {"fresher": (5, 10), "mid": (15, 25), "senior": (30, 60), "lead": (50, 90)},
        "demand": 0.93, "growth": 0.28,
    },
    "Product Manager": {
        "core": ["Product Strategy", "Data Analysis", "Communication", "Agile", "User Research"],
        "advanced": ["OKRs", "A/B Testing", "SQL", "Product Analytics", "Roadmapping"],
        "soft": ["Leadership", "Empathy", "Decision Making", "Storytelling"],
        "salary": {"fresher": (8, 15), "mid": (20, 40), "senior": (40, 80), "lead": (60, 120)},
        "demand": 0.88, "growth": 0.20,
    },
    "DevOps Engineer": {
        "core": ["Linux", "Docker", "Kubernetes", "CI/CD", "Python/Bash"],
        "advanced": ["Terraform", "AWS/GCP/Azure", "Ansible", "Prometheus", "GitOps"],
        "soft": ["Problem Solving", "Collaboration", "Reliability"],
        "salary": {"fresher": (6, 11), "mid": (14, 24), "senior": (28, 50), "lead": (45, 80)},
        "demand": 0.91, "growth": 0.25,
    },
    "Financial Analyst": {
        "core": ["Excel", "Financial Modeling", "Accounting", "Statistics", "SQL"],
        "advanced": ["Python", "Bloomberg", "Power BI", "Valuation", "Risk Analysis"],
        "soft": ["Analytical Thinking", "Attention to Detail", "Communication"],
        "salary": {"fresher": (4, 8), "mid": (10, 20), "senior": (22, 45), "lead": (40, 70)},
        "demand": 0.82, "growth": 0.10,
    },
    "Cloud Architect": {
        "core": ["AWS/GCP/Azure", "Networking", "Security", "Python", "Infrastructure"],
        "advanced": ["Multi-cloud", "Cost Optimization", "Serverless", "Edge Computing"],
        "soft": ["Strategic Thinking", "Communication", "Leadership"],
        "salary": {"fresher": (8, 14), "mid": (20, 38), "senior": (40, 75), "lead": (65, 120)},
        "demand": 0.94, "growth": 0.30,
    },
    "Blockchain Developer": {
        "core": ["Solidity", "Python", "Smart Contracts", "Web3", "Cryptography"],
        "advanced": ["DeFi", "NFTs", "Layer 2", "Consensus Algorithms", "Security Auditing"],
        "soft": ["Innovation Mindset", "Problem Solving", "Self-learning"],
        "salary": {"fresher": (7, 13), "mid": (18, 35), "senior": (35, 70), "lead": (55, 110)},
        "demand": 0.78, "growth": 0.22,
    },
}

CERTIFICATION_PATHS = {
    "Python": ["Python Institute PCAP", "Google IT Automation with Python", "Kaggle Python"],
    "Machine Learning": ["Google ML Crash Course", "Coursera ML Specialization (Andrew Ng)", "fast.ai"],
    "Deep Learning": ["deeplearning.ai Specialization", "PyTorch Official Tutorials", "Hugging Face Course"],
    "AWS": ["AWS Cloud Practitioner", "AWS Solutions Architect Associate", "AWS Developer Associate"],
    "Data Science": ["IBM Data Science Professional", "Kaggle Data Scientist", "DataCamp Career Track"],
    "Cybersecurity": ["CompTIA Security+", "CEH (Certified Ethical Hacker)", "CISSP"],
    "DevOps": ["Docker Certified Associate", "Kubernetes CKA", "AWS DevOps Professional"],
    "SQL": ["Mode SQL Tutorial", "PostgreSQL Official Docs", "LeetCode SQL Problems"],
}

LEARNING_RESOURCES = {
    "free": ["Coursera (audit)", "edX (audit)", "MIT OpenCourseWare", "NPTEL", "YouTube", "freeCodeCamp", "Kaggle", "fast.ai"],
    "paid": ["Udemy", "Pluralsight", "LinkedIn Learning", "O'Reilly", "DataCamp"],
    "practice": ["LeetCode", "HackerRank", "Kaggle Competitions", "GitHub Projects", "Codeforces"],
    "indian": ["NPTEL", "SWAYAM", "IIT Bombay OCW", "IITM BS Degree Online"],
}


# ── Core Analysis Functions ────────────────────────────────────────────────

def compute_skill_match_score(user_skills: list, career: str) -> dict:
    """
    Compute how well a user's skills match a career.
    Returns match percentage and gap analysis.
    """
    if career not in CAREER_SKILLS_MAP:
        return {"error": f"Career '{career}' not found"}

    career_data = CAREER_SKILLS_MAP[career]
    core_skills = [s.lower() for s in career_data["core"]]
    adv_skills = [s.lower() for s in career_data["advanced"]]
    user_lower = [s.lower() for s in user_skills]

    core_matched = [s for s in core_skills if any(u in s or s in u for u in user_lower)]
    adv_matched = [s for s in adv_skills if any(u in s or s in u for u in user_lower)]

    core_score = len(core_matched) / len(core_skills) if core_skills else 0
    adv_score = len(adv_matched) / len(adv_skills) if adv_skills else 0

    # weighted: core is 70%, advanced is 30%
    total_score = round((core_score * 0.7 + adv_score * 0.3) * 100, 1)

    missing_core = [s for s in career_data["core"] if s.lower() not in [m for m in core_matched]]
    missing_adv = [s for s in career_data["advanced"] if s.lower() not in [m for m in adv_matched]]

    return {
        "career": career,
        "match_score": total_score,
        "core_matched": len(core_matched),
        "core_total": len(core_skills),
        "advanced_matched": len(adv_matched),
        "advanced_total": len(adv_skills),
        "missing_core_skills": missing_core,
        "missing_advanced_skills": missing_adv[:3],
        "readiness": get_readiness_level(total_score),
    }


def get_readiness_level(score: float) -> str:
    if score >= 80:
        return "Job Ready"
    elif score >= 60:
        return "Nearly Ready — 2-3 skills to add"
    elif score >= 40:
        return "In Progress — 3-6 months of learning"
    elif score >= 20:
        return "Beginner — 6-12 months roadmap needed"
    else:
        return "Starting Out — Full roadmap needed"


def estimate_salary(career: str, experience_years: int, skills_score: float) -> dict:
    """
    Estimate salary range based on career, experience, and skill score.
    Uses a weighted formula with skill multiplier.
    """
    if career not in CAREER_SKILLS_MAP:
        return {}

    salary_data = CAREER_SKILLS_MAP[career]["salary"]
    skill_multiplier = 0.8 + (skills_score / 100) * 0.4  # 0.8x to 1.2x based on skills

    if experience_years == 0:
        base_min, base_max = salary_data["fresher"]
        level = "Fresher"
    elif experience_years <= 3:
        base_min, base_max = salary_data["mid"]
        level = "Mid-level"
    elif experience_years <= 7:
        base_min, base_max = salary_data["senior"]
        level = "Senior"
    else:
        base_min, base_max = salary_data["lead"]
        level = "Lead/Principal"

    estimated_min = round(base_min * skill_multiplier, 1)
    estimated_max = round(base_max * skill_multiplier, 1)

    return {
        "career": career,
        "level": level,
        "experience_years": experience_years,
        "estimated_min_lpa": estimated_min,
        "estimated_max_lpa": estimated_max,
        "range": f"{estimated_min} - {estimated_max} LPA",
        "monthly_min": round(estimated_min * 100000 / 12),
        "monthly_max": round(estimated_max * 100000 / 12),
        "skill_multiplier": round(skill_multiplier, 2),
        "note": "Indian market estimate. Varies by company, city, and negotiation."
    }


def generate_learning_roadmap(user_skills: list, target_career: str, months_available: int = 12) -> dict:
    """
    Generate a month-by-month learning roadmap to reach a target career.
    """
    if target_career not in CAREER_SKILLS_MAP:
        return {"error": "Career not found"}

    gap = compute_skill_match_score(user_skills, target_career)
    career_data = CAREER_SKILLS_MAP[target_career]

    missing_core = gap["missing_core_skills"]
    missing_adv = gap["missing_advanced_skills"]
    all_missing = missing_core + missing_adv

    skills_per_month = max(1, math.ceil(len(all_missing) / months_available))

    phases = []
    remaining = list(all_missing)

    phase_months = [
        (1, min(3, months_available), "Foundation", missing_core[:3]),
        (4, min(7, months_available), "Core Skills", missing_core[3:] + missing_adv[:2]),
        (8, months_available, "Advanced & Projects", missing_adv[2:]),
    ]

    for start, end, name, skills in phase_months:
        if start > months_available:
            break
        if skills:
            certs = []
            for skill in skills[:2]:
                if skill in CERTIFICATION_PATHS:
                    certs.extend(CERTIFICATION_PATHS[skill][:1])
            phases.append({
                "phase": name,
                "months": f"Month {start}–{min(end, months_available)}",
                "skills_to_learn": skills,
                "certifications": certs,
                "resources": LEARNING_RESOURCES["free"][:3],
                "milestone": f"Complete {len(skills)} skill(s), build 1 project"
            })

    return {
        "target_career": target_career,
        "current_match": gap["match_score"],
        "months_available": months_available,
        "total_skills_to_learn": len(all_missing),
        "current_readiness": gap["readiness"],
        "phases": phases,
        "final_goal": f"Achieve {min(gap['match_score'] + 40, 95)}%+ match score for {target_career}",
        "practice_platforms": LEARNING_RESOURCES["practice"],
        "indian_resources": LEARNING_RESOURCES["indian"],
    }


def rank_careers_for_profile(user_skills: list, user_interests: list, stream: str = "") -> list:
    """
    Rank all careers by fit score for a given user profile.
    Combines skill match + interest alignment + market demand.
    """
    results = []

    interest_lower = [i.lower() for i in user_interests]

    for career, data in CAREER_SKILLS_MAP.items():
        skill_result = compute_skill_match_score(user_skills, career)
        skill_score = skill_result.get("match_score", 0)

        interest_bonus = 0
        career_lower = career.lower()
        for interest in interest_lower:
            if interest in career_lower or any(interest in s.lower() for s in data["core"]):
                interest_bonus += 10

        demand_score = data["demand"] * 100
        growth_score = data["growth"] * 100

        final_score = round(
            skill_score * 0.50 +
            interest_bonus * 0.20 +
            demand_score * 0.15 +
            growth_score * 0.15,
            1
        )

        salary = data["salary"]["fresher"]
        results.append({
            "career": career,
            "fit_score": final_score,
            "skill_match": skill_score,
            "market_demand": f"{int(demand_score)}%",
            "growth_rate": f"{int(data['growth'] * 100)}% YoY",
            "fresher_salary": f"{salary[0]}–{salary[1]} LPA",
            "readiness": skill_result.get("readiness", ""),
            "top_missing_skills": skill_result.get("missing_core_skills", [])[:3],
        })

    results.sort(key=lambda x: x["fit_score"], reverse=True)
    return results[:8]


def analyze_market_trends() -> dict:
    """
    Analyze current Indian job market trends for tech careers.
    Returns demand scores, growth rates, and hot skills.
    """
    trends = {}
    for career, data in CAREER_SKILLS_MAP.items():
        trends[career] = {
            "demand_score": f"{int(data['demand'] * 100)}%",
            "yoy_growth": f"+{int(data['growth'] * 100)}%",
            "hot_skills": data["core"][:3],
            "salary_range_fresher": f"{data['salary']['fresher'][0]}–{data['salary']['fresher'][1]} LPA",
            "salary_range_senior": f"{data['salary']['senior'][0]}–{data['salary']['senior'][1]} LPA",
        }

    sorted_by_demand = sorted(
        trends.items(),
        key=lambda x: float(x[1]["demand_score"].replace("%", "")),
        reverse=True
    )

    hot_skills_counter = Counter()
    for data in CAREER_SKILLS_MAP.values():
        for skill in data["core"]:
            hot_skills_counter[skill] += 1

    return {
        "as_of": datetime.now().strftime("%B %Y"),
        "top_careers_by_demand": [c for c, _ in sorted_by_demand[:5]],
        "hottest_skills": [skill for skill, _ in hot_skills_counter.most_common(10)],
        "career_details": dict(sorted_by_demand),
        "emerging_fields": ["AI/ML Engineering", "Cloud Architecture", "Cybersecurity", "Blockchain"],
        "average_salary_growth": "18-22% YoY for tech roles in India",
        "top_hiring_cities": ["Bangalore", "Hyderabad", "Pune", "Mumbai", "Delhi NCR", "Chennai"],
    }


def generate_career_report(user_profile: dict) -> dict:
    """
    Generate a comprehensive career report for a user profile.
    This is the main function called by the Flask app.
    """
    name = user_profile.get("name", "Student")
    stream = user_profile.get("stream", "")
    skills = user_profile.get("skills", [])
    interests = user_profile.get("interests", [])
    experience = user_profile.get("experience_years", 0)
    target = user_profile.get("target_career", "")

    ranked = rank_careers_for_profile(skills, interests, stream)
    top_career = ranked[0]["career"] if ranked else "Software Engineer"
    analysis = compute_skill_match_score(skills, top_career)
    salary = estimate_salary(top_career, experience, analysis.get("match_score", 50))
    roadmap = generate_learning_roadmap(skills, top_career, months_available=12)
    trends = analyze_market_trends()

    return {
        "generated_at": datetime.now().isoformat(),
        "profile_summary": {
            "stream": stream,
            "skills_count": len(skills),
            "interests": interests,
            "experience_years": experience,
        },
        "top_career_matches": ranked[:5],
        "primary_recommendation": {
            "career": top_career,
            "skill_analysis": analysis,
            "salary_estimate": salary,
        },
        "learning_roadmap": roadmap,
        "market_insights": {
            "hottest_skills": trends["hottest_skills"][:5],
            "top_careers_by_demand": trends["top_careers_by_demand"],
            "top_cities": trends["top_hiring_cities"],
        },
        "action_items": [
            f"Focus on learning: {', '.join(analysis.get('missing_core_skills', [])[:3])}",
            f"Target companies: TCS, Infosys, Wipro (entry) → Flipkart, Swiggy → Google, Microsoft (senior)",
            f"Build 2-3 projects on GitHub to showcase {top_career} skills",
            f"Prepare for interviews using LeetCode (DSA) and system design resources",
        ]
    }


def export_report_json(user_profile: dict, filename: str = "career_report.json") -> str:
    """Export career report as a JSON file."""
    report = generate_career_report(user_profile)
    filepath = os.path.join(os.path.dirname(__file__), filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return filepath


# ── CLI Interface ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  CareerAI — Career Insights Engine")
    print("=" * 60)

    sample_profile = {
        "name": "Nitin Dantani",
        "stream": "Computer Science",
        "skills": ["Python", "Machine Learning", "Data Analysis", "Excel", "SQL"],
        "interests": ["AI", "Technology", "Data Science"],
        "experience_years": 0,
        "target_career": "AI/ML Engineer"
    }

    print("\n1. SKILL MATCH ANALYSIS")
    print("-" * 40)
    match = compute_skill_match_score(sample_profile["skills"], "AI/ML Engineer")
    print(f"   Career: {match['career']}")
    print(f"   Match Score: {match['match_score']}%")
    print(f"   Readiness: {match['readiness']}")
    print(f"   Missing Core Skills: {match['missing_core_skills']}")

    print("\n2. SALARY ESTIMATE")
    print("-" * 40)
    salary = estimate_salary("AI/ML Engineer", 0, match["match_score"])
    print(f"   Level: {salary['level']}")
    print(f"   Estimated Range: {salary['range']}")
    print(f"   Monthly: ₹{salary['monthly_min']:,} – ₹{salary['monthly_max']:,}")

    print("\n3. TOP CAREER MATCHES")
    print("-" * 40)
    ranked = rank_careers_for_profile(
        sample_profile["skills"],
        sample_profile["interests"],
        sample_profile["stream"]
    )
    for i, c in enumerate(ranked[:5], 1):
        print(f"   {i}. {c['career']} — Fit: {c['fit_score']}% | Salary: {c['fresher_salary']}")

    print("\n4. MARKET TRENDS")
    print("-" * 40)
    trends = analyze_market_trends()
    print(f"   Hottest Skills: {', '.join(trends['hottest_skills'][:5])}")
    print(f"   Top Demand Careers: {', '.join(trends['top_careers_by_demand'][:3])}")
    print(f"   Top Hiring Cities: {', '.join(trends['top_hiring_cities'][:4])}")

    print("\n5. LEARNING ROADMAP")
    print("-" * 40)
    roadmap = generate_learning_roadmap(sample_profile["skills"], "AI/ML Engineer", 12)
    print(f"   Current Match: {roadmap['current_match']}%")
    print(f"   Skills to Learn: {roadmap['total_skills_to_learn']}")
    for phase in roadmap["phases"]:
        print(f"   {phase['months']} — {phase['phase']}: {phase['skills_to_learn']}")

    print("\n" + "=" * 60)
    print("  Full report saved to career_report.json")
    print("=" * 60)

    export_report_json(sample_profile)
