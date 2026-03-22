"""
Career guidance data — streams, careers, exams, and skills database.
Used by the AI system prompt for more accurate guidance.
"""

STREAMS = [
    "Science (PCM)",
    "Science (PCB)",
    "Commerce",
    "Arts / Humanities",
    "Computer Science",
    "Engineering",
    "Management",
    "Medical / Paramedical"
]

CAREER_PATHS = {
    "Science (PCM)": [
        "Software Engineer", "Data Scientist", "AI/ML Engineer",
        "Mechanical Engineer", "Civil Engineer", "Aerospace Engineer",
        "Research Scientist", "Mathematician", "Physicist"
    ],
    "Science (PCB)": [
        "Doctor (MBBS)", "Dentist (BDS)", "Pharmacist",
        "Biotechnologist", "Microbiologist", "Nutritionist",
        "Veterinary Doctor", "Research Scientist", "Physiotherapist"
    ],
    "Commerce": [
        "Chartered Accountant (CA)", "Company Secretary (CS)",
        "Investment Banker", "Financial Analyst", "Business Analyst",
        "Marketing Manager", "HR Manager", "Entrepreneur"
    ],
    "Arts / Humanities": [
        "IAS/IPS Officer (UPSC)", "Lawyer", "Journalist",
        "Psychologist", "Social Worker", "Teacher/Professor",
        "Content Writer", "Historian", "Political Scientist"
    ],
    "Computer Science": [
        "Software Developer", "Full Stack Developer", "DevOps Engineer",
        "Cybersecurity Analyst", "Data Engineer", "Cloud Architect",
        "AI/ML Engineer", "Product Manager", "UX Designer"
    ]
}

ENTRANCE_EXAMS = {
    "Engineering": {
        "exam": "JEE Main & Advanced",
        "conducting_body": "NTA / IIT",
        "frequency": "Twice a year (January & April)",
        "top_colleges": ["IIT Bombay", "IIT Delhi", "IIT Madras", "NIT Trichy", "BITS Pilani"]
    },
    "Medical": {
        "exam": "NEET UG",
        "conducting_body": "NTA",
        "frequency": "Once a year (May)",
        "top_colleges": ["AIIMS Delhi", "JIPMER", "CMC Vellore", "AFMC Pune"]
    },
    "Management": {
        "exam": "CAT / XAT / GMAT",
        "conducting_body": "IIMs / XLRI / GMAC",
        "frequency": "CAT: November every year",
        "top_colleges": ["IIM Ahmedabad", "IIM Bangalore", "IIM Calcutta", "XLRI", "FMS Delhi"]
    },
    "Law": {
        "exam": "CLAT",
        "conducting_body": "Consortium of NLUs",
        "frequency": "Once a year (May)",
        "top_colleges": ["NLSIU Bangalore", "NALSAR Hyderabad", "NLU Delhi", "GNLU Gandhinagar"]
    },
    "Civil Services": {
        "exam": "UPSC CSE",
        "conducting_body": "UPSC",
        "frequency": "Once a year (June Prelims)",
        "top_colleges": ["LBSNAA (IAS Training)", "SVPNPA (IPS Training)"]
    },
    "Higher Engineering": {
        "exam": "GATE",
        "conducting_body": "IITs / IISc",
        "frequency": "Once a year (February)",
        "top_colleges": ["IITs", "NITs", "IISc Bangalore", "IIITs"]
    }
}

SALARY_RANGES = {
    "Software Engineer":     {"fresher": "4-8 LPA",  "mid": "10-20 LPA", "senior": "25-50 LPA"},
    "Data Scientist":        {"fresher": "6-10 LPA", "mid": "15-25 LPA", "senior": "30-60 LPA"},
    "AI/ML Engineer":        {"fresher": "6-12 LPA", "mid": "18-35 LPA", "senior": "40-80 LPA"},
    "Doctor (MBBS)":         {"fresher": "5-8 LPA",  "mid": "12-20 LPA", "senior": "25-60 LPA"},
    "Chartered Accountant":  {"fresher": "6-10 LPA", "mid": "15-25 LPA", "senior": "30-70 LPA"},
    "Investment Banker":     {"fresher": "8-15 LPA", "mid": "20-40 LPA", "senior": "50-150 LPA"},
    "IAS Officer":           {"fresher": "7-8 LPA",  "mid": "12-15 LPA", "senior": "18-25 LPA"},
    "Lawyer":                {"fresher": "3-6 LPA",  "mid": "10-20 LPA", "senior": "25-100 LPA"},
    "Cybersecurity Analyst": {"fresher": "5-10 LPA", "mid": "15-25 LPA", "senior": "30-60 LPA"},
    "Product Manager":       {"fresher": "8-15 LPA", "mid": "20-40 LPA", "senior": "50-120 LPA"},
}

TOP_COLLEGES = {
    "Engineering": ["IIT Bombay", "IIT Delhi", "IIT Madras", "IIT Kanpur", "BITS Pilani", "NIT Trichy", "IIIT Hyderabad"],
    "Medical":     ["AIIMS Delhi", "JIPMER Puducherry", "CMC Vellore", "AFMC Pune", "KMC Manipal"],
    "Management":  ["IIM Ahmedabad", "IIM Bangalore", "IIM Calcutta", "XLRI Jamshedpur", "FMS Delhi"],
    "Law":         ["NLSIU Bangalore", "NALSAR Hyderabad", "NLU Delhi", "GNLU Gandhinagar"],
    "Science":     ["IISc Bangalore", "IIT (Research)", "TIFR Mumbai", "IISER Pune"],
    "Arts":        ["JNU Delhi", "DU Delhi", "BHU Varanasi", "Hyderabad Central University"],
}

SYSTEM_PROMPT = """You are an expert AI career guidance counselor for Indian students with deep knowledge of:
- Indian and global career paths, job roles, and industries
- Top universities and colleges (IITs, NITs, IIMs, AIIMS, BITS Pilani, top private universities)
- Current Indian job market trends, salary ranges (in LPA), and demand forecasts for 2025
- Certifications and online courses (Coursera, edX, NPTEL, Udemy, LinkedIn Learning)
- Indian entrance exams: JEE Main/Advanced, NEET, CAT, CLAT, UPSC CSE, GATE, XAT, GMAT, GRE
- Emerging fields: AI/ML, Data Science, Cybersecurity, Cloud Computing, UX Design, FinTech, EdTech

When answering:
1. Give concrete, specific career recommendations — never vague advice
2. For salary: always provide Indian LPA ranges (fresher / mid-level / senior)
3. For colleges: name specific top institutions with their strengths
4. For skills: provide a clear priority-ordered learning roadmap with realistic timelines
5. For exams: mention eligibility, exam pattern, key dates, and preparation tips
6. Structure responses clearly using markdown with headings and bullet points
7. Be honest about competition levels and realistic timelines
8. Always consider the student profile (stream, grades, skills, interests) when provided
9. End every response with 2-3 concrete, actionable next steps
10. Keep responses focused and practical — students need actionable guidance"""
