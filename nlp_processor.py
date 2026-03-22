"""
NLP text processor for career guidance queries.
Handles intent detection, keyword extraction, and query classification
without external NLP libraries — pure Python implementation.
"""

import re
import string

INTENT_KEYWORDS = {
    "salary":       ["salary", "pay", "earn", "income", "lpa", "package", "ctc", "stipend", "compensation"],
    "colleges":     ["college", "university", "institute", "iit", "nit", "iim", "aiims", "admission", "campus"],
    "skills":       ["skill", "learn", "course", "certification", "study", "training", "roadmap", "technology"],
    "careers":      ["career", "job", "role", "profession", "field", "option", "scope", "opportunity", "work"],
    "exams":        ["exam", "entrance", "jee", "neet", "cat", "upsc", "gate", "clat", "gmat", "gre", "test"],
    "roadmap":      ["roadmap", "path", "plan", "future", "timeline", "steps", "journey", "year"],
    "comparison":   ["compare", "better", "vs", "difference", "between", "which", "best", "choose"],
    "internship":   ["internship", "intern", "placement", "fresher", "experience", "campus", "hire"],
}

STREAM_KEYWORDS = {
    "Science (PCM)":          ["pcm", "physics", "chemistry", "maths", "math", "jee", "engineering"],
    "Science (PCB)":          ["pcb", "biology", "neet", "medical", "doctor", "mbbs"],
    "Commerce":               ["commerce", "accounts", "economics", "ca", "cs", "finance"],
    "Arts / Humanities":      ["arts", "humanities", "history", "political", "literature", "upsc"],
    "Computer Science":       ["computer", "cs", "coding", "programming", "software", "it"],
    "Engineering":            ["engineering", "btech", "be", "mechanical", "civil", "electrical"],
    "Management":             ["management", "mba", "bba", "business", "cat", "iim"],
    "Medical / Paramedical":  ["medical", "mbbs", "bds", "pharmacy", "nursing", "paramedical"],
}

SKILL_KEYWORDS = [
    "python", "java", "javascript", "react", "node", "sql", "machine learning",
    "deep learning", "data science", "tensorflow", "pytorch", "excel", "power bi",
    "tableau", "aws", "azure", "docker", "kubernetes", "git", "html", "css",
    "communication", "leadership", "teamwork", "problem solving", "critical thinking"
]


def preprocess(text: str) -> str:
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation.replace("-", "")))
    text = re.sub(r"\s+", " ", text)
    return text


def detect_intents(query: str) -> list:
    clean = preprocess(query)
    found = []
    for intent, keywords in INTENT_KEYWORDS.items():
        if any(kw in clean for kw in keywords):
            found.append(intent)
    return found if found else ["careers"]


def detect_stream(query: str) -> str:
    clean = preprocess(query)
    for stream, keywords in STREAM_KEYWORDS.items():
        if any(kw in clean for kw in keywords):
            return stream
    return ""


def extract_skills(query: str) -> list:
    clean = preprocess(query)
    return [skill for skill in SKILL_KEYWORDS if skill in clean]


def extract_keywords(query: str) -> list:
    clean = preprocess(query)
    stop_words = {"what", "how", "when", "where", "which", "who", "is", "are",
                  "the", "a", "an", "in", "for", "to", "of", "my", "i", "me",
                  "can", "do", "does", "should", "want", "need", "tell", "give"}
    words = clean.split()
    return [w for w in words if w not in stop_words and len(w) > 2]


def classify_query(query: str) -> dict:
    return {
        "original": query,
        "intents": detect_intents(query),
        "stream": detect_stream(query),
        "skills": extract_skills(query),
        "keywords": extract_keywords(query)[:10],
        "word_count": len(query.split()),
        "is_comparison": "compare" in query.lower() or " vs " in query.lower(),
        "is_question": query.strip().endswith("?") or query.lower().startswith(("what", "how", "when", "where", "which", "who")),
    }


def build_enhanced_prompt(query: str, profile: dict) -> str:
    analysis = classify_query(query)
    intents = ", ".join(analysis["intents"])
    parts = [f"User query intent: {intents}"]
    if analysis["stream"]:
        parts.append(f"Detected stream from query: {analysis['stream']}")
    if analysis["skills"]:
        parts.append(f"Skills mentioned in query: {', '.join(analysis['skills'])}")
    if parts:
        return query + "\n\n[Query analysis: " + " | ".join(parts) + "]"
    return query


if __name__ == "__main__":
    test_queries = [
        "What is the salary for a data scientist in India?",
        "Which IIT is best for computer science?",
        "How do I prepare for JEE Advanced?",
        "Compare software engineer vs data scientist career",
    ]
    for q in test_queries:
        result = classify_query(q)
        print(f"Query: {q}")
        print(f"  Intents: {result['intents']}")
        print(f"  Stream: {result['stream']}")
        print(f"  Skills: {result['skills']}")
        print()
