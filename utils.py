"""
Utility functions for the CareerAI application.
"""

def build_profile_context(profile: dict) -> str:
    """Build a profile context string from user profile data."""
    parts = []
    if profile.get("stream"):
        parts.append(f"Stream: {profile['stream']}")
    if profile.get("grade"):
        parts.append(f"Academic performance: {profile['grade']}")
    if profile.get("workexp"):
        parts.append(f"Work experience: {profile['workexp']}")
    if profile.get("skills"):
        parts.append(f"Skills: {', '.join(profile['skills'])}")
    if profile.get("interests"):
        parts.append(f"Interests: {', '.join(profile['interests'])}")
    if parts:
        return "\n\n[Student Profile: " + " | ".join(parts) + "]"
    return ""


def validate_messages(messages: list) -> bool:
    """Validate that messages list is properly formatted."""
    if not messages or not isinstance(messages, list):
        return False
    for msg in messages:
        if not isinstance(msg, dict):
            return False
        if "role" not in msg or "content" not in msg:
            return False
        if msg["role"] not in ["user", "assistant"]:
            return False
    return True


def format_salary(career: str, salary_ranges: dict) -> str:
    """Format salary information for a given career."""
    if career in salary_ranges:
        s = salary_ranges[career]
        return f"Fresher: {s['fresher']} | Mid: {s['mid']} | Senior: {s['senior']}"
    return "Salary varies based on company and experience"


def get_careers_for_stream(stream: str, career_paths: dict) -> list:
    """Get list of career options for a given stream."""
    for key in career_paths:
        if key.lower() in stream.lower():
            return career_paths[key]
    return []
