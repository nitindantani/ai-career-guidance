"""
Career data analyzer — reads career_data.csv and provides
statistics, insights, and trend analysis.
Demonstrates data science skills using pandas and numpy.
"""

import os
import pandas as pd
import numpy as np
from collections import Counter


CSV_PATH = os.path.join(os.path.dirname(__file__), "career_data.csv")


def load_data() -> pd.DataFrame:
    try:
        df = pd.read_csv(CSV_PATH)
        df = df.apply(lambda col: col.str.lower().str.strip() if col.dtype == "object" else col)
        return df
    except FileNotFoundError:
        return pd.DataFrame()


def get_career_distribution(df: pd.DataFrame) -> dict:
    if df.empty or "career_label" not in df.columns:
        return {}
    counts = df["career_label"].value_counts()
    return counts.to_dict()


def get_stream_career_matrix(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    result = {}
    for stream in df["stream"].unique():
        subset = df[df["stream"] == stream]
        result[stream] = subset["career_label"].value_counts().head(3).to_dict()
    return result


def get_top_skills(df: pd.DataFrame, top_n: int = 10) -> list:
    if df.empty or "skills" not in df.columns:
        return []
    all_skills = df["skills"].dropna().tolist()
    skill_words = []
    for s in all_skills:
        skill_words.extend([word.strip() for word in str(s).split(",")])
    counter = Counter(skill_words)
    return counter.most_common(top_n)


def get_dataset_summary(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"error": "No data found"}
    return {
        "total_records": len(df),
        "unique_careers": df["career_label"].nunique() if "career_label" in df.columns else 0,
        "unique_streams": df["stream"].nunique() if "stream" in df.columns else 0,
        "unique_skills": df["skills"].nunique() if "skills" in df.columns else 0,
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
    }


def get_skill_career_correlation(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    result = {}
    for career in df["career_label"].unique():
        subset = df[df["career_label"] == career]
        skills = subset["skills"].value_counts().head(3).index.tolist()
        result[career] = skills
    return result


def analyze_soft_skills(df: pd.DataFrame) -> dict:
    if df.empty or "soft_skill" not in df.columns:
        return {}
    return df["soft_skill"].value_counts().head(10).to_dict()


def get_full_analysis() -> dict:
    df = load_data()
    return {
        "summary": get_dataset_summary(df),
        "career_distribution": get_career_distribution(df),
        "stream_career_matrix": get_stream_career_matrix(df),
        "top_skills": get_top_skills(df),
        "skill_career_map": get_skill_career_correlation(df),
        "top_soft_skills": analyze_soft_skills(df),
    }


def get_recommendations_from_data(stream: str, skill: str) -> list:
    df = load_data()
    if df.empty:
        return []
    filtered = df[
        (df["stream"].str.contains(stream.lower(), na=False)) |
        (df["skills"].str.contains(skill.lower(), na=False))
    ]
    if filtered.empty:
        return df["career_label"].value_counts().head(3).index.tolist()
    return filtered["career_label"].value_counts().head(3).index.tolist()


if __name__ == "__main__":
    analysis = get_full_analysis()
    print("Dataset Summary:")
    for k, v in analysis["summary"].items():
        print(f"  {k}: {v}")
    print("\nCareer Distribution:")
    for career, count in list(analysis["career_distribution"].items())[:5]:
        print(f"  {career}: {count}")
    print("\nTop Skills:")
    for skill, count in analysis["top_skills"][:5]:
        print(f"  {skill}: {count}")
