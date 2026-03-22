from flask import Flask, request, render_template, jsonify, stream_with_context, Response
import os
import json
import logging
from groq import Groq
from config import Config
from career_data import SYSTEM_PROMPT
from utils import build_profile_context, validate_messages
from career_recommender import get_career_context
from nlp_processor import classify_query, build_enhanced_prompt
from model_evaluator import evaluate_response_quality, get_model_summary
from career_insights import (
    compute_skill_match_score,
    rank_careers_for_profile,
    estimate_salary,
    generate_learning_roadmap,
    generate_career_report,
    analyze_market_trends
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
client = Groq(api_key=Config.GROQ_API_KEY)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "app": Config.APP_NAME,
        "version": Config.VERSION,
        "model": Config.MODEL_NAME,
        "api_key_set": bool(Config.GROQ_API_KEY),
        "model_summary": get_model_summary()
    })


@app.route("/insights", methods=["POST"])
def insights():
    """
    Returns pre-computed career insights for the user profile.
    Used to enrich AI responses with data-driven analysis.
    """
    try:
        data = request.get_json()
        profile = data.get("profile", {})
        skills = profile.get("skills", [])
        interests = profile.get("interests", [])
        stream = profile.get("stream", "")
        experience = 0
        workexp = profile.get("workexp", "")
        if "1-3" in workexp:
            experience = 2
        elif "3-7" in workexp:
            experience = 5
        elif "7+" in workexp:
            experience = 8

        ranked = rank_careers_for_profile(skills, interests, stream)
        top_career = ranked[0]["career"] if ranked else "Software Engineer"
        skill_analysis = compute_skill_match_score(skills, top_career)
        salary = estimate_salary(top_career, experience, skill_analysis.get("match_score", 50))
        trends = analyze_market_trends()

        return jsonify({
            "top_career_matches": ranked[:5],
            "skill_analysis": skill_analysis,
            "salary_estimate": salary,
            "hottest_skills": trends["hottest_skills"][:6],
            "top_demand_careers": trends["top_careers_by_demand"][:5],
        })
    except Exception as e:
        logger.error(f"Insights error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/roadmap", methods=["POST"])
def roadmap():
    """Returns a personalized learning roadmap for the user."""
    try:
        data = request.get_json()
        profile = data.get("profile", {})
        target = data.get("target_career", "")
        skills = profile.get("skills", [])
        months = data.get("months", 12)

        if not target:
            ranked = rank_careers_for_profile(skills, profile.get("interests", []))
            target = ranked[0]["career"] if ranked else "Software Engineer"

        result = generate_learning_roadmap(skills, target, months)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Roadmap error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/report", methods=["POST"])
def report():
    """Generates a full career report for the user profile."""
    try:
        data = request.get_json()
        profile = data.get("profile", {})
        result = generate_career_report(profile)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Report error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        messages = data.get("messages", [])
        profile = data.get("profile", {})

        if not validate_messages(messages):
            return jsonify({"error": "Invalid or empty messages"}), 400

        # Step 1: NLP query analysis
        last_query = messages[-1].get("content", "")
        enhanced_query = build_enhanced_prompt(last_query, profile)
        messages[-1]["content"] = enhanced_query

        # Step 2: Add user profile context
        profile_context = build_profile_context(profile)
        if profile_context:
            messages[-1]["content"] += profile_context

        # Step 3: Add rule-based career recommendations
        career_context = get_career_context(profile)
        if career_context:
            messages[-1]["content"] += "\n\n" + career_context

        # Step 4: Add data-driven insights from career_insights.py
        skills = profile.get("skills", [])
        interests = profile.get("interests", [])
        if skills or interests:
            try:
                ranked = rank_careers_for_profile(skills, interests)
                if ranked:
                    top = ranked[0]
                    skill_gap = compute_skill_match_score(skills, top["career"])
                    insights_context = (
                        f"\n\n[Data-driven insights: "
                        f"Top career match = {top['career']} ({top['fit_score']}% fit) | "
                        f"Skill match = {skill_gap['match_score']}% | "
                        f"Readiness = {skill_gap['readiness']} | "
                        f"Missing skills = {', '.join(skill_gap['missing_core_skills'][:3])}]"
                    )
                    messages[-1]["content"] += insights_context
            except Exception as e:
                logger.warning(f"Insights injection skipped: {e}")

        groq_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

        def generate():
            try:
                full_response = []
                stream = client.chat.completions.create(
                    model=Config.MODEL_NAME,
                    messages=groq_messages,
                    max_tokens=Config.MAX_TOKENS,
                    stream=True
                )
                for chunk in stream:
                    text = chunk.choices[0].delta.content
                    if text:
                        full_response.append(text)
                        yield "data: " + json.dumps({"text": text}) + "\n\n"

                full_text = "".join(full_response)
                quality = evaluate_response_quality(full_text)
                logger.info(f"Response quality: {quality['quality_score']}, words: {quality['word_count']}")
                yield "data: [DONE]\n\n"

            except Exception as e:
                logger.error(f"Stream error: {str(e)}")
                yield "data: " + json.dumps({"text": "Error: " + str(e)}) + "\n\n"
                yield "data: [DONE]\n\n"

        return Response(stream_with_context(generate()), mimetype="text/event-stream")

    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=Config.PORT, debug=Config.DEBUG)