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


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    query = data.get("query", "")
    profile = data.get("profile", {})
    query_analysis = classify_query(query)
    career_context = get_career_context(profile)
    return jsonify({
        "query_analysis": query_analysis,
        "career_context": career_context
    })


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        messages = data.get("messages", [])
        profile = data.get("profile", {})

        if not validate_messages(messages):
            return jsonify({"error": "Invalid or empty messages"}), 400

        last_query = messages[-1].get("content", "")
        enhanced_query = build_enhanced_prompt(last_query, profile)
        messages[-1]["content"] = enhanced_query

        profile_context = build_profile_context(profile)
        if profile_context:
            messages[-1]["content"] += profile_context

        career_context = get_career_context(profile)
        if career_context:
            messages[-1]["content"] += "\n\n" + career_context

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
