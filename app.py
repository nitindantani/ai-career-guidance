from flask import Flask, request, render_template, jsonify, stream_with_context, Response
import os
import json
from groq import Groq
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)

SYSTEM_PROMPT = "You are an expert AI career guidance counselor with deep knowledge of Indian and global career paths, top universities, salary ranges, certifications, and entrance exams like JEE, NEET, CAT, UPSC, GATE. Always give specific recommendations, realistic salary ranges, named colleges, and end with 2-3 actionable next steps."

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({"status": "ok", "api_key_set": bool(api_key)})

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        messages = data.get("messages", [])
        profile = data.get("profile", {})
        if not messages:
            return jsonify({"error": "No messages provided"}), 400
        profile_parts = []
        if profile.get("stream"):
            profile_parts.append("Stream: " + profile["stream"])
        if profile.get("grade"):
            profile_parts.append("Grade: " + profile["grade"])
        if profile.get("workexp"):
            profile_parts.append("Experience: " + profile["workexp"])
        if profile.get("skills"):
            profile_parts.append("Skills: " + ", ".join(profile["skills"]))
        if profile.get("interests"):
            profile_parts.append("Interests: " + ", ".join(profile["interests"]))
        if profile_parts:
            messages[-1]["content"] += "\n\n[Profile: " + " | ".join(profile_parts) + "]"
        groq_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
        def generate():
            try:
                stream = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=groq_messages,
                    max_tokens=1024,
                    stream=True
                )
                for chunk in stream:
                    text = chunk.choices[0].delta.content
                    if text:
                        yield "data: " + json.dumps({"text": text}) + "\n\n"
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
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)