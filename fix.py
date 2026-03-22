import os

app_content = """from flask import Flask, request, render_template, jsonify, stream_with_context, Response
import os
import json
import anthropic

app = Flask(__name__)
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
SYSTEM_PROMPT = "You are an expert AI career guidance counselor with deep knowledge of Indian and global career paths, top universities, salary ranges, certifications, and entrance exams like JEE, NEET, CAT, UPSC, GATE. Always give specific recommendations, realistic salary ranges, named colleges, and end with 2-3 actionable next steps."

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
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
        messages[-1]["content"] += "\\n\\n[Profile: " + " | ".join(profile_parts) + "]"
    def generate():
        with client.messages.stream(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=messages
        ) as stream:
            for text in stream.text_stream:
                yield "data: " + json.dumps({"text": text}) + "\\n\\n"
        yield "data: [DONE]\\n\\n"
    return Response(stream_with_context(generate()), mimetype="text/event-stream")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
"""

req_content = """flask==3.1.1
anthropic==0.50.0
gunicorn==23.0.0
"""

with open("app.py", "w", encoding="utf-8") as f:
    f.write(app_content)
print("app.py written OK")

with open("requirements.txt", "w", encoding="utf-8") as f:
    f.write(req_content)
print("requirements.txt written OK")