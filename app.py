from flask import Flask, request, render_template, jsonify, stream_with_context, Response
import os
import json
import anthropic

app = Flask(__name__)

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = """You are an expert AI career guidance counselor with deep knowledge of:
- Indian and global career paths, job roles, and industries
- Top universities and colleges (India: IITs, NITs, IIMs, AIIMS, top private universities)
- Current job market trends, salary ranges, and demand forecasts
- Certifications and online courses (Coursera, edX, NPTEL, Udemy) and skill roadmaps
- Indian entrance exams: JEE, NEET, CAT, CLAT, UPSC, GATE
- Emerging fields: AI/ML, Data Science, Cybersecurity, UX Design

When answering:
1. Give concrete, specific career recommendations
2. For salary: provide Indian ranges (fresher/mid/senior) and global equivalents
3. For colleges: name specific top institutions with admission requirements
4. For skills: provide a priority-ordered learning roadmap with timelines
5. Use markdown formatting with clear sections
6. Always end with 2-3 actionable next steps"""


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
        profile_parts.append(f"Stream: {profile['stream']}")
    if profile.get("grade"):
        profile_parts.append(f"Academic performance: {profile['grade']}")
    if profile.get("workexp"):
        profile_parts.append(f"Work experience: {profile['workexp']}")
    if profile.get("skills"):
        profile_parts.append(f"Skills: {', '.join(profile['skills'])}")
    if profile.get("interests"):
        profile_parts.append(f"Interests: {', '.join(profile['interests'])}")

    if profile_parts:
        messages[-1]["content"] += f"\n\n[My profile: {' | '.join(profile_parts)}]"

    def generate():
        with client.messages.stream(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=messages
        ) as stream:
            for text in stream.text_stream:
                yield f"data: {json.dumps({'text': text})}\n\n"
        yield "data: [DONE]\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
```

Also replace `requirements.txt` with exactly this:
```
flask==3.1.1
anthropic==0.50.0
gunicorn==23.0.0