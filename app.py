from flask import Flask, request, render_template, jsonify, stream_with_context, Response
import os
import json
import anthropic

app = Flask(__name__)

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = """You are an expert AI career guidance counselor with deep knowledge of:
- Indian and global career paths, job roles, and industries
- Top universities and colleges (India: IITs, NITs, IIMs, AIIMS, top private universities; Global: Ivy League, Russell Group, etc.)
- Current job market trends, salary ranges, and demand forecasts for 2024-2025
- Certifications and online courses (Coursera, edX, NPTEL, Udemy, etc.) and skill roadmaps
- Indian entrance exams: JEE, NEET, CAT, CLAT, UPSC, GATE, and more
- Emerging career fields: AI/ML, Data Science, Cybersecurity, UX Design, Sustainable Energy, etc.

When answering:
1. Always give concrete, specific career recommendations — never vague advice
2. For salary: provide realistic Indian salary ranges (fresher / mid-level / senior) and global equivalents where relevant
3. For colleges: name specific top institutions with their strengths and admission requirements
4. For skills: provide a clear, priority-ordered learning roadmap with timelines
5. Structure responses with clear sections using markdown formatting
6. Be honest about competition levels and realistic timelines
7. Always consider the student's specific profile (stream, grades, skills, interests) when provided
8. End every response with 2-3 concrete, actionable next steps
9. Use bullet points for lists and **bold** for important terms
10. Keep responses focused and practical — students need actionable guidance, not essays"""


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
        messages[-1]["content"] += f"\n\n[My profile — {' | '.join(profile_parts)}]"

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

Then also check `requirements.txt` — it should only contain:
```
flask==3.1.1
anthropic==0.50.0
python-dotenv==1.0.1
gunicorn==23.0.0