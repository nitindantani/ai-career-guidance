from flask import Flask, request, render_template
import joblib
import os
import pandas as pd
import logging
from openai import OpenAI

# ---------------- APP SETUP ----------------
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------- LOAD MODEL ----------------
model = joblib.load("career_model.pkl")
encoders = joblib.load("encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")

# ---------------- OPENAI CLIENT ----------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------- LOAD DATA ----------------
df = pd.read_csv("career_data.csv")
df = df.apply(lambda col: col.str.lower().str.strip() if col.dtype == "object" else col)

streams = sorted(df["stream"].dropna().unique())
grouped_data = df.groupby("stream")

# ---------------- HELPERS ----------------
def get_options_for_stream(stream):
    if not stream:
        return {}
    stream = stream.lower().strip()
    if stream in grouped_data.groups:
        g = grouped_data.get_group(stream)
        return {
            "subjects": sorted(g["subject_liked"].unique()),
            "skills": sorted(g["skills"].unique()),
            "soft_skills": sorted(g["soft_skill"].unique()),
            "preferred_fields": sorted(g["preferred_field"].unique()),
        }
    return {}

def safe_encode(encoder, value):
    return encoder.transform([value])[0] if value in encoder.classes_ else -1

def ai_career_fallback(inputs):
    prompt = f"""
Suggest suitable careers for:
Stream: {inputs[0]}
Subject Liked: {inputs[1]}
Technical Skills: {inputs[2]}
Soft Skills: {inputs[3]}
Preferred Field: {inputs[4]}
"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
    )
    return response.choices[0].message.content.strip()

# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return render_template("index.html", streams=streams, suggestions={})

@app.route("/predict", methods=["POST"])
def predict():
    fields = ["stream", "subject_liked", "skills", "soft_skill", "preferred_field"]
    inputs = [request.form.get(f, "").strip().lower() for f in fields]

    # Defensive default
    suggestions = get_options_for_stream(inputs[0]) if inputs[0] else {}

    # Validation
    if not all(inputs):
        return render_template(
            "index.html",
            streams=streams,
            suggestions=suggestions,
            result="Please fill all fields.",
        )

    try:
        encoded = []
        for f, v in zip(fields, inputs):
            val = safe_encode(encoders[f], v)
            if val == -1:
                raise ValueError(f"Unseen value for {f}")
            encoded.append(val)

        pred = model.predict([encoded])[0]
        result = target_encoder.inverse_transform([pred])[0]
        logging.info("Prediction generated using ML model")

    except Exception as e:
        logging.warning(f"ML failed, switching to AI fallback: {e}")
        result = "AI-based suggestion:\n\n" + ai_career_fallback(inputs)

    return render_template(
        "index.html",
        streams=streams,
        suggestions=suggestions,
        result=result,
    )

# ---------------- RUN ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
