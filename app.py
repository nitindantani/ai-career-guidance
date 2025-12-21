from flask import Flask, request, render_template
import joblib
import os
import pandas as pd
import logging
from openai import OpenAI

# --------------------------------------------------
# APP SETUP
# --------------------------------------------------
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# --------------------------------------------------
# LOAD MODEL FILES (SAFE)
# --------------------------------------------------
try:
    model = joblib.load("career_model.pkl")
    encoders = joblib.load("encoders.pkl")
    target_encoder = joblib.load("target_encoder.pkl")
    logging.info("ML models loaded successfully")
except Exception as e:
    logging.error(f"Model loading failed: {e}")
    model = None
    encoders = {}
    target_encoder = None

# --------------------------------------------------
# OPENAI CLIENT (SAFE)
# --------------------------------------------------
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else None

# --------------------------------------------------
# LOAD DATASET (SAFE)
# --------------------------------------------------
try:
    df = pd.read_csv("career_data.csv")
    df = df.apply(lambda col: col.str.lower().str.strip() if col.dtype == "object" else col)
    streams = sorted(df["stream"].dropna().unique())
    grouped_data = df.groupby("stream")
    logging.info("Dataset loaded successfully")
except Exception as e:
    logging.error(f"Dataset loading failed: {e}")
    df = None
    streams = []
    grouped_data = {}

# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def get_options_for_stream(stream):
    try:
        if not stream or stream not in grouped_data:
            return {}
        g = grouped_data.get_group(stream)
        return {
            "subjects": sorted(g["subject_liked"].unique()),
            "skills": sorted(g["skills"].unique()),
            "soft_skills": sorted(g["soft_skill"].unique()),
            "preferred_fields": sorted(g["preferred_field"].unique()),
        }
    except Exception:
        return {}

def safe_encode(encoder, value):
    try:
        return encoder.transform([value])[0] if value in encoder.classes_ else -1
    except Exception:
        return -1

def ai_career_fallback(inputs):
    # AI MUST NEVER CRASH REQUEST
    if not client:
        return (
            "Based on your interests and skills, consider exploring roles "
            "aligned with your preferred field and building relevant experience "
            "through projects and internships."
        )
    try:
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
    except Exception as e:
        logging.error(f"AI fallback failed: {e}")
        return (
            "You may consider roles related to your preferred field. "
            "Strengthening your skills through certifications and practical "
            "experience can help you move forward."
        )

# --------------------------------------------------
# ROUTES
# --------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html", streams=streams, suggestions={})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        fields = ["stream", "subject_liked", "skills", "soft_skill", "preferred_field"]
        inputs = [request.form.get(f, "").strip().lower() for f in fields]

        suggestions = get_options_for_stream(inputs[0]) if inputs[0] else {}

        if not all(inputs):
            return render_template(
                "index.html",
                streams=streams,
                suggestions=suggestions,
                result="Please fill all fields.",
            )

        # ---------------- ML PATH ----------------
        if model and encoders and target_encoder:
            try:
                encoded = []
                for f, v in zip(fields, inputs):
                    enc = encoders.get(f)
                    if not enc:
                        raise ValueError("Encoder missing")
                    val = safe_encode(enc, v)
                    if val == -1:
                        raise ValueError("Unseen value")
                    encoded.append(val)

                pred = model.predict([encoded])[0]
                result = target_encoder.inverse_transform([pred])[0]
                logging.info("Prediction generated via ML model")

            except Exception as ml_error:
                logging.warning(f"ML failed, switching to AI: {ml_error}")
                result = ai_career_fallback(inputs)
        else:
            logging.warning("ML unavailable, using AI fallback")
            result = ai_career_fallback(inputs)

        return render_template(
            "index.html",
            streams=streams,
            suggestions=suggestions,
            result=result,
        )

    except Exception as fatal:
        # THIS PREVENTS 500 ERROR COMPLETELY
        logging.error(f"Fatal predict error: {fatal}")
        return render_template(
            "index.html",
            streams=streams,
            suggestions={},
            result="Something went wrong. Please try again later.",
        )

# --------------------------------------------------
# RUN
# --------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)