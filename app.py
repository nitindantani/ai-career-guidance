from flask import Flask, request, render_template
import joblib
import pandas as pd
import logging
import os

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------- LOAD MODEL ----------------
try:
    model = joblib.load("career_model.pkl")
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Model load failed: {e}")
    model = None

# ---------------- LOAD DATA FOR UI ----------------
df = pd.read_csv("career_data.csv")
df = df.apply(lambda col: col.str.lower().str.strip() if col.dtype == "object" else col)
streams = sorted(df["stream"].unique())

# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return render_template("index.html", streams=streams, suggestions={})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        inputs = {
            "stream": request.form.get("stream", "").lower(),
            "subject_liked": request.form.get("subject_liked", "").lower(),
            "skills": request.form.get("skills", "").lower(),
            "soft_skill": request.form.get("soft_skill", "").lower(),
            "preferred_field": request.form.get("preferred_field", "").lower(),
        }

        if not all(inputs.values()):
            return render_template(
                "index.html",
                streams=streams,
                suggestions={},
                result="Please fill all fields."
            )

        input_df = pd.DataFrame([inputs])

        if model:
            prediction = model.predict(input_df)[0]
            result = f"Recommended career path: {prediction}"
        else:
            result = "Prediction service unavailable."

        return render_template(
            "index.html",
            streams=streams,
            suggestions={},
            result=result
        )

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return render_template(
            "index.html",
            streams=streams,
            suggestions={},
            result="Something went wrong. Please try again."
        )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
