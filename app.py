from flask import Flask, request, render_template
import joblib
import openai
import os
import pandas as pd

# Initialize Flask
app = Flask(__name__)

# Load model and encoders
model = joblib.load("career_model.pkl")
encoders = joblib.load("encoders.pkl")

# Set OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load CSV for suggestions
df = pd.read_csv("career_data.csv")
suggestions = {
    'streams': sorted(df['stream'].dropna().unique()),
    'subjects': sorted(df['subject_liked'].dropna().unique()),
    'skills': sorted(df['skills'].dropna().unique()),
    'soft_skills': sorted(df['soft_skill'].dropna().unique()),
    'preferred_fields': sorted(df['preferred_field'].dropna().unique()),
}

# Home page
@app.route('/')
def home():
    return render_template("index.html", suggestions=suggestions)

# Career Prediction route
@app.route('/predict', methods=["POST"])
def predict():
    inputs = [request.form.get(field) for field in ['stream', 'subject_liked', 'skills', 'soft_skill', 'preferred_field']]
    try:
        encoded = [encoders[col].transform([val])[0] for col, val in zip(encoders, inputs)]
        prediction = model.predict([encoded])[0]
        result = encoders['career_label'].inverse_transform([prediction])[0]
    except Exception:
        prompt = f"""
        A student entered:
        - Stream: {inputs[0]}
        - Subject Liked: {inputs[1]}
        - Technical Skills: {inputs[2]}
        - Soft Skills: {inputs[3]}
        - Preferred Field: {inputs[4]}

        Based on this input, suggest a few suitable career options. Make it helpful and relevant.
        """
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
            result = "Model could not predict accurately. Here's a helpful suggestion:\n\n" + response['choices'][0]['message']['content']
        except Exception:
            result = "Sorry, we couldn't process your input. Please try again."

    return render_template("index.html", suggestions=suggestions, result=result)

# Chat route for AI career Q&A
@app.route('/chat', methods=["POST"])
def chat():
    user_question = request.form.get("question")
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": user_question}]
        )
        answer = response['choices'][0]['message']['content']
    except Exception:
        answer = "Sorry, I couldn't answer your question right now."
    return render_template("index.html", suggestions=suggestions, chat_answer=answer)

# Run app (Render-compatible)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
