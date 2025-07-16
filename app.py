from flask import Flask, request, render_template
import joblib
import openai

# Initialize Flask
app = Flask(__name__)

# Load model and encoders
model = joblib.load("career_model.pkl")
encoders = joblib.load("encoders.pkl")

# OpenAI API Key
import os
openai.api_key = os.getenv("OPENAI_API_KEY")

# Home route
@app.route('/')
def home():
    return render_template("index.html")

# Prediction route
@app.route('/predict', methods=["POST"])
def predict():
    inputs = [request.form.get(field) for field in ['stream', 'subject_liked', 'skills', 'soft_skill', 'preferred_field']]
    try:
        encoded = [encoders[col].transform([val])[0] for col, val in zip(encoders, inputs)]
        prediction = model.predict([encoded])[0]
        result = encoders['career_label'].inverse_transform([prediction])[0]
    except:
        result = "Invalid input."
    return render_template("index.html", result=result)

# 💬 Chat route
@app.route('/chat', methods=["POST"])
def chat():
    user_question = request.form.get("question")
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": user_question}]
    )
    answer = response['choices'][0]['message']['content']
    return render_template("index.html", chat_answer=answer)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
