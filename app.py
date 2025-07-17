from flask import Flask, request, render_template
import joblib
import openai
import os

# Initialize Flask
app = Flask(__name__)

# Load model and encoders
model = joblib.load("career_model.pkl")
encoders = joblib.load("encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")  # ensure you have this

# Set OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Home route
@app.route('/')
def home():
    return render_template("index.html")

# Prediction route
@app.route('/predict', methods=["POST"])
def predict():
    input_fields = ['stream', 'subject_liked', 'skills', 'soft_skill', 'preferred_field']
    inputs = [request.form.get(field) for field in input_fields]

    try:
        # Try encoding and prediction
        encoded = [encoders[col].transform([val])[0] for col, val in zip(encoders, inputs)]
        prediction = model.predict([encoded])[0]
        result = target_encoder.inverse_transform([prediction])[0]
        return render_template("index.html", result=result)
    except Exception as e:
        print("Prediction Error:", e)

        # Fallback: Use OpenAI to suggest a career based on raw input
        prompt = (
            "The user entered the following career-related info:\n"
            f"Stream: {inputs[0]}\n"
            f"Subject Liked: {inputs[1]}\n"
            f"Skills: {inputs[2]}\n"
            f"Soft Skill: {inputs[3]}\n"
            f"Preferred Field: {inputs[4]}\n\n"
            "Based on this, suggest the most suitable career path or advice."
        )

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert career counselor."},
                    {"role": "user", "content": prompt}
                ]
            )
            fallback_answer = response['choices'][0]['message']['content']
        except Exception as api_error:
            print("ChatGPT Fallback Error:", api_error)
            fallback_answer = "Sorry, we couldn’t process your request. Please try again."

        return render_template("index.html", chat_answer=fallback_answer)

# ChatGPT route
@app.route('/chat', methods=["POST"])
def chat():
    user_question = request.form.get("question")
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful AI career counselor."},
                {"role": "user", "content": user_question}
            ]
        )
        answer = response['choices'][0]['message']['content']
    except Exception as e:
        print("ChatGPT Error:", e)
        answer = "Something went wrong while contacting the AI."

    return render_template("index.html", chat_answer=answer)

# Run app
if __name__ == "__main__":
    app.run(debug=True)
