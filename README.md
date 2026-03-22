# 🎓 AI Career Guidance System

A personalized AI-powered career guidance chatbot built with Flask and Groq (Llama 3.3 70B). Helps students and professionals discover the right career path based on their stream, skills, and interests.

🌐 **Live Demo**: [https://ai-career-guidance-hdzl.onrender.com](https://ai-career-guidance-hdzl.onrender.com)

---

## ✨ Features

- 🤖 **AI Chat** — Real-time streaming responses powered by Groq (Llama 3.3 70B)
- 🎯 **Personalized Guidance** — Tailored advice based on stream, grades, skills & interests
- 💰 **Salary Info** — Realistic Indian salary ranges (fresher / mid / senior)
- 🏫 **College Recommendations** — IITs, NITs, IIMs, AIIMS and more
- 📚 **Skill Roadmaps** — Priority-ordered learning paths with timelines
- 🎓 **Entrance Exam Guidance** — JEE, NEET, CAT, UPSC, GATE, CLAT
- 📱 **Mobile Responsive** — Works on all screen sizes
- 🌙 **Dark / Light Mode** — Toggle between themes
- ⚡ **Fast Streaming** — Word-by-word response streaming

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python, Flask |
| AI Model | Groq API (Llama 3.3 70B) |
| Frontend | HTML, CSS, JavaScript |
| Deployment | Render |
| Version Control | GitHub |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Groq API key (free at [console.groq.com](https://console.groq.com))

### Installation

```bash
# Clone the repo
git clone https://github.com/nitindantani/ai-career-guidance.git
cd ai-career-guidance

# Install dependencies
pip install -r requirements.txt

# Set environment variable
set GROQ_API_KEY=your_key_here   # Windows
export GROQ_API_KEY=your_key_here  # Mac/Linux

# Run the app
python app.py
```

Open `http://localhost:10000` in your browser.

---

## 📁 Project Structure

```
ai-career-guidance/
├── app.py                 # Flask backend with Groq API
├── requirements.txt       # Python dependencies
├── render.yaml            # Render deployment config
├── templates/
│   └── index.html         # Frontend chat UI
└── README.md
```

---

## 🔐 Environment Variables

| Variable | Description |
|----------|-------------|
| `GROQ_API_KEY` | Your Groq API key from console.groq.com |

---

## 📸 Screenshots

> Add screenshots of your app here

---

## 🎯 How It Works

1. User fills in their profile (stream, grades, skills, interests)
2. User asks a career question in the chat
3. Flask backend sends the question + profile to Groq API
4. Llama 3.3 70B generates a personalized response
5. Response streams back word-by-word in real time

---

## 🌱 Future Scope

- [ ] Resume upload for deeper personalization
- [ ] User login and chat history
- [ ] Hindi / Gujarati language support
- [ ] Real job listings integration
- [ ] Career roadmap visualizer

---

## 👨‍💻 Author

**Nitin Dantani**  
Saffrony Institute of Technology  
Computer Science and Engineering

---

## 📄 License

MIT License — feel free to use and modify!
