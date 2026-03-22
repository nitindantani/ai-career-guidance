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

## 🧪 AI Model Evolution — How We Got Here

This project went through 3 major AI model iterations before reaching the current version. Here's the full journey:

### Version 1 — Custom ML Model (Random Forest Classifier)
| Property | Detail |
|----------|--------|
| Model | Random Forest Classifier (scikit-learn) |
| Training Data | Custom `career_data.csv` — 20 rows, 6 features |
| Features | stream, subject_liked, skills, soft_skill, preferred_field |
| Accuracy | **~7.5%** |
| Output | Single career label (e.g. "Data Analyst") |
| Problem | Dataset was too small (only 20 rows). The model could only predict careers it had seen in training. No reasoning, no explanations, no salary info, no college recommendations. Completely unreliable. |

### Version 2 — OpenAI GPT-3.5 Turbo
| Property | Detail |
|----------|--------|
| Model | GPT-3.5 Turbo (OpenAI API) |
| Accuracy | Much better — full language understanding |
| Output | Career recommendations with explanations |
| Problem | OpenAI API requires paid credits. No free tier available for this use case. Replaced due to cost. |

### Version 3 — Anthropic Claude (claude-sonnet-4-20250514)
| Property | Detail |
|----------|--------|
| Model | Claude Sonnet by Anthropic |
| Accuracy | Excellent — deep reasoning, Indian context aware |
| Output | Careers + salaries + colleges + skill roadmaps |
| Problem | Anthropic API also requires paid credits. Account had zero balance. Replaced due to cost. |

### Version 4 — Groq + Llama 3.3 70B ✅ Current
| Property | Detail |
|----------|--------|
| Model | Llama 3.3 70B via Groq API |
| Cost | **100% Free** |
| Speed | Ultra-fast inference (Groq's LPU hardware) |
| Accuracy | Excellent — comparable to GPT-4 class models |
| Output | Careers + salaries + colleges + skill roadmaps + entrance exam guidance |
| Streaming | Yes — word-by-word real-time responses |
| Status | ✅ Live and working |

> **Key lesson**: A well-prompted large language model (even free ones) vastly outperforms a custom ML model trained on small datasets for open-ended guidance tasks.

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
set GROQ_API_KEY=your_key_here      # Windows
export GROQ_API_KEY=your_key_here   # Mac/Linux

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
| `GROQ_API_KEY` | Your free Groq API key from console.groq.com |

> ⚠️ Never hardcode API keys in your code. Always use environment variables.

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
- [ ] Real job listings integration (Naukri, LinkedIn)
- [ ] Career roadmap visualizer

---

## 👨‍💻 Author

**Nitin Dantani**
Saffrony Institute of Technology
Computer Science and Engineering

---

## 📄 License

MIT License — feel free to use and modify!
