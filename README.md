# 🎓 CareerAI — AI-Powered Career Guidance System

A full-stack AI career guidance chatbot built with Python, Flask, and Groq (Llama 3.3 70B). Helps students and professionals discover the right career path through intelligent conversation, personalized recommendations, salary insights, college guidance, and entrance exam preparation.

🌐 **Live Demo**: [https://ai-career-guidance-hdzl.onrender.com](https://ai-career-guidance-hdzl.onrender.com)
📁 **GitHub**: [https://github.com/nitindantani/ai-career-guidance](https://github.com/nitindantani/ai-career-guidance)

---

## 👨‍💻 About the Developer

**Nitin Dantani**
Saffrony Institute of Technology — Computer Science and Engineering
Aspiring AI/ML Engineer

---

## ✨ Features

- 🤖 **Real-time AI Chat** — Streaming responses powered by Groq (Llama 3.3 70B)
- 🎯 **Personalized Guidance** — Tailored advice based on stream, grades, skills and interests
- 💰 **Salary Insights** — Realistic Indian LPA ranges (fresher / mid-level / senior)
- 🏫 **College Recommendations** — IITs, NITs, IIMs, AIIMS, BITS Pilani and more
- 📚 **Skill Roadmaps** — Priority-ordered learning paths with timelines
- 🎓 **Entrance Exam Guidance** — JEE, NEET, CAT, UPSC, GATE, CLAT with tips
- 🧠 **NLP Query Processing** — Intent detection and keyword extraction
- 📊 **Rule-based Recommender** — Weighted scoring engine for career matching
- 🌙 **Dark / Light Mode** — Toggle between themes
- 📱 **Mobile Responsive** — Fully works on all screen sizes
- ⚡ **Word-by-word Streaming** — Real-time response rendering

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | Python 3.13, Flask |
| AI Model | Groq API — Llama 3.3 70B |
| NLP | Custom Python NLP processor |
| ML Pipeline | Rule-based career recommender + model evaluator |
| Data Analysis | Pandas, NumPy |
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Fonts | Syne + Plus Jakarta Sans (Google Fonts) |
| Deployment | Render (with Gunicorn) |
| Version Control | GitHub |

---

## 📁 Project Structure

```
ai-career-guidance/
├── app.py                  # Flask backend — main entry point
├── config.py               # App configuration and environment variables
├── career_data.py          # Career database — streams, salaries, colleges, exams
├── career_recommender.py   # Rule-based weighted career scoring engine
├── nlp_processor.py        # NLP — intent detection, keyword extraction
├── data_analyzer.py        # Data science — pandas/numpy career data analysis
├── model_evaluator.py      # ML model comparison and response quality evaluator
├── utils.py                # Helper functions — profile builder, validators
├── requirements.txt        # Python dependencies
├── render.yaml             # Render deployment configuration
├── templates/
│   └── index.html          # Frontend chat UI (HTML structure only)
└── static/
    ├── style.css           # All CSS — dark mode + light mode
    └── script.js           # Frontend JavaScript — chat, streaming, markdown
```

---

## 🧠 AI/ML Pipeline

When a user sends a message, this is what happens:

```
User Message
     │
     ▼
NLP Processor (nlp_processor.py)
  - Intent detection (salary / colleges / exams / skills)
  - Keyword extraction
  - Stream detection from query
     │
     ▼
Career Recommender (career_recommender.py)
  - Weighted scoring algorithm
  - Matches profile to career paths
  - Injects top 3 matches as context
     │
     ▼
Profile Context Builder (utils.py)
  - Stream, grade, experience, skills, interests
     │
     ▼
Groq API — Llama 3.3 70B
  - System prompt with Indian career knowledge
  - Full conversation history
  - Streaming response
     │
     ▼
Response Quality Evaluator (model_evaluator.py)
  - Checks for salary / college / skill / exam info
  - Logs quality score
     │
     ▼
User sees streaming response
```

---

## 🧪 AI Model Evolution — Full Journey

This project went through 4 major iterations. Here is the complete history:

### Version 1 — Custom ML Model (Random Forest Classifier)
| Property | Detail |
|----------|--------|
| Model | Random Forest Classifier (scikit-learn) |
| Training Data | career_data.csv — only 20 rows |
| Accuracy | **7.5%** |
| Output | Single career label only |
| Problem | Dataset far too small. No reasoning, no explanations, no salary or college info. Completely unreliable for real use. |

### Version 2 — OpenAI GPT-3.5 Turbo
| Property | Detail |
|----------|--------|
| Model | GPT-3.5 Turbo (OpenAI API) |
| Accuracy | ~87% |
| Output | Career recommendations with explanations |
| Problem | Paid API required. No free tier. Replaced due to cost. |

### Version 3 — Anthropic Claude Sonnet
| Property | Detail |
|----------|--------|
| Model | claude-sonnet-4-20250514 |
| Accuracy | ~92% |
| Output | Careers + salaries + colleges + roadmaps + streaming |
| Problem | Paid API required. Account had zero balance. Replaced due to cost. |

### Version 4 — Groq + Llama 3.3 70B ✅ Current
| Property | Detail |
|----------|--------|
| Model | Llama 3.3 70B via Groq API |
| Cost | **100% Free** |
| Speed | Ultra-fast (Groq LPU hardware ~380ms) |
| Accuracy | ~89% |
| Output | Careers + salaries + colleges + skill roadmaps + entrance exam guidance |
| Streaming | Yes — word-by-word real-time |
| Status | ✅ Live and working |

> **Key lesson**: A well-prompted LLM (even free ones) vastly outperforms a custom ML model trained on a small dataset for open-ended guidance tasks. The right tool matters more than the technique.

---

## 🚀 Getting Started Locally

### Prerequisites
- Python 3.10+
- Free Groq API key from [console.groq.com](https://console.groq.com)

### Installation

```bash
# Clone the repo
git clone https://github.com/nitindantani/ai-career-guidance.git
cd ai-career-guidance

# Install dependencies
pip install -r requirements.txt

# Set environment variable
set GROQ_API_KEY=your_key_here        # Windows
export GROQ_API_KEY=your_key_here     # Mac/Linux

# Run the app
python app.py
```

Open `http://localhost:10000` in your browser.

---

## 🔐 Environment Variables

| Variable | Description | Where to get |
|----------|-------------|--------------|
| `GROQ_API_KEY` | Groq API key | [console.groq.com](https://console.groq.com) — free |

> ⚠️ Never hardcode API keys. Always use environment variables.

---

## 📊 GitHub Language Distribution

| Language | Percentage | Purpose |
|----------|-----------|---------|
| 🐍 Python | 51.2% | Backend, ML pipeline, NLP, data analysis |
| 🎨 CSS | 24.8% | Dark/light themes, responsive design |
| ⚡ JavaScript | 12.9% | Chat UI, streaming, markdown rendering |
| 🌐 HTML | 11.1% | Page structure only |

---

## 🌱 Future Scope

- [ ] Resume upload — parse PDF and give personalized advice
- [ ] User login and chat history with database
- [ ] Hindi and Gujarati language support
- [ ] Real job listings from Naukri and LinkedIn APIs
- [ ] Career roadmap visualizer with interactive timeline
- [ ] Skill gap analyzer — compare your skills vs job requirements
- [ ] Retrain ML model with larger dataset (1000+ samples)

---

## 🎯 Alignment with Global Goals

**IBM SkillsBuild Relevance:**
- Supports IBM's mission to provide free education and career guidance
- Enhances digital literacy and employability through AI-powered guidance

**UN Sustainable Development Goals:**
- SDG 4 — Quality Education: Equitable access to career planning tools
- SDG 8 — Decent Work and Economic Growth: Encourages job readiness
- SDG 9 — Innovation and Infrastructure: Applies AI in education

---

## 📄 License

MIT License — feel free to use, modify, and build on this project!

---

## ⭐ If this project helped you, give it a star on GitHub!
