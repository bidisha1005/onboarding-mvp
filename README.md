# Onboarding AI Assistant (MVP)

An **AI-powered onboarding assistant** that helps new hires quickly get started with their role.  
It generates **role-specific 30-day learning roadmaps**, answers onboarding-related queries, and retrieves information from internal company documents.

---

## Features

- **Role-Aware Roadmap Generator**  
  Generates a customized 30-day roadmap based on role-specific documents (PDF, TXT).

- **Onboarding Chatbot**  
  Chat with an AI assistant trained on company docs (Notion, Google Drive, GitHub, or uploaded files).

- **RAG (Retrieval-Augmented Generation)**  
  Uses **LangChain + ChromaDB** to fetch relevant knowledge before answering.

- **Streamlit Frontend**  
  Simple UI for uploading docs and interacting with the assistant.

---

## Tech Stack

- **Backend**: Python, FastAPI  
- **AI / LLM**: Groq API (LLaMA / Mistral)  
- **RAG**: LangChain + ChromaDB  
- **Frontend**: Streamlit  

---

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/onboarding-mvp.git
   cd mini_project_rhythm_wave
   ```
2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # macOS/Linux
   .venv\Scripts\activate     # Windows
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Setup environment variables**
   Create a .env file
   ```ini
   GROQ_API_KEY=your_api_key_here
   ```
5. **Run the app**
   ```bash
   streamlit run app1.py
   ```
