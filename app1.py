# app.py
import streamlit as st
import requests
import os
from rag_utils2 import build_vectorstore, ask_with_llm, DOCS_DIR

# ------------------- PAGE CONFIG -------------------
st.set_page_config(page_title="AI Onboarding Assistant Buddy", layout="wide")

# ------------------- SIDEBAR NAVIGATION -------------------
page = st.sidebar.radio(
    "Navigation",
    ["📂 Upload Docs & Generate Roadmap", "💬 RAG Chatbot", "🎯 Role-Aware Assistant"]
)

# ------------------- PAGE 1: UPLOAD DOCS + ROADMAP -------------------
if page == "📂 Upload Docs & Generate Roadmap":
    st.title("📂 Upload Onboarding Docs & Generate Roadmap")

    uploaded_files = st.file_uploader(
        "Upload multiple documents (PDF/TXT)",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    if uploaded_files:
        os.makedirs(DOCS_DIR, exist_ok=True)

        for file in uploaded_files:
            save_path = os.path.join(DOCS_DIR, file.name)
            with open(save_path, "wb") as f:
                f.write(file.getbuffer())
        st.success(f"✅ Uploaded {len(uploaded_files)} files successfully!")

        if st.button("🚀 Build Vectorstore & Generate Roadmap"):
            coll, embeddings = build_vectorstore(DOCS_DIR)
            st.success("✅ Vectorstore built successfully!")

            # Ask LLM to generate roadmap
            roadmap_query = "Generate a 30-day step-by-step onboarding roadmap for a new employee."
            roadmap = ask_with_llm(coll, embeddings, roadmap_query)

            st.subheader("📅 Onboarding Roadmap")
            st.write(roadmap)


# ------------------- PAGE 2: RAG CHATBOT -------------------
elif page == "💬 RAG Chatbot":
    st.title("💬 Onboarding Assistant Chatbot")

    # Load vectorstore
    coll, embeddings = build_vectorstore(DOCS_DIR)

    user_query = st.text_input("Ask me anything about onboarding:")

    if st.button("Ask"):
        if user_query.strip() == "":
            st.warning("⚠️ Please enter a question.")
        else:
            answer = ask_with_llm(coll, embeddings, user_query)
            st.subheader("🤖 Assistant's Answer")
            st.write(answer)


# ------------------- PAGE 3: ROLE-AWARE ASSISTANT -------------------
elif page == "🎯 Role-Aware Assistant":
    st.title("🎯 AI Role-Aware Onboarding Assistant")

    # --- Step 1: Upload Docs ---
    st.header("Upload Your Onboarding Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files.", type=['pdf', 'txt'], accept_multiple_files=True
    )

    if uploaded_files:
        st.success(f"{len(uploaded_files)} file(s) uploaded successfully.")
        UPLOAD_DIR = "./uploaded_docs"
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        for file in uploaded_files:
            with open(os.path.join(UPLOAD_DIR, file.name), "wb") as f:
                f.write(file.getbuffer())
        st.info("📂 Documents ready for querying.")

    # --- Step 2: Select Role & Ask ---
    st.header("Select Role and Ask Your Question")

    roles = [
        "UI/UX Designer",
        "Front End Developer",
        "Back End Developer",
        "Data Analyst",
        "Game Dev Intern"
    ]
    selected_role = st.selectbox("Select your role:", roles)
    user_input = st.text_area("Enter your query here:", placeholder="Type something...")

    if st.button("Submit Query"):
        if user_input.strip() == "":
            st.warning("Please enter some text first.")
        else:
            try:
                response = requests.post(
                    "http://localhost:8000/process",
                    json={
                        "text": user_input,
                        "role": selected_role
                    }
                )
                if response.status_code == 200:
                    st.success(response.json()["result"])
                else:
                    st.error(f"Error {response.status_code}: {response.text}")
            except Exception as e:
                st.error(f"Failed to connect to backend: {e}")
