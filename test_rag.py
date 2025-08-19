# test_rag.py
import os
from rag_utils import build_vectorstore, ask_with_llm

def main():
    docs_path = "data/chroma/docs"
    if not os.path.exists(docs_path):
        print(f"⚠️ Folder '{docs_path}' not found. Please create it and add PDFs or TXTs.")
        return

    print("🔍 Building vectorstore from documents...")
    coll, embeddings = build_vectorstore(docs_path)
    print("✅ Vectorstore ready. Ask questions (type 'exit' to quit).\n")

    while True:
        q = input("Q: ")
        if q.lower() in ["exit", "quit"]:
            print("👋 Exiting...")
            break
        answer = ask_with_llm(coll, embeddings, q)
        print("Answer:", answer, "\n")

if __name__ == "__main__":
    main()
