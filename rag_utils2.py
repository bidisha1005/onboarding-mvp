# rag_utils.py
import os, glob
from dotenv import load_dotenv
from openai import OpenAI
import chromadb

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# For OCR (scanned PDFs)
try:
    import pytesseract
    from pdf2image import convert_from_path
except ImportError:
    pytesseract = None

# -----------------------------
# Load env and setup
# -----------------------------
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Groq client
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# Constants
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
PERSIST_DIR = "./chroma_store"        # Persistent vector storage
DOCS_DIR = "./data/chroma/docs"       # Your uploaded docs folder
COLLECTION_NAME = "docs"


# -----------------------------
# OCR for scanned PDFs
# -----------------------------
def ocr_pdf(file_path):
    """Fallback OCR for scanned PDFs."""
    if pytesseract is None:
        raise ImportError("pytesseract and pdf2image are required for OCR.")
    pages = convert_from_path(file_path)
    text_pages = []
    for page in pages:
        text = pytesseract.image_to_string(page)
        text_pages.append(Document(page_content=text, metadata={"source": os.path.basename(file_path)}))
    return text_pages


# -----------------------------
# Build vectorstore
# -----------------------------
def build_vectorstore(doc_dir: str = DOCS_DIR):
    """
    Build a fresh Chroma vectorstore from uploaded docs.
    Deletes existing collection and rebuilds.
    """

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    client_db = chromadb.PersistentClient(path=PERSIST_DIR)

    # Reset collection
    try:
        client_db.delete_collection(COLLECTION_NAME)
        print("🗑️ Existing collection deleted")
    except Exception:
        pass  # no previous collection

    docs = []
    debug_files = []  # <-- keep track of all loaded docs for debugging

    # PDF docs
    for file in glob.glob(os.path.join(doc_dir, "*.pdf")):
        loader = PyPDFLoader(file)
        loaded_docs = loader.load()
        if not loaded_docs and pytesseract is not None:
            print(f"⚠️ PDF seems scanned, running OCR: {file}")
            loaded_docs = ocr_pdf(file)

        print(f"📄 Loaded {len(loaded_docs)} pages from {file}")
        for doc in loaded_docs:
            doc.metadata = {"source": os.path.basename(file)}
        docs.extend(loaded_docs)
        debug_files.append({"file": os.path.basename(file), "pages": len(loaded_docs)})

    # TXT docs
    for file in glob.glob(os.path.join(doc_dir, "*.txt")):
        loader = TextLoader(file)
        loaded_docs = loader.load()

        print(f"📄 Loaded {len(loaded_docs)} pages from {file}")
        for doc in loaded_docs:
            doc.metadata = {"source": os.path.basename(file)}
        docs.extend(loaded_docs)
        debug_files.append({"file": os.path.basename(file), "pages": len(loaded_docs)})

    # Fallback if no docs
    if not docs:
        print("⚠️ No docs found, using fallback text")
        docs = [Document(page_content="This is a fallback onboarding guide document.", metadata={"source": "fallback"})]
        debug_files.append({"file": "fallback", "pages": 1})

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"✂️ Split into {len(chunks)} chunks")

    # Create new collection
    coll = client_db.create_collection(COLLECTION_NAME)

    for i, chunk in enumerate(chunks):
        coll.add(
            documents=[chunk.page_content],
            ids=[str(i)],
            embeddings=[embeddings.embed_query(chunk.page_content)],
            metadatas=[chunk.metadata]
        )

    print(f"✅ New vectorstore built and persisted to {PERSIST_DIR}")
    print("📊 Debug – Files loaded into vectorstore:", debug_files)

    return coll, embeddings


# -----------------------------
# Query functions
# -----------------------------
def ask_llm(prompt: str, model="llama3-70b-8192") -> str:
    """Send prompt to Groq's Llama model and return response."""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return resp.choices[0].message.content


def ask_with_llm(coll, embeddings, query: str) -> str:
    """Retrieve RAG context and ask LLM."""
    results = coll.query(query_texts=[query], n_results=5)

    context = ""
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        context += f"(From {meta.get('source','unknown')}):\n{doc}\n\n"

    prompt = f"""
You are an AI Onboarding Assistant for new employees.  
Your role is to answer questions strictly based on the provided onboarding documents.  

Context:
{context}

Guidelines:
- Provide clear, step-by-step answers when possible.  
- If the question relates to tools, accounts, training, or policies, explain in simple terms.  
- If the answer is not found in the context, say:  
  "I couldn’t find this information in the onboarding guide."  
- Do not make up information or provide external details.  

Question: {query}

Answer:
"""
    return ask_llm(prompt)


# -----------------------------
# Debug test
# -----------------------------
if __name__ == "__main__":
    coll, embeddings = build_vectorstore(DOCS_DIR)
    answer = ask_with_llm(coll, embeddings, "What are the first steps in onboarding?")
    print("Answer:", answer)
