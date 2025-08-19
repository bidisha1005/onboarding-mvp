# rag_utils1.py
import os, glob
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI
from dotenv import load_dotenv
import chromadb

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
PERSIST_DIR = "./chroma_store"   # 👈 Persistent storage path


def build_vectorstore(doc_dir: str):
    """Build or load a persistent Chroma vectorstore."""

    # Init embeddings
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

    # Persistent client
    client_db = chromadb.PersistentClient(path=PERSIST_DIR)
    coll_name = "docs"

    # Try reusing existing collection
    try:
        coll = client_db.get_collection(coll_name)
        print("✅ Reusing existing vectorstore from disk")
        return coll, embeddings
    except:
        print("📦 No existing vectorstore found, creating new one...")

    # Load documents
    docs = []
    for file in glob.glob(os.path.join(doc_dir, "*.pdf")):
        loader = PyPDFLoader(file)
        docs.extend(loader.load())
    for file in glob.glob(os.path.join(doc_dir, "*.txt")):
        loader = TextLoader(file)
        docs.extend(loader.load())

    if not docs:
        print("⚠️ No docs found, using fallback text")
        from langchain.schema import Document
        docs = [Document(page_content="This is a fallback onboarding guide document.")]

    # Split text
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # Create new collection
    coll = client_db.create_collection(coll_name)

    for i, chunk in enumerate(chunks):
        coll.add(
            documents=[chunk.page_content],
            ids=[str(i)],
            embeddings=[embeddings.embed_query(chunk.page_content)]
        )

    print("✅ New vectorstore built and persisted to disk")
    return coll, embeddings


def ask_llm(prompt: str, model="llama3-70b-8192") -> str:
    """Send prompt to Groq's Llama model and return response."""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return resp.choices[0].message.content


def ask_with_llm(coll, embeddings, query: str) -> str:
    # Get RAG context
    results = coll.query(
        query_texts=[query],
        n_results=3
    )
    context = "\n".join(results["documents"][0])

    # Build prompt
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

loader = PyPDFLoader("sample.pdf")
docs = loader.load()
print(len(docs))
print(docs[0].page_content[:500])