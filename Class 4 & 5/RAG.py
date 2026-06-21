import os
import time
from uuid import uuid4
from dotenv import load_dotenv

from pinecone import Pinecone, ServerlessSpec
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)

# =========================
# 1. LOAD ENV VARIABLES
# =========================
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in .env file")

# =========================
# 2. PINECONE INIT
# =========================
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "rag-index"

if index_name not in pc.list_indexes().names():
    print("Creating Pinecone Index...")

    pc.create_index(
        name=index_name,
        dimension=3072,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ),
    )

    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pc.Index(index_name)

# =========================
# 3. EMBEDDINGS
# =========================
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview"
)

# =========================
# 4. VECTOR STORE
# =========================
vector_store = PineconeVectorStore(
    index=index,
    embedding=embeddings
)

# =========================
# 5. SAMPLE DATA
# =========================
documents = [
    Document(
        page_content="I had chocolate chip pancakes this morning.",
        metadata={"source": "tweet"},
    ),
    Document(
        page_content="Weather is cloudy and cold tomorrow.",
        metadata={"source": "news"},
    ),
    Document(
        page_content="LangChain makes LLM applications easier.",
        metadata={"source": "tweet"},
    ),
]

# Insert only once
ADD_SAMPLE_DATA = False

if ADD_SAMPLE_DATA:
    ids = [str(uuid4()) for _ in documents]
    vector_store.add_documents(
        documents=documents,
        ids=ids
    )
    print("Documents inserted successfully.")

# =========================
# 6. LLM
# =========================
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7
)

# =========================
# 7. RAG FUNCTION
# =========================
def rag_answer(query: str):

    results = vector_store.similarity_search_with_score(
        query=query,
        k=3
    )

    context = "\n\n".join(
        [doc.page_content for doc, score in results]
    )

    prompt = f"""
You are a helpful AI assistant.

Answer the question using only the provided context.

Context:
{context}

Question:
{query}

Answer:
"""

    response = llm.invoke(prompt)

    return response.content


# =========================
# 8. CHAT LOOP
# =========================
print("\n🤖 RAG Chatbot Started")
print("Type 'exit' to quit\n")

while True:

    query = input("You: ").strip()

    if query.lower() == "exit":
        print("\n👋 Chat Ended")
        break

    try:
        answer = rag_answer(query)
        print(f"\nBot: {answer}\n")

    except Exception as e:
        print(f"\nError: {e}\n")