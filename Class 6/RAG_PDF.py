import os
import time
from uuid import uuid4

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)

from langchain_pinecone import PineconeVectorStore


# =========================
# LOAD ENV VARIABLES
# =========================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found")


# =========================
# PINECONE SETUP
# =========================

pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "pdf-rag-index"

if index_name not in pc.list_indexes().names():

    print("Creating Pinecone Index...")

    pc.create_index(
        name=index_name,
        dimension=3072,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

    time.sleep(10)

index = pc.Index(index_name)


# =========================
# EMBEDDING MODEL
# =========================

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)


# =========================
# VECTOR STORE
# =========================

vector_store = PineconeVectorStore(
    index=index,
    embedding=embeddings
)


# =========================
# LLM
# =========================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3
)


# =========================
# LOAD PDF
# =========================

pdf_path = r"Class 6\Document\Python_OOP.pdf"

print(f"Loading PDF: {pdf_path}")

loader = PyPDFLoader(pdf_path)

documents = loader.load()

print(f"\nLoaded {len(documents)} pages")


# =========================
# CHUNKING
# =========================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks")


# =========================
# STORE IN PINECONE
# =========================

ids = [str(uuid4()) for _ in chunks]

vector_store.add_documents(
    documents=chunks,
    ids=ids
)
print("PDF stored successfully")


# =========================
# RAG FUNCTION
# =========================

def ask_rag(question):

    results = vector_store.similarity_search(
        question,
        k=3
    )

    context = "\n\n".join(
        [doc.page_content for doc in results]
    )

    prompt = f"""
You are a helpful assistant.

Answer ONLY from the provided context.

If the answer is not present in the context,
say:

'I could not find this information in the PDF.'

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    return response.content


# =========================
# CHAT LOOP
# =========================

print("\nPDF RAG Chat Started")
print("Type 'exit' to quit\n")

while True:

    query = input("You: ")

    if query.lower() == "exit":
        print("\nGoodbye")
        break

    try:

        answer = ask_rag(query)

        print("\nBot:", answer)
        print()

    except Exception as e:

        print("\nError:", str(e))

