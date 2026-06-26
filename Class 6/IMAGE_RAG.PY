import os
from dotenv import load_dotenv
from PIL import Image
import chromadb
from google import genai
from embeddings import EmbeddingModel

# ==========================
# LOAD ENV
# ==========================

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
)

# ==========================
# CHROMADB
# ==========================

db = chromadb.PersistentClient(
    path="Class 6/chroma_db"
)

collection = db.get_collection(
    name="image_collection"
)

# ==========================
# LOAD CLIP MODEL
# ==========================

model = EmbeddingModel()

# ==========================
# SEARCH FUNCTION
# ==========================
def retrieve_image(query):

    embedding = model.text_embedding(query)

    results = collection.query(
        query_embeddings=[embedding],
        n_results=1
    )

    if len(results["ids"][0]) == 0:
        return None

    metadata = results["metadatas"][0][0]
    distance = results["distances"][0][0]

    return {
        "id": results["ids"][0][0],
        "path": metadata["path"],
        "distance": distance
    }



# ==========================
# Preview Images
# ==========================
def preview_image(image):

    print("\n========== Retrieved Image ==========\n")

    print(f"Image    : {image['id']}")
    print(f"Distance : {image['distance']:.3f}")
    print(f"Path     : {image['path']}")

    img = Image.open(image["path"])
    img.show()

# ==========================
# GEMINI VISION
# ==========================

def ask_image(image, question):

    img = Image.open(image["path"])

    prompt = f"""
You are an AI assistant working as an Image RAG system.

The attached image was retrieved from the vector database.

Use ONLY this image to answer the user's question.

Rules:
- Analyze the image carefully.
- Answer naturally in English.
- If the answer is not visible in the image, say:
  "I cannot determine this from the retrieved image."
- Never return JSON.
- Never return coordinates.
- Never return bounding boxes.

User Question:
{question}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[prompt, img]
    )

    if hasattr(response, "text") and response.text:
        return response.text

    return "No textual response returned."


# ==========================
# CHAT LOOP
# ==========================

print("\n===== Image RAG Started =====")
print("Type 'exit' to quit.\n")

while True:

    question = input("You: ")

    if question.lower() == "exit":
        print("\nGoodbye!")
        break

    try:

        image = retrieve_image(question)

        if image is None:
            print("\nNo relevant image found.\n")
            continue

        preview_image(image)

        answer = ask_image(image, question)

        print("\nBot:", answer)
        print()

    except Exception as e:
        print("\nError:", e)