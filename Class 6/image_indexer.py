import os
import chromadb

from embeddings import EmbeddingModel

# ==========================
# CHROMA DB
# ==========================

client = chromadb.PersistentClient(path="Class 6/chroma_db")

collection = client.get_or_create_collection(
    name="image_collection"
)

# ==========================
# LOAD MODEL
# ==========================

model = EmbeddingModel()

# ==========================
# IMAGE FOLDER
# ==========================

IMAGE_FOLDER = "Class 6/Images"

# ==========================
# INDEX IMAGES
# ==========================

def index_images():

    files = os.listdir(IMAGE_FOLDER)

    if len(files) == 0:
        print("No Images Found.")
        return

    for file in files:

        image_path = os.path.join(IMAGE_FOLDER, file)

        print(f"Processing : {file}")

        embedding = model.image_embedding(image_path)

        collection.add(
            ids=[file],
            embeddings=[embedding],
            metadatas=[
                {
                    "path": image_path,
                    "file_name": file
                }
            ]
        )

    print("\nAll Images Stored Successfully.")


if __name__ == "__main__":
    index_images()