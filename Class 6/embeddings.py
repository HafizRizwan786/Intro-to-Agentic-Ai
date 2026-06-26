from sentence_transformers import SentenceTransformer
from PIL import Image

class EmbeddingModel:

    def __init__(self):
        print("Loading CLIP Model...")
        self.model = SentenceTransformer("clip-ViT-B-32")
        print("Model Loaded Successfully.\n")

    # ----------------------------
    # Convert Image -> Vector
    # ----------------------------
    def image_embedding(self, image_path):
        image = Image.open(image_path).convert("RGB")
        embedding = self.model.encode(image)
        return embedding.tolist()

    # ----------------------------
    # Convert Text -> Vector
    # ----------------------------
    def text_embedding(self, text):
        embedding = self.model.encode(text)
        return embedding.tolist()