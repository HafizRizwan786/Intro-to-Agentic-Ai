from embeddings import EmbeddingModel

model = EmbeddingModel()

vector = model.text_embedding("Cat")

print(len(vector))
print(vector[:10])