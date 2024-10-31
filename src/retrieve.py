from sentence_transformers import SentenceTransformer, util

class Retriever:
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)
        self.documents = []

    def load_documents(self, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            self.documents = [line.strip() for line in f]

    def retrieve(self, query, top_k=5):
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        doc_embeddings = self.model.encode(self.documents, convert_to_tensor=True)
        scores = util.cos_sim(query_embedding, doc_embeddings)[0]
        best_scores = sorted(zip(scores, self.documents), reverse=True)[:top_k]
        return [doc for score, doc in best_scores]

