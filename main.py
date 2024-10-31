from src.retrieve import Retriever
from src.generate import Generator
from src.models.rag_model import RAGModel

if __name__ == "__main__":
    retriever = Retriever()
    retriever.load_documents("data/processed_documents.txt")
    generator = Generator()
    rag = RAGModel(retriever, generator)

    query = "What is deep learning?"
    answer = rag.answer_query(query)
    print("Query:", query)
    print("Answer:", answer)