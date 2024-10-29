from retrieve import Retriever

if __name__ == "__main__":
    retriever = Retriever()
    retriever.load_documents("data/processed_documents.txt")