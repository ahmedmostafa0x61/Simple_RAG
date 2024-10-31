
class RAGModel:
    def __init__(self, retriever, generator):
        self.retriever = retriever
        self.generator = generator

    def answer_query(self, query):
        relevant_docs = self.retriever.retrieve(query)
        context = " ".join(relevant_docs)
        response = self.generator.generate_response(context)
        return response
