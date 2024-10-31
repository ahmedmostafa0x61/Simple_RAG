from transformers import T5ForConditionalGeneration, T5Tokenizer

class Generator:
    def __init__(self, model_name="t5-small"):
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

    def generate_response(self, context):
        input_text = "question: " + context + " </s>"
        inputs = self.tokenizer(input_text, return_tensors="pt")
        output = self.model.generate(inputs["input_ids"], max_length=50)
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return response
