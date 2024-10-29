import os
from transformers import AutoTokenizer

def preprocess_documents(input_folder, output_file="processed_documents.txt"):
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    with open(output_file, "w", encoding="utf-8") as out_f:
        for filename in os.listdir(input_folder):
            if filename.endswith(".txt"):
                with open(os.path.join(input_folder, filename), "r", encoding="utf-8") as f:
                    text = f.read()
                    tokens = tokenizer.tokenize(text)
                    out_f.write(" ".join(tokens) + "\n")

if __name__ == "__main__":
    preprocess_documents("documents")
    print("Documents preprocessed and saved.")