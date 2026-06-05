from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

with open("./Wikipedia Entries/39477.chunks.txt", "r") as f:
    content = f.readlines()
content = "".join(content).split("###")[1:]
content = [s[19:] for s in content]

embeddings = model.tokenizer.encode(content)
print([len(e) for e in embeddings])
