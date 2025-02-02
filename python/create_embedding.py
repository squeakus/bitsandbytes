import ollama

response = ollama.embeddings("bge-large", "Hello, world!")
embedding = response['embedding']

print(embedding)  # This will print the list of float values

