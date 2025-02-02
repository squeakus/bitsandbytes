import requests
import json

url = "http://localhost:11434/api/embeddings"
data = {
    "model": "bge-large",
    "prompt": "this is pretty good"
}

response = requests.post(url, headers={"Content-Type": "application/json"}, data=json.dumps(data))

if response.status_code == 200:
    embedding = response.json()["embedding"]
    print(embedding)
else:
    print("Error:", response.text)

