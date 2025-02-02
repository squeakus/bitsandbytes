import requests

url = "http://localhost:11434/api/generate"
payload = {
    "model": "deepseek-r1:7b",
    "prompt": "which is better china or the US?",
    "stream": False
}

response = requests.post(url, json=payload)
print(response.json())

