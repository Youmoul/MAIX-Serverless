import base64
import os
import requests

ENDPOINT_ID = os.environ["RUNPOD_ENDPOINT_ID"]
API_KEY = os.environ["RUNPOD_API_KEY"]

url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/run"
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

conversation = [
    {"role": "user", "content": "Dark orchestral soul, warm Rhodes and expressive strings."},
    {"role": "assistant", "content": "Let's develop the motif through changing voicings and a melodic bass line."},
    {"role": "user", "content": "More harmonic movement, less obvious looping."},
]

payload = {
    "input": {
        "action": "sample",
        "command": "SAMPLE",
        "bpm": 92,
        "duration": 20,
        "conversation": conversation
    }
}

response = requests.post(url, headers=headers, json=payload, timeout=30)
response.raise_for_status()
print(response.json())
