import os
from openai import OpenAI

# 1. SETUP - Read from environment variables
API_URL = os.getenv("HF_SPACE_URL", "https://asreesaila-mojo_monday.hf.space/v1")
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set. Please set it with: export HF_TOKEN='your_token'")

# 2. CONNECT
client = OpenAI(
    base_url=API_URL,
    api_key=HF_TOKEN
)

# 3. CHAT
print("Connecting to Space...")
response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a poem in 4 lines about Docker containers."}
    ],
    max_tokens=100
)

print("\nResponse from Server:")
print(response.choices[0].message.content)