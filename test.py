import requests

# URL of your locally running Guardrail API
URL = "http://127.0.0.1:8000/api/v1/validate"

# Simulate a 5-dimensional embedding model space (e.g., [val1, val2, val3, val4, val5])
# In reality, this would be a 1536-dim vector from OpenAI or 384-dim from HuggingFace
mock_llm_response_vector = [0.9, 0.1, 0.0, 0.3, 0.7]

mock_retrieved_chunks = [
    [0.12, 0.19, 0.79, 0.88, 0.01],  
    [0.05, 0.22, 0.85, 0.91, 0.03]       # Chunk 2: Completely different vector (Irrelevant)
]

# Create the JSON payload according to our Pydantic contract
payload = {
    "llm_response_vector": mock_llm_response_vector,
    "retrieved_chunk_vectors": mock_retrieved_chunks,
    "threshold": 0.25
}

# Send the POST request to your microservice
print("Sending vectors to Guardrail Engine...")
response = requests.post(URL, json=payload)

# Print the result
print("\nResponse Status Code:", response.status_code)
print("Engine Output:", response.json())