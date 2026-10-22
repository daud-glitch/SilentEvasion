"""Test script for the SLM shim mock server."""
import subprocess
import time
import urllib.request
import json

# Start the shim in background
proc = subprocess.Popen(
    ["./venv/Scripts/python.exe", "src/shim_colab.py", "--mock"],
    cwd="slm_robustness",
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)

# Wait for it to start
time.sleep(3)

try:
    # Test /health
    resp = urllib.request.urlopen("http://localhost:8000/health")
    print("health:", resp.read().decode())

    # Test /v1/models
    resp = urllib.request.urlopen("http://localhost:8000/v1/models")
    print("models:", resp.read().decode())

    # Test /v1/chat/completions
    payload = json.dumps({
        "model": "qwen2.5-3b",
        "messages": [{"role": "user", "content": "Go until jurong point, crazy.. Available only in bugis n great world la e buffet..."}],
        "temperature": 0.0,
        "max_tokens": 8,
    }).encode()
    req = urllib.request.Request(
        "http://localhost:8000/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req)
    result = resp.read().decode()
    print("chat completion:", result[:300])

except Exception as e:
    print("Error:", e)

# Stop the server
proc.terminate()
proc.wait(timeout=5)
print("server stopped")