import socket
import json
import time
import torch
import numpy as np
from transformers import pipeline
from colorama import Fore, Style, init

init(autoreset=True)

HOST = "127.0.0.1"
PORT = 65433  # Port 65433 for Traditional Synchronous Server

print(f"{Fore.RED}[Traditional Server] Initializing Synchronous In-Loop Server...{Style.RESET_ALL}")

# Load local Hugging Face transformer model on CPU
print(f"{Fore.YELLOW}[Traditional Server] Loading Hugging Face inference pipeline...{Style.RESET_ALL}")
try:
    nlp_model = pipeline("feature-extraction", model="distilbert-base-uncased")
    print(f"{Fore.GREEN}[Traditional Server] DistilBERT loaded successfully.{Style.RESET_ALL}")
except Exception as e:
    print(f"{Fore.RED}[Traditional Server] Error loading pipeline: {e}{Style.RESET_ALL}")
    nlp_model = None

def compute_direct_action(state_vector):
    """
    Traditional Direct GenAI Inference:
    Executes a full forward pass through the transformer model on EVERY frame.
    """
    start_time = time.perf_counter()
    
    d_enemy = state_vector[9]
    health = state_vector[10]
    
    actions = ["COVER", "ATTACK", "RETREAT", "FLANK"]
    scores = []
    
    for action in actions:
        prompt = f"Agent health: {health:.1f}, enemy distance: {d_enemy:.1f}. Action: {action}"
        if nlp_model:
            embeddings = nlp_model(prompt)
            score = float(np.linalg.norm(np.mean(embeddings[0], axis=0)) % 1.0)
        else:
            score = np.random.uniform(0.1, 0.9)
        scores.append(score)
        
    best_action = actions[np.argmax(scores)]
    inference_time_ms = (time.perf_counter() - start_time) * 1000.0
    
    return best_action, inference_time_ms

def start_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    server_socket.listen(5)
    print(f"{Fore.GREEN}[Traditional Server] Listening on {HOST}:{PORT}... Ready for synchronous per-frame queries.{Style.RESET_ALL}\n")
    
    while True:
        conn, addr = server_socket.accept()
        data = conn.recv(4096)
        if not data:
            conn.close()
            continue
            
        request = json.loads(data.decode('utf-8'))
        state_vector = np.array(request["state_vector"])
        
        action, latency_ms = compute_direct_action(state_vector)
        
        response = {
            "action": action,
            "latency_ms": round(latency_ms, 2)
        }
        conn.sendall(json.dumps(response).encode('utf-8'))
        conn.close()

if __name__ == "__main__":
    start_server()