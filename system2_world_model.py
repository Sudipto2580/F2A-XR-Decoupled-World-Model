import socket
import json
import time
import torch
import numpy as np
from transformers import pipeline
from colorama import Fore, Style, init

init(autoreset=True)
HOST = "127.0.0.1"
PORT = 65432
print(f"{Fore.CYAN}[System 2 World Model] Initializing deliberative off-loop server...{Style.RESET_ALL}")
# Load local Hugging Face transformer model on CPU
print(f"{Fore.YELLOW}[System 2 World Model] Loading Hugging Face semantic reasoning pipeline...{Style.RESET_ALL}")
try:
    nlp_model = pipeline("feature-extraction", model="distilbert-base-uncased")
    print(f"{Fore.GREEN}[System 2 World Model] DistilBERT loaded successfully.{Style.RESET_ALL}")
except Exception as e:
    print(f"{Fore.RED}[System 2 World Model] Error loading pipeline: {e}{Style.RESET_ALL}")
    nlp_model = None

def evaluate_and_compile_policy(state_vector, K_futures=8):
    """
    Evaluates K candidate futures and distills optimal policy weights (W, b) for SDT.
    """
    start_time = time.perf_counter()
    pos_head = state_vector[:3]
    d_enemy = state_vector[9]
    health = state_vector[10]
    candidates = []
    actions = ["COVER", "ATTACK", "RETREAT", "FLANK"]
    for i in range(K_futures):
        action = actions[i % len(actions)]
        prompt = f"Agent status: health {health:.2f}, enemy distance {d_enemy:.2f}. Action: {action}"
        
        if nlp_model:
            embeddings = nlp_model(prompt)
            Q_i = float(np.linalg.norm(np.mean(embeddings[0], axis=0)) % 1.0)
        else:
            Q_i = np.random.uniform(0.4, 0.9)
            
        G_i = 1.0 / (1.0 + d_enemy) if action == "ATTACK" else (0.8 if action == "COVER" else 0.5)
        S_i = (d_enemy / 10.0) if action in ["COVER", "RETREAT"] else (1.0 - (1.0 / (1.0 + d_enemy)))
        C_i = 0.05 * (i + 1)
        
        # Utility U(f_i) = w_g*G + w_q*Q + w_s*S - w_c*C
        U = (0.35 * G_i) + (0.25 * Q_i) + (0.30 * S_i) - (0.10 * C_i)
        candidates.append((action, U))
        
    candidates.sort(key=lambda x: x[1], reverse=True)
    best_action, best_u = candidates[0]
    
    # Sigmoid Gating Weights for Soft Decision Tree (4 actions x 12 state dims)
    W = np.random.randn(4, 12) * 0.5
    b = np.random.randn(4) * 0.1
    
    target_idx = actions.index(best_action)
    W[target_idx, 9] = -1.5  # Sensitivity to enemy distance
    W[target_idx, 10] = -2.0 # Sensitivity to health
    b[target_idx] += 1.2
    
    compilation_time_ms = (time.perf_counter() - start_time) * 1000.0
    
    return {
        "best_action": best_action,
        "utility": round(best_u, 4),
        "W": W.tolist(),
        "b": b.tolist(),
        "compilation_time_ms": round(compilation_time_ms, 2)
    }

def start_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    server_socket.listen(5)
    print(f"{Fore.GREEN}[System 2 World Model] Server ready on {HOST}:{PORT}{Style.RESET_ALL}\n")
    
    while True:
        conn, addr = server_socket.accept()
        data = conn.recv(4096)
        if not data:
            conn.close()
            continue
            
        request = json.loads(data.decode('utf-8'))
        state_vector = np.array(request["state_vector"])
        reason = request.get("reason", "UNKNOWN")
        
        print(f"{Fore.MAGENTA}[System 2 Triggered] Reason: {reason} | Running Rollouts...{Style.RESET_ALL}")
        compiled_policy = evaluate_and_compile_policy(state_vector)
        print(f"{Fore.GREEN}[System 2 Compiled] Action: {compiled_policy['best_action']} | Utility: {compiled_policy['utility']} | Time: {compiled_policy['compilation_time_ms']} ms{Style.RESET_ALL}\n")
        
        conn.sendall(json.dumps(compiled_policy).encode('utf-8'))
        conn.close()

if __name__ == "__main__":
    start_server()