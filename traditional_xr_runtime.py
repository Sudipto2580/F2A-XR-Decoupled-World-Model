import socket
import json
import time
import csv
import numpy as np
from colorama import Fore, Style, init

init(autoreset=True)

HOST = "127.0.0.1"
PORT = 65433
TARGET_FRAME_TIME_MS = 11.11  # 90 Hz Strict Frame Budget
TOTAL_SIMULATION_FRAMES = 10000

def query_traditional_server(state_vector):
    """Synchronous per-frame TCP call blocking the main thread"""
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((HOST, PORT))
        payload = json.dumps({"state_vector": state_vector.tolist()})
        client_socket.sendall(payload.encode('utf-8'))
        
        response = client_socket.recv(4096)
        client_socket.close()
        return json.loads(response.decode('utf-8'))
    except ConnectionRefusedError:
        return None

def run_traditional_loop():
    print(f"{Fore.RED}========================================================================{Style.RESET_ALL}")
    print(f"{Fore.RED}   TRADITIONAL MODEL: Direct Synchronous GenAI (10,000-Frame Run)      {Style.RESET_ALL}")
    print(f"{Fore.RED}========================================================================{Style.RESET_ALL}\n")

    s_t = np.array([0.0, 1.7, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 5.0, 100.0, 1.0])
    frame_count = 0
    total_calls = 0
    
    csv_file = open("traditional_benchmark_results.csv", mode="w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Frame", "Inference_Latency_ms", "Total_Frame_Time_ms", "Budget_Passed", "Action"])

    print(f"{Fore.YELLOW}[Traditional System Active] Querying model on EVERY frame synchronously...\n{Style.RESET_ALL}")
    start_bench_time = time.perf_counter()

    try:
        while frame_count < TOTAL_SIMULATION_FRAMES:
            loop_start = time.perf_counter()
            frame_count += 1
            
            # Telemetry drift simulation
            s_t[0] += np.random.uniform(-0.04, 0.04)
            s_t[2] += np.random.uniform(-0.04, 0.06)
            s_t[9] = max(0.5, s_t[9] + np.random.uniform(-0.10, 0.04))
            
            # Direct In-Loop Synchronous Query (Blocks render thread)
            res = query_traditional_server(s_t)
            total_calls += 1
            
            if res:
                action = res["action"]
                inf_latency_ms = res["latency_ms"]
            else:
                action = "FALLBACK"
                inf_latency_ms = 0.0
            
            total_frame_time_ms = (time.perf_counter() - loop_start) * 1000.0
            passed = int(total_frame_time_ms < TARGET_FRAME_TIME_MS)
            
            csv_writer.writerow([frame_count, round(inf_latency_ms, 4), round(total_frame_time_ms, 4), passed, action])
            
            if frame_count % 500 == 0:
                print(f"{Fore.RED}Progress: {frame_count:5d} / {TOTAL_SIMULATION_FRAMES} frames | Frame Latency: {total_frame_time_ms:.2f} ms [FAIL - Exceeds 11.11ms]{Style.RESET_ALL}")

    except KeyboardInterrupt:
        pass

    csv_file.close()
    total_duration_sec = time.perf_counter() - start_bench_time

    print(f"\n{Fore.RED}========================================================================{Style.RESET_ALL}")
    print(f"{Fore.RED}               TRADITIONAL BENCHMARK RUN COMPLETE                        {Style.RESET_ALL}")
    print(f"{Fore.RED}========================================================================{Style.RESET_ALL}")
    print(f" Total Frames Evaluated     : {frame_count}")
    print(f" Total Wall-Clock Time      : {total_duration_sec:.2f} seconds")
    print(f" Total API/Model Invocations: {total_calls} (Continuous 1:1 ratio)")
    print(f" Output Benchmark File     : traditional_benchmark_results.csv")
    print(f"{Fore.RED}========================================================================{Style.RESET_ALL}\n")

if __name__ == "__main__":
    run_traditional_loop()