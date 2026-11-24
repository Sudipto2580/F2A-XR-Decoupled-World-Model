import socket
import json
import time
import csv
import numpy as np
from colorama import Fore, Style, init

init(autoreset=True)

HOST = "127.0.0.1"
PORT = 65432
TARGET_FRAME_TIME_MS = 11.11  # 90 Hz Frame Budget
DRIFT_THRESHOLD_DELTA = 1.2    # ETPR Spatial Drift Threshold
TOTAL_SIMULATION_FRAMES = 10000

class SoftDecisionTreePolicy:
    """Microsecond Soft Decision Tree Policy Executor"""
    def __init__(self):
        self.W = np.random.randn(4, 12) * 0.1
        self.b = np.zeros(4)
        self.actions = ["COVER", "ATTACK", "RETREAT", "FLANK"]

    def update_weights(self, W_new, b_new):
        self.W = np.array(W_new)
        self.b = np.array(b_new)

    def evaluate(self, state_vector):
        logits = np.dot(self.W, state_vector) + self.b
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / np.sum(exp_logits)
        action_idx = np.argmax(probabilities)
        return self.actions[action_idx], probabilities

def request_system2_regeneration(state_vector, reason):
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((HOST, PORT))
        payload = json.dumps({"state_vector": state_vector.tolist(), "reason": reason})
        client_socket.sendall(payload.encode('utf-8'))
        response = client_socket.recv(4096)
        client_socket.close()
        return json.loads(response.decode('utf-8'))
    except ConnectionRefusedError:
        return None

def run_xr_loop():
    print(f"{Fore.CYAN}========================================================================{Style.RESET_ALL}")
    print(f"{Fore.CYAN}   F2A-XR Architecture: 10,000-Frame Production Benchmark Execution     {Style.RESET_ALL}")
    print(f"{Fore.CYAN}========================================================================{Style.RESET_ALL}\n")

    policy_tree = SoftDecisionTreePolicy()
    s_t = np.array([0.0, 1.7, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 5.0, 100.0, 1.0])
    s_compiled = np.copy(s_t)
    
    frame_count = 0
    system2_invocations = 0
    
    csv_file = open("benchmark_results.csv", mode="w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Frame", "SDT_Latency_ms", "Total_Frame_Time_ms", "Drift_Delta", "ETPR_Triggered", "Action"])

    print(f"{Fore.YELLOW}[Initialization] Requesting initial policy from System 2...{Style.RESET_ALL}")
    init_res = request_system2_regeneration(s_t, reason="INITIALIZATION")
    if init_res:
        policy_tree.update_weights(init_res["W"], init_res["b"])
        system2_invocations += 1

    print(f"{Fore.GREEN}[System 1 Active] Running 10,000 frames... (Logging progress every 1,000 frames)\n{Style.RESET_ALL}")

    start_bench_time = time.perf_counter()

    try:
        while frame_count < TOTAL_SIMULATION_FRAMES:
            loop_start = time.perf_counter()
            frame_count += 1
            
            # Simulate 6-DOF spatial telemetry drift
            s_t[0] += np.random.uniform(-0.04, 0.04)
            s_t[2] += np.random.uniform(-0.04, 0.06)
            s_t[9] = max(0.5, s_t[9] + np.random.uniform(-0.10, 0.04))
            
            # Inject dynamic event spikes every 2,500 frames
            if frame_count % 2500 == 0:
                s_t[9] = 0.7  # Enemy rush proximity spike
                s_t[10] = max(10.0, s_t[10] - 25.0)
                
            # --- IN-LOOP SDT EXECUTION (< 0.05 ms BUDGET) ---
            sdt_start = time.perf_counter()
            action, proba = policy_tree.evaluate(s_t)
            sdt_latency_ms = (time.perf_counter() - sdt_start) * 1000.0
            
            # --- ETPR Spatial Drift Monitoring ---
            drift_delta = float(np.linalg.norm(s_t[:3] - s_compiled[:3]) + abs(s_t[9] - s_compiled[9]))
            
            triggered = False
            trigger_reason = ""
            if drift_delta > DRIFT_THRESHOLD_DELTA:
                triggered = True
                trigger_reason = f"SPATIAL DRIFT (Δs = {drift_delta:.2f} > {DRIFT_THRESHOLD_DELTA})"
            elif frame_count % 2500 == 0:
                triggered = True
                trigger_reason = "CRITICAL GAME EVENT (Health/Proximity Spike)"
                
            if triggered:
                reg_res = request_system2_regeneration(s_t, reason=trigger_reason)
                if reg_res:
                    policy_tree.update_weights(reg_res["W"], reg_res["b"])
                    s_compiled = np.copy(s_t)
                    system2_invocations += 1
            
            total_frame_time_ms = (time.perf_counter() - loop_start) * 1000.0
            
            # Save data row
            csv_writer.writerow([frame_count, round(sdt_latency_ms, 5), round(total_frame_time_ms, 4), round(drift_delta, 4), int(triggered), action])
            
            # Milestone Progress Log
            if frame_count % 1000 == 0:
                print(f"{Fore.GREEN}Progress: {frame_count:5d} / {TOTAL_SIMULATION_FRAMES} frames | System 2 Invocations: {system2_invocations:3d} | Current Drift: {drift_delta:.2f}{Style.RESET_ALL}")
                
            # 90 Hz Rate Limiting (Simulating real headset clock)
            sleep_time = max(0.0, (TARGET_FRAME_TIME_MS / 1000.0) - (time.perf_counter() - loop_start))
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        pass

    csv_file.close()
    total_duration_sec = time.perf_counter() - start_bench_time

    print(f"\n{Fore.CYAN}========================================================================{Style.RESET_ALL}")
    print(f"{Fore.CYAN}                 10,000-FRAME BENCHMARK COMPLETE                        {Style.RESET_ALL}")
    print(f"{Fore.CYAN}========================================================================{Style.RESET_ALL}")
    print(f" Total Frames Evaluated      : {frame_count}")
    print(f" Total Wall-Clock Time       : {total_duration_sec:.2f} seconds")
    print(f" System 2 API Invocations   : {system2_invocations} (vs {frame_count} in Direct LLM)")
    print(f" Cost Amortization Savings  : {((1.0 - (system2_invocations / frame_count)) * 100):.2f}%")
    print(f" Benchmark CSV Exported To  : benchmark_results.csv")
    print(f"{Fore.CYAN}========================================================================{Style.RESET_ALL}\n")

if __name__ == "__main__":
    run_xr_loop()