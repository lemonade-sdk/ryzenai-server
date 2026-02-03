#!/usr/bin/env python3
"""
RyzenAI Server Performance Analysis Tool

This script performs comprehensive performance analysis of the ryzenai-server
including latency, throughput, memory usage, and GPU utilization measurements.

Usage:
    python performance_analysis.py --model-path /path/to/model --backend [mlx-rocm|mlx-cuda|onnx-cpu]
"""

import argparse
import time
import statistics
import subprocess
import threading
import requests
import json
import psutil
import os
from pathlib import Path
import signal
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed


class PerformanceAnalyzer:
    def __init__(self, server_url="http://127.0.0.1:8080"):
        self.server_url = server_url
        self.server_process = None
        self.monitoring = False
        self.monitor_thread = None
        self.metrics = {
            'latency': [],
            'throughput': [],
            'memory_usage': [],
            'cpu_usage': [],
            'gpu_memory': [],
            'gpu_utilization': []
        }

    def start_server(self, model_path, backend_type, host="127.0.0.1", port=8080):
        """Start the ryzenai-server process."""
        cmd = [
            "./build/bin/ryzenai-server",
            "-m", model_path,
            "--backend", backend_type,
            "--host", host,
            "--port", str(port)
        ]

        print(f"Starting server: {' '.join(cmd)}")
        self.server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid if os.name != 'nt' else None
        )

        # Wait for server to be ready
        for _ in range(30):  # 30 second timeout
            try:
                response = requests.get(f"{self.server_url}/health", timeout=2)
                if response.status_code == 200:
                    print("Server is ready")
                    return True
            except:
                pass
            time.sleep(1)

        print("Server failed to start")
        return False

    def stop_server(self):
        """Stop the ryzenai-server process."""
        if self.server_process:
            if os.name == 'nt':
                self.server_process.terminate()
            else:
                os.killpg(os.getpgid(self.server_process.pid), signal.SIGTERM)
            self.server_process.wait()
            print("Server stopped")

    def monitor_resources(self, duration=60):
        """Monitor system resources during performance testing."""
        self.monitoring = True

        def monitor():
            start_time = time.time()
            while self.monitoring and (time.time() - start_time) < duration:
                # CPU and memory usage
                self.metrics['cpu_usage'].append(psutil.cpu_percent(interval=1))
                self.metrics['memory_usage'].append(psutil.virtual_memory().percent)

                # GPU metrics (if available)
                try:
                    gpu_mem = subprocess.run(
                        ["rocm-smi", "--showmeminfo", "vram"],
                        capture_output=True, text=True, timeout=5
                    )
                    if gpu_mem.returncode == 0:
                        lines = gpu_mem.stdout.split('\n')
                        for line in lines:
                            if 'VRAM Total Used Memory' in line:
                                used_mb = int(line.split(':')[1].strip()) // (1024*1024)
                                self.metrics['gpu_memory'].append(used_mb)
                except:
                    pass

                time.sleep(1)

        self.monitor_thread = threading.Thread(target=monitor)
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()

    def measure_latency(self, prompt="What is the capital of France?", num_runs=10):
        """Measure inference latency."""
        latencies = []

        for i in range(num_runs):
            start_time = time.time()

            response = requests.post(
                f"{self.server_url}/v1/chat/completions",
                json={
                    "model": "test",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 50,
                    "temperature": 0.1
                },
                timeout=30
            )

            end_time = time.time()

            if response.status_code == 200:
                latency = (end_time - start_time) * 1000  # Convert to milliseconds
                latencies.append(latency)
                print(".2f")
            else:
                print(f"Request {i+1} failed with status {response.status_code}")

        return latencies

    def measure_throughput(self, prompt="Hello world", num_requests=20, concurrent=4):
        """Measure throughput with concurrent requests."""
        def single_request(req_id):
            start_time = time.time()

            response = requests.post(
                f"{self.server_url}/v1/chat/completions",
                json={
                    "model": "test",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 20,
                    "temperature": 0.1
                },
                timeout=60
            )

            end_time = time.time()

            if response.status_code == 200:
                return (end_time - start_time) * 1000
            else:
                print(f"Request {req_id} failed")
                return None

        # Run concurrent requests
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=concurrent) as executor:
            futures = [executor.submit(single_request, i) for i in range(num_requests)]
            latencies = []
            for future in as_completed(futures):
                latency = future.result()
                if latency is not None:
                    latencies.append(latency)

        total_time = time.time() - start_time
        successful_requests = len(latencies)

        throughput = successful_requests / total_time  # requests per second

        return {
            'throughput_rps': throughput,
            'total_requests': num_requests,
            'successful_requests': successful_requests,
            'total_time': total_time,
            'avg_latency': statistics.mean(latencies) if latencies else 0,
            'p95_latency': statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies) if latencies else 0
        }

    def run_comprehensive_analysis(self, model_path, backend_type):
        """Run comprehensive performance analysis."""
        print("=== RyzenAI Server Performance Analysis ===")
        print(f"Model: {model_path}")
        print(f"Backend: {backend_type}")
        print()

        # Start server
        if not self.start_server(model_path, backend_type):
            print("Failed to start server")
            return

        try:
            # Test 1: Latency measurement
            print("1. Measuring Latency...")
            latencies = self.measure_latency(num_runs=10)
            if latencies:
                print(".2f")
                print(".2f")
                print(".2f")
                print()

            # Test 2: Throughput measurement
            print("2. Measuring Throughput...")
            throughput_results = self.measure_throughput(num_requests=20, concurrent=4)
            print(".2f")
            print(".2f")
            print(".2f")
            print()

            # Test 3: Resource monitoring during load
            print("3. Monitoring Resources During Load...")
            self.monitor_resources(duration=30)

            # Generate some load
            def generate_load():
                for _ in range(10):
                    requests.post(
                        f"{self.server_url}/v1/chat/completions",
                        json={
                            "model": "test",
                            "messages": [{"role": "user", "content": "Tell me a short joke"}],
                            "max_tokens": 30
                        },
                        timeout=30
                    )
                    time.sleep(0.5)

            load_thread = threading.Thread(target=generate_load)
            load_thread.start()
            load_thread.join()

            self.stop_monitoring()

            # Resource usage summary
            if self.metrics['cpu_usage']:
                print(".1f")
            if self.metrics['memory_usage']:
                print(".1f")
            if self.metrics['gpu_memory']:
                print(f"Average GPU Memory: {statistics.mean(self.metrics['gpu_memory']):.0f} MB")
            print()

            # Test 4: Memory efficiency test
            print("4. Memory Efficiency Test...")
            # Test with different context sizes
            context_sizes = [512, 1024, 2048, 4096]
            for ctx_size in context_sizes:
                try:
                    response = requests.post(
                        f"{self.server_url}/v1/chat/completions",
                        json={
                            "model": "test",
                            "messages": [{"role": "user", "content": "x" * (ctx_size // 4)}],  # Approximate token count
                            "max_tokens": 50
                        },
                        timeout=60
                    )
                    if response.status_code == 200:
                        print(f"  Context size {ctx_size}: OK")
                    else:
                        print(f"  Context size {ctx_size}: Failed ({response.status_code})")
                except Exception as e:
                    print(f"  Context size {ctx_size}: Error - {e}")
            print()

        finally:
            self.stop_server()

        print("=== Analysis Complete ===")


def main():
    parser = argparse.ArgumentParser(description="RyzenAI Server Performance Analysis")
    parser.add_argument("--model-path", required=True, help="Path to the model directory")
    parser.add_argument("--backend", required=True,
                       choices=["mlx-rocm", "mlx-cuda", "onnx-cpu"],
                       help="Backend type to test")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=8080, help="Server port")

    args = parser.parse_args()

    analyzer = PerformanceAnalyzer(f"http://{args.host}:{args.port}")
    analyzer.run_comprehensive_analysis(args.model_path, args.backend)


if __name__ == "__main__":
    main()