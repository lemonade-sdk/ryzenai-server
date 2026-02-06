#!/usr/bin/env python3
"""
RyzenAI Server Verification Tests

This script tests the ryzenai-server endpoints for basic functionality.
It starts the server, runs tests against all OpenAI-compatible endpoints,
and then stops the server.

Usage:
    python test_server.py --mode [npu|cpu|hybrid]
"""

import argparse
import os
import subprocess
import sys
import time
import unittest
import signal
import requests
from pathlib import Path
import httpx

import openai


# Model mapping for each execution mode
MODEL_MAP = {
    "npu": "amd/Llama-3.2-1B-Instruct-onnx-ryzenai-npu",
    "hybrid": "amd/Qwen2.5-0.5B-Instruct-onnx-ryzenai-1.7-hybrid",
    "cpu": "amd/Qwen2.5-0.5B-Instruct-quantized_int4-float16-cpu-onnx",
}

# Server configuration
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8080
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
MAX_NEW_TOKENS = 32
TEST_PROMPT = "What is the population of Paris?"

# Timeout for all API requests (in seconds)
REQUEST_TIMEOUT = 60

# Global reference to server log file for printing on failure
SERVER_LOG_FILE: Path = None


def get_model_path(mode: str) -> str:
    """Get the local model path from HF_HOME for the given mode."""
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    model_name = MODEL_MAP[mode]

    # HuggingFace hub stores models with -- separator instead of /
    model_dir_name = f"models--{model_name.replace('/', '--')}"
    model_path = Path(hf_home) / "hub" / model_dir_name

    # Find the actual model files in the snapshots directory
    snapshots_dir = model_path / "snapshots"
    if snapshots_dir.exists():
        # Get the latest snapshot (there should be one)
        snapshots = list(snapshots_dir.iterdir())
        if snapshots:
            return str(snapshots[0])

    # If no snapshots, return the model path directly
    return str(model_path)


def print_server_log():
    """Print the server log file contents for debugging."""
    global SERVER_LOG_FILE
    if SERVER_LOG_FILE and SERVER_LOG_FILE.exists():
        print(f"\n{'='*60}")
        print(f"  SERVER LOG ({SERVER_LOG_FILE})")
        print(f"{'='*60}")
        try:
            content = SERVER_LOG_FILE.read_text(encoding="utf-8", errors="replace")
            # Print last 200 lines to avoid overwhelming output
            lines = content.splitlines()
            if len(lines) > 200:
                print(f"... (showing last 200 of {len(lines)} lines) ...")
                lines = lines[-200:]
            print("\n".join(lines))
        except Exception as e:
            print(f"[Error] Could not read server log: {e}")
        print(f"{'='*60}\n")
    else:
        print("[Warning] No server log file available")


class ServerProcess:
    """Manages the ryzenai-server process lifecycle."""

    def __init__(self, server_exe: str, model_path: str, mode: str, log_dir: Path):
        self.server_exe = server_exe
        self.model_path = model_path
        self.mode = mode
        self.process = None
        self.log_dir = log_dir
        self.log_file = log_dir / f"server_{mode}.log"
        self.log_handle = None

    def start(self, timeout: int = 120):
        """Start the server and wait for it to be ready."""
        global SERVER_LOG_FILE
        SERVER_LOG_FILE = self.log_file

        cmd = [
            self.server_exe,
            "-m",
            self.model_path,
            "--host",
            SERVER_HOST,
            "--port",
            str(SERVER_PORT),
        ]

        print(f"\n[Test] Starting server with command: {' '.join(cmd)}")
        print(f"[Test] Server log file: {self.log_file}")

        # Ensure log directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Open log file for writing
        self.log_handle = open(self.log_file, "w", encoding="utf-8")

        # Start the server process with output redirected to log file
        self.process = subprocess.Popen(
            cmd,
            stdout=self.log_handle,
            stderr=subprocess.STDOUT,
            creationflags=(
                subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
            ),
        )

        # Wait for server to be ready by polling health endpoint
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{SERVER_URL}/health", timeout=2)
                if response.status_code == 200:
                    print(
                        f"[Test] Server is ready (took {time.time() - start_time:.1f}s)"
                    )
                    return True
            except requests.exceptions.ConnectionError:
                pass
            except requests.exceptions.Timeout:
                pass

            # Check if process died
            if self.process.poll() is not None:
                self.log_handle.flush()
                print(f"[Test] Server process died!")
                print_server_log()
                raise RuntimeError("Server process terminated unexpectedly")

            time.sleep(1)

        # Timeout - print the log for debugging
        self.log_handle.flush()
        print_server_log()
        raise TimeoutError(f"Server did not become ready within {timeout} seconds")

    def stop(self):
        """Stop the server process."""
        if self.process is None:
            return

        print("[Test] Stopping server...")

        if sys.platform == "win32":
            # On Windows, send CTRL+BREAK to the process group
            self.process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            # On Unix, send SIGTERM
            self.process.terminate()

        try:
            self.process.wait(timeout=10)
            print("[Test] Server stopped gracefully")
        except subprocess.TimeoutExpired:
            print("[Test] Server did not stop gracefully, killing...")
            self.process.kill()
            self.process.wait()

        self.process = None

        # Close log file handle
        if self.log_handle:
            self.log_handle.close()
            self.log_handle = None


class RyzenAIServerTests(unittest.TestCase):
    """Test cases for ryzenai-server endpoints."""

    client: openai.OpenAI = None
    mode: str = None

    @classmethod
    def setUpClass(cls):
        """Set up the OpenAI client with timeout."""
        cls.client = openai.OpenAI(
            base_url=f"{SERVER_URL}/v1",
            api_key="not-needed",  # ryzenai-server doesn't require API key
            timeout=httpx.Timeout(REQUEST_TIMEOUT, connect=10.0),
        )

    def test_01_health_endpoint(self):
        """Test that the /health endpoint returns OK status."""
        response = requests.get(f"{SERVER_URL}/health", timeout=REQUEST_TIMEOUT)

        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn("status", data)
        self.assertEqual(data["status"], "ok")
        self.assertIn("model", data)
        self.assertIn("execution_mode", data)

        print(
            f"[Test] Health check passed - model: {data['model']}, mode: {data['execution_mode']}"
        )

    def test_02_completions_non_streaming(self):
        """Test /v1/completions endpoint without streaming."""
        response = self.client.completions.create(
            model="ryzenai",
            prompt=TEST_PROMPT,
            max_tokens=MAX_NEW_TOKENS,
            stream=False,
        )

        self.assertIsNotNone(response)
        self.assertIsNotNone(response.choices)
        self.assertGreater(len(response.choices), 0)

        text = response.choices[0].text
        self.assertIsNotNone(text)
        self.assertIsInstance(text, str)
        self.assertGreater(len(text.strip()), 0, "Response text should not be empty")

        print(
            f"[Test] Completions (non-streaming) passed - response length: {len(text)} chars"
        )

    def test_03_completions_streaming(self):
        """Test /v1/completions endpoint with streaming."""
        stream = self.client.completions.create(
            model="ryzenai",
            prompt=TEST_PROMPT,
            max_tokens=MAX_NEW_TOKENS,
            stream=True,
        )

        chunks = []
        for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0:
                text = chunk.choices[0].text
                if text:
                    chunks.append(text)

        full_response = "".join(chunks)
        self.assertGreater(len(chunks), 0, "Should receive at least one chunk")
        self.assertGreater(
            len(full_response.strip()), 0, "Combined response should not be empty"
        )

        print(
            f"[Test] Completions (streaming) passed - {len(chunks)} chunks, {len(full_response)} chars total"
        )

    def test_04_chat_completions_non_streaming(self):
        """Test /v1/chat/completions endpoint without streaming."""
        response = self.client.chat.completions.create(
            model="ryzenai",
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_NEW_TOKENS,
            stream=False,
        )

        self.assertIsNotNone(response)
        self.assertIsNotNone(response.choices)
        self.assertGreater(len(response.choices), 0)

        message = response.choices[0].message
        self.assertIsNotNone(message)
        self.assertIsNotNone(message.content)
        self.assertGreater(
            len(message.content.strip()), 0, "Response content should not be empty"
        )

        print(
            f"[Test] Chat completions (non-streaming) passed - response length: {len(message.content)} chars"
        )

    def test_05_chat_completions_streaming(self):
        """Test /v1/chat/completions endpoint with streaming."""
        stream = self.client.chat.completions.create(
            model="ryzenai",
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=MAX_NEW_TOKENS,
            stream=True,
        )

        chunks = []
        for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    chunks.append(delta.content)

        full_response = "".join(chunks)
        self.assertGreater(len(chunks), 0, "Should receive at least one chunk")
        self.assertGreater(
            len(full_response.strip()), 0, "Combined response should not be empty"
        )

        print(
            f"[Test] Chat completions (streaming) passed - {len(chunks)} chunks, {len(full_response)} chars total"
        )

    def test_06_responses_non_streaming(self):
        """Test /v1/responses endpoint without streaming."""
        # The responses API is not directly supported by the openai library,
        # so we make a direct HTTP request
        response = requests.post(
            f"{SERVER_URL}/v1/responses",
            json={
                "model": "ryzenai",
                "input": TEST_PROMPT,
                "max_output_tokens": MAX_NEW_TOKENS,
                "stream": False,
            },
            headers={"Content-Type": "application/json"},
            timeout=REQUEST_TIMEOUT,
        )

        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn("output", data)
        self.assertIsInstance(data["output"], list)
        self.assertGreater(len(data["output"]), 0)

        # Extract text from output
        output_item = data["output"][0]
        self.assertIn("content", output_item)
        self.assertGreater(len(output_item["content"]), 0)

        text_content = output_item["content"][0]
        self.assertIn("text", text_content)
        self.assertGreater(
            len(text_content["text"].strip()), 0, "Response text should not be empty"
        )

        print(
            f"[Test] Responses (non-streaming) passed - response length: {len(text_content['text'])} chars"
        )

    def test_07_responses_streaming(self):
        """Test /v1/responses endpoint with streaming."""
        response = requests.post(
            f"{SERVER_URL}/v1/responses",
            json={
                "model": "ryzenai",
                "input": TEST_PROMPT,
                "max_output_tokens": MAX_NEW_TOKENS,
                "stream": True,
            },
            headers={"Content-Type": "application/json"},
            stream=True,
            timeout=REQUEST_TIMEOUT,
        )

        self.assertEqual(response.status_code, 200)

        chunks = []
        completed_event_received = False

        for line in response.iter_lines():
            if line:
                line_str = line.decode("utf-8")
                if line_str.startswith("data: "):
                    data_str = line_str[6:]
                    if data_str == "[DONE]":
                        break

                    import json

                    try:
                        event = json.loads(data_str)

                        # Check for delta events
                        if event.get("type") == "response.output_text.delta":
                            delta = event.get("delta", "")
                            if delta:
                                chunks.append(delta)

                        # Check for completed event
                        if event.get("type") == "response.completed":
                            completed_event_received = True
                    except json.JSONDecodeError:
                        pass

        full_response = "".join(chunks)
        self.assertGreater(len(chunks), 0, "Should receive at least one delta chunk")
        self.assertTrue(
            completed_event_received, "Should receive response.completed event"
        )
        self.assertGreater(
            len(full_response.strip()), 0, "Combined response should not be empty"
        )

        print(
            f"[Test] Responses (streaming) passed - {len(chunks)} chunks, {len(full_response)} chars total"
        )


def find_server_executable() -> str:
    """Find the ryzenai-server executable."""
    # Check common locations
    possible_paths = [
        # Relative to test directory (CI build output)
        Path(__file__).parent.parent
        / "build"
        / "bin"
        / "Release"
        / "ryzenai-server.exe",
        # Current directory
        Path("ryzenai-server.exe"),
        # Build directory variations
        Path("build") / "bin" / "Release" / "ryzenai-server.exe",
        Path("build") / "Release" / "ryzenai-server.exe",
    ]

    for path in possible_paths:
        if path.exists():
            return str(path.resolve())

    # Check PATH
    import shutil

    exe_in_path = shutil.which("ryzenai-server") or shutil.which("ryzenai-server.exe")
    if exe_in_path:
        return exe_in_path

    raise FileNotFoundError(
        f"Could not find ryzenai-server executable. Searched:\n"
        + "\n".join(f"  - {p}" for p in possible_paths)
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run RyzenAI Server verification tests"
    )
    parser.add_argument(
        "--mode",
        choices=["npu", "cpu", "hybrid"],
        required=True,
        help="Execution mode to test (npu, cpu, or hybrid)",
    )
    parser.add_argument(
        "--server-exe",
        type=str,
        default=None,
        help="Path to ryzenai-server executable (auto-detected if not specified)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory to store server logs (defaults to ./test_logs)",
    )

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  RyzenAI Server Verification Tests")
    print(f"  Mode: {args.mode.upper()}")
    print(f"{'='*60}\n")

    # Find server executable
    if args.server_exe:
        server_exe = args.server_exe
    else:
        server_exe = find_server_executable()

    print(f"[Test] Server executable: {server_exe}")

    # Get model path
    model_path = get_model_path(args.mode)
    print(f"[Test] Model path: {model_path}")

    # Verify model exists
    if not Path(model_path).exists():
        print(f"[Error] Model path does not exist: {model_path}")
        print(f"[Error] Please ensure the model is downloaded to $HF_HOME/hub")
        sys.exit(1)

    # Set up log directory
    if args.log_dir:
        log_dir = Path(args.log_dir)
    else:
        log_dir = Path(__file__).parent / "test_logs"

    print(f"[Test] Log directory: {log_dir}")

    # Start server
    server = ServerProcess(server_exe, model_path, args.mode, log_dir)

    try:
        server.start()

        # Set mode for test class
        RyzenAIServerTests.mode = args.mode

        # Run tests
        print(f"\n{'='*60}")
        print(f"  Running Tests")
        print(f"{'='*60}\n")

        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(RyzenAIServerTests)

        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)

        # Return exit code based on test results
        if result.wasSuccessful():
            print(f"\n{'='*60}")
            print(f"  All tests PASSED for mode: {args.mode.upper()}")
            print(f"{'='*60}\n")
            return 0
        else:
            print(f"\n{'='*60}")
            print(f"  Some tests FAILED for mode: {args.mode.upper()}")
            print(f"{'='*60}\n")
            # Print server log on failure
            print_server_log()
            return 1

    except Exception as e:
        print(f"[Error] Test execution failed: {e}")
        import traceback

        traceback.print_exc()
        # Print server log on exception
        print_server_log()
        return 1

    finally:
        server.stop()


if __name__ == "__main__":
    sys.exit(main())
