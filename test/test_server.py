#!/usr/bin/env python3
"""
RyzenAI Server Verification Tests

Starts ryzenai-server on one model profile, runs the OpenAI-compatible endpoint
tests that profile supports, and stops the server.

Usage:
    python test_server.py --profile [text|tools|gemma4-e2b|gemma4-e4b|qwen3-vl]
    python test_server.py --profile gemma4-e2b --model-path C:\\models\\gemma-4-e2b
"""

import argparse
import base64
import json
import os
import signal
import struct
import subprocess
import sys
import time
import unittest
import zlib
from pathlib import Path

import httpx
import openai
import requests

# Each profile is one 1.9.0 NPU checkpoint plus the features it is expected to
# support. Tests for features a profile lacks are skipped.
PROFILES = {
    "text": {
        "checkpoint": "amd/Llama-3.2-1B-Instruct_rai_1.9.0_npu",
        "features": {"text"},
    },
    "tools": {
        "checkpoint": "amd/Llama-3.2-1B-Instruct_rai_1.9.0_npu",
        "features": {"text", "tools"},
    },
    "gemma4-e2b": {
        "checkpoint": "amd/gemma-4-e2b-it-mobius-int4_rai_1.9.0_npu",
        "features": {"text", "vision", "audio"},
    },
    "gemma4-e4b": {
        "checkpoint": "amd/gemma-4-e4b-it-mobius-int4_rai_1.9.0_npu",
        "features": {"text", "vision", "audio"},
    },
    "qwen3-vl": {
        "checkpoint": "amd/Qwen3-VL-4B-Instruct_rai_1.9.0_npu",
        "features": {"text", "vision"},
    },
}

# Server configuration
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8080
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
MAX_NEW_TOKENS = 32
TEST_PROMPT = "What is the population of Paris?"

# Generous enough for the first image/audio request, which compiles the encoders
REQUEST_TIMEOUT = 300

SPEECH_WAV = Path(__file__).parent / "assets" / "speech.wav"
SPEECH_KEYWORD = "working"  # speech.wav says "Just seeing if this is working."

CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "calculator_calculate",
        "description": "Evaluate a math expression",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The expression to evaluate",
                }
            },
            "required": ["expression"],
        },
    },
}

# Global reference to server log file for printing on failure
SERVER_LOG_FILE: Path = None


def resolve_model_path(checkpoint: str) -> str:
    """Return the local snapshot directory for a checkpoint, downloading it if needed."""
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=checkpoint)


def solid_png(rgb, size=64) -> bytes:
    """Encode a solid-color RGB PNG without needing an imaging library."""

    def chunk(tag, data):
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    row = b"\x00" + bytes(rgb) * size
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(row * size))
        + chunk(b"IEND", b"")
    )


def image_part(png: bytes) -> dict:
    data = base64.b64encode(png).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}}


def audio_part(wav: bytes) -> dict:
    data = base64.b64encode(wav).decode()
    return {"type": "input_audio", "input_audio": {"data": data, "format": "wav"}}


RED_IMAGE = image_part(solid_png((220, 20, 20)))
COLOR_QUESTION = {
    "type": "text",
    "text": "What color is this image? Answer with a single word.",
}


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

    def __init__(self, server_exe: str, model_path: str, profile: str, log_dir: Path):
        self.server_exe = server_exe
        self.model_path = model_path
        self.process = None
        self.log_dir = log_dir
        self.log_file = log_dir / f"server_{profile}.log"
        self.log_handle = None

    def start(self, timeout: int = 300):
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

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_handle = open(self.log_file, "w", encoding="utf-8")

        self.process = subprocess.Popen(
            cmd,
            stdout=self.log_handle,
            stderr=subprocess.STDOUT,
            creationflags=(
                subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
            ),
        )

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{SERVER_URL}/health", timeout=2)
                if response.status_code == 200:
                    print(
                        f"[Test] Server is ready (took {time.time() - start_time:.1f}s)"
                    )
                    return True
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                pass

            if self.process.poll() is not None:
                self.log_handle.flush()
                print("[Test] Server process died!")
                print_server_log()
                raise RuntimeError("Server process terminated unexpectedly")

            time.sleep(1)

        self.log_handle.flush()
        print_server_log()
        raise TimeoutError(f"Server did not become ready within {timeout} seconds")

    def stop(self):
        """Stop the server process."""
        if self.process is None:
            return

        print("[Test] Stopping server...")

        if sys.platform == "win32":
            self.process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            self.process.terminate()

        try:
            self.process.wait(timeout=10)
            print("[Test] Server stopped gracefully")
        except subprocess.TimeoutExpired:
            print("[Test] Server did not stop gracefully, killing...")
            self.process.kill()
            self.process.wait()

        self.process = None

        if self.log_handle:
            self.log_handle.close()
            self.log_handle = None


def requires(feature: str):
    """Skip the test unless the active profile supports `feature`."""

    def decorator(test_func):
        def wrapper(self, *args, **kwargs):
            if feature not in self.features:
                self.skipTest(f"profile does not support {feature}")
            return test_func(self, *args, **kwargs)

        wrapper.__name__ = test_func.__name__
        wrapper.__doc__ = test_func.__doc__
        return wrapper

    return decorator


class RyzenAIServerTests(unittest.TestCase):
    """Test cases for ryzenai-server endpoints."""

    client: openai.OpenAI = None
    features: set = set()

    @classmethod
    def setUpClass(cls):
        cls.client = openai.OpenAI(
            base_url=f"{SERVER_URL}/v1",
            api_key="not-needed",  # ryzenai-server doesn't require API key
            timeout=httpx.Timeout(REQUEST_TIMEOUT, connect=10.0),
        )

    def chat(self, messages, **kwargs):
        response = self.client.chat.completions.create(
            model="ryzenai",
            messages=messages,
            max_tokens=kwargs.pop("max_tokens", MAX_NEW_TOKENS),
            temperature=0.0,
            **kwargs,
        )
        return response.choices[0].message

    def chat_stream_text(self, messages):
        stream = self.client.chat.completions.create(
            model="ryzenai",
            messages=messages,
            max_tokens=MAX_NEW_TOKENS,
            temperature=0.0,
            stream=True,
        )
        return "".join(
            chunk.choices[0].delta.content or ""
            for chunk in stream
            if chunk.choices and chunk.choices[0].delta
        )

    def raw_chat(self, messages):
        return requests.post(
            f"{SERVER_URL}/v1/chat/completions",
            json={"model": "ryzenai", "messages": messages, "max_tokens": 8},
            timeout=REQUEST_TIMEOUT,
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
        message = self.chat([{"role": "user", "content": TEST_PROMPT}])

        self.assertIsNotNone(message.content)
        self.assertGreater(
            len(message.content.strip()), 0, "Response content should not be empty"
        )

        print(
            f"[Test] Chat completions (non-streaming) passed - response length: {len(message.content)} chars"
        )

    def test_05_chat_completions_streaming(self):
        """Test /v1/chat/completions endpoint with streaming."""
        full_response = self.chat_stream_text(
            [{"role": "user", "content": TEST_PROMPT}]
        )
        self.assertGreater(
            len(full_response.strip()), 0, "Combined response should not be empty"
        )

        print(
            f"[Test] Chat completions (streaming) passed - {len(full_response)} chars total"
        )

    def test_06_responses_non_streaming(self):
        """Test /v1/responses endpoint without streaming."""
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
            if not line:
                continue
            line_str = line.decode("utf-8")
            if not line_str.startswith("data: "):
                continue
            data_str = line_str[6:]
            if data_str == "[DONE]":
                break
            try:
                event = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "response.output_text.delta" and event.get("delta"):
                chunks.append(event["delta"])
            if event.get("type") == "response.completed":
                completed_event_received = True

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

    def test_08_chat_answer_is_correct(self):
        """A factual question gets a correct answer, so garbage output fails."""
        message = self.chat(
            [
                {
                    "role": "user",
                    "content": "What is the capital of France? Answer with one word.",
                }
            ]
        )
        self.assertIn("paris", message.content.lower(), message.content)

    def test_08b_long_output_stays_coherent(self):
        """Continuation must stay coherent past the first token (broken NPU checkpoints produce correct first token then garbage)."""
        response = self.client.completions.create(
            model="ryzenai",
            prompt="1, 2, 3, 4, 5,",
            max_tokens=20,
            temperature=0.0,
        )
        text = response.choices[0].text
        digits = "".join(c for c in text if c.isdigit() or c == ",")
        self.assertTrue(digits.startswith("6,7,8,9,10"), repr(text))

    def test_09_chat_text_content_parts(self):
        """Content sent as an array of text parts is accepted and answered."""
        message = self.chat(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is the capital of France?"},
                        {"type": "text", "text": "Answer with one word."},
                    ],
                }
            ]
        )
        self.assertIn("paris", message.content.lower(), message.content)

    @requires("tools")
    def test_10_tool_calls(self):
        """The model calls the provided tool with the requested argument."""
        message = self.chat(
            [
                {
                    "role": "user",
                    "content": "Run the calculator_calculate tool with expression set to 1+1",
                }
            ],
            tools=[CALCULATOR_TOOL],
            max_tokens=100,
        )
        self.assertTrue(
            message.tool_calls, f"Expected tool_calls; content was: {message.content!r}"
        )
        call = message.tool_calls[0].function
        self.assertEqual(call.name, "calculator_calculate")
        args = json.loads(call.arguments)
        # Llama uses "parameters" key internally but server normalises to the value dict;
        # check the expression appears somewhere in the args values.
        self.assertIn(
            "1+1",
            list(args.values()),
            f"Expected expression '1+1' in tool args: {args}",
        )

    @requires("vision")
    def test_11_image_input(self):
        """The model sees an inline image and names its color."""
        message = self.chat([{"role": "user", "content": [RED_IMAGE, COLOR_QUESTION]}])
        self.assertIn("red", message.content.lower(), message.content)

    @requires("vision")
    def test_12_image_input_streaming(self):
        """Streaming responses work for image input."""
        text = self.chat_stream_text(
            [{"role": "user", "content": [RED_IMAGE, COLOR_QUESTION]}]
        )
        self.assertIn("red", text.lower(), text)

    @requires("vision")
    def test_13_image_with_history(self):
        """System prompt and earlier turns are kept alongside an image turn."""
        message = self.chat(
            [
                {"role": "system", "content": "You are a concise assistant."},
                {"role": "user", "content": "Hi! I will show you a picture next."},
                {"role": "assistant", "content": "Sure, go ahead."},
                {"role": "user", "content": [RED_IMAGE, COLOR_QUESTION]},
            ]
        )
        self.assertIn("red", message.content.lower(), message.content)

    @requires("audio")
    def test_14_audio_input(self):
        """The model hears an inline WAV clip and repeats what was said."""
        message = self.chat(
            [
                {
                    "role": "user",
                    "content": [
                        audio_part(SPEECH_WAV.read_bytes()),
                        {"type": "text", "text": "Transcribe this audio exactly."},
                    ],
                }
            ]
        )
        self.assertIn(SPEECH_KEYWORD, message.content.lower(), message.content)

    @requires("audio")
    def test_15_image_and_audio_input(self):
        """One message can carry both an image and an audio clip."""
        message = self.chat(
            [
                {
                    "role": "user",
                    "content": [
                        RED_IMAGE,
                        audio_part(SPEECH_WAV.read_bytes()),
                        {
                            "type": "text",
                            "text": "First transcribe the audio, then name the color of the image.",
                        },
                    ],
                }
            ],
            max_tokens=64,
        )
        text = message.content.lower()
        self.assertIn(SPEECH_KEYWORD, text, message.content)
        self.assertIn("red", text, message.content)

    def test_16_media_rejected_without_encoder(self):
        """Media the model can't consume is rejected with 400 instead of being dropped."""
        if "vision" not in self.features:
            response = self.raw_chat(
                [{"role": "user", "content": [RED_IMAGE, COLOR_QUESTION]}]
            )
        elif "audio" not in self.features:
            response = self.raw_chat(
                [{"role": "user", "content": [audio_part(SPEECH_WAV.read_bytes())]}]
            )
        else:
            self.skipTest("profile accepts both images and audio")
        self.assertEqual(response.status_code, 400, response.text)

    def test_17_remote_image_url_rejected(self):
        """Only base64 data URLs are supported; remote URLs get a clear 400."""
        response = self.raw_chat(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/a.png"},
                        },
                        COLOR_QUESTION,
                    ],
                }
            ]
        )
        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("data URL", response.text)


def find_server_executable() -> str:
    """Find the ryzenai-server executable."""
    possible_paths = [
        Path(__file__).parent.parent
        / "build"
        / "bin"
        / "Release"
        / "ryzenai-server.exe",
        Path("ryzenai-server.exe"),
        Path("build") / "bin" / "Release" / "ryzenai-server.exe",
        Path("build") / "Release" / "ryzenai-server.exe",
    ]

    for path in possible_paths:
        if path.exists():
            return str(path.resolve())

    import shutil

    exe_in_path = shutil.which("ryzenai-server") or shutil.which("ryzenai-server.exe")
    if exe_in_path:
        return exe_in_path

    raise FileNotFoundError(
        "Could not find ryzenai-server executable. Searched:\n"
        + "\n".join(f"  - {p}" for p in possible_paths)
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run RyzenAI Server verification tests"
    )
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        required=True,
        help="Model profile to test",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Local model directory (default: download the profile's checkpoint from Hugging Face)",
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
    parser.add_argument(
        "-k",
        dest="pattern",
        default=None,
        help="Only run tests whose name contains this substring",
    )

    args = parser.parse_args()
    profile = PROFILES[args.profile]

    print(f"\n{'='*60}")
    print("  RyzenAI Server Verification Tests")
    print(f"  Profile: {args.profile} ({profile['checkpoint']})")
    print(f"  Features: {', '.join(sorted(profile['features']))}")
    print(f"{'='*60}\n")

    server_exe = args.server_exe or find_server_executable()
    print(f"[Test] Server executable: {server_exe}")

    model_path = args.model_path or resolve_model_path(profile["checkpoint"])
    print(f"[Test] Model path: {model_path}")
    if not Path(model_path).exists():
        print(f"[Error] Model path does not exist: {model_path}")
        sys.exit(1)

    log_dir = (
        Path(args.log_dir) if args.log_dir else Path(__file__).parent / "test_logs"
    )
    print(f"[Test] Log directory: {log_dir}")

    server = ServerProcess(server_exe, model_path, args.profile, log_dir)

    try:
        server.start()
        RyzenAIServerTests.features = profile["features"]

        loader = unittest.TestLoader()
        if args.pattern:
            loader.testNamePatterns = [f"*{args.pattern}*"]
        suite = loader.loadTestsFromTestCase(RyzenAIServerTests)
        result = unittest.TextTestRunner(verbosity=2).run(suite)

        if result.wasSuccessful():
            print(f"\n  All tests PASSED for profile: {args.profile}\n")
            return 0
        print(f"\n  Some tests FAILED for profile: {args.profile}\n")
        print_server_log()
        return 1

    except Exception as e:
        print(f"[Error] Test execution failed: {e}")
        import traceback

        traceback.print_exc()
        print_server_log()
        return 1

    finally:
        server.stop()


if __name__ == "__main__":
    sys.exit(main())
