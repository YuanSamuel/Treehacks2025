import argparse
import asyncio
import websockets
import json
import numpy as np
import whisper
import torch
from datetime import datetime

def list_devices():
    devices = {"cpu": "CPU"}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            device_name = torch.cuda.get_device_name(i)
            vram = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            devices[f"cuda:{i}"] = f"CUDA (GPU {i}) - {device_name} ({vram:.2f} GB VRAM)"
    if torch.backends.mps.is_available():
        devices["mps"] = "MPS (Mac Metal)"
    return devices

def select_device():
    devices = list_devices()
    print("Available devices:")
    for i, (key, value) in enumerate(devices.items()):
        print(f"{i}: {value} ({key})")
    while True:
        try:
            choice = int(input("Select a device by number: "))
            if 0 <= choice < len(devices):
                return list(devices.keys())[choice]
            else:
                print("Invalid choice. Please select a valid number.")
        except ValueError:
            print("Invalid input. Please enter a number.")

class TranscriptionServer:
    def __init__(self, model, device, phrase_timeout=3):
        self.device = device
        self.phrase_timeout = phrase_timeout
        self.audio_model = whisper.load_model(model, device=device)
        print("Whisper model loaded on", device)

    async def handler(self, websocket, path=None):
        print("Client connected.")
        audio_buffer = bytearray()
        buffer_start_time = None

        async def process_buffer():
            nonlocal audio_buffer, buffer_start_time
            if audio_buffer and buffer_start_time:
                elapsed = (datetime.utcnow() - buffer_start_time).total_seconds()
                if elapsed >= self.phrase_timeout:
                    try:
                        # Convert raw bytes to a NumPy array.
                        audio_np = np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0
                        use_fp16 = self.device.startswith("cuda")
                        result = self.audio_model.transcribe(audio_np, fp16=use_fp16)
                        text = result.get("text", "").strip()
                        response = json.dumps({"transcription": text})
                        await websocket.send(response)
                    except Exception as e:
                        print("Error during transcription:", e)
                    # Reset buffer after processing.
                    audio_buffer.clear()
                    buffer_start_time = None

        try:
            while True:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                except asyncio.TimeoutError:
                    message = None

                if message:
                    try:
                        data = json.loads(message)
                        if "audio" in data:
                            audio_bytes = bytes(data["audio"])
                            if buffer_start_time is None:
                                buffer_start_time = datetime.utcnow()
                            audio_buffer.extend(audio_bytes)
                    except Exception as e:
                        print("Error processing received audio data:", e)

                # Process the accumulated buffer if needed.
                await process_buffer()
        except websockets.exceptions.ConnectionClosed as e:
            print("Client disconnected. Code:", e.code, "Reason:", e.reason)
        except Exception as e:
            print("Unexpected error in handler:", e)

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="medium", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large", "turbo"])
    parser.add_argument("--non_english", action="store_true", help="Don't use the English model.")
    parser.add_argument("--phrase_timeout", default=3, help="Silence timeout before processing a phrase.", type=float)
    args = parser.parse_args()

    device = select_device()
    model = args.model
    if args.model != "large" and not args.non_english:
        model = model + ".en"

    server_instance = TranscriptionServer(model, device, phrase_timeout=args.phrase_timeout)

    try:
        async with websockets.serve(
            server_instance.handler,
            "0.0.0.0",
            8080,
            ping_interval=20
        ) as server:
            print("WebSocket server running on ws://0.0.0.0:8080")
            await asyncio.Future()  # Run forever.
    except Exception as e:
        print("Server encountered an error:", e)

if __name__ == "__main__":
    asyncio.run(main())
