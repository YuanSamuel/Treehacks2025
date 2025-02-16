import argparse
import asyncio
import websockets
import numpy as np
import whisper
import torch
import json

from datetime import datetime, timedelta
from queue import Queue


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
    return list(devices.keys())[0]  # Default to first available device


class TranscriptionServer:
    def __init__(self, model, device, phrase_timeout=3):
        self.audio_model = whisper.load_model(model, device=device)
        self.transcription = []
        self.phrase_timeout = phrase_timeout
        self.phrase_time = None
        self.data_queue = Queue()
        print("Whisper model loaded.")

    async def handler(self, websocket, path):
        print("Client connected.")
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    if "audio" in data:
                        audio_bytes = bytes(data["audio"])
                        self.data_queue.put(audio_bytes)

                        now = datetime.utcnow()
                        phrase_complete = False
                        if self.phrase_time and now - self.phrase_time > timedelta(seconds=self.phrase_timeout):
                            phrase_complete = True
                        self.phrase_time = now

                        audio_chunks = []
                        while not self.data_queue.empty():
                            audio_chunks.append(self.data_queue.get())
                        audio_data = b"".join(audio_chunks)

                        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                        result = self.audio_model.transcribe(audio_np, fp16=("cuda" in device))
                        text = result['text'].strip()

                        if phrase_complete:
                            self.transcription.append(f'{text} ')
                        else:
                            if self.transcription:
                                self.transcription[-1] += f'{text} '
                            else:
                                self.transcription.append(f'{text} ')

                        response = json.dumps({"transcription": text})
                        await websocket.send(response)

                except Exception as e:
                    print(f"Error processing audio: {e}")
        except websockets.exceptions.ConnectionClosed:
            print("Client disconnected.")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="medium", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large", "turbo"])
    parser.add_argument("--non_english", action='store_true', help="Don't use the English model.")
    parser.add_argument("--phrase_timeout", default=3, help="Silence timeout before starting a new phrase.", type=float)
    args = parser.parse_args()

    device = select_device()
    model = args.model
    if args.model != "large" and not args.non_english:
        model = model + ".en"

    server = TranscriptionServer(model, device, phrase_timeout=args.phrase_timeout)
    
    async with websockets.serve(server.handler, "0.0.0.0", 8080):
        print("WebSocket server running on ws://0.0.0.0:8080")
        await asyncio.Future()  # Keep running


if __name__ == "__main__":
    asyncio.run(main())
