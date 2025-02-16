import argparse
import asyncio
import os
import json
import numpy as np
import speech_recognition as sr
import websockets
from datetime import datetime, timedelta
from queue import Queue
from time import sleep

def select_microphone():
    available_mics = sr.Microphone.list_microphone_names()
    if not available_mics:
        print("No microphones detected. Please check your audio setup.")
        exit(1)
    print("Available microphone devices:")
    for index, name in enumerate(available_mics):
        print(f"[{index}] {name}")
    while True:
        try:
            choice = int(input("Select the microphone index to use: "))
            if 0 <= choice < len(available_mics):
                return sr.Microphone(device_index=choice)
            else:
                print("Invalid choice. Please select a valid index.")
        except ValueError:
            print("Invalid input. Please enter a number.")

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record_timeout", default=0.5, type=float,
                        help="Duration (in seconds) for each recording chunk.")
    parser.add_argument("--phrase_timeout", default=3, type=float,
                        help="Time (in seconds) with no new audio before considering the phrase complete.")
    parser.add_argument("--server_uri", default="ws://localhost:8080",
                        help="WebSocket URI of the transcription server.")
    args = parser.parse_args()

    # Set up the recognizer and microphone
    recorder = sr.Recognizer()
    recorder.energy_threshold = 1000
    recorder.dynamic_energy_threshold = False

    # Use user-selected microphone (or default)
    source = select_microphone()
    with source:
        recorder.adjust_for_ambient_noise(source)

    data_queue = Queue()
    phrase_time = None

    def record_callback(_, audio: sr.AudioData):
        data = audio.get_raw_data()
        data_queue.put(data)

    # Start background recording (each chunk lasts record_timeout seconds)
    recorder.listen_in_background(source, record_callback, phrase_time_limit=args.record_timeout)
    print("Listening...")

    transcription_lines = ['']

    # Connect to the server
    async with websockets.connect(args.server_uri, ping_interval=20) as websocket:
        while True:
            # If new audio data is available, collect it and send it
            if not data_queue.empty():
                now = datetime.utcnow()
                # Check if the gap since the last batch signals a phrase break
                phrase_complete = False
                if phrase_time and now - phrase_time > timedelta(seconds=args.phrase_timeout):
                    phrase_complete = True
                phrase_time = now

                audio_chunks = []
                while not data_queue.empty():
                    audio_chunks.append(data_queue.get())
                audio_data = b''.join(audio_chunks)

                # Convert audio bytes to a list of integers for JSON transport
                message = json.dumps({"audio": list(audio_data)})
                await websocket.send(message)

            # Check for transcription responses from the server
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                data = json.loads(response)
                text = data.get("transcription", "")
                if text:
                    if phrase_complete:
                        transcription_lines.append(text + " ")
                    else:
                        transcription_lines[-1] += text + " "
                    os.system('cls' if os.name == 'nt' else 'clear')
                    for line in transcription_lines:
                        print(line)
            except asyncio.TimeoutError:
                # No transcription received; give the CPU a short break.
                await asyncio.sleep(0.25)
            except websockets.exceptions.ConnectionClosed:
                print("Server connection closed")
                break

if __name__ == "__main__":
    asyncio.run(main())
