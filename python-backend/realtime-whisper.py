import argparse
import os
import numpy as np
import speech_recognition as sr
import whisper
import torch

from datetime import datetime, timedelta
from queue import Queue
from time import sleep
from sys import platform

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
    print("Available devices for running Whisper:")
    for i, (key, name) in enumerate(devices.items()):
        print(f"[{i}] {name}")
    
    while True:
        try:
            choice = int(input("Select the device number to use: "))
            if 0 <= choice < len(devices):
                return list(devices.keys())[choice]
            else:
                print("Invalid choice. Please select a valid device number.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="medium", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large", "turbo"])
    parser.add_argument("--non_english", action='store_true',
                        help="Don't use the english model.")
    parser.add_argument("--energy_threshold", default=1000,
                        help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=0.5,
                        help="How real time the recording is in seconds.", type=float)
    parser.add_argument("--phrase_timeout", default=3,
                        help="How much empty space between recordings before we "
                             "consider it a new line in the transcription.", type=float)
    if 'linux' in platform:
        parser.add_argument("--default_microphone", default='pulse',
                            help="Default microphone name for SpeechRecognition. "
                                 "Run this with 'list' to view available Microphones.", type=str)
    args = parser.parse_args()

    device = select_device()
    
    phrase_time = None
    data_queue = Queue()
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    recorder.dynamic_energy_threshold = False

    if 'linux' in platform:
        mic_name = args.default_microphone
        available_mics = sr.Microphone.list_microphone_names()
        
        if not mic_name or mic_name == 'list':
            print("Available microphone devices:")
            for index, name in enumerate(available_mics):
                print(f"[{index}] {name}")
            exit(1)  # Exit to prevent unassigned variable

        source = None
        for index, name in enumerate(available_mics):
            if mic_name in name:
                source = sr.Microphone(sample_rate=16000, device_index=index)
                break

        if source is None:
            print(f"Microphone '{mic_name}' not found. Please check `list`.")
            exit(1)  # Exit to prevent error

    else:
        source = sr.Microphone(sample_rate=16000)

    model = args.model
    if args.model != "large" and not args.non_english:
        model = model + ".en"
    audio_model = whisper.load_model(model, device=device)

    record_timeout = args.record_timeout
    phrase_timeout = args.phrase_timeout
    transcription = ['']

    with source:
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio: sr.AudioData) -> None:
        data = audio.get_raw_data()
        data_queue.put(data)

    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    print("Model loaded.")
    while True:
        try:
            now = datetime.utcnow()
            if not data_queue.empty():
                phrase_complete = False
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    phrase_complete = True
                phrase_time = now
                
                audio_chunks = []
                while not data_queue.empty():
                    audio_chunks.append(data_queue.get())
                audio_data = b''.join(audio_chunks)

                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                result = audio_model.transcribe(audio_np, fp16=("cuda" in device))
                text = result['text'].strip()

                if phrase_complete:
                    transcription.append(f'{text} ')
                else:
                    transcription[-1] += f'{text} '

                os.system('cls' if os.name=='nt' else 'clear')
                for line in transcription:
                    print(line)
                print('', end='', flush=True)
            else:
                sleep(0.25)
        except KeyboardInterrupt:
            break
        
if __name__ == "__main__":
    main()
