import argparse
import os
import numpy as np
import whisper
import torch
import sounddevice as sd
from queue import Queue
from time import sleep, time
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

def select_input_device():
    """
    Lists all devices that have input channels and (if on Linux)
    prompts the user to select one. On other platforms the default
    input device is used.
    """
    devices = sd.query_devices()
    input_indices = [i for i, dev in enumerate(devices) if dev['max_input_channels'] > 0]
    if not input_indices:
        print("No input devices available. Please check your audio setup.")
        exit(1)
    print("Available input devices:")
    for i in input_indices:
        print(f"[{i}] {devices[i]['name']}")
    while True:
        try:
            choice = int(input("Select the device number to use: "))
            if choice in input_indices:
                return choice
            else:
                print("Invalid choice. Please select a valid input device index.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="medium", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large", "turbo"])
    parser.add_argument("--non_english", action='store_true',
                        help="Don't use the english model.")
    parser.add_argument("--energy_threshold", default=100,
                        help="Energy level for mic to detect (in int16 units).", type=int)
    parser.add_argument("--record_timeout", default=0.5,
                        help="Time interval (in seconds) for processing audio chunks.", type=float)
    parser.add_argument("--phrase_timeout", default=3,
                        help="Timeout (in seconds) between recordings to consider it a new phrase.", type=float)
    args = parser.parse_args()

    # GPU/CPU selection for Whisper
    device = select_device()
    
    # Select input device (only prompt on Linux)
    input_device = select_input_device()
    if input_device is not None:
        sd_device = input_device
    else:
        sd_device = None

    # Set desired sample rate and channels.
    device_info = sd.query_devices(sd_device, 'input')
    fs = int(device_info['default_samplerate'])
    channels = 1

    # Adjust model name based on language settings.
    model_name = args.model
    if args.model != "large" and not args.non_english:
        model_name = model_name + ".en"
    audio_model = whisper.load_model(model_name, device=device)
    print("Model loaded.")

    # Queue to store audio chunks.
    audio_queue = Queue()
    # Time (in seconds) when the last chunk was processed.
    last_process_time = None
    # Keep track of the transcription so far.
    transcription = ['']

    def callback(indata, frames, time_info, status):
        if status:
            print(status)
        # Compute RMS energy of the current block.
        rms = np.sqrt(np.mean(indata**2))
        # Convert the int16 threshold to the float32 scale.
        threshold = args.energy_threshold / 32768.0
        # If the block is too quiet, skip it.
        if rms < threshold:
            return
        # Otherwise, push a copy of the audio block onto the queue.
        audio_queue.put(indata.copy())

    print("Recording... Press Ctrl+C to stop.")
    with sd.InputStream(samplerate=fs, device=sd_device, channels=channels,
                        dtype='float32', callback=callback):
        try:
            while True:
                now = time()
                if not audio_queue.empty():
                    # Decide whether this block should start a new phrase.
                    if last_process_time is not None and now - last_process_time > args.phrase_timeout:
                        phrase_complete = True
                    else:
                        phrase_complete = False
                    last_process_time = now

                    # Gather all queued audio chunks.
                    chunks = []
                    while not audio_queue.empty():
                        chunks.append(audio_queue.get())
                    # Concatenate along the time axis.
                    audio_data = np.concatenate(chunks, axis=0)
                    # If the audio data is 2D with one channel, flatten it.
                    if audio_data.ndim == 2 and audio_data.shape[1] == 1:
                        audio_data = audio_data.flatten()
                    
                    # Transcribe the audio using Whisper.
                    result = audio_model.transcribe(audio_data, fp16=("cuda" in device))
                    text = result['text'].strip()

                    if phrase_complete:
                        transcription.append(f'{text} ')
                    else:
                        transcription[-1] += f'{text} '

                    # Clear the screen and print the updated transcription.
                    os.system('cls' if os.name == 'nt' else 'clear')
                    for line in transcription:
                        print(line)
                else:
                    sleep(0.25)
        except KeyboardInterrupt:
            print("Stopping...")

if __name__ == "__main__":
    main()
