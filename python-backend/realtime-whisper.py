import argparse
import os
import numpy as np
import whisper
import torch
import sounddevice as sd
import scipy.signal  # For resampling
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
                        help="Don't use the English model.")
    parser.add_argument("--energy_threshold", default=500,
                        help="Energy level for mic to detect (in int16 units).", type=int)
    parser.add_argument("--buffer_duration", default=10,
                        help="Duration (in seconds) of the rolling audio buffer.", type=float)
    args = parser.parse_args()

    # GPU/CPU selection for Whisper
    device = select_device()
    
    # Select input device (only prompt on Linux)
    input_device = select_input_device()
    sd_device = input_device if input_device is not None else None

    # Set target sample rate for Whisper.
    target_fs = 16000

    # Query the default sample rate of the input device.
    device_info = sd.query_devices(sd_device, 'input')
    device_fs = int(device_info['default_samplerate'])
    print(f"Device default sample rate: {device_fs} Hz. Will resample to {target_fs} Hz.")
    
    channels = 1

    # Adjust model name based on language settings.
    model_name = args.model
    if args.model not in ["large", "turbo"] and not args.non_english:
        model_name = model_name + ".en"
    audio_model = whisper.load_model(model_name, device=device)
    print("Model loaded.")

    # Initialize a rolling buffer (for audio at the target sample rate)
    rolling_audio = np.zeros((0,), dtype=np.float32)
    # A temporary buffer to store new audio chunks from the callback
    new_audio_buffer = []

    # The callback simply appends all incoming audio.
    def callback(indata, frames, time_info, status):
        nonlocal new_audio_buffer
        if status:
            print(status)
        # Append the incoming audio chunk (regardless of energy)
        new_audio_buffer.append(indata.copy())

    print("Recording... Press Ctrl+C to stop.")
    with sd.InputStream(samplerate=device_fs, device=sd_device, channels=channels,
                        dtype="float32", callback=callback):
        try:
            while True:
                if new_audio_buffer:
                    # Concatenate the new chunks
                    chunks = new_audio_buffer.copy()
                    new_audio_buffer.clear()
                    audio_chunk = np.concatenate(chunks, axis=0)
                    # If the audio has a shape (N, 1), flatten it.
                    if audio_chunk.ndim == 2 and audio_chunk.shape[1] == 1:
                        audio_chunk = audio_chunk.flatten()
                    
                    # Resample the chunk to 16 kHz if needed.
                    if device_fs != target_fs:
                        audio_chunk = scipy.signal.resample_poly(audio_chunk, target_fs, device_fs)
                    
                    # Append the new chunk to the rolling buffer.
                    rolling_audio = np.concatenate([rolling_audio, audio_chunk])
                    # Trim the rolling buffer to keep only the last buffer_duration seconds.
                    max_samples = int(args.buffer_duration * target_fs)
                    if rolling_audio.shape[0] > max_samples:
                        rolling_audio = rolling_audio[-max_samples:]
                    
                    # Transcribe the current rolling buffer.
                    result = audio_model.transcribe(rolling_audio, fp16=("cuda" in device))
                    text = result['text'].strip()
                    
                    # Clear the screen and print the rolling transcription.
                    os.system("cls" if os.name == "nt" else "clear")
                    print(text)
                
                sleep(0.01)
        except KeyboardInterrupt:
            print("Stopping...")

if __name__ == "__main__":
    main()
