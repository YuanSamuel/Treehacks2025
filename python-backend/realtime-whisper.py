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
    parser.add_argument("--energy_threshold", default=600,
                        help="Energy level for mic to detect (in int16 units).", type=int)
    parser.add_argument("--phrase_timeout", default=3,
                        help="Timeout (in seconds) between processed chunks to consider it a new phrase.", type=float)
    parser.add_argument("--silence_timeout", default=0.25,
                        help="Time (in seconds) of silence (below energy threshold) to trigger processing of the accumulated chunk.", type=float)
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
    if args.model != "large" and not args.non_english:
        model_name = model_name + ".en"
    audio_model = whisper.load_model(model_name, device=device)
    print("Model loaded.")

    # Variables for silence-based audio accumulation.
    audio_buffer = []          # List to accumulate audio chunks.
    last_active_time = time()  # Time when audio was last above threshold.
    last_process_time = None   # Time when last chunk was processed.
    transcription = ['']       # List to hold transcription lines.

    # The callback simply appends audio chunks when the energy is high.
    def callback(indata, frames, time_info, status):
        nonlocal last_active_time, audio_buffer
        if status:
            print(status)
        # Compute RMS energy of the current block.
        rms = np.sqrt(np.mean(indata**2))
        # Convert the int16 threshold to float scale.
        threshold = args.energy_threshold / 32768.0
        now = time()
        # If the energy is high enough, update the last active time and accumulate the chunk.
        if rms >= threshold:
            last_active_time = now
            audio_buffer.append(indata.copy())

    print("Recording... Press Ctrl+C to stop.")
    with sd.InputStream(samplerate=device_fs, device=sd_device, channels=channels,
                        dtype="float32", callback=callback):
        try:
            while True:
                now = time()
                # Check if there's accumulated audio and if silence has persisted long enough.
                if audio_buffer and (now - last_active_time) >= args.silence_timeout:
                    # Copy and clear the audio buffer.
                    chunks = audio_buffer.copy()
                    audio_buffer.clear()

                    # Concatenate all chunks into one audio array.
                    audio_data = np.concatenate(chunks, axis=0)
                    # If necessary, flatten the audio array.
                    if audio_data.ndim == 2 and audio_data.shape[1] == 1:
                        audio_data = audio_data.flatten()
                    
                    # Resample the audio to 16 kHz if needed.
                    if device_fs != target_fs:
                        audio_data = scipy.signal.resample_poly(audio_data, target_fs, device_fs)
                    
                    # Ensure the audio data is in float32 format.
                    audio_data = audio_data.astype(np.float32)
                    
                    # Transcribe the accumulated audio chunk using Whisper.
                    result = audio_model.transcribe(audio_data, fp16=("cuda" in device))
                    text = result['text'].strip()
                    
                    # Determine if we should start a new phrase or append to the current one.
                    if (last_process_time is None) or ((now - last_process_time) > args.phrase_timeout):
                        transcription.append(text + " ")
                    else:
                        transcription[-1] += text + " "
                    
                    last_process_time = now
                    
                    # Clear the screen and print the updated transcription.
                    os.system("cls" if os.name == "nt" else "clear")
                    for line in transcription:
                        print(line)
                
                sleep(0.1)
        except KeyboardInterrupt:
            print("Stopping...")

if __name__ == "__main__":
    main()
