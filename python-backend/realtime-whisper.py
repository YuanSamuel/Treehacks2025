import argparse
import os
import numpy as np
import whisper
import torch
import sounddevice as sd
import scipy.signal  # For resampling
from time import sleep, time
from sys import platform
import websocket  # Added for websocket communication

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
    Lists input devices. If PulseAudio is running and has available devices,
    only those are shown. Otherwise, falls back to listing all input devices.
    """
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    pulse_api_index = None
    # Look for a host API that includes "pulse" in its name.
    for idx, ha in enumerate(hostapis):
        if "pulse" in ha['name'].lower():
            pulse_api_index = idx
            break
    
    # If PulseAudio is found, filter for PulseAudio input devices.
    if pulse_api_index is not None:
        input_indices = [i for i, dev in enumerate(devices)
                         if dev['max_input_channels'] > 0 and dev['hostapi'] == pulse_api_index]
        if input_indices:
            print("Available PulseAudio input devices:")
            for i in input_indices:
                print(f"[{i}] {devices[i]['name']}")
            while True:
                try:
                    choice = int(input("Select the device number to use: "))
                    if choice in input_indices:
                        return choice
                    else:
                        print("Invalid choice. Please select a valid PulseAudio input device index.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
    
    # Fallback: if no PulseAudio devices are found, list all input devices.
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

    # GPU/CPU selection for Whisper.
    device = select_device()
    
    # Select input device (preferring PulseAudio devices if available).
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

    # WebSocket setup.
    use_ws = True
    ws_url = "ws://localhost:8765"
    ws = None

    # Try connecting to the WebSocket server initially.
    try:
        ws = websocket.create_connection(ws_url)
        print(f"Connected to WebSocket server at {ws_url}")
    except Exception as e:
        print("Initial connection to WebSocket server failed:", e)
        ws = None

    # Prompt for confirmation if WebSocket connection is established.
    if ws is not None:
        confirm = input("WebSocket connection established. Do you want to continue using it? (y/n): ").strip().lower()
        if confirm not in ("y", "yes"):
            print("WebSocket connection will not be used.")
            ws.close()
            ws = None
            use_ws = False

    # Helper function: try connecting once (non-blocking in main loop).
    def try_connect_ws(url):
        try:
            connection = websocket.create_connection(url)
            print(f"Connected to WebSocket server at {url}")
            return connection
        except Exception as e:
            print(f"Failed to connect to WebSocket server at {url}: {e}")
            return None

    # Variables for reconnection timing.
    last_ws_attempt = 0
    reconnect_interval = 5  # seconds

    # Initialize a rolling buffer (for audio at the target sample rate).
    rolling_audio = np.zeros((0,), dtype=np.float32)
    # A temporary buffer to store new audio chunks from the callback.
    new_audio_buffer = []

    # The callback simply appends all incoming audio.
    def callback(indata, frames, time_info, status):
        nonlocal new_audio_buffer
        if status:
            print(status)
        new_audio_buffer.append(indata.copy())

    print("Recording... Press Ctrl+C to stop.")
    try:
        with sd.InputStream(samplerate=device_fs, device=sd_device, channels=channels,
                            dtype="float32", callback=callback):
            while True:
                if new_audio_buffer:
                    # Concatenate the new chunks.
                    chunks = new_audio_buffer.copy()
                    new_audio_buffer.clear()
                    audio_chunk = np.concatenate(chunks, axis=0)
                    # If the audio is shaped (N, 1), flatten it.
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
                    
                    # WebSocket communication.
                    if use_ws:
                        # Only try to reconnect if enough time has passed since the last attempt.
                        if ws is None and (time() - last_ws_attempt) >= reconnect_interval:
                            print("Attempting to reconnect to WebSocket server...")
                            ws = try_connect_ws(ws_url)
                            last_ws_attempt = time()
                        if ws is not None:
                            try:
                                ws.send(text)
                            except Exception as e:
                                print("Failed to send message over WebSocket:", e)
                                try:
                                    ws.close()
                                except Exception:
                                    pass
                                ws = None
                                last_ws_attempt = time()  # Reset reconnect timer.
                sleep(0.01)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        if ws is not None:
            ws.close()

if __name__ == "__main__":
    main()
