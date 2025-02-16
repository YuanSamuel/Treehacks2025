import argparse
import threading
import numpy as np
import sounddevice as sd
import whisper
import tensorflow as tf
import tensorflow_hub as hub
import socket
import csv
import math
import time
import torch
import usb.core
from tuning import Tuning

last_class_time = {}
THROTTLE = 2
APPROVED_CATEGORIES = {
    "yell": ["shout", "whoop", "yell"],
    "clap": ["wood", "chop", "crack", "clapping", "applause", "hands", "door", "bouncing", "hammer"],
    "speech": ["speech", "narration", "conversation", "monologue"],
    "siren": ["siren", "alarm", "buzzer", "vehicle horn, car horn, honking"]
}

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
            choice = int(input("Select the device number to use for transcription: "))
            if 0 <= choice < len(devices):
                return list(devices.keys())[choice]
            else:
                print("Invalid choice. Please select a valid device number.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def select_input_device():
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
            choice = int(input("Select the device number to use for transcription: "))
            if choice in input_indices:
                return choice
            else:
                print("Invalid choice. Please select a valid input device index.")
        except ValueError:
            print("Invalid input. Please enter a number.")

# Wrap the audio buffer in a dictionary so updates are seen by all threads.
def audio_callback(indata, frames, time_info, status, audio_buffer, lock):
    if status:
        print("[AUDIO] Status:", status)
    new_data = indata.flatten() if indata.ndim == 2 else indata
    with lock:
        # Append new data to the shared buffer.
        audio_buffer["data"] = np.concatenate([audio_buffer["data"], new_data])
        max_samples = int(5.0 * 16000)
        if audio_buffer["data"].shape[0] > max_samples:
            # Instead of keeping the latest samples, clear the buffer to start fresh.
            audio_buffer["data"] = np.empty((0,), dtype=np.float32)

def transcription_thread(stop_event, audio_model, device, sock, sock_lock, audio_buffer, lock):
    print("[TRANSCRIPTION] Thread started.")
    while not stop_event.is_set():
        with lock:
            if audio_buffer["data"].shape[0] == 0:
                # Nothing to transcribe, skip this iteration.
                buffer_copy = None
            else:
                buffer_copy = audio_buffer["data"].copy()
        if buffer_copy is not None:
            try:
                transcribe_result = audio_model.transcribe(buffer_copy, fp16=("cuda" in device))
                transcript_text = transcribe_result['text'].strip()
                print("[TRANSCRIPTION]", transcript_text)
                # Send transcript message.
                send_message(sock, sock_lock, {"type": "transcript", "payload": transcript_text})
            except Exception as e:
                print("[TRANSCRIPTION] Error:", e)
        time.sleep(0.01)

def classification_thread(stop_event, yamnet_model, class_names, sock, sock_lock, audio_buffer, lock):
    target_fs = 16000
    num_class_samples = int(1.0 * target_fs)
    # Assuming Tuning is defined/imported appropriately:
    dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
    if not dev:
        print("[ANGLE] USB device not found!")
        return
    mic_tuning = Tuning(dev)
    print("[CLASSIFICATION] Thread started.")
    while not stop_event.is_set():
        with lock:
            if audio_buffer["data"].shape[0] < num_class_samples:
                buffer_copy = None
            else:
                buffer_copy = audio_buffer["data"][-num_class_samples:].copy()
        if buffer_copy is not None:
            waveform = tf.convert_to_tensor(buffer_copy, dtype=tf.float32)
            scores, _, _ = yamnet_model(waveform)
            mean_scores = np.mean(scores.numpy(), axis=0)
            top_index = np.argmax(mean_scores)
            classification = f"{class_names[top_index]}"
            print("[CLASSIFICATION]", classification)

            try:
                direction = mic_tuning.direction
            except Exception as e:
                print(f"[ANGLE] Error: {e}")
                direction = "Unknown"
                
            # Compute volume as dB using the RMS of the waveform.
            rms_value = np.sqrt(np.mean(np.square(buffer_copy)))
            volume_db = 20 * math.log10(rms_value + 1e-9)  # in dB

            # Normalize dB value between 0 and 3.
            # Quiet: -53 dB -> 0, Loud: -8 dB -> 3.
            normalized_volume = (volume_db + 53) / (45) * 3
            # Clamp between 0 and 3.
            normalized_volume = max(0, min(3, normalized_volume))

            print("[CLASSIFICATION]", direction, f"{normalized_volume:.3f}", classification)
            # Send the message with proper payload (direction, volume in dB, classification).
            send_message(sock, sock_lock, {
                "type": "classification",
                "payload": (direction, f"{normalized_volume:.3f}", classification)
            })
        time.sleep(1)

def send_message(sock: socket.socket, sock_lock: threading.Lock, message: dict):
    """
    Sends a message over the persistent TCP connection.

    The message parameter should be a dictionary with two properties:
      - "type": Either "transcript" or "classification".
      - "payload":
          - If type is "transcript": a string.
          - If type is "classification": a tuple of (angle: int/str, volume: float, class_name: str).
    """
    try:
        if message["type"] == "transcript":
            # Encode transcript message.
            encoded_message = f"0|{message['payload']}"
        elif message["type"] == "classification":
            # Extract tuple components: (angle, volume, class_name)
            angle, volume, class_name = message['payload']
            class_lower = class_name.lower()

            # Determine if class_name contains any of the approved keywords.
            final_class = None
            for label, keywords in APPROVED_CATEGORIES.items():
                if any(keyword in class_lower for keyword in keywords):
                    final_class = label
                    break

            # If no approved keyword is found, skip the message.
            if not final_class:
                print(f"[SEND] Seeing class: [{class_lower}], can't classify")
                return

            # Throttle messages per class.
            current_time = time.time()
            if final_class in last_class_time and (current_time - last_class_time[final_class] < THROTTLE):
                return

            print(f"[SEND] Sending classification: [{final_class}]")
            last_class_time[final_class] = current_time
            encoded_message = f"1|{angle}|{volume}|{final_class}"
        else:
            print("[SEND] Unknown message type")
            return

        with sock_lock:
            # Send the message using the persistent connection.
            sock.sendall(encoded_message.encode('utf-8'))
    except Exception as e:
        print("[SEND] Error sending message:", e, message)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="tiny", choices=["tiny", "base", "small", "medium", "large", "turbo"])
    parser.add_argument("--metaquest_host", type=str, default="127.0.0.1")
    parser.add_argument("--metaquest_port", type=int, default=7000)
    parser.add_argument("--non_english", action='store_true', help="Do not use English-specific Whisper model")
    parser.add_argument("--trans_input_device", type=int, default=None)
    parser.add_argument("--yamnet_csv", type=str, default="./yamnet_local/yamnet_class_map.csv")
    args = parser.parse_args()
    
    stop_event = threading.Event()
    device = select_device()
    if args.trans_input_device is None:
        args.trans_input_device = select_input_device()
    
    device_fs = 16000
    # Shared audio buffer and lock.
    audio_buffer = {"data": np.zeros((0,), dtype=np.float32)}
    lock = threading.Lock()
    
    print("[MAIN] Loading models...")
    model_name = args.model
    if args.model not in ["large", "turbo"] and not args.non_english:
        model_name = model_name + ".en"
    audio_model = whisper.load_model(model_name, device=device)

    yamnet_model = hub.load("./yamnet_local")
    with open(args.yamnet_csv, 'r') as f:
        class_names = [row[2] for row in csv.reader(f)][1:]
    print("[MAIN] Models loaded.")
    
    # Establish a persistent TCP connection to MetaQuest.
    try:
        persistent_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print(f"[MAIN] Connecting to {args.metaquest_host}:{args.metaquest_port}...")
        persistent_sock.connect((args.metaquest_host, args.metaquest_port))
        print("[MAIN] Persistent connection established.")
    except Exception as e:
        print("[MAIN] Could not establish persistent connection:", e)
        return

    # Create a lock to protect writes on the persistent socket.
    persistent_sock_lock = threading.Lock()
    
    print("[MAIN] Starting audio input stream...")
    stream = sd.InputStream(
        samplerate=device_fs,
        device=args.trans_input_device,
        channels=1,
        dtype="float32",
        callback=lambda indata, frames, time_info, status: audio_callback(indata, frames, time_info, status, audio_buffer, lock)
    )
    with stream:
        t1 = threading.Thread(target=transcription_thread, args=(stop_event, audio_model, device, persistent_sock, persistent_sock_lock, audio_buffer, lock))
        t2 = threading.Thread(target=classification_thread, args=(stop_event, yamnet_model, class_names, persistent_sock, persistent_sock_lock, audio_buffer, lock))
        t1.start()
        t2.start()
        try:
            while not stop_event.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            stop_event.set()
        t1.join()
        t2.join()
    persistent_sock.close()
    print("[MAIN] Exiting.")

if __name__ == "__main__":
    main()
