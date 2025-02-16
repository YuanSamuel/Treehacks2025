import argparse
import threading
import numpy as np
import sounddevice as sd
import whisper
import tensorflow as tf
import tensorflow_hub as hub
import socket
import csv
import time
import torch

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

def audio_callback(indata, frames, time_info, status, rolling_audio, lock):
    if status:
        print("[AUDIO] Status:", status)
    new_data = indata.flatten() if indata.ndim == 2 else indata
    with lock:
        rolling_audio[:] = np.concatenate([rolling_audio, new_data])
        max_samples = int(10.0 * 16000)
        if rolling_audio.shape[0] > max_samples:
            rolling_audio[:] = rolling_audio[-max_samples:]

def transcription_thread(stop_event, audio_model, device, host, port, rolling_audio, lock):
    print("[TRANSCRIPTION] Thread started.")
    while not stop_event.is_set():
        with lock:
            if rolling_audio.shape[0] == 0:
                continue
            buffer_copy = rolling_audio.copy()
        try:
            transcribe_result = audio_model.transcribe(buffer_copy, fp16=("cuda" in device))
            transcript_text = transcribe_result['text'].strip()
            print("[TRANSCRIPTION]", transcript_text)
            send_message(host, port, f"Transcript: {transcript_text}")
        except Exception as e:
            print("[TRANSCRIPTION] Error:", e)
        time.sleep(1)

def classification_thread(stop_event, yamnet_model, class_names, host, port, rolling_audio, lock):
    target_fs = 16000
    num_class_samples = int(1.0 * target_fs)
    print("[CLASSIFICATION] Thread started.")
    while not stop_event.is_set():
        with lock:
            if rolling_audio.shape[0] < num_class_samples:
                continue
            buffer_copy = rolling_audio[-num_class_samples:].copy()
        waveform = tf.convert_to_tensor(buffer_copy, dtype=tf.float32)
        scores, _, _ = yamnet_model(waveform)
        mean_scores = np.mean(scores.numpy(), axis=0)
        top_index = np.argmax(mean_scores)
        top_score = mean_scores[top_index]
        classification = f"{class_names[top_index]}: {top_score:.3f}"
        print("[CLASSIFICATION]", classification)
        send_message(host, port, f"Class: {classification} | Angle: N/A | Volume: {np.max(buffer_copy):.3f}")
        time.sleep(1)

def send_message(host, port, message):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((host, port))
            sock.sendall(message.encode('utf-8'))
    except Exception as e:
        print("[SEND] Error:", e)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="tiny", choices=["tiny", "base", "small", "medium", "large", "turbo"])
    parser.add_argument("--metaquest_host", type=str, default="127.0.0.1")
    parser.add_argument("--metaquest_port", type=int, default=7000)
    parser.add_argument("--trans_input_device", type=int, default=None)
    parser.add_argument("--yamnet_csv", type=str, default="./yamnet_local/yamnet_class_map.csv")
    args = parser.parse_args()
    
    stop_event = threading.Event()
    device = select_device()
    if args.trans_input_device is None:
        args.trans_input_device = select_input_device()
    
    device_fs = 16000
    rolling_audio = np.zeros((0,), dtype=np.float32)
    lock = threading.Lock()
    
    print("[MAIN] Loading models...")
    audio_model = whisper.load_model(args.model, device=device)
    yamnet_model = hub.load("./yamnet_local")
    with open(args.yamnet_csv, 'r') as f:
        class_names = [row[2] for row in csv.reader(f)][1:]
    print("[MAIN] Models loaded.")
    
    print("[MAIN] Starting audio input stream...")
    with sd.InputStream(samplerate=device_fs, device=args.trans_input_device,
                        channels=1, dtype="float32", callback=lambda indata, frames, time_info, status:
                        audio_callback(indata, frames, time_info, status, rolling_audio, lock)):
        threading.Thread(target=transcription_thread, args=(stop_event, audio_model, device, args.metaquest_host, args.metaquest_port, rolling_audio, lock)).start()
        threading.Thread(target=classification_thread, args=(stop_event, yamnet_model, class_names, args.metaquest_host, args.metaquest_port, rolling_audio, lock)).start()
        try:
            while not stop_event.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            stop_event.set()
    print("[MAIN] Exiting.")

if __name__ == "__main__":
    main()
