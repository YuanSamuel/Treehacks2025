#!/usr/bin/env python3
import argparse
import os
import threading
import time
from time import sleep, time
import csv
import math
import socket
import numpy as np
import torch
import sounddevice as sd
import scipy.signal  # For resampling
import whisper
import tensorflow as tf
import tensorflow_hub as hub
import usb.core
import usb.util
from tuning import Tuning  # Assumes you have this module

# ----------------------
# Global variables for shared audio and transcription
# ----------------------
rolling_audio = np.zeros((0,), dtype=np.float32)
latest_transcription = ""  # Updated only in the central (combined) loop

# ----------------------
# Utility functions for device selection
# ----------------------
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
    hostapis = sd.query_hostapis()
    pulse_api_index = None
    for idx, ha in enumerate(hostapis):
        if "pulse" in ha['name'].lower():
            pulse_api_index = idx
            break
    if pulse_api_index is not None:
        input_indices = [i for i, dev in enumerate(devices)
                         if dev['max_input_channels'] > 0 and dev['hostapi'] == pulse_api_index]
        if input_indices:
            print("Available PulseAudio input devices:")
            for i in input_indices:
                print(f"[{i}] {devices[i]['name']}")
            while True:
                try:
                    choice = int(input("Select the device number to use for transcription: "))
                    if choice in input_indices:
                        return choice
                    else:
                        print("Invalid choice. Please select a valid PulseAudio input device index.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
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

# ----------------------
# Combined processing: transcription and classification using a single input stream
# ----------------------
def combined_processing_loop(stop_event, args, device, audio_model, device_fs, target_fs, yamnet_model, class_names):
    # Classification uses the last 1 second of audio
    classification_duration = 1.0  # seconds
    num_class_samples = int(target_fs * classification_duration)
    
    def callback(indata, frames, time_info, status):
        global rolling_audio
        if status:
            print("[COMBINED] Audio stream status:", status)
        # Append new samples (flatten if needed)
        new_data = indata.flatten() if indata.ndim == 2 else indata
        rolling_audio = np.concatenate([rolling_audio, new_data])
    
    print("[COMBINED] Starting combined processing. Press Ctrl+C to stop.")
    try:
        with sd.InputStream(samplerate=device_fs, device=args.trans_input_device,
                            channels=1, dtype="float32", callback=callback):
            while not stop_event.is_set():
                sleep(0.05)  # Allow some audio to accumulate
                current_buffer = rolling_audio.copy()
                # Trim buffer to last 'buffer_duration' seconds if needed
                max_total_samples = int(args.buffer_duration * target_fs)
                if current_buffer.shape[0] > max_total_samples:
                    current_buffer = current_buffer[-max_total_samples:]
                    rolling_audio = current_buffer.copy()
                
                if current_buffer.shape[0] == 0:
                    continue
                
                # Transcription: use the entire rolling buffer
                try:
                    transcribe_result = audio_model.transcribe(current_buffer, fp16=("cuda" in device))
                    transcript_text = transcribe_result['text'].strip()
                    latest_transcription_local = transcript_text
                    print("[COMBINED] Transcription:", transcript_text)
                except Exception as e:
                    print("[COMBINED] Transcription error:", e)
                    latest_transcription_local = ""
                
                # Classification: use the last num_class_samples if available
                if current_buffer.shape[0] >= num_class_samples:
                    classification_chunk = current_buffer[-num_class_samples:]
                    waveform = tf.convert_to_tensor(classification_chunk, dtype=tf.float32)
                    scores, embeddings, spectrogram = yamnet_model(waveform)
                    mean_scores = np.mean(scores.numpy(), axis=0)
                    top_index = np.argsort(mean_scores)[-1]
                    top_score = mean_scores[top_index]
                    angle_info = current_angle if 'current_angle' in globals() and current_angle is not None else "N/A"
                    
                    combined_message = (f"Direction: {angle_info} | "
                                        f"Prediction: {class_names[top_index]}: {top_score:.3f} | "
                                        f"Transcript: {latest_transcription_local}")
                    print("[COMBINED] Final Message:", combined_message)
                    # Central thread sends out the message (e.g., via TCP)
                    send_message(args.metaquest_host, args.metaquest_port, combined_message)
    except Exception as e:
        print("[COMBINED] Error in combined processing loop:", e)
    print("[COMBINED] Combined processing loop stopped.")

# ----------------------
# Angle Detection Thread
# ----------------------
def angle_thread(stop_event):
    global current_angle
    current_angle = None
    dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
    if not dev:
        print("[ANGLE] USB device not found!")
        return
    mic_tuning = Tuning(dev)
    print("[ANGLE] Starting angle detection thread.")
    try:
        while not stop_event.is_set():
            direction = mic_tuning.direction
            speech_detected = mic_tuning.read('SPEECHDETECTED')
            current_angle = direction
            print(f"[ANGLE] Direction: {direction} | SpeechDetected: {speech_detected}")
            time.sleep(1)
    except Exception as e:
        print(f"[ANGLE] Error: {e}")
    print("[ANGLE] Angle detection thread stopped.")

# ----------------------
# Helper for YAMNet: Load class names from CSV
# ----------------------
def load_class_names(csv_path):
    class_names = []
    skipped = False
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not skipped:
                skipped = True
                continue
            class_names.append(row[2])
    return class_names

# ----------------------
# Send message via TCP
# ----------------------
def send_message(host: str, port: int, message: str):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            print(f"[SEND] Connecting to {host}:{port}...")
            sock.connect((host, port))
            print(f"[SEND] Sending message: {message}")
            sock.sendall(message.encode('utf-8'))
    except Exception as e:
        print("[SEND] Error sending message:", e)

# ----------------------
# Main function
# ----------------------
def main():
    parser = argparse.ArgumentParser(description="Combined Audio Transcription and Classification with Single Input Stream")
    parser.add_argument("--model", default="tiny", choices=["tiny", "base", "small", "medium", "large", "turbo"],
                        help="Whisper model to use")
    parser.add_argument("--non_english", action='store_true',
                        help="Do not use English-specific Whisper model")
    parser.add_argument("--buffer_duration", default=10.0, type=float,
                        help="Duration (in seconds) of the rolling audio buffer for transcription")
    parser.add_argument("--metaquest_host", type=str, default="127.0.0.1",
                        help="MetaQuest host IP address")
    parser.add_argument("--metaquest_port", type=int, default=7000,
                        help="MetaQuest port number")
    parser.add_argument("--trans_input_device", type=int, default=None,
                        help="Input device ID for transcription (if not provided, will prompt)")
    parser.add_argument("--yamnet_csv", type=str, default="./yamnet_local/yamnet_class_map.csv",
                        help="Path to the YAMNet class map CSV")
    args = parser.parse_args()

    stop_event = threading.Event()

    # Select device for Whisper (GPU/CPU)
    device = select_device()
    if args.trans_input_device is None:
        args.trans_input_device = select_input_device()
    device_info = sd.query_devices(args.trans_input_device, 'input')
    hostapi_index = device_info['hostapi']
    hostapi_name = sd.query_hostapis()[hostapi_index]['name']
    print(f"[MAIN] Using {'PulseAudio' if 'pulse' in hostapi_name.lower() else 'non-PulseAudio'} input device: {device_info['name']}")
    device_fs = int(device_info['default_samplerate'])
    target_fs = 16000
    print(f"[MAIN] Device sample rate: {device_fs} Hz; Resampling to {target_fs} Hz.")

    # Adjust model name for Whisper
    model_name = args.model
    if args.model not in ["large", "turbo"] and not args.non_english:
        model_name += ".en"
    print(f"[MAIN] Loading Whisper model ({model_name}) on {device}...")
    audio_model = whisper.load_model(model_name, device=device)
    print("[MAIN] Whisper model loaded.")

    # Load YAMNet model and class names
    print("[MAIN] Loading YAMNet model...")
    yamnet_model = hub.load('./yamnet_local')
    class_names = load_class_names(args.yamnet_csv)
    print("[MAIN] YAMNet model loaded.")

    # Start angle detection thread
    angle_thread_obj = threading.Thread(target=angle_thread, args=(stop_event,))
    angle_thread_obj.start()

    # Start the combined processing loop (which handles transcription, classification, and sending)
    combined_processing_loop(stop_event, args, device, audio_model, device_fs, target_fs, yamnet_model, class_names)

    try:
        while not stop_event.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("KeyboardInterrupt detected. Stopping...")
        stop_event.set()

    angle_thread_obj.join()
    print("All threads stopped. Exiting.")

if __name__ == "__main__":
    main()
