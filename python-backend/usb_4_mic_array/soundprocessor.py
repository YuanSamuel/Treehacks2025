#!/usr/bin/env python3
import threading
import time
import csv
import numpy as np
import sounddevice as sd
import tensorflow as tf
import tensorflow_hub as hub
import usb.core
import usb.util
import math
import socket
import argparse
from tuning import Tuning

# Global variable to hold the current angle from the angle_thread.
current_angle = None

def load_class_names(csv_path):
    """
    Load the YAMNet class names from a CSV file.
    The CSV is assumed to have rows of the form: index, mid, display_name.
    """
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

def angle_thread(stop_event):
    """
    Continuously update the sound direction (angle) and speech detection status.
    """
    global current_angle
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

def send_message(host: str, port: int, message: str):
    """
    Connect to the specified host and port, send a message,
    and print any response from the server.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            print(f"[QuestUDP] Connecting to {host}:{port}...")
            sock.connect((host, port))
            print("[QuestUDP] Connected.")
            print(f"[QuestUDP] Sending message: {message}")
            sock.sendall(message.encode('utf-8'))
    except ConnectionRefusedError:
        print("[QuestUDP] Connection refused. Is the Unity server running on the Quest?")
    except socket.timeout:
        print("[QuestUDP] Connection timed out.")
    except Exception as e:
        print("[QuestUDP] An error occurred:", e)

def sound_classification_thread(stop_event, device_id, metaquest_host, metaquest_port):
    """
    Continuously record audio, compute its RMS (volume), run inference with YAMNet,
    and send the direction, magnitude, and top prediction to MetaQuest via UDP.
    """
    sample_rate = 16000  # YAMNet expects 16 kHz mono audio.
    duration = 1.0       # seconds per inference
    num_samples = int(sample_rate * duration)

    print("[PREDICTION] Loading YAMNet model...")
    yamnet_model = hub.load('../yamnet_local')
    class_map_path = "../yamnet_local/yamnet_class_map.csv"
    class_names = load_class_names(class_map_path)
    
    print(f"[PREDICTION] Starting audio stream on device {device_id} (sample rate {sample_rate} Hz)...")
    
    try:
        with sd.InputStream(device=device_id, channels=6, samplerate=sample_rate) as stream:
            while not stop_event.is_set():
                audio_data, overflowed = stream.read(num_samples)
                if overflowed:
                    print("[PREDICTION] Warning: Audio buffer has overflowed!")
                
                # Select channel 0 (you could also add logic to choose the highest volume channel)
                channel_index = 0
                mono_audio = audio_data[:, channel_index].astype(np.float32)
                
                # Compute RMS (volume) and convert to dB.
                rms_value = np.sqrt(np.mean(mono_audio**2))
                volume_db = 20 * math.log10(rms_value + 1e-9)
                
                waveform = tf.convert_to_tensor(mono_audio)
                
                # Run inference with YAMNet.
                scores, embeddings, spectrogram = yamnet_model(waveform)
                mean_scores = np.mean(scores, axis=0)
                top_index = np.argsort(mean_scores)[-1]
                top_score = mean_scores[top_index]
                
                angle_info = current_angle if current_angle is not None else "N/A"
                
                # Build the message with direction, magnitude, and prediction.
                message = (f"Direction: {angle_info} | Magnitude: {volume_db:.2f} dB | "
                           f"Prediction: {class_names[top_index]}: {top_score:.3f}")
                
                print("[PREDICTION]", message)
                
                # Send the message to MetaQuest via UDP.
                send_message(metaquest_host, metaquest_port, message)
    except Exception as e:
        print(f"[PREDICTION] Error: {e}")

def main():
    # List available sound devices.
    print("Available sound devices:")
    try:
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            print(f"{i}: {dev['name']}")
    except Exception as e:
        print("Error querying devices:", e)
    
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(description="Sound detection and UDP messaging to MetaQuest.")
    parser.add_argument('--device', type=int, default=None, help="Sound device ID to use for recording.")
    parser.add_argument('--metaquest-host', type=str, default="127.0.0.1", help="MetaQuest host IP address.")
    parser.add_argument('--metaquest-port', type=int, default=7000, help="MetaQuest port number.")
    args = parser.parse_args()
    
    if args.device is None:
        try:
            args.device = int(input("Enter the sound device ID: "))
        except ValueError:
            print("Invalid device ID. Exiting.")
            return

    stop_event = threading.Event()
    
    # Start the angle detection and sound classification threads.
    angle_thread_obj = threading.Thread(target=angle_thread, args=(stop_event,))
    prediction_thread_obj = threading.Thread(target=sound_classification_thread,
                                             args=(stop_event, args.device, args.metaquest_host, args.metaquest_port))
    
    angle_thread_obj.start()
    prediction_thread_obj.start()
    
    try:
        # Main thread idles until interrupted.
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nExiting. Stopping threads...")
        stop_event.set()
    
    angle_thread_obj.join()
    prediction_thread_obj.join()
    print("Threads stopped. Exiting program.")

if __name__ == '__main__':
    main()
