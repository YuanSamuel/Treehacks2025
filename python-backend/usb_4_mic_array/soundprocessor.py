import threading
import time
import csv
import numpy as np
import sounddevice as sd
import tensorflow as tf
import tensorflow_hub as hub
import usb.core
import usb.util
from tuning import Tuning

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
    Continuously prints the angle (direction) of the sound and the speech detection status.
    """
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
            print(f"[ANGLE] Direction: {direction} | SpeechDetected: {speech_detected}")
            time.sleep(1)
    except Exception as e:
        print(f"[ANGLE] Error: {e}")

def sound_classification_thread(stop_event, device_id):
    """
    Continuously records audio from the specified device, processes it using the local YAMNet model,
    and prints the top predictions.
    """
    # YAMNet expects 16 kHz mono audio.
    sample_rate = 16000
    duration = 1.0  # seconds of audio per inference
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
                
                # Select channel 0 (adjust if needed) and ensure float32 type.
                mono_audio = audio_data[:, 0].astype(np.float32)
                waveform = tf.convert_to_tensor(mono_audio)
                
                # Run inference with YAMNet.
                scores, embeddings, spectrogram = yamnet_model(waveform)
                mean_scores = np.mean(scores, axis=0)
                top_indices = np.argsort(mean_scores)[-2:][::-1]
                
                print("\n[PREDICTION] Top predictions:")
                for i in top_indices:
                    print(f"  {class_names[i]}: {mean_scores[i]:.3f}")
    except Exception as e:
        print(f"[PREDICTION] Error: {e}")

def main():
    # Ask the user for the audio device ID.
    try:
        device_id = int(input("Enter the sound device ID: "))
    except ValueError:
        print("Invalid device ID. Exiting.")
        return
    
    stop_event = threading.Event()
    
    # Create and start both threads.
    angle_thread_obj = threading.Thread(target=angle_thread, args=(stop_event,))
    prediction_thread_obj = threading.Thread(target=sound_classification_thread, args=(stop_event, device_id))
    
    angle_thread_obj.start()
    prediction_thread_obj.start()
    
    try:
        # Main thread idles until a KeyboardInterrupt (Ctrl+C) is detected.
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
