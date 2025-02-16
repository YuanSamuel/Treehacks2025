import queue
import threading
import time

import numpy as np
import sounddevice as sd
import tensorflow as tf
import tensorflow_hub as hub

# -----------------------------------------------------
# 1. Load YAMNet model + class map
# -----------------------------------------------------
print("Loading YAMNet model...")
yamnet_model = hub.load('./yamnet_local')

# Download the YAMNet class map for labeling
class_map_path = "yamnet_local/yamnet_class_map.csv"
# Parse the CSV to get human-readable class names
class_names = [line.strip().split(',')[2] for line in open(class_map_path).readlines()[1:]]

# -----------------------------------------------------
# 2. Prepare a queue for audio chunks
# -----------------------------------------------------
audio_queue = queue.Queue()

# -----------------------------------------------------
# 3. Classification worker thread
# -----------------------------------------------------
def classification_worker():
    """
    Continuously read audio chunks from the queue, process them,
    and run YAMNet classification in (near) real-time.
    """
    print("Classification worker started...")
    buffer_size = 16000  # We'll accumulate ~1 second of audio
    chunk_buffer = np.empty((0,), dtype=np.float32)

    while True:
        # Get next chunk from the queue (blocks until chunk is available)
        chunk = audio_queue.get()
        if chunk is None:
            print("Classification worker stopped.")
            break

        # Accumulate chunk into our buffer
        chunk_buffer = np.concatenate((chunk_buffer, chunk), axis=0)

        # If we have at least 1 second of audio, run classification
        if len(chunk_buffer) >= buffer_size:
            one_second_audio = chunk_buffer[:buffer_size]
            chunk_buffer = chunk_buffer[buffer_size:]  # leftover for next round

            # YAMNet expects [any_length] float32 waveform at 16kHz
            waveform = one_second_audio.astype(np.float32)

            # Run inference
            scores, embeddings, spectrogram = yamnet_model(waveform)
            scores_np = scores.numpy()

            # Average scores across time frames
            mean_scores = np.mean(scores_np, axis=0)

            # Pick top 1 prediction
            top_pred = np.argsort(mean_scores)[-1:][::-1]

            # Print results with timestamp
            timestamp = time.strftime("%H:%M:%S", time.localtime())
            print(f"[{timestamp}] Channel 5 - Top prediction:")
            print(f"  {class_names[top_pred[0]]} (confidence: {mean_scores[top_pred[0]]:.2f})")
            print("-" * 50)

# -----------------------------------------------------
# 4. Audio callback: listen only to the mapped channel (channel 5)
# -----------------------------------------------------
def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"Audio callback status: {status}")

    # With mapping in effect, indata already has shape (frames, 1)
    audio_data = indata[:, 0].copy()
    audio_queue.put(audio_data)

# -----------------------------------------------------
# Main function: start stream, start worker, run
# -----------------------------------------------------
def main():
    # List available audio devices
    print("Available audio devices:")
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        print(f"{i}: {dev['name']}")

    # Get device selection from user
    device_index = int(input("Enter input device index: "))

    # Verify device supports 16kHz sample rate with one channel
    try:
        sd.check_input_settings(device=device_index, samplerate=16000, channels=1)
    except sd.PortAudioError as e:
        print(f"Error: {e}")
        return

    # Start classification worker in a background thread
    worker_thread = threading.Thread(target=classification_worker, daemon=True)
    worker_thread.start()

    # Open an InputStream with selected device
    # Request only 1 channel and map that channel to device channel index 4 (channel 5)
    with sd.InputStream(device=device_index,
                        channels=1,          # Request 1 channel
                        samplerate=16000,
                        callback=audio_callback,
                        mapping=[5]):        # Map that channel from device channel 5 (0-indexed)
        print("Recording from channel 5... Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nStopping...")

    # Signal worker to stop
    audio_queue.put(None)
    worker_thread.join()

if __name__ == "__main__":
    main()
