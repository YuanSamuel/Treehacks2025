import queue
import threading
import time

import numpy as np
import sounddevice as sd
import tensorflow as tf
import tensorflow_hub as hub
import librosa

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
            # Sentinel value to indicate we should stop
            print("Classification worker stopped.")
            break

        # Accumulate chunk into our buffer
        chunk_buffer = np.concatenate((chunk_buffer, chunk), axis=0)

        # If we have at least 1 second of audio, run classification
        if len(chunk_buffer) >= buffer_size:
            # Take 1 second from the front, keep leftover for overlap
            one_second_audio = chunk_buffer[:buffer_size]
            chunk_buffer = chunk_buffer[buffer_size:]  # leftover for next round

            # 4. Run YAMNet
            # YAMNet expects [any_length] float32 waveform at 16kHz
            waveform = one_second_audio.astype(np.float32)

            # Run inference
            scores, embeddings, spectrogram = yamnet_model(waveform)
            scores_np = scores.numpy()

            # Average scores across time frames
            mean_scores = np.mean(scores_np, axis=0)

            # Pick top 3
            top3_i = np.argsort(mean_scores)[-3:][::-1]

            # Print results
            print("Top predictions:")
            for i in top3_i:
                print(f"  {class_names[i]} (confidence: {mean_scores[i]:.2f})")
            print("-" * 50)

# -----------------------------------------------------
# 4. Audio callback: push chunks to the queue
# -----------------------------------------------------
def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"Audio callback status: {status}")
    # 'indata' is float32 audio at 16kHz, shape: (frames, channels)
    # We'll assume 1 channel for simplicity. If stereo, convert to mono.
    if indata.shape[1] > 1:
        indata_mono = np.mean(indata, axis=1)
    else:
        indata_mono = indata[:, 0]

    # Put chunk into the queue
    audio_queue.put(indata_mono.copy())

# -----------------------------------------------------
# Main function: start stream, start worker, run
# -----------------------------------------------------
def main():
    # Start classification worker in a background thread
    worker_thread = threading.Thread(target=classification_worker, daemon=True)
    worker_thread.start()

    # Sample rate = 16k for YAMNet
    samplerate = 16000

    # Open an InputStream
    with sd.InputStream(channels=1,
                        samplerate=samplerate,
                        callback=audio_callback):
        print("Recording... Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nStopping...")

    # Signal worker to stop by sending None
    audio_queue.put(None)
    worker_thread.join()

if __name__ == "__main__":
    main()
