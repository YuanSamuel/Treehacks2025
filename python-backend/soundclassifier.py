import csv
import time
import numpy as np
import sounddevice as sd
import tensorflow as tf
import tensorflow_hub as hub

def load_class_names(csv_path):
    """Load the YAMNet class names from a CSV file.
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
            # row[2] is the display name.
            class_names.append(row[2])
    return class_names

def main():
    # Let the user specify the input device id.
    device_id = int(input("Enter the sound device ID: "))
    
    # YAMNet expects 16 kHz mono audio.
    sample_rate = 16000
    duration = 1.0  # seconds of audio to process each inference
    num_samples = int(sample_rate * duration)

    # Load the YAMNet model and class mapping.
    print("Loading YAMNet model...")
    yamnet_model = hub.load('./yamnet_local')
    class_map_path = "yamnet_local/yamnet_class_map.csv"
    class_names = load_class_names(class_map_path)
    
    print(f"Starting audio stream on device {device_id} (using sample rate {sample_rate} Hz)...")
    
    # Open an input stream with 6 channels (even though we only need channel 5).
    # Note: channels are zero-indexed in Python.
    with sd.InputStream(device=device_id, channels=6, samplerate=sample_rate) as stream:
        while True:
            try:
                # Read one second of audio (num_samples frames)
                audio_data, overflowed = stream.read(num_samples)
                if overflowed:
                    print("Warning: Audio buffer has overflowed!")
                
                # audio_data has shape (num_samples, 6).
                # We select channel 6 (i.e. column 5) and flatten to 1-D.
                mono_audio = audio_data[:, 0]
                
                # Ensure the data is float32 (YAMNet expects float32 waveform).
                mono_audio = mono_audio.astype(np.float32)
                
                # Optionally, you can convert to a TensorFlow tensor:
                waveform = tf.convert_to_tensor(mono_audio)
                
                # Run YAMNet inference.
                # The model returns:
                #   scores: shape (num_patches, num_classes)
                #   embeddings: (num_patches, embedding_size)
                #   spectrogram: (num_patches, num_bands)
                scores, embeddings, spectrogram = yamnet_model(waveform)
                
                # Average the per-frame scores to get a single prediction per class.
                mean_scores = np.mean(scores, axis=0)
                
                # Get the indices of the top 2 predicted classes.
                top_indices = np.argsort(mean_scores)[-2:][::-1]
                
                print("\nTop predictions:")
                for i in top_indices:
                    print(f"  {class_names[i]}: {mean_scores[i]:.3f}")
                
                # The stream.read() call blocks until 1 second of audio is collected,
                # so no additional sleep is necessary.
            except KeyboardInterrupt:
                print("\nExiting.")
                break

if __name__ == '__main__':
    main()
