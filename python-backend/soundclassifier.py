#!/usr/bin/env python3

import sys
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import soundfile as sf
import librosa

def main():
    # Check for command-line argument
    if len(sys.argv) < 2:
        print("Usage: python soundclassifer.py /path/to/audio.wav")
        sys.exit(1)
    
    # Path to the WAV file
    wav_file = sys.argv[1]
    
    # Load YAMNet model from TensorFlow Hub
    print("Loading YAMNet model...")
    yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
    
    # Download the YAMNet class map for labeling
    class_map_path = "yamnet_class_map.csv"
    # Parse the CSV to get human-readable class names
    class_names = [line.strip().split(',')[2] for line in open(class_map_path).readlines()[1:]]
    
    # Load the WAV file
    print(f"Reading audio data from: {wav_file}")
    waveform, sample_rate = sf.read(wav_file)
    
    # Convert to mono if stereo
    if len(waveform.shape) > 1:
        waveform = np.mean(waveform, axis=1)
    
    # YAMNet expects 16 kHz audio
    if sample_rate != 16000:
        print(f"Resampling from {sample_rate} Hz to 16000 Hz...")
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000
    
    # Convert waveform to float32
    waveform = waveform.astype(np.float32)
    
    # Run the YAMNet model
    print("Classifying audio with YAMNet...")
    scores, embeddings, spectrogram = yamnet_model(waveform)
    scores_np = scores.numpy()
    
    # Average scores across all time frames
    mean_scores = np.mean(scores_np, axis=0)
    
    # Identify top 3 classes
    top3_i = np.argsort(mean_scores)[-3:][::-1]
    
    print("\n=== YAMNet Top 3 Predictions ===")
    for i in top3_i:
        print(f"{class_names[i]} (Confidence: {mean_scores[i]:.2f})")

if __name__ == "__main__":
    main()
