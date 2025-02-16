import sounddevice as sd
import numpy as np
import time
import scipy.io.wavfile as wav

def record_microphones(device_indices, mic_names, duration=10, samplerate=16000):
    """
    Records audio from multiple microphones simultaneously for a fixed duration
    and saves each recording as a separate WAV file.
    """
    n_mics = len(device_indices)
    
    # Create a dictionary to store recorded audio data
    recordings = {name: [] for name in mic_names}

    def callback(indata, frames, time_info, status, mic_index, mic_name):
        """
        Callback function to capture audio and store it in the recordings dictionary.
        """
        if status:
            print(f"Microphone {mic_name} status: {status}", flush=True)
        recordings[mic_name].append(indata.copy())

    streams = []
    
    try:
        # Open an InputStream for each microphone
        for i in range(n_mics):
            stream = sd.InputStream(
                device=device_indices[i],
                channels=6,
                samplerate=samplerate,
                callback=lambda indata, frames, time_info, status, i=i: 
                    callback(indata, frames, time_info, status, i, mic_names[i])
            )
            streams.append(stream)
            stream.start()

        print(f"\nRecording for {duration} seconds...")
        time.sleep(duration)

    finally:
        # Stop all streams
        for stream in streams:
            stream.stop()
            stream.close()

        print("\nSaving recordings...")

        # Save each microphone's data as a WAV file
        for mic_name, data in recordings.items():
            if data:
                # Convert list of arrays to a single NumPy array
                audio_data = np.concatenate(data, axis=0)
                # Normalize to int16 for saving as WAV
                audio_data = (audio_data * 32767).astype(np.int16)
                # Save the file
                filename = f"{mic_name.replace(' ', '_')}.wav"
                wav.write(filename, samplerate, audio_data)
                print(f"Saved: {filename}")

    print("\nRecording complete!")

def main():
    # Print available devices to help user pick indices
    print("Available audio devices:")
    print(sd.query_devices())

    # Ask the user for device indices (comma-separated)
    devices_str = input("\nEnter device indices (comma-separated) for the microphones you want to use: ")
    device_indices = [int(x.strip()) for x in devices_str.split(",")]
    n_mics = len(device_indices)

    if n_mics < 1:
        print("Please provide at least one device index.")
        return

    # Ask for user-friendly names for each microphone
    mic_names_str = input("\nEnter names for each microphone (comma-separated, in the same order as the indices): ")
    mic_names = [x.strip() for x in mic_names_str.split(",")]

    if len(mic_names) != n_mics:
        print("Error: The number of names must match the number of device indices.")
        return

    # Record and save audio from all selected microphones
    record_microphones(device_indices, mic_names, duration=10)

if __name__ == "__main__":
    main()
