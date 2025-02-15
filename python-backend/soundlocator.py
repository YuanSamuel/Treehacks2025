import sounddevice as sd
import numpy as np
import threading
import time
import argparse

# We use an Event to signal when we want all threads to stop.
stop_event = threading.Event()

def audio_callback(indata, frames, time_info, status, mic_index, shared_amplitudes):
    """
    Callback function called by the InputStream whenever audio is available.
    It updates the shared_amplitudes list with the RMS amplitude for this mic_index.
    """
    if status:
        print(f"Microphone {mic_index} callback status: {status}", flush=True)
    # Compute RMS amplitude:
    amplitude = np.sqrt(np.mean(indata**2))
    shared_amplitudes[mic_index] = amplitude

def record_microphone(mic_index, device_index, shared_amplitudes, samplerate=44100):
    """
    Opens an InputStream for the specified device_index, using a callback
    that updates the global shared_amplitudes for mic_index.
    """
    def callback_wrapper(indata, frames, time_info, status):
        audio_callback(indata, frames, time_info, status, mic_index, shared_amplitudes)

    # Open the input stream with 1 channel (mono) from the given device
    with sd.InputStream(device=device_index,
                        channels=1,
                        samplerate=samplerate,
                        callback=callback_wrapper):
        # Keep the stream alive until stop_event is set.
        while not stop_event.is_set():
            sd.sleep(100)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Monitor microphone amplitudes with an optional threshold.")
    parser.add_argument("-t", "--threshold", type=float, default=None, help="Amplitude threshold for printing (optional).")
    args = parser.parse_args()
    threshold = args.threshold  # Get the threshold value (or None if not provided)

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

    # Initialize a shared list of amplitudes for each mic
    shared_amplitudes = [0.0] * n_mics

    # Create and start one thread per microphone
    threads = []
    for i in range(n_mics):
        t = threading.Thread(
            target=record_microphone,
            args=(i, device_indices[i], shared_amplitudes)
        )
        t.start()
        threads.append(t)

    print(f"\nRecording from {n_mics} devices... Press Ctrl+C to stop.")
    if threshold:
        print(f"Only printing when amplitude exceeds {threshold}.")

    try:
        while True:
            if np.sum(shared_amplitudes) > 0:  # Avoid division by zero
                loudest_mic_index = np.argmax(shared_amplitudes)
                loudest_mic_name = mic_names[loudest_mic_index]
            else:
                loudest_mic_name = "None"

            # Check if any amplitude exceeds the threshold (if provided)
            if threshold is None or any(amp > threshold for amp in shared_amplitudes):
                print(
                    f"Loudest microphone: {loudest_mic_name} "
                    f"amplitudes={shared_amplitudes})"
                )

            time.sleep(0.2)  # Adjust frequency of printing as desired

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        # Signal threads to stop and wait for them to finish
        stop_event.set()
        for t in threads:
            t.join()

if __name__ == "__main__":
    main()
