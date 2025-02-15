import sounddevice as sd
import numpy as np
import threading
import time

# We use an Event to signal when we want all threads to stop.
stop_event = threading.Event()

def softmax(x):
    """Compute softmax values for an array x."""
    exp_x = np.exp(x - np.max(x))  # Stability improvement by subtracting max
    return exp_x / np.sum(exp_x)

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
    # Print available devices to help user pick indices
    print("Available audio devices:")
    print(sd.query_devices())

    # Ask the user for device indices (comma-separated)
    devices_str = input(
        "Enter device indices (comma-separated) for the microphones you want to use: "
    )
    device_indices = [int(x.strip()) for x in devices_str.split(",")]
    n_mics = len(device_indices)
    
    if n_mics < 1:
        print("Please provide at least one device index.")
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

    print(f"Recording from {n_mics} devices... Press Ctrl+C to stop.")

    try:
        while True:
            # Compute softmax of amplitudes
            if np.sum(shared_amplitudes) > 0:  # Avoid division by zero
                softmax_values = softmax(shared_amplitudes)
                loudest_mic = np.argmax(softmax_values)
            else:
                softmax_values = [0.0] * n_mics  # If all zero, no confidence in any mic
                loudest_mic = None

            print(
                f"Loudest microphone index: {loudest_mic} "
                f"(softmax={softmax_values}, amplitudes={shared_amplitudes})"
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
