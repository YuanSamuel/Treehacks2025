import sounddevice as sd
import numpy as np
import threading
import time

# Global list that will store the latest amplitude for each microphone.
# In this example, we assume exactly 4 microphones.
shared_amplitudes = [0.0, 0.0, 0.0, 0.0]

# We'll use an Event to signal when we want all threads to stop.
stop_event = threading.Event()

def audio_callback(indata, frames, time_info, status, mic_index):
    """Called whenever the audio buffer is filled.
    Updates the shared_amplitudes with the latest RMS amplitude for this mic."""
    if status:
        print(f"Microphone {mic_index} callback status: {status}", flush=True)
    # Compute RMS amplitude:
    amplitude = np.sqrt(np.mean(indata**2))
    shared_amplitudes[mic_index] = amplitude

def record_microphone(mic_index, device_index, samplerate=44100):
    """
    Opens an InputStream for the specified device_index, using a callback
    that updates the global shared_amplitudes for mic_index.
    """
    def callback_wrapper(indata, frames, time_info, status):
        audio_callback(indata, frames, time_info, status, mic_index)

    # Open the input stream with 1 channel (mono) from the given device
    # Adjust channels, samplerate, blocksize, etc. as needed.
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

    # Ask the user for 4 device indices (or you can hardcode them).
    devices_str = input(
        "Enter the 4 device indices (comma-separated) for the microphones you want to use: "
    )
    device_indices = [int(x.strip()) for x in devices_str.split(",")]
    
    if len(device_indices) != 4:
        print("Please provide exactly 4 device indices.")
        return

    # Create and start one thread per microphone
    threads = []
    for i in range(4):
        t = threading.Thread(target=record_microphone, args=(i, device_indices[i]))
        t.start()
        threads.append(t)

    print("Recording from four devices... Press Ctrl+C to stop.")

    try:
        while True:
            # Determine which microphone is the loudest at this moment
            loudest_mic = np.argmax(shared_amplitudes)
            # (You could also consider thresholds or smoothing if you prefer.)
            print(f"Loudest microphone index: {loudest_mic} (amplitudes={shared_amplitudes})")
            time.sleep(0.2)  # Adjust how frequently you want to print
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        # Signal threads to stop and wait for them to finish
        stop_event.set()
        for t in threads:
            t.join()

if __name__ == "__main__":
    main()
