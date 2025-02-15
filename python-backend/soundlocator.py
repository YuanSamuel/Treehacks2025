import sounddevice as sd
import numpy as np
import threading
import time
import matplotlib.pyplot as plt

stop_event = threading.Event()

def softmax(x):
    """Compute softmax values for an array x."""
    exp_x = np.exp(x - np.max(x))  # Stability improvement by subtracting max
    return exp_x / np.sum(exp_x)

def audio_callback(indata, frames, time_info, status, mic_index, shared_amplitudes):
    """
    Callback function called by the InputStream whenever audio is available.
    It updates the shared_amplitudes list with the RMS amplitude for mic_index.
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

    with sd.InputStream(device=device_index,
                        channels=1,
                        samplerate=samplerate,
                        callback=callback_wrapper):
        while not stop_event.is_set():
            sd.sleep(100)

def get_directions(n_mics):
    """
    Returns a dictionary that maps each mic index -> (x, y) coordinate.
    This is the 'direction vector' for that microphone.
    """
    if n_mics == 2:
        # Left, Right
        return {
            0: (-1,  0),
            1: ( 1,  0)
        }
    elif n_mics == 3:
        # Triangular: bottom-left, bottom-right, top
        return {
            0: (-1, -1),  # Bottom-left
            1: ( 1, -1),  # Bottom-right
            2: ( 0,  1)   # Top
        }
    elif n_mics == 4:
        # NW, NE, SW, SE
        return {
            0: (-1,  1),
            1: ( 1,  1),
            2: (-1, -1),
            3: ( 1, -1)
        }
    else:
        # Fallback (no defined layout). 
        return {i: (0,0) for i in range(n_mics)}

def setup_plot():
    """
    Sets up an interactive matplotlib plot, returns (fig, ax).
    """
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_aspect('equal', 'box')  # Make x/y scale the same
    ax.grid(True)
    ax.set_title("Live Direction (Top Two Mics Combined)")
    plt.show()  # Ensure a window pops up
    return fig, ax

def update_plot_top_two(ax, softmax_values, n_mics):
    """
    1. Identify the top two microphones by softmax value.
    2. Sum their direction vectors (scaled by amplitude).
    3. Draw a single arrow from (0,0) to that summed vector.
    """
    directions = get_directions(n_mics)
    if not directions:
        return  # No defined directions for this number of mics

    # If we have fewer than 2 mics, or softmax_values is empty, just skip
    if len(softmax_values) < 2:
        return

    # Identify indices of the top two mics
    top_two_indices = np.argsort(softmax_values)[-2:]  # Last two are the largest

    # Sum the direction vectors, each scaled by its softmax amplitude
    resultant_x = 0.0
    resultant_y = 0.0
    for idx in top_two_indices:
        dx, dy = directions.get(idx, (0,0))
        amplitude = softmax_values[idx]
        resultant_x += dx * amplitude
        resultant_y += dy * amplitude

    # Clear old arrows
    ax.cla()
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_aspect('equal', 'box')
    ax.grid(True)
    ax.set_title("Live Direction (Top Two Mics Combined)")

    # Draw arrow from origin to (resultant_x, resultant_y)
    ax.arrow(
        0, 0,
        resultant_x, resultant_y,
        head_width=0.1,
        length_includes_head=True,
        color='r'
    )

    # Update the plot
    plt.draw()
    plt.pause(0.01)

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

    shared_amplitudes = [0.0] * n_mics

    # Launch a thread per microphone
    threads = []
    for i in range(n_mics):
        t = threading.Thread(
            target=record_microphone,
            args=(i, device_indices[i], shared_amplitudes)
        )
        t.start()
        threads.append(t)

    print(f"Recording from {n_mics} devices... Press Ctrl+C to stop.")

    # Set up the real-time plot
    fig, ax = setup_plot()

    try:
        while True:
            # Avoid division by zero
            if np.sum(shared_amplitudes) > 0:
                softmax_values = softmax(shared_amplitudes)
            else:
                softmax_values = [0.0] * n_mics

            # Print out the status (amplitudes & softmax)
            print(
                f"Amplitudes: {shared_amplitudes}\n"
                f"Softmax:    {softmax_values}\n"
            )

            # Update the direction arrow for the top two mics
            update_plot_top_two(ax, softmax_values, n_mics)

            time.sleep(0.2)

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        stop_event.set()
        for t in threads:
            t.join()

if __name__ == "__main__":
    main()
