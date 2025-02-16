import sounddevice as sd
import numpy as np
import queue
import threading
import math
import time

###############################################################################
# 1) Configuration: sample rate, mic array geometry, etc.
###############################################################################

SAMPLE_RATE = 48000   # Hz
BLOCK_SIZE = 1024     # Samples per block (adjust as needed)
DISTANCE = 0.2        # Meters between adjacent mics in the square
SPEED_OF_SOUND = 343  # m/s (approx. at 20°C)

# Mic positions (X, Y) for a square layout.
#  (Mic0 at NW, Mic1 at NE, Mic2 at SW, Mic3 at SE)
mic_positions = np.array([
    [-DISTANCE/2,  DISTANCE/2],  # Mic0
    [ DISTANCE/2,  DISTANCE/2],  # Mic1
    [-DISTANCE/2, -DISTANCE/2],  # Mic2
    [ DISTANCE/2, -DISTANCE/2],  # Mic3
])

###############################################################################
# 2) GCC-PHAT function (common method to estimate time delay between signals)
###############################################################################
def gcc_phat(sig, refsig, fs=1, max_tau=None, interp=1):
    """
    Compute the time delay between 'sig' and 'refsig' using the
    Generalized Cross Correlation - Phase Transform (GCC-PHAT).
    Returns the delay (in seconds).

    :param sig:     1D numpy array of the signal of interest
    :param refsig:  1D numpy array of the reference signal
    :param fs:      sampling rate
    :param max_tau: maximum allowed time delay (seconds)
    :param interp:  upsampling factor for cross-correlation
    """
    n = len(sig) + len(refsig)
    # FFT size (next power of 2 can be used, but here we do n directly)
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)

    # Apply PHAT weighting
    denom = np.abs(R)
    denom = np.maximum(denom, 1e-8)  # avoid /0
    R /= denom

    # Inverse FFT
    cc = np.fft.irfft(R, n=n * interp)
    max_shift = int(interp * n / 2)
    if max_tau:
        max_shift = np.minimum(int(interp * fs * max_tau), max_shift)

    # Find the shift index of the max cross-correlation peak
    shift_range = np.arange(-max_shift, max_shift+1)
    cc_segment = cc[-max_shift: max_shift+1]
    shift = np.argmax(np.abs(cc_segment)) - max_shift
    tau = shift / float(interp * fs)
    return tau

###############################################################################
# 3) Predict TDOAs for a given angle (0-360°) and compare with measured TDOAs
###############################################################################
def predict_tdoas(angle_deg, mic_positions, ref_index=0):
    """
    Given an angle in degrees (0-360), compute the expected TDOA (mic_i vs mic_ref)
    for each mic i>0, relative to the reference mic (ref_index).
    Returns a list [tau_1ref, tau_2ref, tau_3ref] for mics 1,2,3 vs mic0 (for example).

    :param angle_deg:      incoming wave angle in degrees (0 = +X axis, 90 = +Y axis)
    :param mic_positions:  Nx2 array of mic positions
    :param ref_index:      which mic is the reference (0 by default)
    """
    angle_rad = np.deg2rad(angle_deg)

    # Unit direction vector from which the wave is traveling *towards* the array.
    # Convention: 0° means wave traveling along +X axis, 90° means wave traveling along +Y, etc.
    # However, you might want to invert if you define 0° as "to the right" or "to the east."
    direction = np.array([math.cos(angle_rad), math.sin(angle_rad)], dtype=np.float32)

    # We'll measure TDOA_i = ( (mic_i - mic_ref) . direction ) / c
    # Because the wave arrives at mic_ref first, then mic_i after some delay (or negative if it arrives earlier).
    ref_pos = mic_positions[ref_index]
    tdoas = []
    for i in range(len(mic_positions)):
        if i == ref_index:
            continue
        delta_pos = mic_positions[i] - ref_pos
        # Project delta_pos onto 'direction'
        dist_diff = np.dot(delta_pos, direction)
        tau = dist_diff / SPEED_OF_SOUND
        tdoas.append(tau)
    return tdoas

def angle_error(angle_deg, measured_tdoas):
    """
    Compute the sum of squared error between measured TDOAs and predicted TDOAs
    for a given angle. 'measured_tdoas' is [tau_01, tau_02, tau_03] for mics 1,2,3 vs 0.
    """
    # Predicted TDOAs for the same mic order: mic1 vs mic0, mic2 vs mic0, mic3 vs mic0
    predicted = predict_tdoas(angle_deg, mic_positions, ref_index=0)
    # Compare
    if len(predicted) != len(measured_tdoas):
        return 1e9  # mismatch in length, large error
    error = 0.0
    for p, m in zip(predicted, measured_tdoas):
        error += (p - m)**2
    return error

def estimate_angle_from_tdoas(tau_01, tau_02, tau_03):
    """
    Brute force search from 0 to 359 degrees, picking the angle with minimal SSE to measured TDOAs.

    :param tau_01, tau_02, tau_03: measured TDOAs for (mic1 vs mic0), (mic2 vs mic0), (mic3 vs mic0).
    :return: best_angle (float), minimal_error
    """
    measured = [tau_01, tau_02, tau_03]
    best_angle = 0.0
    min_error = 1e9

    # You can refine by smaller steps if you need more precision (e.g., 0.5°, 0.1°, etc.)
    for angle_deg in range(360):
        err = angle_error(angle_deg, measured)
        if err < min_error:
            min_error = err
            best_angle = angle_deg
    return best_angle, min_error

###############################################################################
# 4) Main: Record audio in 4 channels, compute TDOAs, estimate angle
###############################################################################

# A thread-safe queue for audio blocks
audio_queue = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    # 'indata' is shape (frames, 4) if channels=4
    if status:
        print(f"Status: {status}", flush=True)
    audio_queue.put(indata.copy())

def main():
    # Open a 4-channel input stream at our chosen sample rate / block size
    # Adjust 'device' parameter if needed (e.g., device=YourMicArrayIndex).
    stream = sd.InputStream(
        channels=4,
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        callback=audio_callback
    )

    print("Starting 4-channel stream. Press Ctrl+C to stop.")
    with stream:
        try:
            while True:
                # Get one block from the queue (blocking)
                block = audio_queue.get()
                # block.shape => (BLOCK_SIZE, 4)
                mic0 = block[:, 0]
                mic1 = block[:, 1]
                mic2 = block[:, 2]
                mic3 = block[:, 3]

                # Compute TDOA of mic1 vs mic0, mic2 vs mic0, mic3 vs mic0
                tau_01 = gcc_phat(mic1, mic0, fs=SAMPLE_RATE)
                tau_02 = gcc_phat(mic2, mic0, fs=SAMPLE_RATE)
                tau_03 = gcc_phat(mic3, mic0, fs=SAMPLE_RATE)

                # Estimate the angle (0-359) that best fits these TDOAs
                angle_deg, err = estimate_angle_from_tdoas(tau_01, tau_02, tau_03)

                print(f"Measured TDOAs => (01: {tau_01:.6f}s, 02: {tau_02:.6f}s, 03: {tau_03:.6f}s)")
                print(f"Estimated Angle => {angle_deg:.1f}°, Error={err:.2e}\n")

                # Sleep or loop immediately. Adjust as needed for your throughput.
                time.sleep(0.05)

        except KeyboardInterrupt:
            print("Stopping...")

if __name__ == "__main__":
    main()
