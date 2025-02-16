#!/usr/bin/env python3
import sounddevice as sd
import numpy as np
import pyroomacoustics as pra

# 1) Define mic geometry (2 x 4 array for four mics in 2D)
mic_positions = np.array([
    [ 0.033,  0.033],
    [ 0.033, -0.033],
    [-0.033, -0.033],
    [-0.033,  0.033],
]).T  # shape: (2, 4)

# Number of mics we want to use:
NUM_MICS = 4

# 2) Audio settings
SAMPLE_RATE = 16000
BLOCK_SIZE  = 1024  # samples per block to read

# 3) Create a DOA object from Pyroomacoustics
# - We specify the mic array geometry and the wave speed (default ~343 m/s at 20°C).
# - 'MUSIC' is one of several algorithms supported (e.g. SRP, CSSM, WAVES, etc.).
doa_estimator = pra.doa.music.MUSIC(
    L=BLOCK_SIZE,
    fs=SAMPLE_RATE,
    nfft=256,             # FFT size for the algorithm; adjust as needed
    c=343.0,              # Speed of sound (m/s)
    num_src=1,            # Number of sources we want to localize
    dim=2,                # 2D scenario
    microphones=mic_positions
)

# 4) Select your ReSpeaker 4-Mic device
#    Use sounddevice.query_devices() to find the correct index, e.g.:
#    print(sd.query_devices())
DEVICE_INDEX = None  # or set to an integer if you know your device index

def audio_callback(indata, frames, time_info, status):
    """
    This callback is invoked whenever a new audio block is available.
    'indata' is shape (frames, channels).
    We assume the first 4 channels correspond to the physical mics.
    """
    if status:
        print(status, flush=True)

    # Convert to float64 for pyroomacoustics
    audio_data = np.array(indata[:, 0:NUM_MICS], dtype=np.float64).T
    # shape now: (NUM_MICS, frames), i.e. (4, 1024)

    # 5) Pass the data block to the DOA estimator
    doa_estimator.locate_sources(audio_data)

    # 6) Retrieve the estimated DOA. In 2D, pyroomacoustics returns an azimuth angle in radians.
    #    Typically 0 rad is from x-axis, measured counterclockwise.
    #    Convert to degrees and maybe shift so 0° is "front".
    azimuth = np.degrees(doa_estimator.azimuth_recon[0])

    # Optional: wrap angle to 0-360
    azimuth = azimuth % 360

    # Print or otherwise handle the angle
    print(f"Estimated DOA: {azimuth:.1f}°")

def main():
    print("Starting audio stream for DOA estimation...")
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        channels=6,  # The 4-mic array might show 6 input channels total
        device=DEVICE_INDEX,
        dtype='int16',  # or 'float32' depending on device
        callback=audio_callback
    ):
        print("Press Ctrl+C to stop.")
        while True:
            sd.sleep(1000)

if __name__ == "__main__":
    main()
