
import pyaudio
import struct

# Initialize PyAudio
p = pyaudio.PyAudio()

# Get info about the default input device
info = p.get_default_input_device_info()
num_channels = info['maxInputChannels']

print(f"Default input device has {num_channels} channel(s).")

# Check if the device is stereo
if num_channels == 2:
    print("This input device supports stereo (Left/Right).")

    # Open a stereo stream (16-bit PCM, 44.1kHz)
    stream = p.open(
        format=pyaudio.paInt16,
        channels=2,
        rate=44100,
        input=True,
        frames_per_buffer=1024
    )

    print("Recording a short snippet for analysis...")
    data = stream.read(1024)  # Read one buffer worth of frames

    # Stop & close the stream
    stream.stop_stream()
    stream.close()

    # We'll parse the first 10 frames of the buffer just to demonstrate
    left_samples = []
    right_samples = []

    # Each frame has 2 channels (Left, Right), and each sample is 16 bits = 2 bytes.
    # So each frame is 4 bytes total.
    for i in range(10):
        # Calculate where in the byte array this frame starts:
        frame_offset = i * 4  # 4 bytes per frame in 16-bit stereo
        # Extract left (2 bytes) and right (2 bytes) samples
        left_val = struct.unpack('<h', data[frame_offset:frame_offset+2])[0]
        right_val = struct.unpack('<h', data[frame_offset+2:frame_offset+4])[0]

        left_samples.append(left_val)
        right_samples.append(right_val)

    print("Left channel samples (first 10 frames):", left_samples)
    print("Right channel samples (first 10 frames):", right_samples)

else:
    print("This device is not in stereo mode (it might be mono or more than 2 channels).")

# Terminate PyAudio
p.terminate()

