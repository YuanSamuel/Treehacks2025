import sounddevice as sd
import wave

# Parameters
SAMPLE_RATE = 44100  # Standard CD-quality sample rate
DURATION = 5         # Record for 5 seconds
CHANNELS = 1         # Mono (typical for a single mic)

# If you want to target a specific device, set it here:
# device_index_or_name = 2   # Example: integer device index from sd.query_devices()
# Or device_index_or_name = "USB Audio Device"
# But if your mic is the default input device, you can omit 'device' below.

# Start recording
print("Recording started...")
recording = sd.rec(frames=int(SAMPLE_RATE * DURATION),
                   samplerate=SAMPLE_RATE,
                   channels=CHANNELS,
                   dtype='int16')
sd.wait()  # Wait until recording is finished

print("Recording finished. Saving to output.wav...")

# Save to a WAV file
with wave.open("output.wav", "wb") as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(2)  # 'int16' is 2 bytes
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(recording.tobytes())

print("Saved output.wav")
