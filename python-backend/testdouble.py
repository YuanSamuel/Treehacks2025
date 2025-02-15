import sounddevice as sd
import wave
import threading
import time

def record_microphone(device_id, filename, stop_event):
    # Query device info
    device_info = sd.query_devices(device_id)
    samplerate = int(device_info['default_samplerate'])
    channels = device_info['max_input_channels']
    
    # Set up WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(2)  # 16-bit PCM
    wf.setframerate(samplerate)

    # Callback to write audio data
    def callback(indata, frames, time, status):
        if status:
            print(f"Device {device_id}: {status}")
        wf.writeframes(indata.tobytes())

    # Start audio stream
    stream = sd.InputStream(
        device=device_id,
        samplerate=samplerate,
        channels=channels,
        dtype='int16',
        callback=callback
    )
    stream.start()

    # Keep the thread alive until stop_event is set
    while not stop_event.is_set():
        time.sleep(0.1)

    # Cleanup
    stream.stop()
    stream.close()
    wf.close()
    print(f"Stopped recording from device {device_id}")

# List available devices
print("Available Devices:")
print(sd.query_devices())

# Get device IDs from user
device1 = int(input("Enter device ID for first microphone: "))
device2 = int(input("Enter device ID for second microphone: "))

# Create stop event
stop_event = threading.Event()

# Start recording threads
thread1 = threading.Thread(target=record_microphone, args=(device1, 'mic1.wav', stop_event))
thread2 = threading.Thread(target=record_microphone, args=(device2, 'mic2.wav', stop_event))

thread1.start()
thread2.start()

# Wait for keyboard interrupt to stop
try:
    print("Recording... Press Ctrl+C to stop.")
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nStopping recordings...")
    stop_event.set()
    thread1.join()
    thread2.join()