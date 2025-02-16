import asyncio
import websockets
import json
import speech_recognition as sr

def select_microphone():
    available_mics = sr.Microphone.list_microphone_names()

    if not available_mics:
        print("No microphones detected. Please check your audio setup.")
        exit(1)

    print("Available microphone devices:")
    for index, name in enumerate(available_mics):
        print(f"[{index}] {name}")

    while True:
        try:
            choice = int(input("Select the microphone index to use: "))
            if 0 <= choice < len(available_mics):
                return sr.Microphone(device_index=choice)
            else:
                print("Invalid choice. Please select a valid index.")
        except ValueError:
            print("Invalid input. Please enter a number.")

async def send_audio():
    uri = "ws://localhost:8888"
    mic = select_microphone()
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 1000
    recognizer.dynamic_energy_threshold = False
    
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Listening...")

    async with websockets.connect(uri) as websocket:
        def callback(_, audio: sr.AudioData):
            audio_bytes = audio.get_raw_data()
            message = json.dumps({"audio": list(audio_bytes)})
            asyncio.run_coroutine_threadsafe(websocket.send(message), asyncio.get_event_loop())

        recognizer.listen_in_background(mic, callback)

        try:
            while True:
                response = await websocket.recv()
                data = json.loads(response)
                print(f"Transcription: {data.get('transcription', '')}")
        except websockets.exceptions.ConnectionClosed:
            print("Connection to server lost.")

if __name__ == "__main__":
    asyncio.run(send_audio())
