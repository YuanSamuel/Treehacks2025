#!/usr/bin/env python3
import socket
import time
import asyncio
import websockets
import argparse

def send_message(host: str = '127.0.0.1', port: int = 7000, message: str = "Hello from Jetson!"):
    """
    Connects to the specified host and port, sends a message,
    and prints any response from the server.
    """
    try:
        # Create a TCP/IP socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            print(f"Connecting to {host}:{port}...")
            sock.connect((host, port))
            print("Connected.")

            # Send the message
            print(f"[QuestTCP] Sending message: {message}")
            sock.sendall(message.encode('utf-8'))

            # Wait for a response (adjust buffer size as needed)
            response = sock.recv(1024)
            if response:
                print("Received response:", response.decode('utf-8'))
            else:
                print("No response received from the server.")

    except ConnectionRefusedError:
        print("Connection refused. Is the Unity server running on the Quest?")
    except socket.timeout:
        print("Connection timed out.")
    except Exception as e:
        print("An error occurred:", e)

def main():
    # Interactive loop for sending messages manually.
    while True:
        send_message()
        # Ask if you want to send another message
        user_input = input("Send another message? (y/n): ").strip().lower()
        if user_input != 'y':
            print("Exiting.")
            break
        # Optionally wait before sending another message
        time.sleep(1)

async def proxy_message(websocket, path):
    """
    Handler for incoming WebSocket messages.
    For each message received, call the send_message function to
    proxy the message to the TCP server.
    """
    async for message in websocket:
        print(f"Received WebSocket message: {message}")
        # Run send_message in a separate thread to avoid blocking the event loop.
        await asyncio.to_thread(send_message, message=message)
        # Optionally send a confirmation back to the WebSocket client.
        await websocket.send("TCP message sent successfully.")

async def run_websocket_server():
    """
    Starts the WebSocket server on localhost at port 8765.
    """
    async with websockets.serve(proxy_message, "localhost", 8765):
        print("WebSocket server started on ws://localhost:8765")
        await asyncio.Future()  # run forever

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="TCP message sender with WebSocket proxy.")
    parser.add_argument('--ws', action='store_true', help="Run as a WebSocket server.")
    args = parser.parse_args()

    if args.ws:
        # Run the WebSocket server if --ws is provided.
        asyncio.run(run_websocket_server())
    else:
        # Otherwise, run the interactive terminal mode.
        main()
