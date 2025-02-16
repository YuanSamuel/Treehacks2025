#!/usr/bin/env python3
import socket
import time

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
    # Optionally, you can loop to allow sending multiple messages.
    while True:
        send_message()
        # Ask if you want to send another message
        user_input = input("Send another message? (y/n): ").strip().lower()
        if user_input != 'y':
            print("Exiting.")
            break
        # Optionally wait before sending another message
        time.sleep(1)

if __name__ == '__main__':
    main()