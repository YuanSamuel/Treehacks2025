import socket
import json

# Server configuration: listen on all interfaces
LISTEN_IP = "127.0.0.1"
LISTEN_PORT = 9000

def odas_server():
    # Create a TCP/IP socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Allow the socket to reuse the address
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # Bind the socket to the IP and port
    server_socket.bind((LISTEN_IP, LISTEN_PORT))
    server_socket.listen(5)
    print(f"Server listening on {LISTEN_IP}:{LISTEN_PORT}")

    try:
        while True:
            # Wait for an incoming connection
            conn, addr = server_socket.accept()
            print(f"Accepted connection from {addr}")
            with conn:
                while True:
                    # Receive data in chunks of 4096 bytes
                    data = conn.recv(4096)
                    if not data:
                        # No more data from the client
                        break
                    try:
                        # Try to decode and parse the received data as JSON
                        json_data = json.loads(data.decode('utf-8'))
                        print("Received JSON data:", json_data)
                    except json.JSONDecodeError:
                        # If JSON parsing fails, print raw data
                        print("Received non-JSON data:", data)
            print("Connection closed.")
    except KeyboardInterrupt:
        print("Server interrupted by user.")
    except Exception as e:
        print("An error occurred:", e)
    finally:
        server_socket.close()
        print("Server socket closed.")

if __name__ == "__main__":
    odas_server()