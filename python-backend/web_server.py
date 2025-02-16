import socket
import time

HOST = ''  # Listen on all network interfaces
PORT = 12345  # Arbitrary non-privileged port

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    print(f"Server listening on port {PORT}")
    
    while True:
        conn, addr = s.accept()
        with conn:
            print(f"Connected by {addr}")
            try:
                while True:
                    conn.sendall(b"HELLO WORLD\n")
                    time.sleep(1)  # Send every 1 second
            except (ConnectionResetError, BrokenPipeError):
                print("Client disconnected")
            finally:
                conn.close()