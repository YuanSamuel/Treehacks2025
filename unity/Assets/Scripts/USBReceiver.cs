using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;
using UnityEngine.UI; // Needed for UI elements
using TMPro;

public class USBReceiver : MonoBehaviour
{
    public int port = 7000; // Port must match the adb forward configuration

    private TcpListener tcpListener;
    private Thread listenerThread;

    // This is a thread-safe variable if you want to update the UI later.
    public string receivedMessage = "";

    // Reference to the UI Text component that will display the message.
    public TMP_Text displayText;

    void Start()
    {
        // Start the listener thread.
        listenerThread = new Thread(ListenForConnections);
        listenerThread.IsBackground = true;
        listenerThread.Start();
    }

    void ListenForConnections()
    {
        try
        {
            // Listen on any IP address for the specified port.
            tcpListener = new TcpListener(IPAddress.Any, port);
            tcpListener.Start();
            Debug.Log("TCP Server is listening on port " + port);

            while (true)
            {
                // Accept incoming connection. This call blocks until a connection is received.
                TcpClient client = tcpListener.AcceptTcpClient();
                Debug.Log("Client connected.");
                // Handle the client connection in a separate thread
                Thread clientThread = new Thread(() => HandleClient(client));
                clientThread.IsBackground = true;
                clientThread.Start();
            }
        }
        catch (SocketException ex)
        {
            Debug.Log("SocketException: " + ex);
        }
        catch (Exception ex)
        {
            Debug.Log("Exception: " + ex);
        }
    }

    void HandleClient(TcpClient client)
    {
        try
        {
            NetworkStream stream = client.GetStream();
            byte[] buffer = new byte[1024];

            while (client.Connected)
            {
                // Read data from the stream
                int length = stream.Read(buffer, 0, buffer.Length);
                if (length == 0)
                {
                    // Connection closed
                    break;
                }

                // Convert bytes received into a string
                string data = Encoding.ASCII.GetString(buffer, 0, length);
                Debug.Log("Received: " + data);
                
                // Optionally, update a variable accessible from the main thread
                receivedMessage = data;
            }
        }
        catch (Exception ex)
        {
            Debug.Log("Client handling exception: " + ex);
        }
        finally
        {
            client.Close();
        }
    }

    void Update()
    {
        // Update the UI Text element each frame
        if (displayText != null)
        {
            displayText.text = receivedMessage;
        }
    }

    void OnApplicationQuit()
    {
        // Clean up: stop the listener and abort the thread.
        if (tcpListener != null)
        {
            tcpListener.Stop();
        }
        if (listenerThread != null && listenerThread.IsAlive)
        {
            listenerThread.Abort();
        }
    }
}
