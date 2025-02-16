using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;
using TMPro;

public class CaptionManager : MonoBehaviour
{
    [Header("UI References")]
    public TextMeshProUGUI captionText;     // Reference to the TextMeshPro caption text
    public RectTransform backgroundRect;    // Reference to the background RectTransform

    public float backgroundPadding = 20f;     // Extra space around text

    // Thread-safe queue for messages coming from the network thread
    private ConcurrentQueue<string> messageQueue = new ConcurrentQueue<string>();

    // TCP listener fields
    private TcpListener tcpListener;
    private TcpClient connectedClient;
    private NetworkStream networkStream;
    private Thread listenThread;
    private bool isRunning = false;

    // Accumulating buffer for incomplete messages
    private StringBuilder accumulatedMessage = new StringBuilder();

    // List to hold objects of type "1" messages
    private List<SoundEvent> soundEvents = new List<SoundEvent>();

    private void Start()
    {
        Vector2 preferredValues = captionText.GetPreferredValues(captionText.text);
        if (backgroundRect != null)
        {
            backgroundRect.sizeDelta = new Vector2(preferredValues.x + backgroundPadding, backgroundRect.sizeDelta.y);
        }

        // Start the TCP listener on port 7000
        isRunning = true;
        try
        {
            tcpListener = new TcpListener(IPAddress.Any, 7000);
            tcpListener.Start();
            listenThread = new Thread(ListenForClient);
            listenThread.IsBackground = true;
            listenThread.Start();
        }
        catch (Exception ex)
        {
            Debug.LogError("Error starting TCP listener: " + ex.Message);
            return;
        }
    }

    private void Update()
    {
        // Process any messages received from the network thread
        while (messageQueue.TryDequeue(out string newCaption))
        {
            ProcessMessage(newCaption);
        }
    }

    /// <summary>
    /// Listens for incoming TCP client connection (only one client is expected).
    /// </summary>
    private void ListenForClient()
    {
        try
        {
            // Accept a single incoming client connection
            connectedClient = tcpListener.AcceptTcpClient();
            networkStream = connectedClient.GetStream();
            Debug.Log("Client connected.");

            // Begin asynchronous read from the network stream
            ReadDataAsync();
        }
        catch (Exception ex)
        {
            Debug.LogError("Error in ListenForClient: " + ex.Message);
        }
    }

    /// <summary>
    /// Asynchronously reads data from the connected client and accumulates it into a complete message.
    /// </summary>
    private async void ReadDataAsync()
    {
        byte[] buffer = new byte[1024];

        try
        {
            while (isRunning && connectedClient.Connected)
            {
                int bytesRead = await networkStream.ReadAsync(buffer, 0, buffer.Length);

                Debug.Log("Reading from TCP, read " + bytesRead + " bytes");

                if (bytesRead > 0)
                {
                    string chunk = Encoding.UTF8.GetString(buffer, 0, bytesRead);
                    accumulatedMessage.Append(chunk);

                    // Process complete messages if any
                    ProcessMessages();
                }
            }
        }
        catch (Exception ex)
        {
            Debug.LogError("Error reading data from client: " + ex.Message);
        }
        finally
        {
            // Clean up when client disconnects
            connectedClient.Close();
            networkStream.Close();
            Debug.Log("Client disconnected.");
        }
    }

    /// <summary>
    /// Processes the accumulated message and splits it into complete messages.
    /// </summary>
    private void ProcessMessages()
    {
        string[] messages = accumulatedMessage.ToString().Split(new[] { "\n" }, StringSplitOptions.RemoveEmptyEntries);

        foreach (var message in messages)
        {
            Debug.Log("Got full message " + message);
            // Enqueue the complete message for processing on the main thread
            messageQueue.Enqueue(message);
        }

        // Clear the accumulated message buffer, leaving unprocessed leftover data for the next read
        accumulatedMessage.Clear();
    }

    /// <summary>
    /// Process the message and update accordingly
    /// </summary>
    private void ProcessMessage(string message)
    {
        string[] parts = message.Split('|');
        
        if (parts.Length < 2)
        {
            Debug.LogWarning("Invalid message format: " + message);
            return;
        }

        switch (parts[0])
        {
            case "0":
                // Update the caption with the string following the delimiter
                string caption = parts.Length > 1 ? parts[1] : "";
                UpdateCaption(caption);
                break;

            case "1":
                if (parts.Length < 4)
                {
                    Debug.LogWarning("Invalid message format for type '1': " + message);
                    return;
                }

                // Parse the data and create an object of MyObject
                int angle = int.Parse(parts[1]);
                float volume = float.Parse(parts[2]);
                string classType = parts[3];

                Radar.Instance.AddRadarObject(angle, volume, classType);
                break;

            default:
                Debug.LogWarning("Unknown message type: " + message);
                break;
        }
    }

    /// <summary>
    /// Updates the caption text and adjusts the background size.
    /// </summary>
    /// <param name="newText">The new caption text.</param>
    private void UpdateCaption(string newText)
    {
        if (captionText == null)
        {
            Debug.LogError("Caption Text is not assigned!");
            return;
        }

        // Update the text
        captionText.text = newText;

        // Adjust the background size based on the text's preferred width
        Vector2 preferredValues = captionText.GetPreferredValues(newText);
        if (backgroundRect != null)
        {
            backgroundRect.sizeDelta = new Vector2(preferredValues.x + backgroundPadding, backgroundRect.sizeDelta.y);
        }
    }

    private void OnDestroy()
    {
        isRunning = false;
        if (listenThread != null && listenThread.IsAlive)
        {
            listenThread.Abort();
        }
        if (tcpListener != null)
        {
            tcpListener.Stop();
        }
    }
}

// Class to represent data from a "1" message type
public class SoundEvent
{
    public int Angle { get; private set; }
    public float Volume { get; private set; }
    public string ClassType { get; private set; }

    public SoundEvent(int angle, float volume, string classType)
    {
        Angle = angle;
        Volume = volume;
        ClassType = classType;
    }
}
