using UnityEngine;

public class AutoTestRadar : MonoBehaviour
{
    private readonly string[] iconTypes = new string[] { "speech", "clap", "siren", "unknown" };

    void Start()
    {
        // Start calling AddRandomRadarIcon every 1 second, after an initial 1 second delay.
        InvokeRepeating(nameof(AddRandomRadarIcon), 1f, 1f);
    }

    void AddRandomRadarIcon()
    {
        // Generate a random angle between 0 and 360.
        int randomAngle = Random.Range(0, 361);
        // Generate a random relative distance between 0 and 3.
        float randomDistance = Random.Range(0f, 3f);
        // Specify the icon type. Ensure this matches one defined in your IconManager.
        string iconType = iconTypes[Random.Range(0, iconTypes.Length)];

        // Call the AddRadarObject method from the Radar singleton instance.
        if (Radar.Instance != null)
        {
            Radar.Instance.AddRadarObject(randomAngle, randomDistance, iconType);
            Debug.Log($"Added radar object: Angle = {randomAngle}, Distance = {randomDistance}, Icon Type = {iconType}");
        }
        else
        {
            Debug.LogWarning("Radar instance not found!");
        }
    }
}
