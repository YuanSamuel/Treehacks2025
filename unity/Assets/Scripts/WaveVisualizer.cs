using UnityEngine;

public class WaveVisualizer : MonoBehaviour
{
    public static WaveVisualizer Instance;

    [Header("References")]
    public GameObject wavePrefab;       // Assign your ring prefab here
    public Transform headsetTransform;  // Assign your VR camera or main camera

    [Header("Spawn Settings")]
    public float defaultSpawnDistance = 2f;  // Distance in front of user
    public float lateralSpawnOffset = 1f;    // How far to the left/right

    void Awake()
    {
        // Simple Singleton so we can call WaveVisualizer.Instance.SpawnWave()
        if (Instance == null) Instance = this;
        else Destroy(gameObject);
    }

    /// <summary>
    /// Spawns the wave in front of the camera with optional random lateral offset.
    /// </summary>
    public void SpawnWave()
    {
        if (!wavePrefab || !headsetTransform) return;

        // Start directly in front of user
        Vector3 spawnPos = headsetTransform.position + (headsetTransform.forward * defaultSpawnDistance);

        // Random left/right offset
        float sideOffset = Random.Range(-lateralSpawnOffset, lateralSpawnOffset);
        spawnPos += headsetTransform.right * sideOffset;

        // Instantiate the prefab
        Instantiate(wavePrefab, spawnPos, Quaternion.identity);
    }
}
