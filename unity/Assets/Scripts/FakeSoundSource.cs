using UnityEngine;

public class FakeSoundSource : MonoBehaviour
{
    [Header("Spawn Timing")]
    public bool spawnOnKeyPress = true;
    public KeyCode spawnKey = KeyCode.Space;

    public bool spawnPeriodically = false;
    public float spawnInterval = 2f;

    private float _timer;

    void Update()
    {
        // 1) Key press
        if (spawnOnKeyPress && Input.GetKeyDown(spawnKey))
        {
            SpawnWave();
        }

        // 2) Periodic spawn
        if (spawnPeriodically)
        {
            _timer += Time.deltaTime;
            if (_timer >= spawnInterval)
            {
                _timer = 0f;
                SpawnWave();
            }
        }
    }

    void SpawnWave()
    {
        // Just call our manager’s SpawnWave function
        WaveVisualizer.Instance.SpawnWave();
    }
}
