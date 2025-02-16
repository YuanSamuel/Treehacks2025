using UnityEngine;

public class WaveBehavior : MonoBehaviour
{
    public float expandSpeed = 1.5f;   // How quickly the ring expands outward
    public float fadeDuration = 1.5f;  // How long until it's fully transparent

    private Material _mat;
    private float _timer = 0f;
    private Vector3 _initialScale;

    void Start()
    {
        // Grab the material so we can change color/alpha
        _mat = GetComponent<MeshRenderer>().material;
        // Store the initial scale if needed
        _initialScale = transform.localScale;
    }

    void Update()
    {
        _timer += Time.deltaTime;

        // 1) Expand
        float scaleFactor = 1f + expandSpeed * _timer;
        transform.localScale = _initialScale * scaleFactor;

        // 2) Fade out
        float alpha = Mathf.Lerp(1f, 0f, _timer / fadeDuration);
        Color c = _mat.color;
        c.a = alpha;
        _mat.color = c;

        // 3) Destroy when fully faded
        if (_timer >= fadeDuration)
        {
            Destroy(gameObject);
        }
    }
}
