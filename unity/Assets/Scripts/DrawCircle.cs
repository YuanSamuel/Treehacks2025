using UnityEngine;

[RequireComponent(typeof(LineRenderer))]
public class DrawCircle : MonoBehaviour {
    public int segments = 100;
    public float radius = 1f;
    public float lineWidth = 0.02f;
    
    void Start() {
        LineRenderer line = GetComponent<LineRenderer>();
        line.useWorldSpace = false;
        line.loop = true;
        line.startWidth = lineWidth;
        line.endWidth = lineWidth;
        line.positionCount = segments;
        
        for (int i = 0; i < segments; i++) {
            float angle = 2 * Mathf.PI * i / segments;
            float x = Mathf.Cos(angle) * radius;
            float z = Mathf.Sin(angle) * radius;
            line.SetPosition(i, new Vector3(x, 0, z));
        }
    }
}
