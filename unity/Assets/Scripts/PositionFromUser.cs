using UnityEngine;

public class PositionRelativeToUser : MonoBehaviour
{
    [Header("Configuration")]
    // Assign your XR Camera (often your Main Camera)
    public Camera userCamera;
    // Distance from the camera where you want the object to appear
    public float distanceFromCamera = 2.5f;

    void Update()
    {
        if (userCamera != null)
        {
            // Position the object at a point in front of the camera
            Vector3 viewportCenter = new Vector3(0.5f, 0.5f, distanceFromCamera);
            Vector3 worldPosition = userCamera.ViewportToWorldPoint(viewportCenter);
            transform.position = worldPosition;

            // Make the object face away from the camera
            Vector3 directionAwayFromCamera = transform.position - userCamera.transform.position;
            transform.rotation = Quaternion.LookRotation(directionAwayFromCamera);

            // Debugging info
            Debug.Log($"Model Position: {transform.position}, Distance: {distanceFromCamera}");
        }
    }
}
