using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class Radar : MonoBehaviour
{   
    public static Radar Instance { get; private set; }

    [Header("References")]
    // This should be your camera's head or the UI center.
    public Transform player;

    [Header("Radar Settings")]
    // The UI radar’s radius in pixels (or units) — icon positions are scaled so that a relative distance of 3 lands at the edge.
    public float radarRadius = 100f;

    // Internal class to track each dynamic icon’s data.
    private class IconData
    {
        public GameObject icon;
        // Angle in degrees (0–360) where 0 is to the right (on the UI) and increases counterclockwise.
        public float angle;
        // Relative distance from the center (0–3). A value of 3 means the icon is at the outer edge of the radar.
        public float relativeDistance;
    }

    // List to hold all currently active radar icons.
    private List<IconData> iconDataList = new List<IconData>();

    void Awake()
    {
        // Ensure there's only one instance of Radar.
        if (Instance == null)
            Instance = this;
        else
            Destroy(gameObject);
    }

    void Update()
    {
        // Update the anchored positions of all active icons so they rotate with the player’s view.
        // (If you want the radar to be fixed instead, you could remove the subtraction of player.eulerAngles.y.)
        foreach (var data in iconDataList.ToArray())
        {
            UpdateIconPosition(data);
        }
    }

    /// <summary>
    /// Call this method to add a new radar icon.
    /// </summary>
    /// <param name="angle">An integer angle (0–360) on the circle.</param>
    /// <param name="relativeDistance">A float from 0–3 representing how far out the icon should appear.</param>
    /// <param name="iconType">A string key for IconManager to get the desired prefab.</param>
    public void AddRadarObject(int angle, float relativeDistance, string iconType)
    {
        // Instantiate the icon prefab (assumed to be a UI element) as a child of this radar object.
        GameObject newIcon = Instantiate(IconManager.Instance.GetIconPrefab(iconType), transform);

        // Create and store this icon’s data.
        IconData data = new IconData
        {
            icon = newIcon,
            angle = angle,
            relativeDistance = relativeDistance
        };
        iconDataList.Add(data);

        // Set the icon’s initial position.
        UpdateIconPosition(data);

        // Start fading out the icon over 4 seconds.
        StartCoroutine(FadeOutIcon(data, 4f));
    }

    /// <summary>
    /// Calculates and sets the UI anchored position for an icon.
    /// </summary>
    /// <param name="data">The icon’s data.</param>
    private void UpdateIconPosition(IconData data)
    {
        if (data.icon == null)
            return;

        RectTransform rt = data.icon.GetComponent<RectTransform>();

        // Convert the relative distance (0–3) into a distance in UI units.
        float adjustedDistance = (data.relativeDistance / 3f) * radarRadius;

        // Convert the given angle to radians.
        // We subtract the player's current y-angle so that the radar rotates with the player.
        float angleRad = data.angle * Mathf.Deg2Rad;

        // Compute the position on the circle.
        Vector2 anchoredPos = new Vector2(
            Mathf.Cos(angleRad) * adjustedDistance,
            Mathf.Sin(angleRad) * adjustedDistance);

        rt.anchoredPosition = anchoredPos;

        // --- Billboard Effect ---
        // Calculate a direction vector from the icon to the player.
        Vector3 direction = player.position - data.icon.transform.position;
        direction.y = 0; // keep the icon upright

        // Only update if there's a valid direction (avoids zero-length errors)
        if (direction != Vector3.zero)
        {
            // Set the icon's rotation so its forward side faces the camera.
            // You might need to flip the direction (using -direction) if your sprite appears backwards.
            data.icon.transform.rotation = Quaternion.LookRotation(-direction, Vector3.up);
        }

    }

    /// <summary>
    /// Gradually fades out an icon’s image over a given duration.
    /// Once complete, the icon is removed from the radar.
    /// </summary>
    /// <param name="data">The icon’s data.</param>
    /// <param name="duration">Fade duration in seconds.</param>
    private IEnumerator FadeOutIcon(IconData data, float duration)
    {
        Image img = data.icon.GetComponent<Image>();
        if (img == null)
            yield break;

        Color initialColor = img.color;
        float elapsedTime = 0f;

        while (elapsedTime < duration)
        {
            elapsedTime += Time.deltaTime;
            float alpha = Mathf.Lerp(initialColor.a, 0f, elapsedTime / duration);
            img.color = new Color(initialColor.r, initialColor.g, initialColor.b, alpha);
            yield return null;
        }

        // After fading, remove the icon and clean up.
        iconDataList.Remove(data);
        Destroy(data.icon);
    }
}
