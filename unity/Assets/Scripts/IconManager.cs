using System.Collections.Generic;
using UnityEngine;

public class IconManager : MonoBehaviour
{
    // The singleton instance.
    public static IconManager Instance { get; private set; }

    // Assign these prefab references in the Inspector.
    public GameObject speechIconPrefab;
    public GameObject clapIconPrefab;
    public GameObject sirenIconPrefab;
    public GameObject unknownIconPrefab;

    // This dictionary maps string keys to icon prefabs.
    private Dictionary<string, GameObject> iconMapping = new Dictionary<string, GameObject>();

    void Awake()
    {
        // Implement the singleton pattern.
        if (Instance != null && Instance != this)
        {
            Destroy(gameObject);
            return;
        }
        Instance = this;
        // Optionally, persist this object across scenes.
        DontDestroyOnLoad(gameObject);


        // Initialize the mapping.
        iconMapping["speech"] = speechIconPrefab;
        iconMapping["clap"] = clapIconPrefab;
        iconMapping["siren"] = sirenIconPrefab;
        iconMapping["unknown"] = unknownIconPrefab;

        // Now instantiate each icon.
        // You could also just store these instantiated icons somewhere if needed.
        foreach (KeyValuePair<string, GameObject> pair in iconMapping)
        {
            // Instantiate the icon as a child of iconsParent if assigned; otherwise, in world space.
            GameObject iconInstance = Instantiate(pair.Value);
            // Optionally set the name or any data on the instantiated icon.
            iconInstance.name = pair.Key + "Icon";
        }
    }

    /// <summary>
    /// Public method to get an icon prefab by its key.
    /// </summary>
    public GameObject GetIconPrefab(string key)
    {
        return iconMapping.GetValueOrDefault(key, unknownIconPrefab);
    }

}
