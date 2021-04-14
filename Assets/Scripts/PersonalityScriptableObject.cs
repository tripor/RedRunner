using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[CreateAssetMenu(fileName = "PersonalityType", menuName = "PersonalityType")]
public class PersonalityScriptableObject : ScriptableObject
{
    [Tooltip("Secção Repetida (nº de vezes que tenho de ver secções diferentes")]
    [Min(0)]
    public float toleranceForRepetingLevels = 1;
    [Tooltip("Importancia de moedas")]
    [Min(0)]
    public float coinsImportance = 1;
    [Tooltip("Importancia dos chests")]
    [Min(0)]
    public float chestsImportance = 1;
    [Tooltip("Importancia de ter vidas")]
    [Min(0)]
    public float lifeImportance = 1;
    [Tooltip("Importancia de perder vidas")]
    [Min(0)]
    public float lifeLostImportance = 1;
    [Tooltip("Importancia da velocidade/tempo numa secção")]
    [Min(0)]
    public float speedImportance = 1;
    [Tooltip("Importancia do score")]
    [Min(0)]
    public float scoreImportance = 1;
    [Tooltip("Importancia de ter um novo score")]
    [Min(0)]
    public float newScoreImportance = 1;
    [Tooltip("Disconto de moedas totais")]
    [Min(0)]
    public float totalCoinsDiscount = 1;
    [Space]

    [Tooltip("Prefered concentration")]
    public float concentrationLevelPrefered = 1;
    public List<float> concentration = new List<float>() { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
    [Tooltip("Prefered skill")]
    public float skillLevelPrefered = 1;
    public List<float> skill = new List<float>() { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
    [Tooltip("Prefered challenge")]
    public float challengeLevelPrefered = 1;
    public List<float> challenge = new List<float>() { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
    [Tooltip("Prefered immersion")]
    public float immersionLevelPrefered = 1;
    public List<float> immersion = new List<float>() { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
}
