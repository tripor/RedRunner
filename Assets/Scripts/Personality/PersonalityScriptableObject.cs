using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[Serializable]
public class RandomNumber
{
    public RandomNumber(int minumum, int maximum)
    {
        this.minimum = minumum;
        this.maximum = maximum;
    }
    public int minimum;
    public int maximum;
}

[CreateAssetMenu(fileName = "PersonalityType", menuName = "PersonalityType")]
public class PersonalityScriptableObject : ScriptableObject
{

    [Tooltip("Sec��o Repetida (n� de vezes que tenho de ver sec��es diferentes")]
    [Min(0)]
    public float toleranceForRepetingLevels = 1;
    [Tooltip("Importancia de ver sec��es novas")]
    [Min(0)]
    public float newLevelsImportance = 1;
    [Tooltip("Importancia de cada moeda")]
    [Min(0)]
    public float coinsImportance = 1;
    public List<RandomNumber> coinsSection = new List<RandomNumber>() { new RandomNumber(1, 1), new RandomNumber(1, 1), new RandomNumber(1, 1), new RandomNumber(1, 1), new RandomNumber(1, 1), new RandomNumber(1, 1), new RandomNumber(1, 1), new RandomNumber(1, 1), new RandomNumber(1, 1), new RandomNumber(1, 1), new RandomNumber(1, 1), new RandomNumber(1, 1), new RandomNumber(1, 1), new RandomNumber(1, 1), new RandomNumber(1, 1), new RandomNumber(1, 1) };
    [Tooltip("Importancia de cada chest")]
    [Min(0)]
    public float chestsImportance = 1;
    public List<RandomNumber> chestsSection = new List<RandomNumber>() { new RandomNumber(0, 0), new RandomNumber(0, 0), new RandomNumber(0, 0), new RandomNumber(0, 0), new RandomNumber(0, 0), new RandomNumber(0, 0), new RandomNumber(0, 0), new RandomNumber(0, 0), new RandomNumber(0, 0), new RandomNumber(0, 0), new RandomNumber(0, 0), new RandomNumber(0, 0), new RandomNumber(0, 0), new RandomNumber(0, 0), new RandomNumber(0, 0), new RandomNumber(0, 0) };
    [Tooltip("Importancia de ter vidas")]
    [Min(0)]
    public float lifeImportance = 1;
    [Tooltip("Importancia de perder vidas")]
    [Min(0)]
    public float lifeLostImportance = 1;
    public List<RandomNumber> lifeLostSection = new List<RandomNumber>() { new RandomNumber(0, 0), new RandomNumber(0, 0), new RandomNumber(0, 0), new RandomNumber(0, 0), new RandomNumber(0, 0), new RandomNumber(0, 0), new RandomNumber(0, 0), new RandomNumber(0, 0), new RandomNumber(0, 0), new RandomNumber(0, 0), new RandomNumber(0, 0), new RandomNumber(0, 0), new RandomNumber(0, 0), new RandomNumber(0, 0), new RandomNumber(0, 0), new RandomNumber(0, 0) };
    [Tooltip("Importancia da velocidade/tempo numa sec��o")]
    [Min(0)]
    public float speedImportance = 1;
    public List<RandomNumber> speedSection = new List<RandomNumber>() { new RandomNumber(0, 0), new RandomNumber(0, 0), new RandomNumber(0, 0), new RandomNumber(0, 0), new RandomNumber(0, 0), new RandomNumber(0, 0), new RandomNumber(0, 0), new RandomNumber(0, 0), new RandomNumber(0, 0), new RandomNumber(0, 0), new RandomNumber(0, 0), new RandomNumber(0, 0), new RandomNumber(0, 0), new RandomNumber(0, 0), new RandomNumber(0, 0), new RandomNumber(0, 0) };
    [Tooltip("Importancia de ter um novo score")]
    [Min(0)]
    public float newScoreImportance = 1;
    [Space]

    [Tooltip("Prefered concentration")]
    public float concentrationLevelPrefered = 1;
    public float concentrationImportance = 1;
    public List<float> concentration = new List<float>() { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
    [Tooltip("Prefered skill")]
    public float skillLevelPrefered = 1;
    public float skillImportance = 1;
    public List<float> skill = new List<float>() { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
    [Tooltip("Prefered challenge")]
    public float challengeLevelPrefered = 1;
    public float challengeImportance = 1;
    public List<float> challenge = new List<float>() { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
    [Tooltip("Prefered immersion")]
    public float immersionLevelPrefered = 1;
    public float immersionImportance = 1;
    public List<float> immersion = new List<float>() { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
}
