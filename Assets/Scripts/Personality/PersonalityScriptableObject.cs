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
[Serializable]
public class RandomNumberFloat
{
    public RandomNumberFloat(float minumum, float maximum)
    {
        this.minimum = minumum;
        this.maximum = maximum;
    }
    public float minimum;
    public float maximum;
}

[Serializable]
public class LifeLostType
{
    public LifeLostType(int water, int static_enemy, int moving_enemy)
    {
        this.water = water;
        this.static_enemy = static_enemy;
        this.moving_enemy = moving_enemy;
    }
    public int water;
    public int static_enemy;
    public int moving_enemy;

    public int typeToLost()
    {
        List<int> choose = new List<int>();
        for (int i = 0; i < water; i++)
        {
            choose.Add(1);
        }
        for (int i = 0; i < static_enemy; i++)
        {
            choose.Add(2);
        }
        for (int i = 0; i < moving_enemy; i++)
        {
            choose.Add(3);
        }
        int randomNumber = UnityEngine.Random.Range(0, water + static_enemy + moving_enemy);
        return choose[randomNumber];
    }
}

[CreateAssetMenu(fileName = "PersonalityType", menuName = "PersonalityType")]
public class PersonalityScriptableObject : ScriptableObject
{

    [Tooltip("Secção Repetida (nº de vezes que tenho de ver secções diferentes")]
    [Min(0)]
    public float toleranceForRepetingLevels = 3;
    [Tooltip("Importancia de ver secções novas")]
    [Min(0)]
    public float newLevelsImportance = 2;
    [Tooltip("Importancia de cada moeda")]
    [Min(0)]
    public float coinsImportance = 3;
    public List<RandomNumber> coinsSection = new List<RandomNumber>() { new RandomNumber(1, 1), new RandomNumber(2, 3), new RandomNumber(4, 4), new RandomNumber(1, 1), new RandomNumber(4, 6), new RandomNumber(3, 3), new RandomNumber(6, 13), new RandomNumber(5, 5), new RandomNumber(9, 9), new RandomNumber(8, 14), new RandomNumber(5, 5), new RandomNumber(6, 9), new RandomNumber(4, 4), new RandomNumber(5, 5), new RandomNumber(0, 6), new RandomNumber(0, 0) };
    [Tooltip("Importancia de cada chest")]
    [Min(0)]
    public float chestsImportance = 3;
    public List<RandomNumber> chestsSection = new List<RandomNumber>() { new RandomNumber(0, 0), new RandomNumber(0, 0), new RandomNumber(0, 0), new RandomNumber(0, 0), new RandomNumber(0, 0), new RandomNumber(0, 0), new RandomNumber(0, 1), new RandomNumber(0, 0), new RandomNumber(0, 0), new RandomNumber(0, 1), new RandomNumber(0, 0), new RandomNumber(0, 0), new RandomNumber(0, 0), new RandomNumber(0, 0), new RandomNumber(0, 1), new RandomNumber(0, 0) };
    [Tooltip("Importancia de ter vidas")]
    [Min(0)]
    public float lifeImportance = 1;
    [Tooltip("Importancia de perder vidas")]
    [Min(0)]
    public float lifeLostImportance = 3;
    public List<RandomNumber> lifeLostSection = new List<RandomNumber>() { new RandomNumber(0, 3), new RandomNumber(0, 3), new RandomNumber(0, 3), new RandomNumber(0, 3), new RandomNumber(0, 3), new RandomNumber(0, 3), new RandomNumber(0, 3), new RandomNumber(0, 3), new RandomNumber(0, 3), new RandomNumber(0, 3), new RandomNumber(0, 3), new RandomNumber(0, 3), new RandomNumber(0, 3), new RandomNumber(0, 3), new RandomNumber(0, 3), new RandomNumber(0, 3) };
    [Tooltip("Water, Static, Moving")]
    public List<LifeLostType> lifeLostTypeSection = new List<LifeLostType>() { new LifeLostType(0, 1, 0), new LifeLostType(0, 0, 1), new LifeLostType(0, 1, 0), new LifeLostType(0, 0, 0), new LifeLostType(1, 1, 0), new LifeLostType(1, 1, 0), new LifeLostType(1, 0, 1), new LifeLostType(0, 1, 0), new LifeLostType(1, 1, 0), new LifeLostType(1, 1, 0), new LifeLostType(1, 0, 1), new LifeLostType(1, 0, 1), new LifeLostType(1, 1, 1), new LifeLostType(1, 0, 1), new LifeLostType(1, 0, 1), new LifeLostType(0, 0, 1) };
    [Tooltip("Tempo gasto em cada secção")]
    public List<RandomNumberFloat> timeSection = new List<RandomNumberFloat>() { new RandomNumberFloat(2.0f, 5.0f), new RandomNumberFloat(7.0f, 16.0f), new RandomNumberFloat(6.0f, 11.0f), new RandomNumberFloat(4.6f, 7.0f), new RandomNumberFloat(13.0f, 28.0f), new RandomNumberFloat(13.5f, 20.0f), new RandomNumberFloat(29.0f, 60.0f), new RandomNumberFloat(21.5f, 25.0f), new RandomNumberFloat(38.5f, 46.0f), new RandomNumberFloat(38.5f, 47.0f), new RandomNumberFloat(24.0f, 32.0f), new RandomNumberFloat(41.5f, 62.0f), new RandomNumberFloat(27.0f, 34.0f), new RandomNumberFloat(33.0f, 38.0f), new RandomNumberFloat(18.5f, 35.0f), new RandomNumberFloat(12.0f, 25.0f) };
    [Tooltip("Importancia da velocidade/tempo numa secção")]
    [Min(0)]
    public float speedImportance = 1;
    public float speedPreference = 7.0f;
    public List<RandomNumberFloat> speedSection = new List<RandomNumberFloat>() { new RandomNumberFloat(4.0f, 8.5f), new RandomNumberFloat(4.0f, 8.8f), new RandomNumberFloat(4.0f, 5.8f), new RandomNumberFloat(7.5f, 8.5f), new RandomNumberFloat(3.8f, 8.7f), new RandomNumberFloat(6.0f, 7.9f), new RandomNumberFloat(3.0f, 6.4f), new RandomNumberFloat(7.5f, 8.9f), new RandomNumberFloat(5.5f, 7.3f), new RandomNumberFloat(6.5f, 8.1f), new RandomNumberFloat(5.0f, 7.3f), new RandomNumberFloat(4.2f, 6.3f), new RandomNumberFloat(6.5f, 8.4f), new RandomNumberFloat(8.0f, 8.9f), new RandomNumberFloat(4.2f, 8.2f), new RandomNumberFloat(3.8f, 8.9f) };
    [Tooltip("Saltos numa secção")]
    public List<RandomNumber> jumpsSection = new List<RandomNumber>() { new RandomNumber(1, 3), new RandomNumber(1, 5), new RandomNumber(4, 6), new RandomNumber(1, 2), new RandomNumber(6, 9), new RandomNumber(7, 9), new RandomNumber(12, 22), new RandomNumber(5, 6), new RandomNumber(18, 23), new RandomNumber(13, 16), new RandomNumber(10, 14), new RandomNumber(17, 25), new RandomNumber(8, 12), new RandomNumber(4, 5), new RandomNumber(5, 8), new RandomNumber(0, 1) };
    [Tooltip("Probabilidade de fazer backtrack entre 0% e 100%")]
    public List<float> backtrackProbability = new List<float>() { 0, 100, 0, 0, 100, 0, 100, 0, 0, 100, 0, 100, 0, 0, 100, 0 };
    [Tooltip("Importancia de ter um novo score")]
    [Min(0)]
    public float newScoreImportance = 3;
    [Space]

    public float lossEachSection = 1;
    [Tooltip("Prefered concentration")]
    public float concentrationLevelPrefered = 0;
    public float concentrationImportance = 1;
    public List<float> concentration = new List<float>() { 0, -1, 2, -3, 1, 2, 4, 1, 5, 2, 6, 6, 3, -2, -1, 2 };
    [Tooltip("Prefered skill")]
    public float skillLevelPrefered = 0;
    public float skillImportance = 1;
    public List<float> skill = new List<float>() { 1, 0, 3, -3, 2, 3, 3, 1, 4, 2, 6, 5, 4, -1, 0, 1 };
    [Tooltip("Prefered challenge")]
    public float challengeLevelPrefered = 0;
    public float challengeImportance = 1;
    public List<float> challenge = new List<float>() { 1, -2, 2, -4, 1, 2, 3, 0, 5, 2, 6, 6, 4, -2, -2, 0 };
    [Tooltip("Prefered immersion")]
    public float immersionLevelPrefered = 0;
    public float immersionImportance = 1;
    public List<float> immersion = new List<float>() { 2, -2, 2, 0, 2, 3, 1, -2, 4, 0, 6, 3, 0, -1, -2, 0 };
}
