using RedRunner.TerrainGeneration;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PersonalitySimulator : MonoBehaviour
{
    public List<PersonalityScriptableObject> personalities;
    public List<Block> blocks;

    public int currentPersonality = 0;
    private List<int> sectionPassed = new List<int>();
    private float currentConcentration = 0;
    private Queue<float> concentrationQueue;
    private float currentSkill = 0;
    private Queue<float> skillQueue;
    private float currentChallenge = 0;
    private Queue<float> challengeQueue;
    private float currentImmersion = 0;
    private bool endGame = false;
    private bool firstSection = true;
    public int currentSection = -1;
    private bool waitingForNewSection = false;

    public static PersonalitySimulator Instance;

    public AdaptivityAI ai;

    private float maxWidth = 0;

    #region Observations
    [System.NonSerialized] public List<float> maxScorePersonalities;
    [System.NonSerialized] public float currentScore = 0;

    [System.NonSerialized] public int currentLives = 3;
    [System.NonSerialized] public List<int> lifeLostWaterPersonalities;
    [System.NonSerialized] public List<int> lifeLostMovingPersonalities;
    [System.NonSerialized] public List<int> lifeLostStaticPersonalities;
    [System.NonSerialized] public List<int> lifeLostPersonalities;
    [System.NonSerialized] public int lastSectionLifeLost = 0;
    [System.NonSerialized] public int lastSectionLifeLostWater = 0;
    [System.NonSerialized] public int lastSectionLifeLostStatic = 0;
    [System.NonSerialized] public int lastSectionLifeLostMoving = 0;

    [System.NonSerialized] public List<int> coinsPersonalities;
    [System.NonSerialized] public List<int> chestsPersonalities;
    [System.NonSerialized] public int coinsGame = 0;
    [System.NonSerialized] public int chestGame = 0;

    [System.NonSerialized] public float averageVelocityGames = 0;
    [System.NonSerialized] public float m2VelocityGames = 0;
    [System.NonSerialized] public int averageVelocityGames_i = 0;
    [System.NonSerialized] public List<float> averageVelocityPersonalities;
    [System.NonSerialized] public List<float> m2VelocityPersonalities;
    [System.NonSerialized] public List<int> averageVelocityPersonalities_i;

    [System.NonSerialized] public int jumpsGame = 0;

    [System.NonSerialized] public int backtracksGame = 0;

    [System.NonSerialized] public float timeSpent = 2;

    #endregion


    private void Awake()
    {
        if (PersonalitySimulator.Instance != null)
        {
            Destroy(PersonalitySimulator.Instance);
        }
        PersonalitySimulator.Instance = this;
    }

    public float rewardOfPersonality(int section, float width)
    {
        float sectionReward = 0;

        // Repeting section reward
        bool repetingSection = false;
        for (int i = sectionPassed.Count - 1; i > 0 && i > sectionPassed.Count - 1 - personalities[currentPersonality].toleranceForRepetingLevels; i--)
        {
            if (sectionPassed[i] == section)
            {
                repetingSection = true;
                break;
            }
        }
        sectionPassed.Add(section);
        if (repetingSection)
        {
            sectionReward -= personalities[currentPersonality].newLevelsImportance;
        }

        // Coins collected in section
        int randomCoinsSections = Random.Range(personalities[currentPersonality].coinsSection[section].minimum, personalities[currentPersonality].coinsSection[section].maximum + 1);
        sectionReward += personalities[currentPersonality].coinsImportance * randomCoinsSections;
        coinsGame += randomCoinsSections;
        coinsPersonalities[currentPersonality] += randomCoinsSections;

        // Chests collected in section
        int randomChestSections = Random.Range(personalities[currentPersonality].chestsSection[section].minimum, personalities[currentPersonality].chestsSection[section].maximum + 1);
        sectionReward += personalities[currentPersonality].chestsImportance * randomChestSections;
        chestGame += randomChestSections;
        chestsPersonalities[currentPersonality] += randomChestSections;

        // Lives lost in section
        int randomLiveLostSections = Random.Range(personalities[currentPersonality].lifeLostSection[section].minimum, personalities[currentPersonality].lifeLostSection[section].maximum + 1);
        int trueLost = randomLiveLostSections;
        if (currentLives - randomLiveLostSections <= 0)
        {
            trueLost = currentLives;
            sectionReward -= personalities[currentPersonality].lifeLostImportance * currentLives;
            currentLives = 0;

            float randomPlaceInSection = Random.Range(0, width);

            if (currentScore + randomPlaceInSection > this.maxScorePersonalities[currentPersonality])
            {
                sectionReward += personalities[currentPersonality].newScoreImportance;
                this.maxScorePersonalities[currentPersonality] = currentScore + randomPlaceInSection;
            }
            endGame = true;
        }
        else
        {
            currentLives -= randomLiveLostSections;
            sectionReward -= personalities[currentPersonality].lifeLostImportance * randomLiveLostSections;
        }
        lifeLostPersonalities[currentPersonality] += trueLost;
        lastSectionLifeLost = trueLost;
        for (int i = 0; i < trueLost; i++)
        {
            int typeLost = personalities[currentPersonality].lifeLostTypeSection[section].typeToLost();
            if (typeLost == 1)
            {
                lifeLostWaterPersonalities[currentPersonality]++;
                lastSectionLifeLostWater++;
            }
            if (typeLost == 2)
            {
                lifeLostStaticPersonalities[currentPersonality]++;
                lastSectionLifeLostStatic++;
            }
            if (typeLost == 3)
            {
                lifeLostMovingPersonalities[currentPersonality]++;
                lastSectionLifeLostMoving++;
            }
        }

        // Time
        float randomTimeSections = Random.Range((float)personalities[currentPersonality].timeSection[section].minimum, (float)personalities[currentPersonality].timeSection[section].maximum);
        timeSpent += 1 + randomTimeSections + randomTimeSections * trueLost * 0.6f;
        if (timeSpent > 300 && !endGame)
        {
            float randomPlaceInSection = Random.Range(0, width);

            if (currentScore + randomPlaceInSection > this.maxScorePersonalities[currentPersonality])
            {
                sectionReward += personalities[currentPersonality].newScoreImportance;
                this.maxScorePersonalities[currentPersonality] = currentScore + randomPlaceInSection;
            }
            endGame = true;
        }


        // Importance of having lives
        sectionReward += currentLives * personalities[currentPersonality].lifeImportance;

        // Average velocity in section
        float randomVelocitySections = Random.Range((float)personalities[currentPersonality].speedSection[section].minimum, (float)personalities[currentPersonality].speedSection[section].maximum);

        averageVelocityGames_i++;
        averageVelocityPersonalities_i[currentPersonality]++;
        var delta = randomVelocitySections - averageVelocityGames;
        averageVelocityGames += delta / averageVelocityGames_i;
        var deltaP = randomVelocitySections - averageVelocityPersonalities[currentPersonality];
        averageVelocityPersonalities[currentPersonality] += deltaP / averageVelocityPersonalities_i[currentPersonality];
        var delta2 = randomVelocitySections - averageVelocityGames;
        m2VelocityGames += delta * delta2;
        var delta2P = randomVelocitySections - averageVelocityPersonalities[currentPersonality];
        m2VelocityPersonalities[currentPersonality] += deltaP * delta2P;

        float percentageVelocity = averageVelocityGames / personalities[currentPersonality].speedPreference;
        float percentageVelocityPer = averageVelocityPersonalities[currentPersonality] / personalities[currentPersonality].speedPreference;
        if (!float.IsNaN(percentageVelocity) && !float.IsNaN(percentageVelocityPer))
        {
            if (percentageVelocity < 1) sectionReward -= ((1 - percentageVelocity) * personalities[currentPersonality].speedImportance) / 2;
            else sectionReward += (percentageVelocity * personalities[currentPersonality].speedImportance) / 2;
            if (percentageVelocityPer < 1) sectionReward -= ((1 - percentageVelocityPer) * personalities[currentPersonality].speedImportance) / 2;
            else sectionReward += (percentageVelocityPer * personalities[currentPersonality].speedImportance) / 2;
        }

        currentScore += width;

        // Jumps
        int randomJumpsSections = Random.Range(personalities[currentPersonality].jumpsSection[section].minimum, personalities[currentPersonality].jumpsSection[section].maximum + 1);
        jumpsGame += randomJumpsSections + Mathf.RoundToInt(randomJumpsSections * trueLost * 0.8f);

        // BackTrack
        int randomBacktrackSections = Random.Range(1, 101);
        if (randomBacktrackSections <= personalities[currentPersonality].backtrackProbability[section]) backtracksGame += 1 + trueLost;

        float average;

        // Concentration
        if (concentrationQueue.Count >= personalities[currentPersonality].toleranceForRepetingLevels)
        {
            currentConcentration -= concentrationQueue.Dequeue();
        }
        concentrationQueue.Enqueue(personalities[currentPersonality].concentration[section]);
        currentConcentration += personalities[currentPersonality].concentration[section];
        average = currentConcentration / concentrationQueue.Count;
        sectionReward += (1 / (Mathf.Abs(personalities[currentPersonality].concentrationLevelPrefered - average) + 1)) * personalities[currentPersonality].concentrationImportance;

        // Skill
        if (skillQueue.Count >= personalities[currentPersonality].toleranceForRepetingLevels)
        {
            currentSkill -= skillQueue.Dequeue();
        }
        skillQueue.Enqueue(personalities[currentPersonality].skill[section]);
        currentSkill += personalities[currentPersonality].skill[section];
        average = currentSkill / skillQueue.Count;
        sectionReward += (1 / (Mathf.Abs(personalities[currentPersonality].skillLevelPrefered - average) + 1)) * personalities[currentPersonality].skillImportance;

        // Challenge
        if (challengeQueue.Count >= personalities[currentPersonality].toleranceForRepetingLevels)
        {
            currentChallenge -= challengeQueue.Dequeue();
        }
        challengeQueue.Enqueue(personalities[currentPersonality].challenge[section]);
        currentChallenge += personalities[currentPersonality].challenge[section];
        average = currentChallenge / challengeQueue.Count;
        sectionReward += (1 / (Mathf.Abs(personalities[currentPersonality].challengeLevelPrefered - average) + 1)) * personalities[currentPersonality].challengeImportance;

        // Immersion
        /*
        currentImmersion += personalities[currentPersonality].immersion[section];
        sectionReward += (1 / (Mathf.Abs(personalities[currentPersonality].immersionLevelPrefered - currentImmersion) + 1)) * personalities[currentPersonality].immersionImportance;
        currentImmersion -= personalities[currentPersonality].lossEachSection;
        */
        sectionReward = (width / maxWidth) * sectionReward;
        return sectionReward;
    }

    private void Start()
    {
        for (int i = 0; i < blocks.Count; i++)
        {
            if (blocks[i].Width > maxWidth) maxWidth = blocks[i].Width;
        }
        this.maxScorePersonalities = new List<float>();
        lifeLostWaterPersonalities = new List<int>();
        lifeLostMovingPersonalities = new List<int>();
        lifeLostStaticPersonalities = new List<int>();
        lifeLostPersonalities = new List<int>();
        averageVelocityPersonalities = new List<float>();
        m2VelocityPersonalities = new List<float>();
        averageVelocityPersonalities_i = new List<int>();
        coinsPersonalities = new List<int>();
        chestsPersonalities = new List<int>();
        sectionPassed = new List<int>();
        concentrationQueue = new Queue<float>();
        skillQueue = new Queue<float>();
        challengeQueue = new Queue<float>();
        for (int i = 0; i < this.personalities.Count; i++)
        {
            this.maxScorePersonalities.Add(0);
            lifeLostWaterPersonalities.Add(0);
            lifeLostMovingPersonalities.Add(0);
            lifeLostStaticPersonalities.Add(0);
            lifeLostPersonalities.Add(0);
            averageVelocityPersonalities.Add(0);
            m2VelocityPersonalities.Add(0);
            averageVelocityPersonalities_i.Add(0);
            coinsPersonalities.Add(0);
            chestsPersonalities.Add(0);
        }
        resetGame();
    }

    public void nextSection(int section)
    {
        currentSection = section;
        Debug.Log(currentPersonality + " " + section);
        waitingForNewSection = false;
    }

    public void resetGame()
    {
        currentLives = 3;
        currentScore = 0;
        currentConcentration = 0;
        concentrationQueue = new Queue<float>();
        currentSkill = 0;
        skillQueue = new Queue<float>();
        currentChallenge = 0;
        challengeQueue = new Queue<float>();
        currentImmersion = 0;
        endGame = false;
        firstSection = true;
        waitingForNewSection = false;
        sectionPassed = new List<int>();

        coinsGame = 0;
        chestGame = 0;

        lastSectionLifeLost = 0;
        lastSectionLifeLostWater = 0;
        lastSectionLifeLostStatic = 0;
        lastSectionLifeLostMoving = 0;

        timeSpent = 2;
        averageVelocityGames_i = 0;
        averageVelocityGames = 0;
        m2VelocityGames = 0;

        jumpsGame = 0;
        backtracksGame = 0;

        currentPersonality = Random.Range(0, personalities.Count);
        int percentageResetMaxScore = Random.Range(0, 100);
        if (percentageResetMaxScore >= 80)
        {
            lifeLostWaterPersonalities[currentPersonality] = 0;
            lifeLostMovingPersonalities[currentPersonality] = 0;
            lifeLostStaticPersonalities[currentPersonality] = 0;
            lifeLostPersonalities[currentPersonality] = 0;
            maxScorePersonalities[currentPersonality] = 0;
            averageVelocityPersonalities[currentPersonality] = 0;
            m2VelocityPersonalities[currentPersonality] = 0;
            averageVelocityPersonalities_i[currentPersonality] = 0;
            coinsPersonalities[currentPersonality] = 0;
            chestsPersonalities[currentPersonality] = 0;
        }
    }

    private void Update()
    {
        if (!waitingForNewSection)
        {
            if (firstSection)
            {
                this.ai.RequestDecision();
                waitingForNewSection = true;
                firstSection = false;
            }
            else
            {
                float reward = this.rewardOfPersonality(currentSection, blocks[currentSection].Width);
                this.ai.AddReward(reward);
                if (endGame)
                {
                    this.ai.EndEpisode();
                }
                else
                {
                    this.ai.RequestDecision();
                    waitingForNewSection = true;
                }
            }
        }


    }
}
