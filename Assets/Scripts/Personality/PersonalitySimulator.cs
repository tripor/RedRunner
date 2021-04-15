using RedRunner.TerrainGeneration;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PersonalitySimulator : MonoBehaviour
{
    public List<PersonalityScriptableObject> personalities;
    public List<Block> blocks;

    private int currentPersonality = 0;
    private List<int> sectionPassed = new List<int>();
    private List<float> maxScorePersonalities;
    private int currentLives = 3;
    private float currentScore = 0;
    private float currentConcentration = 0;
    private float currentSkill = 0;
    private float currentChallenge = 0;
    private float currentImmersion = 0;
    private bool endGame = false;
    private bool firstSection = true;
    private int currentSection = 3;
    private bool waitingForNewSection = false;

    public static PersonalitySimulator Instance;

    public AdaptivityAI ai;

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
        if (!repetingSection)
        {
            sectionReward += personalities[currentPersonality].newLevelsImportance;
        }

        // Coins collected in section
        int randomCoinsSections = Random.Range(personalities[currentPersonality].coinsSection[section].minimum, personalities[currentPersonality].coinsSection[section].maximum);
        sectionReward += personalities[currentPersonality].coinsImportance * randomCoinsSections;

        // Chests collected in section
        int randomChestSections = Random.Range(personalities[currentPersonality].chestsSection[section].minimum, personalities[currentPersonality].chestsSection[section].maximum);
        sectionReward += personalities[currentPersonality].chestsImportance * randomChestSections;

        // Lives lost in section
        int randomLiveLostSections = Random.Range(personalities[currentPersonality].lifeLostSection[section].minimum, personalities[currentPersonality].lifeLostSection[section].maximum);
        if (currentLives - randomLiveLostSections <= 0)
        {
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


        // Importance of having lives
        sectionReward += currentLives * personalities[currentPersonality].lifeImportance;

        // Average velocity in section
        int randomVelocitySections = Random.Range(personalities[currentPersonality].speedSection[section].minimum, personalities[currentPersonality].speedSection[section].maximum);

        float percentageVelocity = randomVelocitySections / ((personalities[currentPersonality].speedSection[section].minimum + personalities[currentPersonality].speedSection[section].maximum) / 2);
        if (percentageVelocity < 1) sectionReward -= (1 - percentageVelocity) * personalities[currentPersonality].speedImportance;
        else sectionReward += percentageVelocity * personalities[currentPersonality].speedImportance;

        currentScore += width;

        // Concentration
        currentConcentration += personalities[currentPersonality].concentration[section];
        sectionReward += (1 / Mathf.Abs(personalities[currentPersonality].concentrationLevelPrefered - currentConcentration)) * personalities[currentPersonality].concentrationImportance;

        // Skill
        currentSkill += personalities[currentPersonality].skill[section];
        sectionReward += (1 / Mathf.Abs(personalities[currentPersonality].skillLevelPrefered - currentSkill)) * personalities[currentPersonality].skillImportance;

        // Challenge
        currentChallenge += personalities[currentPersonality].challenge[section];
        sectionReward += (1 / Mathf.Abs(personalities[currentPersonality].challengeLevelPrefered - currentChallenge)) * personalities[currentPersonality].challengeImportance;

        // Immersion
        currentImmersion += personalities[currentPersonality].immersion[section];
        sectionReward += (1 / Mathf.Abs(personalities[currentPersonality].immersionLevelPrefered - currentImmersion)) * personalities[currentPersonality].immersionImportance;

        return sectionReward;
    }

    private void Start()
    {
        this.maxScorePersonalities = new List<float>();
        for (int i = 0; i < this.personalities.Count; i++)
        {
            this.maxScorePersonalities.Add(0);
        }
        resetGame();
    }

    public void nextSection(int section)
    {
        currentSection = section;
        waitingForNewSection = false;
    }

    public void resetGame()
    {
        currentLives = 3;
        currentScore = 0;
        currentConcentration = 0;
        currentSkill = 0;
        currentChallenge = 0;
        currentImmersion = 0;
        endGame = false;
        firstSection = true;
        waitingForNewSection = false;

        currentPersonality = Random.Range(0, personalities.Count);
        int percentageResetMaxScore = Random.Range(0, 100);
        if (percentageResetMaxScore >= 80) maxScorePersonalities[currentPersonality] = 0;
    }

    private void Update()
    {
        if (!waitingForNewSection)
        {
            if (firstSection)
            {
                this.ai.RequestDecision();
                waitingForNewSection = true;
            }
            else
            {
                float reward = rewardOfPersonality(currentSection, blocks[currentSection].Width);
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
