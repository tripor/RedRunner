using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class AdaptivityAI : Agent
{

    public override void CollectObservations(VectorSensor sensor)
    {
        int currentPersonality = PersonalitySimulator.Instance.currentPersonality;
        sensor.AddObservation(PersonalitySimulator.Instance.maxScorePersonalities[currentPersonality]);
        sensor.AddObservation(PersonalitySimulator.Instance.currentScore);
        sensor.AddObservation(PersonalitySimulator.Instance.currentLives);
        sensor.AddObservation(PersonalitySimulator.Instance.lifeLostWaterPersonalities[currentPersonality]);
        sensor.AddObservation(PersonalitySimulator.Instance.lifeLostMovingPersonalities[currentPersonality]);
        sensor.AddObservation(PersonalitySimulator.Instance.lifeLostStaticPersonalities[currentPersonality]);
        sensor.AddObservation(PersonalitySimulator.Instance.lifeLostPersonalities[currentPersonality]);
        sensor.AddObservation(PersonalitySimulator.Instance.lastSectionLifeLost);
        sensor.AddObservation(PersonalitySimulator.Instance.lastSectionLifeLostWater);
        sensor.AddObservation(PersonalitySimulator.Instance.lastSectionLifeLostStatic);
        sensor.AddObservation(PersonalitySimulator.Instance.lastSectionLifeLostMoving);
        sensor.AddObservation(PersonalitySimulator.Instance.coinsPersonalities[currentPersonality]);
        sensor.AddObservation(PersonalitySimulator.Instance.chestsPersonalities[currentPersonality]);
        sensor.AddObservation(PersonalitySimulator.Instance.coinsGame);
        sensor.AddObservation(PersonalitySimulator.Instance.chestGame);
        sensor.AddObservation(PersonalitySimulator.Instance.averageVelocityGames);
        sensor.AddObservation(PersonalitySimulator.Instance.averageVelocityPersonalities[currentPersonality]);
        if (PersonalitySimulator.Instance.averageVelocityGames_i > 1)
        {
            float variance = PersonalitySimulator.Instance.m2VelocityGames / PersonalitySimulator.Instance.averageVelocityGames_i;
            if (!float.IsNaN(variance))
            {
                sensor.AddObservation(variance);
                sensor.AddObservation(Mathf.Sqrt(variance));
            }
            else
            {
                sensor.AddObservation(0);
                sensor.AddObservation(0);
            }
        }
        else
        {
            sensor.AddObservation(0);
            sensor.AddObservation(0);
        }
        if (PersonalitySimulator.Instance.averageVelocityPersonalities_i[currentPersonality] > 1)
        {
            float variance = PersonalitySimulator.Instance.m2VelocityPersonalities[currentPersonality] / PersonalitySimulator.Instance.averageVelocityPersonalities_i[currentPersonality];
            if (!float.IsNaN(variance))
            {
                sensor.AddObservation(variance);
                sensor.AddObservation(Mathf.Sqrt(variance));
            }
            else
            {
                sensor.AddObservation(0);
                sensor.AddObservation(0);
            }
        }
        else
        {
            sensor.AddObservation(0);
            sensor.AddObservation(0);
        }
        sensor.AddObservation(PersonalitySimulator.Instance.jumpsGame);
        sensor.AddObservation(PersonalitySimulator.Instance.backtracksGame);
        sensor.AddObservation(PersonalitySimulator.Instance.timeSpent);
        sensor.AddObservation(PersonalitySimulator.Instance.currentSection);
    }
    public override void OnActionReceived(ActionBuffers actions)
    {
        int value = actions.DiscreteActions[0];
        PersonalitySimulator.Instance.nextSection(value);
    }

    public override void OnEpisodeBegin()
    {
        PersonalitySimulator.Instance.resetGame();
    }
}
