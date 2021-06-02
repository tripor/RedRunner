using System.Collections;
using System.Collections.Generic;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;

public class AdaptivityAIGame : Agent
{
    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(RedRunner.GameManager.Singleton.bestScore);
        sensor.AddObservation(RedRunner.GameManager.Singleton.currentScore);
        sensor.AddObservation(RedRunner.GameManager.Singleton.CurrentLives);
        sensor.AddObservation(RedRunner.GameManager.Singleton.LivesLostWaterUser);
        sensor.AddObservation(RedRunner.GameManager.Singleton.LivesLostMovingUser);
        sensor.AddObservation(RedRunner.GameManager.Singleton.LivesLostStaticUser);
        sensor.AddObservation(RedRunner.GameManager.Singleton.LivesLostUser);
        sensor.AddObservation(RedRunner.GameManager.Singleton.LivesLostGame);
        sensor.AddObservation(RedRunner.GameManager.Singleton.LivesLostWaterGame);
        sensor.AddObservation(RedRunner.GameManager.Singleton.LivesLostStaticGame);
        sensor.AddObservation(RedRunner.GameManager.Singleton.LivesLostMovingGame);
        sensor.AddObservation(RedRunner.GameManager.Singleton.CoinsUser);
        sensor.AddObservation(RedRunner.GameManager.Singleton.ChestUser);
        sensor.AddObservation(RedRunner.GameManager.Singleton.CoinsGame);
        sensor.AddObservation(RedRunner.GameManager.Singleton.ChestGame);
        sensor.AddObservation(RedRunner.GameManager.Singleton.AverageVelocityGame);
        sensor.AddObservation(RedRunner.GameManager.Singleton.AverageVelocityUser);
        sensor.AddObservation(RedRunner.GameManager.Singleton.VarianceVelocityGame);
        sensor.AddObservation(RedRunner.GameManager.Singleton.StandardDeviationVelocityGame);
        sensor.AddObservation(RedRunner.GameManager.Singleton.VarianceVelocityUser);
        sensor.AddObservation(RedRunner.GameManager.Singleton.StandardDeviationVelocityUser);
        sensor.AddObservation(RedRunner.GameManager.Singleton.JumpsGame);
        sensor.AddObservation(RedRunner.GameManager.Singleton.BacktracksGame);
        sensor.AddObservation(RedRunner.GameManager.Singleton.totalGameTime);
        if (RedRunner.GameManager.Singleton.CurrentSection == -1)
            sensor.AddObservation(RedRunner.GameManager.Singleton.CurrentSection);
        else
            sensor.AddObservation(RedRunner.GameManager.Singleton.CurrentSection - 1);
    }
    public override void OnActionReceived(ActionBuffers actions)
    {
        int value = actions.DiscreteActions[0];
        RedRunner.TerrainGeneration.TerrainGenerator.Singleton.GenerateMiddle(value);
    }

    public override void OnEpisodeBegin()
    {
    }

}
