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
        // sensor.AddObservation(RedRunner.GameManager.Singleton.m_Coin.Value); information from the past
        //sensor.AddObservation(RedRunner.GameManager.Singleton.coinsGame); // current collected coins in game
        //sensor.AddObservation(m_character.Lives); // current lives
        //sensor.AddObservation(RedRunner.GameManager.Singleton.currentScore); // current score
        // sensor.AddObservation(RedRunner.GameManager.Singleton.bestScore); information from the past
        //sensor.AddObservation(RedRunner.GameManager.Singleton.currentGameTime); // current game time
        //sensor.AddObservation(RedRunner.GameManager.Singleton.CurrentSectionIdentifier); // current section identifier
        //sensor.AddObservation(RedRunner.GameManager.Singleton.CurrentSectionPosition); // current position on section
        //sensor.AddObservation(m_character.Rigidbody2D.velocity.x); // current velocity
    }
    public override void OnActionReceived(ActionBuffers actions)
    {
        /*
        int value = actions.DiscreteActions[0];
        switch (value)
        {
            case 0:
                m_character.Move(0);
                break;
            case 1:
                m_character.Move(-1);
                break;
            case 2:
                m_character.Move(1);
                break;
            case 3:
                Debug.Log("Jump");
                m_character.Jump();
                break;
            case 4:
                Debug.Log("Jump");
                m_character.Move(-1);
                m_character.Jump();
                break;
            case 5:
                Debug.Log("Jump");
                m_character.Move(1);
                m_character.Jump();
                break;
            default:
                break;
        }*/
    }

    public override void OnEpisodeBegin()
    {
        /*
        Debug.Log("Episode Begin");
        RedRunner.GameManager.Singleton.EndGame(false);
        RedRunner.GameManager.Singleton.Reset();
        if (m_character.PlayerAiTraining)
            RedRunner.GameManager.Singleton.StartGame();
        time = 0;
        */
    }
}
