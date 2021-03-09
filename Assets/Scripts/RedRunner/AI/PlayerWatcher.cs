using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;

namespace GameAI
{
    public class PlayerWatcher : Agent
    {
        private bool is_game_running = false;
        [SerializeField]
        private RedRunner.Characters.Character m_character;
        public override void Initialize()
        {
            Debug.Log("Init");
            is_game_running = false;
        }

        public override void CollectObservations(VectorSensor sensor)
        {
            sensor.AddObservation(RedRunner.GameManager.Singleton.m_Coin.Value);
            sensor.AddObservation(RedRunner.GameManager.Singleton.coinsGame);
            sensor.AddObservation(m_character.Lives);
            sensor.AddObservation(RedRunner.GameManager.Singleton.currentScore);
            sensor.AddObservation(RedRunner.GameManager.Singleton.bestScore);
            sensor.AddObservation(RedRunner.GameManager.Singleton.currentGameTime);
            if (m_character.InputType == -1) sensor.AddObservation(0);
            else sensor.AddObservation(m_character.InputType);
            m_character.InputType = -1;
        }

        public override void OnActionReceived(float[] vectorAction)
        {
            base.OnActionReceived(vectorAction);
        }

        public override void Heuristic(float[] actionsOut)
        {
        }

        public void endGame()
        {
            EndEpisode();
        }

        public override void OnEpisodeBegin()
        {
            Debug.Log("Episode Begin");
        }
        private void Update()
        {
            is_game_running = RedRunner.GameManager.Singleton.gameRunning;
            if (is_game_running)
            {
                RequestDecision();
            }
        }
    }
}

