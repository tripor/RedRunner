using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using UnityStandardAssets.CrossPlatformInput;

namespace GameAI
{
    public class PlayerWatcher : Agent
    {
        private bool is_game_running = false;
        [SerializeField]
        private RedRunner.Characters.Character m_character;

        private int jump = 0;
        private float horizontal_movement = 0;

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
        }
        public override void OnActionReceived(ActionBuffers actions)
        {
            float movement = actions.ContinuousActions[0];
            int jump = actions.DiscreteActions[0];
            if (m_character.PlayerAiTraining)
            {
                m_character.Move(movement);
                if (jump == 1) m_character.Jump();
            }
        }

        public override void Heuristic(in ActionBuffers actionsOut)
        {
            var da = actionsOut.DiscreteActions;
            da[0] = this.jump;
            this.jump = 0;
            var ca = actionsOut.ContinuousActions;
            ca[0] = this.horizontal_movement;
            this.horizontal_movement = 0;
        }

        public void endGame()
        {
            EndEpisode();
        }

        public override void OnEpisodeBegin()
        {
            Debug.Log("Episode Begin");
            RedRunner.GameManager.Singleton.EndGame(false);
            RedRunner.GameManager.Singleton.Reset();
            if (m_character.PlayerAiTraining)
                RedRunner.GameManager.Singleton.StartGame();
        }
        private void Update()
        {
            is_game_running = RedRunner.GameManager.Singleton.gameRunning;
            if (is_game_running)
            {
                if (!m_character.PlayerAiTraining)
                {
                    float input_horizontal = CrossPlatformInputManager.GetAxis("Horizontal");
                    if (Mathf.Abs(input_horizontal) > Mathf.Abs(this.horizontal_movement))
                        this.horizontal_movement = input_horizontal;
                    m_character.Move(input_horizontal);
                    if (CrossPlatformInputManager.GetButtonDown("Jump"))
                    {
                        this.jump = 1;
                        m_character.Jump();
                    }
                    else
                    {
                        if (this.jump != 1)
                            this.jump = 0;
                    }
                }
                RequestDecision();
            }
        }
    }
}

