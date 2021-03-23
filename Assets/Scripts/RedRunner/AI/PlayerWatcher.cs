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
            int value = actions.DiscreteActions[0];
            if (m_character.PlayerAiTraining)
            {
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
                        m_character.Jump();
                        break;
                    case 4:
                        m_character.Move(-1);
                        m_character.Jump();
                        break;
                    case 5:
                        m_character.Move(1);
                        m_character.Jump();
                        break;
                    default:
                        break;
                }
            }
        }

        public override void Heuristic(in ActionBuffers actionsOut)
        {
            var da = actionsOut.DiscreteActions;
            int value = 0;
            if (this.jump == 1 && this.horizontal_movement < 0)
            {
                value = 4;
            }
            else if (this.jump == 1 && this.horizontal_movement > 0)
            {
                value = 5;
            }
            else if (this.jump == 0 && this.horizontal_movement < 0)
            {
                value = 1;
            }
            else if (this.jump == 0 && this.horizontal_movement > 0)
            {
                value = 2;
            }
            else if (this.jump == 1)
            {
                value = 3;
            }
            da[0] = value;
            this.jump = 0;
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

