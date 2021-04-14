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

        private float time = 0;
        private bool jump_once = false;

        public override void Initialize()
        {
            Debug.Log("Init");
            is_game_running = false;
            time = 0;
        }

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
            }
        }

        public override void Heuristic(in ActionBuffers actionsOut)
        {
            var da = actionsOut.DiscreteActions;
            if (!m_character.GroundCheck.IsGrounded) jump_once = false;
            float input_horizontal = 0;
            if (Input.GetKey(KeyCode.RightArrow) || Input.GetKey(KeyCode.D))
                input_horizontal = 1;
            else if (Input.GetKey(KeyCode.LeftArrow) || Input.GetKey(KeyCode.A))
                input_horizontal = -1;
            int jump = 0;
            if ((Input.GetKey(KeyCode.UpArrow) || Input.GetKey(KeyCode.W)) && m_character.GroundCheck.IsGrounded && !jump_once)
            {
                jump_once = true;
                jump = 1;
            }
            int value = 0;
            if (jump == 1 && input_horizontal < 0)
            {
                value = 4;
            }
            else if (jump == 1 && input_horizontal > 0)
            {
                value = 5;
            }
            else if (jump == 0 && input_horizontal < 0)
            {
                value = 1;
            }
            else if (jump == 0 && input_horizontal > 0)
            {
                value = 2;
            }
            else if (jump == 1)
            {
                value = 3;
            }
            da[0] = value;
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
            time = 0;
        }

        public void incrementReward(float increment)
        {
            this.AddReward(increment);
        }
        private void Update()
        {
            is_game_running = RedRunner.GameManager.Singleton.gameRunning;
            this.transform.position = m_character.transform.position;
            if (is_game_running)
            {
                if (Mathf.Abs(m_character.Rigidbody2D.velocity.x) < 1)
                {
                    time += Time.deltaTime;
                    if (time > 7)
                    {
                        this.incrementReward(-0.05f);
                    }
                }
                else time = 0;
            }
        }
    }
}

