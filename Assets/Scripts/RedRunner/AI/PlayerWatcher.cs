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
        public override void Initialize()
        {
            Debug.Log("Init");
            is_game_running = false;
        }

        public override void CollectObservations(VectorSensor sensor)
        {
            base.CollectObservations(sensor);
        }

        public override void OnActionReceived(float[] vectorAction)
        {
            base.OnActionReceived(vectorAction);
        }

        public override void Heuristic(float[] actionsOut)
        {
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

